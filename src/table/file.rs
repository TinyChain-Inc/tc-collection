use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use b_table::{Range, Row, TableLock};
use collate::{try_diff, try_merge, Collate, Collator as TxnCollator};
use collate::Collator;
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_ir::{Id, Transact, TxnId};
use tc_value::Value;

use super::schema::{TableIndexSchema, TableSchema};
use super::stream::Rows;
use super::view::{Limited, Selection, TableSlice};
use crate::btree::{StorageConfig, PersistentFile};

fn background_error(err: impl fmt::Display) -> txn_lock::Error {
    txn_lock::Error::Background(err.to_string())
}

/// Collator for merging row streams that are ordered by specific column indices.
///
/// When `order` is empty (natural primary-key order), `indices` is set to
/// `[0, 1, ..., key_len-1]` so the collator compares the primary-key prefix.
/// When `order` is non-empty, `indices` holds the positions of the order
/// columns within the row (which is always in primary-column order: key
/// columns followed by value columns).
#[derive(Clone, Debug, Eq, PartialEq)]
struct RowCollator {
    indices: Vec<usize>,
    reverse: bool,
}

impl RowCollator {
    /// Build a collator for the given schema and order specification.
    ///
    /// If `order` is empty, the collator compares by the primary key columns
    /// (indices `0..key_len`). Otherwise, it compares by the positions of the
    /// named order columns within the full row (key + value columns).
    fn for_order(schema: &TableSchema, order: &[Id], reverse: bool) -> Self {
        let key = schema.key();
        let values = schema.values();
        let all: Vec<&Id> = key.iter().chain(values.iter()).collect();

        let indices: Vec<usize> = if order.is_empty() {
            (0..key.len()).collect()
        } else {
            order
                .iter()
                .filter_map(|col| all.iter().position(|name| *name == col))
                .collect()
        };

        Self { indices, reverse }
    }
}

impl Collate for RowCollator {
    type Value = Vec<Value>;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> std::cmp::Ordering {
        let mut ord = std::cmp::Ordering::Equal;
        for &i in &self.indices {
            let l = left.get(i);
            let r = right.get(i);
            ord = match (l, r) {
                (Some(l), Some(r)) => l.cmp(r),
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            };
            if ord != std::cmp::Ordering::Equal {
                break;
            }
        }
        if self.reverse {
            ord.reverse()
        } else {
            ord
        }
    }
}

type TableVersion = TableLock<TableSchema, TableIndexSchema, Collator<Value>, PersistentFile>;

#[derive(Clone)]
struct TableStore {
    table: TableVersion,
}

impl TableStore {
    fn schema(&self) -> &TableSchema {
        self.table.schema()
    }

    fn from_dir(
        dir: DirLock<PersistentFile>,
        schema: TableSchema,
        _storage: StorageConfig,
    ) -> std::io::Result<Self> {
        let collator = Collator::<Value>::default();
        let table = TableLock::load(schema, collator, dir)?;
        Ok(Self { table })
    }

    async fn sync(&self) -> std::io::Result<()> {
        self.table.sync().await
    }

    async fn row_stream(
        &self,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
    ) -> std::io::Result<BoxStream<'static, Result<Row<Value>, std::io::Error>>> {
        let view = self.table.read().await;
        let rows = view.rows(range, order, reverse, None).await?;
        Ok(Box::pin(rows))
    }

    async fn get_row(&self, key: &[Value]) -> std::io::Result<Option<Row<Value>>> {
        let schema = self.schema().clone();
        let range = schema.range_from_key(key).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, e)
        })?;
        let view = self.table.read().await;
        let mut rows = view.rows(range, &[], false, None).await?;
        rows.try_next().await
    }

    async fn upsert_row(&self, key: Vec<Value>, values: Vec<Value>) -> std::io::Result<()> {
        let mut view = self.table.write().await;
        view.upsert(key, values)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    async fn delete_row(&self, key: &[Value]) -> std::io::Result<()> {
        let mut view = self.table.write().await;
        view.delete_row(key).await?;
        Ok(())
    }

    async fn apply_delta(&self, delta: &Delta) -> std::io::Result<()> {
        {
            let inserts_read = delta.inserts.table.read().await;
            let mut view = self.table.write().await;
            view.merge(inserts_read).await?;
        }

        {
            let deletes_read = delta.deletes.table.read().await;
            let mut view = self.table.write().await;
            view.delete_all(deletes_read).await?;
        }

        Ok(())
    }
}

#[derive(Clone)]
struct Delta {
    inserts: TableStore,
    deletes: TableStore,
}

impl Delta {
    async fn upsert(&self, key: Vec<Value>, values: Vec<Value>) -> std::io::Result<()> {
        self.deletes.delete_row(&key).await?;
        self.inserts.delete_row(&key).await?;
        self.inserts.upsert_row(key, values).await
    }

    async fn delete_from_inserts(&self, key: &[Value]) -> std::io::Result<()> {
        self.inserts.delete_row(key).await
    }

    async fn add_to_deletes(&self, key: &[Value], values: Vec<Value>) -> std::io::Result<()> {
        self.deletes.upsert_row(key.to_vec(), values).await
    }

    async fn get_inserted_row(&self, key: &[Value]) -> std::io::Result<Option<Row<Value>>> {
        self.inserts.get_row(key).await
    }

    async fn already_deleted(&self, key: &[Value]) -> std::io::Result<bool> {
        self.deletes.get_row(key).await.map(|row| row.is_some())
    }

    async fn merge_into<'a>(
        &'a self,
        rows: BoxStream<'a, Result<Vec<Value>, std::io::Error>>,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
        collator: RowCollator,
    ) -> BoxStream<'a, Result<Vec<Value>, std::io::Error>> {
        let inserted = self
            .inserts
            .row_stream(range.clone(), order, reverse)
            .await
            .expect("stream insert delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        let merged = try_merge(collator.clone(), inserted, rows).boxed();

        let deleted = self
            .deletes
            .row_stream(range, order, reverse)
            .await
            .expect("stream delete delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        try_diff(collator, merged, deleted).boxed()
    }

    /// Owned variant of [`merge_into`](Self::merge_into) that produces a
    /// `'static` stream suitable for returning from `rows()`.
    ///
    /// `self` is consumed (Delta is `Clone`) so the returned stream does not
    /// borrow from the caller's stack.
    async fn merge_into_owned(
        self,
        rows: BoxStream<'static, Result<Vec<Value>, std::io::Error>>,
        range: Range<tc_ir::Id, Value>,
        order: Vec<tc_ir::Id>,
        reverse: bool,
        collator: RowCollator,
    ) -> BoxStream<'static, Result<Vec<Value>, std::io::Error>> {
        let inserted = self
            .inserts
            .row_stream(range.clone(), &order, reverse)
            .await
            .expect("stream insert delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        let merged = try_merge(collator.clone(), inserted, rows).boxed();

        let deleted = self
            .deletes
            .row_stream(range, &order, reverse)
            .await
            .expect("stream delete delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        try_diff(collator, merged, deleted).boxed()
    }
}

#[derive(Clone)]
struct State {
    persistent: TableStore,
    committed: BTreeMap<TxnId, Delta>,
    pending: BTreeMap<TxnId, Delta>,
    finalized: Option<TxnId>,
    txn_root: DirLock<PersistentFile>,
}

#[derive(Clone)]
struct VisibleSnapshot {
    persistent: TableStore,
    deltas: Vec<Delta>,
}

#[derive(Clone)]
pub struct PersistentTable {
    state: Arc<RwLock<State>>,
    semaphore: txn_lock::semaphore::Semaphore<
        TxnId,
        TxnCollator<Vec<Value>>,
        txn_lock::set::Range<Vec<Value>>,
    >,
    schema: TableSchema,
}

impl fmt::Debug for PersistentTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read().expect("state read lock");
        f.debug_struct("PersistentTable")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl PersistentTable {
    pub fn new(
        persistent_dir: DirLock<PersistentFile>,
        txn_root: DirLock<PersistentFile>,
        schema: TableSchema,
    ) -> Self {
        let persistent = Self::load_store(persistent_dir, schema.clone());

        let state = State {
            persistent,
            committed: BTreeMap::new(),
            pending: BTreeMap::new(),
            finalized: None,
            txn_root,
        };

        Self {
            state: Arc::new(RwLock::new(state)),
            semaphore: txn_lock::semaphore::Semaphore::new(TxnCollator::default()),
            schema,
        }
    }

    pub fn schema(&self) -> &TableSchema {
        &self.schema
    }

    pub fn finalized(&self) -> Option<TxnId> {
        self.state.read().expect("state read lock").finalized
    }

    /// Sync the canonical (persistent) state to disk.
    ///
    /// This flushes any in-memory modifications to the filesystem so they
    /// survive a restart. Pending and committed deltas are not synced —
    /// Chain owns replay durability for those.
    pub async fn sync(&self) -> std::io::Result<()> {
        let persistent = {
            let state = self.state.read().expect("state read lock");
            state.persistent.clone()
        };
        persistent.sync().await
    }

    pub async fn upsert_row(
        &self,
        txn_id: TxnId,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> Result<(), txn_lock::Error> {
        let key = b_table::Schema::validate_key(&self.schema, key).map_err(background_error)?;
        let values = b_table::Schema::validate_values(&self.schema, values).map_err(background_error)?;

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn_id).await?;
        pending.upsert(key, values).await.map_err(background_error)?;

        Ok(())
    }

    pub async fn delete_row(
        &self,
        txn_id: TxnId,
        key: Vec<Value>,
    ) -> Result<(), txn_lock::Error> {
        let key = b_table::Schema::validate_key(&self.schema, key).map_err(background_error)?;

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn_id).await?;

        if pending.already_deleted(&key).await.map_err(background_error)? {
            return Ok(());
        }

        let mut row = pending.get_inserted_row(&key).await.map_err(background_error)?;

        if row.is_none() {
            let snapshot = self.visible_snapshot(txn_id);
            row = self.resolve_row(&snapshot, &key).await;
        }

        pending.delete_from_inserts(&key).await.map_err(background_error)?;

        if let Some(mut row) = row {
            let key_len = self.schema.key().len();
            let values: Vec<Value> = row.drain(key_len..).collect();
            pending
                .add_to_deletes(&key, values)
                .await
                .map_err(background_error)?;
        }

        Ok(())
    }

    pub async fn read_row(&self, txn_id: TxnId, key: &[Value]) -> Option<Row<Value>> {
        let _permit = self
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await;

        let snapshot = self.visible_snapshot(txn_id);
        self.resolve_row(&snapshot, key).await
    }

    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> bool {
        let _permit = self
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await;

        let snapshot = self.visible_snapshot(txn_id);
        self.is_row_visible(&snapshot, key).await
    }

    pub async fn count(&self, txn_id: TxnId) -> u64 {
        let mut count = 0_u64;
        self.for_each_row_in_order(txn_id, Range::default(), &[], false, |_| {
            count += 1;
        })
        .await;
        count
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> bool {
        !self
            .any_row_in(txn_id, Range::default(), &[], false)
            .await
    }

    pub async fn for_each_row_in_order<F>(
        &self,
        txn_id: TxnId,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
        mut on_row: F,
    ) where
        F: FnMut(Row<Value>),
    {
        let _permit = self
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await;

        let collator = RowCollator::for_order(&self.schema, order, reverse);

        self.for_each_visible_row_until(txn_id, range, order, reverse, collator, |row| {
            on_row(row);
            true
        })
        .await;
    }

    pub async fn count_in(
        &self,
        txn_id: TxnId,
        range: Range<tc_ir::Id, Value>,
    ) -> u64 {
        let mut count = 0_u64;
        self.for_each_row_in_order(txn_id, range, &[], false, |_| {
            count += 1;
        })
        .await;
        count
    }

    /// Return `true` if there are no visible rows in `range` at `txn_id`.
    pub async fn is_empty_in(
        &self,
        txn_id: TxnId,
        range: Range<tc_ir::Id, Value>,
    ) -> bool {
        !self.any_row_in(txn_id, range, &[], false).await
    }

    /// Construct a permit-bound row stream over the visible state at `txn_id`.
    ///
    /// The stream is fully lazy — rows are produced on demand by polling the
    /// returned [`Rows`]. The read permit is held for the lifetime of the
    /// stream so the transactional snapshot stays coherent.
    ///
    /// If the given `range` is not supported by any index, this returns an
    /// `Unsupported` I/O error wrapped in a background transactional error.
    pub async fn rows(
        &self,
        txn_id: TxnId,
        range: Range<Id, Value>,
        order: Vec<Id>,
        reverse: bool,
    ) -> Result<Rows, txn_lock::Error> {
        let permit = self
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await;

        let snapshot = self.visible_snapshot(txn_id);
        let collator = RowCollator::for_order(&self.schema, &order, reverse);

        let mut visible: BoxStream<'static, Result<Vec<Value>, std::io::Error>> = snapshot
            .persistent
            .row_stream(range.clone(), &order, reverse)
            .await
            .map_err(background_error)?
            .map_ok(|row| row.to_vec())
            .boxed();

        for delta in snapshot.deltas.into_iter() {
            visible = delta
                .merge_into_owned(visible, range.clone(), order.clone(), reverse, collator.clone())
                .await;
        }

        let stream = visible.map_ok(Row::from_vec).boxed();
        Ok(Rows::new(stream, permit))
    }

    /// Create a range + order + reverse view over this table.
    ///
    /// The view is structural — it holds no row data. Row streaming, count,
    /// and containment checks delegate to this table with the view's bounds.
    pub fn slice(
        &self,
        range: Range<Id, Value>,
        order: &[Id],
        reverse: bool,
    ) -> TableSlice {
        TableSlice::new(self.clone(), range, order.to_vec(), reverse)
    }

    /// Create an ordered view over this table using the given `columns`.
    ///
    /// Equivalent to `slice(Range::default(), columns, reverse)`.
    pub fn order_by(&self, columns: &[Id], reverse: bool) -> TableSlice {
        self.slice(Range::default(), columns, reverse)
    }

    /// Create a row-cap view that yields at most `n` rows.
    pub fn limit(&self, n: u64) -> Limited {
        self.slice(Range::default(), &[], false).limit(n)
    }

    /// Create a column-projection view that yields only `columns`.
    pub fn select(&self, columns: &[Id]) -> Selection {
        self.slice(Range::default(), &[], false)
            .select(columns.to_vec())
    }

    /// Update all visible rows in `range` at `txn_id` with the given column
    /// `values`.
    ///
    /// Only value columns (not key columns) may be updated.  Rows are streamed
    /// from the visible snapshot, updated in-place, and upserted into the
    /// pending delta — the full affected set is never buffered in a `Vec`
    /// (v1 no-materialization invariant).  A single `Range::All` write permit
    /// is acquired upfront so no per-key semaphore re-acquisition is needed
    /// during the streamed update loop.
    ///
    /// Ported from v1 `TableFile::update`.
    pub async fn update(
        &self,
        txn_id: TxnId,
        range: Range<Id, Value>,
        values: tc_ir::Map<Value>,
    ) -> Result<(), txn_lock::Error> {
        let value_columns = self.schema.values();
        for name in values.keys() {
            if !value_columns.contains(name) {
                return Err(background_error(format!(
                    "cannot update key column {name}"
                )));
            }
        }

        let key_len = self.schema.key().len();
        let collator = RowCollator::for_order(&self.schema, &[], false);

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn_id).await?;

        let snapshot = self.visible_snapshot(txn_id);

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> = snapshot
            .persistent
            .row_stream(range.clone(), &[], false)
            .await
            .map_err(background_error)?
            .map_ok(|row| row.to_vec())
            .boxed();

        for delta in &snapshot.deltas {
            visible = delta
                .merge_into(visible, range.clone(), &[], false, collator.clone())
                .await;
        }

        while let Some(mut row) = visible.try_next().await.map_err(background_error)? {
            for (i, name) in value_columns.iter().enumerate() {
                if let Some(value) = values.get(name) {
                    row[key_len + i] = value.clone();
                }
            }

            let key: Vec<Value> = row[..key_len].to_vec();
            let updated_values: Vec<Value> = row[key_len..].to_vec();

            pending.upsert(key, updated_values).await.map_err(background_error)?;
        }

        Ok(())
    }

    /// Copy all rows from `source` into this table at `txn_id`.
    ///
    /// Rows are streamed from the source table's visible snapshot and upserted
    /// one-by-one into this table's pending delta — no full source is ever
    /// buffered (v1 no-materialization invariant).
    ///
    /// The source table must have a compatible schema (same key and value
    /// column count and types).
    ///
    /// Ported from v1 `CopyFrom<FE, T> for TableFile`.
    pub async fn copy_from(
        &self,
        txn_id: TxnId,
        source: &PersistentTable,
    ) -> Result<(), txn_lock::Error> {
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn_id).await?;

        let key_len = self.schema.key().len();

        let mut rows = source
            .rows(txn_id, Range::default(), Vec::new(), false)
            .await?;

        while let Some(row) = rows.try_next().await.map_err(background_error)? {
            let row_vec = row.into_vec();
            let key: Vec<Value> = row_vec[..key_len].to_vec();
            let values: Vec<Value> = row_vec[key_len..].to_vec();
            pending.upsert(key, values).await.map_err(background_error)?;
        }

        Ok(())
    }

    /// Delete all visible rows in `range` at `txn_id`.
    ///
    /// Rows are streamed and deleted one-by-one into the pending delta — the
    /// full affected set is never buffered in a `Vec` (v1 no-materialization
    /// invariant). A single `Range::All` write permit is acquired upfront so
    /// no per-key semaphore re-acquisition is needed during the streamed
    /// delete loop.
    pub async fn truncate(
        &self,
        txn_id: TxnId,
        range: Range<Id, Value>,
    ) -> Result<(), txn_lock::Error> {
        let key_len = self.schema.key().len();
        let collator = RowCollator::for_order(&self.schema, &[], false);

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn_id).await?;

        let snapshot = self.visible_snapshot(txn_id);

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> = snapshot
            .persistent
            .row_stream(range.clone(), &[], false)
            .await
            .expect("stream persistent rows for truncate")
            .map_ok(|row| row.to_vec())
            .boxed();

        for delta in &snapshot.deltas {
            visible = delta
                .merge_into(visible, range.clone(), &[], false, collator.clone())
                .await;
        }

        while let Some(row) = visible.try_next().await.expect("read truncate stream") {
            let key: Vec<Value> = row[..key_len].to_vec();
            let values: Vec<Value> = row[key_len..].to_vec();

            pending
                .delete_from_inserts(&key)
                .await
                .map_err(background_error)?;
            pending
                .add_to_deletes(&key, values)
                .await
                .map_err(background_error)?;
        }

        Ok(())
    }

    async fn any_row_in(
        &self,
        txn_id: TxnId,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
    ) -> bool {
        let collator = RowCollator::for_order(&self.schema, order, reverse);

        let mut found = false;
        self.for_each_visible_row_until(txn_id, range, order, reverse, collator, |_| {
            found = true;
            false
        })
        .await;

        found
    }

    async fn for_each_visible_row_until<F>(
        &self,
        txn_id: TxnId,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
        collator: RowCollator,
        mut on_row: F,
    ) where
        F: FnMut(Row<Value>) -> bool,
    {
        let snapshot = self.visible_snapshot(txn_id);

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> = snapshot
            .persistent
            .row_stream(range.clone(), order, reverse)
            .await
            .expect("stream persistent rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        for delta in &snapshot.deltas {
            visible = delta
                .merge_into(visible, range.clone(), order, reverse, collator.clone())
                .await;
        }

        while let Some(row) = visible.try_next().await.expect("read visible row stream") {
            let row = Row::from_vec(row);
            if !on_row(row) {
                break;
            }
        }
    }

    fn visible_snapshot(&self, txn_id: TxnId) -> VisibleSnapshot {
        let state = self.state.read().expect("state read lock");
        let mut deltas = state
            .committed
            .iter()
            .filter_map(|(id, delta)| (*id <= txn_id).then_some(delta.clone()))
            .collect::<Vec<_>>();

        if let Some(delta) = state.pending.get(&txn_id).cloned() {
            deltas.push(delta);
        }

        VisibleSnapshot {
            persistent: state.persistent.clone(),
            deltas,
        }
    }

    async fn acquire_read_permit(
        &self,
        txn_id: TxnId,
        range: txn_lock::set::Range<Vec<Value>>,
    ) -> txn_lock::semaphore::PermitRead<txn_lock::set::Range<Vec<Value>>> {
        self.semaphore
            .read(txn_id, range)
            .await
            .expect("acquire read permit")
    }

    #[inline]
    fn release_txn_reservation(&self, txn_id: TxnId) {
        self.semaphore.finalize(&txn_id, false);
    }

    #[inline]
    fn release_txn_frontier(&self, txn_id: TxnId) {
        self.semaphore.finalize(&txn_id, true);
    }

    async fn resolve_row(
        &self,
        snapshot: &VisibleSnapshot,
        key: &[Value],
    ) -> Option<Row<Value>> {
        let mut row = snapshot
            .persistent
            .get_row(key)
            .await
            .expect("check persistent visibility");

        for delta in &snapshot.deltas {
            if delta
                .deletes
                .get_row(key)
                .await
                .expect("check delete delta")
                .is_some()
            {
                row = None;
            }

            if let Some(inserted) = delta
                .inserts
                .get_row(key)
                .await
                .expect("check insert delta")
            {
                row = Some(inserted);
            }
        }

        row
    }

    async fn is_row_visible(&self, snapshot: &VisibleSnapshot, key: &[Value]) -> bool {
        self.resolve_row(snapshot, key).await.is_some()
    }

    fn assert_writable_state(state: &State, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
            return Err(txn_lock::Error::Outdated);
        }

        if state.committed.contains_key(&txn_id) {
            return Err(txn_lock::Error::Committed);
        }

        Ok(())
    }

    fn load_store(
        persistent_dir: DirLock<PersistentFile>,
        schema: TableSchema,
    ) -> TableStore {
        let storage = schema.storage().clone();
        TableStore::from_dir(persistent_dir, schema, storage)
            .expect("load persistent Table store")
    }

    async fn pending_delta_for_txn(&self, txn_id: TxnId) -> Result<Delta, txn_lock::Error> {
        let (txn_root, schema, storage) = {
            let state = self.state.write().expect("state write lock");
            Self::assert_writable_state(&state, txn_id)?;

            if let Some(pending) = state.pending.get(&txn_id).cloned() {
                return Ok(pending);
            }

            (
                state.txn_root.clone(),
                state.persistent.schema().clone(),
                state.persistent.schema().storage().clone(),
            )
        };

        let txn_dir = {
            let mut root = txn_root.write().await;
            root.get_or_create_dir(txn_id.to_string())
                .map_err(background_error)?
        };

        let pending_dir = {
            let mut txn_dir = txn_dir.write().await;
            txn_dir
                .get_or_create_dir("pending".to_string())
                .map_err(background_error)?
        };

        let (inserts_dir, deletes_dir) = {
            let mut pending_dir = pending_dir.write().await;
            let inserts = pending_dir
                .get_or_create_dir("inserts".to_string())
                .map_err(background_error)?;
            let deletes = pending_dir
                .get_or_create_dir("deletes".to_string())
                .map_err(background_error)?;
            (inserts, deletes)
        };

        let delta = Delta {
            inserts: TableStore::from_dir(inserts_dir, schema.clone(), storage.clone())
                .map_err(background_error)?,
            deletes: TableStore::from_dir(deletes_dir, schema, storage)
                .map_err(background_error)?,
        };

        let mut state = self.state.write().expect("state write lock");
        Self::assert_writable_state(&state, txn_id)?;

        if let Some(existing) = state.pending.get(&txn_id).cloned() {
            return Ok(existing);
        }

        state.pending.insert(txn_id, delta.clone());

        Ok(delta)
    }
}

impl PersistentTable {
    /// Commit the pending delta at `txn_id`.
    ///
    /// Returns `Err(Outdated)` if the txn is at or before the finalize frontier,
    /// `Ok(())` for a duplicate commit (idempotent no-op), or `Err(Conflict)` if
    /// the semaphore cannot be acquired due to a future overlapping read.
    pub fn commit(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let mut state = self.state.write().expect("state write lock");

        if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
            drop(state);
            self.release_txn_reservation(txn_id);
            return Err(txn_lock::Error::Outdated);
        }

        if state.committed.contains_key(&txn_id) {
            drop(state);
            self.release_txn_reservation(txn_id);
            return Ok(());
        }

        if let Some(delta) = state.pending.remove(&txn_id) {
            state.committed.insert(txn_id, delta);
        }

        drop(state);
        self.release_txn_reservation(txn_id);

        Ok(())
    }

    /// Roll back the pending delta at `txn_id`.
    ///
    /// Returns `Err(Outdated)` if the txn is at or before the finalize frontier,
    /// `Err(Conflict)` if the txn is already committed, or `Err(Conflict)` if the
    /// semaphore cannot be acquired due to a future overlapping read.
    pub fn rollback(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let mut state = self.state.write().expect("state write lock");

        if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
            drop(state);
            self.release_txn_reservation(txn_id);
            return Err(txn_lock::Error::Outdated);
        }

        if state.committed.contains_key(&txn_id) {
            drop(state);
            self.release_txn_reservation(txn_id);
            return Err(txn_lock::Error::Conflict);
        }

        state.pending.remove(&txn_id);

        drop(state);
        self.release_txn_reservation(txn_id);

        Ok(())
    }

    /// Finalize all committed deltas up to `txn_id` into persistent state.
    ///
    /// Finalize is monotonic. A stale finalize (≤ frontier) is a no-op.
    ///
    /// Unlike commit/rollback, finalize does **not** acquire a semaphore write
    /// permit. Finalize is a lifecycle operation that merges already-committed
    /// data into canon — it does not introduce new pending writes. Acquiring a
    /// write permit via `try_write` would conflict with future read reservations
    /// (e.g. readers at txn N+1), which is incorrect: finalize at txn N should
    /// proceed even when later transactions hold read permits. Synchronization
    /// is provided by the state write lock, and `semaphore.finalize(drop_past=true)`
    /// cleans up semaphore versions ≤ `txn_id`.
    pub async fn finalize(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        let (persistent, committed_to_apply) = {
            let state = self.state.write().expect("state write lock");

            if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
                return Ok(());
            }

            let committed_to_apply = state
                .committed
                .iter()
                .filter_map(|(id, delta)| (*id <= txn_id).then_some(delta.clone()))
                .collect::<Vec<_>>();

            (state.persistent.clone(), committed_to_apply)
        };

        for delta in &committed_to_apply {
            persistent
                .apply_delta(delta)
                .await
                .map_err(background_error)?;
        }

        {
            let mut state = self.state.write().expect("state write lock");
            state.committed.retain(|id, _| *id > txn_id);
            state.pending.retain(|id, _| *id > txn_id);
            state.finalized = Some(state.finalized.map_or(txn_id, |prior| prior.max(txn_id)));
        }

        self.release_txn_frontier(txn_id);

        Ok(())
    }
}

impl Transact for PersistentTable {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        PersistentTable::commit(self, txn_id).expect("Table commit failed");
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            PersistentTable::rollback(self, txn_id).expect("Table rollback failed");
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            PersistentTable::finalize(self, txn_id)
                .await
                .expect("Table finalize failed");
        }
    }
}
