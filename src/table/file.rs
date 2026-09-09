use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use b_table::{Range, Row, TableLock};
use collate::{Collate, try_diff, try_merge};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_ir::{Id, Transact, TxnId};
use tc_value::{Value, ValueCollator};

use super::schema::{TableIndexSchema, TableSchema};
use super::stream::Rows;
use super::view::{Limited, Selection, TableSlice};

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
#[derive(Clone, Eq, PartialEq)]
struct RowCollator {
    indices: Vec<usize>,
    reverse: bool,
    values: ValueCollator,
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

        Self {
            indices,
            reverse,
            values: ValueCollator::default(),
        }
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
                (Some(l), Some(r)) => self.values.cmp(l, r),
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            };
            if ord != std::cmp::Ordering::Equal {
                break;
            }
        }
        if self.reverse { ord.reverse() } else { ord }
    }
}

type RowsResult = std::io::Result<BoxStream<'static, Result<Row<Value>, std::io::Error>>>;
pub(crate) type TableFile<F> = TableLock<TableSchema, TableIndexSchema, ValueCollator, F>;

pub(crate) async fn row_stream<F: crate::CollectionFile>(
    table: &TableFile<F>,
    range: Range<Id, Value>,
    order: &[Id],
    reverse: bool,
) -> RowsResult {
    let view = table.read().await;
    view.rows(range, order, reverse, None)
        .await
        .map(|rows| Box::pin(rows) as _)
}

pub(crate) async fn get_row<F: crate::CollectionFile>(
    table: &TableFile<F>,
    key: &[Value],
) -> std::io::Result<Option<Row<Value>>> {
    let range = table
        .schema()
        .range_from_key(key)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    table
        .read()
        .await
        .rows(range, &[], false, None)
        .await?
        .try_next()
        .await
}

pub(crate) async fn upsert<F: crate::CollectionFile>(
    table: &TableFile<F>,
    key: Vec<Value>,
    values: Vec<Value>,
) -> std::io::Result<()> {
    table
        .write()
        .await
        .upsert(key, values)
        .await
        .map(|_| ())
        .map_err(std::io::Error::other)
}

pub(crate) async fn delete_row<F: crate::CollectionFile>(
    table: &TableFile<F>,
    key: &[Value],
) -> std::io::Result<()> {
    table.write().await.delete_row(key).await.map(|_| ())
}

pub(crate) async fn count<F: crate::CollectionFile>(
    table: &TableFile<F>,
    range: Range<Id, Value>,
) -> std::io::Result<u64> {
    table.read().await.count(range).await
}

pub(crate) async fn is_empty<F: crate::CollectionFile>(
    table: &TableFile<F>,
    range: Range<Id, Value>,
) -> std::io::Result<bool> {
    table.read().await.is_empty(range).await
}

#[derive(Clone)]
struct Delta<F: crate::CollectionFile> {
    inserts: TableFile<F>,
    deletes: TableFile<F>,
}

impl<F: crate::CollectionFile> Delta<F> {
    async fn upsert(&self, key: Vec<Value>, values: Vec<Value>) -> std::io::Result<()> {
        delete_row(&self.deletes, &key).await?;
        delete_row(&self.inserts, &key).await?;
        upsert(&self.inserts, key, values).await
    }

    async fn delete_from_inserts(&self, key: &[Value]) -> std::io::Result<()> {
        delete_row(&self.inserts, key).await
    }

    async fn add_to_deletes(&self, key: &[Value], values: Vec<Value>) -> std::io::Result<()> {
        upsert(&self.deletes, key.to_vec(), values).await
    }

    async fn get_inserted_row(&self, key: &[Value]) -> std::io::Result<Option<Row<Value>>> {
        get_row(&self.inserts, key).await
    }

    async fn already_deleted(&self, key: &[Value]) -> std::io::Result<bool> {
        get_row(&self.deletes, key).await.map(|row| row.is_some())
    }

    async fn merge_into<'a>(
        &'a self,
        rows: BoxStream<'a, Result<Vec<Value>, std::io::Error>>,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
        collator: RowCollator,
    ) -> BoxStream<'a, Result<Vec<Value>, std::io::Error>> {
        let inserted = row_stream(&self.inserts, range.clone(), order, reverse)
            .await
            .expect("stream insert delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        let merged = try_merge(collator.clone(), inserted, rows).boxed();

        let deleted = row_stream(&self.deletes, range, order, reverse)
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
        let inserted = row_stream(&self.inserts, range.clone(), &order, reverse)
            .await
            .expect("stream insert delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        let merged = try_merge(collator.clone(), inserted, rows).boxed();

        let deleted = row_stream(&self.deletes, range, &order, reverse)
            .await
            .expect("stream delete delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        try_diff(collator, merged, deleted).boxed()
    }
}

async fn apply_delta<F: crate::CollectionFile>(
    persistent: &TableFile<F>,
    delta: &Delta<F>,
) -> std::io::Result<()> {
    let key_len = persistent.schema().key().len();
    let mut inserts = row_stream(&delta.inserts, Range::default(), &[], false).await?;
    while let Some(row) = inserts.try_next().await? {
        upsert(persistent, row[..key_len].to_vec(), row[key_len..].to_vec()).await?;
    }

    let mut deletes = row_stream(&delta.deletes, Range::default(), &[], false).await?;
    while let Some(row) = deletes.try_next().await? {
        delete_row(persistent, &row[..key_len]).await?;
    }

    Ok(())
}

#[derive(Clone)]
struct State<F: crate::CollectionFile> {
    persistent: TableFile<F>,
    committed: BTreeMap<TxnId, Delta<F>>,
    pending: BTreeMap<TxnId, Delta<F>>,
    finalized: Option<TxnId>,
}

#[derive(Clone)]
struct VisibleSnapshot<F: crate::CollectionFile> {
    persistent: TableFile<F>,
    deltas: Vec<Delta<F>>,
}

pub struct PersistentTable<Txn: crate::StorageContext> {
    state: Arc<RwLock<State<Txn::File>>>,
    semaphore: txn_lock::semaphore::Semaphore<
        TxnId,
        b_tree::Collator<ValueCollator>,
        txn_lock::set::Range<Vec<Value>>,
    >,
    schema: TableSchema,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: crate::StorageContext> Clone for PersistentTable<Txn> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
            schema: self.schema.clone(),
            txn: PhantomData,
        }
    }
}

impl<Txn: crate::StorageContext> fmt::Debug for PersistentTable<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read().expect("state read lock");
        f.debug_struct("PersistentTable")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl<Txn: crate::StorageContext> PersistentTable<Txn> {
    pub fn new(persistent_dir: DirLock<Txn::File>, schema: TableSchema) -> Self {
        let persistent = Self::load_store(persistent_dir.clone(), schema.clone());

        let state = State {
            persistent,
            committed: BTreeMap::new(),
            pending: BTreeMap::new(),
            finalized: None,
        };

        Self {
            state: Arc::new(RwLock::new(state)),
            semaphore: txn_lock::semaphore::Semaphore::new(b_tree::Collator::new(
                ValueCollator::default(),
            )),
            schema,
            txn: PhantomData,
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
    /// survive a restart. Pending and committed deltas are not synced; their
    /// durable ordering and replay belong to the caller.
    pub async fn sync(&self) -> std::io::Result<()> {
        let persistent = {
            let state = self.state.read().expect("state read lock");
            state.persistent.clone()
        };
        persistent.sync().await
    }

    pub async fn upsert_row(
        &self,
        txn: &Txn,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> Result<(), txn_lock::Error>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(&self.schema, key).map_err(background_error)?;
        let values =
            b_table::Schema::validate_values(&self.schema, values).map_err(background_error)?;

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending
            .upsert(key, values)
            .await
            .map_err(background_error)?;

        Ok(())
    }

    pub async fn insert_row(
        &self,
        txn: &Txn,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> tc_error::TCResult<()>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(&self.schema, key)?;
        let values = b_table::Schema::validate_values(&self.schema, values)?;

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))
            .map_err(tc_error::TCError::from)?;

        if self
            .resolve_row(&self.visible_snapshot(txn_id), &key)
            .await
            .is_some()
        {
            return Err(tc_error::TCError::bad_request(format!(
                "cannot insert Table row: key {key:?} already exists"
            )));
        }

        self.pending_delta_for_txn(txn)
            .await
            .map_err(tc_error::TCError::from)?
            .upsert(key, values)
            .await
            .map_err(tc_error::TCError::from)?;

        Ok(())
    }

    pub async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> Result<(), txn_lock::Error>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(&self.schema, key).map_err(background_error)?;

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;

        if pending
            .already_deleted(&key)
            .await
            .map_err(background_error)?
        {
            return Ok(());
        }

        let mut row = pending
            .get_inserted_row(&key)
            .await
            .map_err(background_error)?;

        if row.is_none() {
            let snapshot = self.visible_snapshot(txn_id);
            row = self.resolve_row(&snapshot, &key).await;
        }

        pending
            .delete_from_inserts(&key)
            .await
            .map_err(background_error)?;

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
        !self.any_row_in(txn_id, Range::default(), &[], false).await
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

    pub async fn count_in(&self, txn_id: TxnId, range: Range<tc_ir::Id, Value>) -> u64 {
        let mut count = 0_u64;
        self.for_each_row_in_order(txn_id, range, &[], false, |_| {
            count += 1;
        })
        .await;
        count
    }

    /// Return `true` if there are no visible rows in `range` at `txn_id`.
    pub async fn is_empty_in(&self, txn_id: TxnId, range: Range<tc_ir::Id, Value>) -> bool {
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

        let mut visible: BoxStream<'static, Result<Vec<Value>, std::io::Error>> =
            row_stream(&snapshot.persistent, range.clone(), &order, reverse)
                .await
                .map_err(background_error)?
                .map_ok(|row| row.to_vec())
                .boxed();

        for delta in snapshot.deltas.into_iter() {
            visible = delta
                .merge_into_owned(
                    visible,
                    range.clone(),
                    order.clone(),
                    reverse,
                    collator.clone(),
                )
                .await;
        }

        let stream = visible.map_ok(Row::from_vec).boxed();
        Ok(Rows::new(stream, permit))
    }

    /// Create a range + order + reverse view over this table.
    ///
    /// The view is structural — it holds no row data. Row streaming, count,
    /// and containment checks delegate to this table with the view's bounds.
    pub fn slice(&self, range: Range<Id, Value>, order: &[Id], reverse: bool) -> TableSlice<Txn> {
        TableSlice::new(self.clone(), range, order.to_vec(), reverse)
    }

    /// Create an ordered view over this table using the given `columns`.
    ///
    /// Equivalent to `slice(Range::default(), columns, reverse)`.
    pub fn order_by(&self, columns: &[Id], reverse: bool) -> TableSlice<Txn> {
        self.slice(Range::default(), columns, reverse)
    }

    /// Create a row-cap view that yields at most `n` rows.
    pub fn limit(&self, n: u64) -> Limited<Txn> {
        self.slice(Range::default(), &[], false).limit(n)
    }

    /// Create a column-projection view that yields only `columns`.
    pub fn select(&self, columns: &[Id]) -> tc_error::TCResult<Selection<Txn>> {
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
        txn: &Txn,
        range: Range<Id, Value>,
        values: tc_ir::Map<Value>,
    ) -> Result<(), txn_lock::Error>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        let value_columns = self.schema.values();
        for name in values.keys() {
            if !value_columns.contains(name) {
                return Err(background_error(format!("cannot update key column {name}")));
            }
        }

        let key_len = self.schema.key().len();
        let collator = RowCollator::for_order(&self.schema, &[], false);

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn).await?;

        let snapshot = self.visible_snapshot(txn_id);

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> =
            row_stream(&snapshot.persistent, range.clone(), &[], false)
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

            pending
                .upsert(key, updated_values)
                .await
                .map_err(background_error)?;
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
    pub async fn truncate(&self, txn: &Txn, range: Range<Id, Value>) -> Result<(), txn_lock::Error>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        let key_len = self.schema.key().len();
        let collator = RowCollator::for_order(&self.schema, &[], false);

        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn).await?;

        let snapshot = self.visible_snapshot(txn_id);

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> =
            row_stream(&snapshot.persistent, range.clone(), &[], false)
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

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> =
            row_stream(&snapshot.persistent, range.clone(), order, reverse)
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

    fn visible_snapshot(&self, txn_id: TxnId) -> VisibleSnapshot<Txn::File> {
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
        snapshot: &VisibleSnapshot<Txn::File>,
        key: &[Value],
    ) -> Option<Row<Value>> {
        let mut row = get_row(&snapshot.persistent, key)
            .await
            .expect("check persistent visibility");

        for delta in &snapshot.deltas {
            if get_row(&delta.deletes, key)
                .await
                .expect("check delete delta")
                .is_some()
            {
                row = None;
            }

            if let Some(inserted) = get_row(&delta.inserts, key)
                .await
                .expect("check insert delta")
            {
                row = Some(inserted);
            }
        }

        row
    }

    async fn is_row_visible(&self, snapshot: &VisibleSnapshot<Txn::File>, key: &[Value]) -> bool {
        self.resolve_row(snapshot, key).await.is_some()
    }

    fn assert_writable_state(
        state: &State<Txn::File>,
        txn_id: TxnId,
    ) -> Result<(), txn_lock::Error> {
        if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
            return Err(txn_lock::Error::Outdated);
        }

        if state.committed.contains_key(&txn_id) {
            return Err(txn_lock::Error::Committed);
        }

        Ok(())
    }

    fn load_store(persistent_dir: DirLock<Txn::File>, schema: TableSchema) -> TableFile<Txn::File> {
        TableLock::load(schema, ValueCollator::default(), persistent_dir)
            .expect("load persistent Table store")
    }

    async fn pending_delta_for_txn(&self, txn: &Txn) -> Result<Delta<Txn::File>, txn_lock::Error> {
        let txn_id = txn.id();
        let schema = {
            let state = self.state.write().expect("state write lock");
            Self::assert_writable_state(&state, txn_id)?;

            if let Some(pending) = state.pending.get(&txn_id).cloned() {
                return Ok(pending);
            }

            state.persistent.schema().clone()
        };

        let txn_dir = txn
            .subcontext_unique()
            .context()
            .await
            .map_err(background_error)?;

        let (inserts_dir, deletes_dir) = {
            let mut txn_dir = txn_dir.write().await;
            let inserts = txn_dir
                .get_or_create_dir("inserts".to_string())
                .map_err(background_error)?;
            let deletes = txn_dir
                .get_or_create_dir("deletes".to_string())
                .map_err(background_error)?;
            (inserts, deletes)
        };

        let delta = Delta {
            inserts: TableLock::load(schema.clone(), ValueCollator::default(), inserts_dir)
                .map_err(background_error)?,
            deletes: TableLock::load(schema, ValueCollator::default(), deletes_dir)
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

impl<Txn: crate::StorageContext> PersistentTable<Txn> {
    /// Commit the pending delta at `txn_id`.
    ///
    /// Returns `Err(Outdated)` if the txn is at or before the finalize frontier,
    /// `Ok(())` for a duplicate commit (idempotent no-op), or `Err(Conflict)` if
    /// the semaphore cannot be acquired due to a future overlapping read.
    pub fn commit(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let result = {
            let mut state = self.state.write().expect("state write lock");
            if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
                Err(txn_lock::Error::Outdated)
            } else if state.committed.contains_key(&txn_id) {
                Ok(())
            } else {
                if let Some(delta) = state.pending.remove(&txn_id) {
                    state.committed.insert(txn_id, delta);
                }
                Ok(())
            }
        };
        self.release_txn_reservation(txn_id);
        result
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

        let result = {
            let mut state = self.state.write().expect("state write lock");
            if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
                Err(txn_lock::Error::Outdated)
            } else if state.committed.contains_key(&txn_id) {
                Err(txn_lock::Error::Conflict)
            } else {
                state.pending.remove(&txn_id);
                Ok(())
            }
        };
        self.release_txn_reservation(txn_id);
        result
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
            apply_delta(&persistent, delta)
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

impl<Txn: crate::StorageContext> Transact for PersistentTable<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        PersistentTable::commit(self, txn_id).map_err(Into::into)
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { PersistentTable::rollback(self, txn_id).map_err(Into::into) }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            PersistentTable::finalize(self, txn_id)
                .await
                .map_err(Into::into)
        }
    }
}
