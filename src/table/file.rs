use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use b_table::{Range, Row, TableLock};
use collate::{Collate, try_diff, try_merge};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_ir::{Id, Transact, TxnId};
use tc_value::{Value, ValueCollator};

use crate::persistence::{CollectionOwner, PersistentDelta, VisibleSnapshot, background_error};

use super::schema::{TableIndexSchema, TableSchema};
use super::stream::Rows;
use super::view::{Limited, Selection, TableSlice};

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
    /// Start an ordinary delta with empty insert and delete storage.
    async fn create(schema: TableSchema, dir: DirLock<F>) -> std::io::Result<Self> {
        let (inserts, deletes) = crate::persistence::create_delta_dirs(&dir).await?;
        Ok(Self {
            inserts: TableLock::create(schema.clone(), ValueCollator::default(), inserts)?,
            deletes: TableLock::create(schema, ValueCollator::default(), deletes)?,
        })
    }

    /// Seed a restoration delta in a separate workspace: empty inserts and
    /// canonical contents in deletes, so keys absent from the snapshot are removed.
    /// The caller applies visible committed deltas to deletes, inserts the snapshot,
    /// and installs the pending replacement only after construction succeeds.
    /// Failure or cancellation leaves the existing pending delta untouched.
    async fn for_restore(canonical: &TableFile<F>, dir: DirLock<F>) -> std::io::Result<Self> {
        let (inserts, deletes) = crate::persistence::create_delta_dirs(&dir).await?;
        Ok(Self {
            inserts: TableLock::create(
                canonical.schema().clone(),
                ValueCollator::default(),
                inserts,
            )?,
            deletes: canonical.copy_into(deletes).await?,
        })
    }

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
        &self,
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
}

impl<F: crate::CollectionFile> PersistentDelta for Delta<F> {
    type Native = TableFile<F>;

    async fn apply_to(&self, persistent: &TableFile<F>) -> std::io::Result<()> {
        let key_len = persistent.schema().key().len();
        let mut inserts = row_stream(&self.inserts, Range::default(), &[], false).await?;
        while let Some(row) = inserts.try_next().await? {
            upsert(persistent, row[..key_len].to_vec(), row[key_len..].to_vec()).await?;
        }

        let mut deletes = row_stream(&self.deletes, Range::default(), &[], false).await?;
        while let Some(row) = deletes.try_next().await? {
            delete_row(persistent, &row[..key_len]).await?;
        }

        Ok(())
    }
}

pub struct PersistentTable<Txn: crate::StorageContext> {
    owner: Arc<CollectionOwner<Delta<Txn::File>>>,
    schema: TableSchema,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: crate::StorageContext> Clone for PersistentTable<Txn> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            schema: self.schema.clone(),
            txn: PhantomData,
        }
    }
}

impl<Txn: crate::StorageContext> fmt::Debug for PersistentTable<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.owner.state.read().expect("state read lock");
        f.debug_struct("PersistentTable")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl<Txn: crate::StorageContext> PersistentTable<Txn> {
    pub fn new(persistent_dir: DirLock<Txn::File>, schema: TableSchema) -> Self {
        Self::try_new(persistent_dir, schema).expect("create persistent Table store")
    }

    /// Create in empty caller-delegated storage.
    pub fn try_new(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let persistent =
            TableLock::create(schema.clone(), ValueCollator::default(), persistent_dir)?;
        Ok(Self::from_store(schema, persistent))
    }

    pub async fn load(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let persistent = TableLock::load(schema.clone(), ValueCollator::default(), persistent_dir)?;
        persistent.validate().await?;
        Ok(Self::from_store(schema, persistent))
    }

    fn from_store(schema: TableSchema, persistent: TableFile<Txn::File>) -> Self {
        Self {
            owner: Arc::new(CollectionOwner::new(persistent)),
            schema,
            txn: PhantomData,
        }
    }

    pub fn schema(&self) -> &TableSchema {
        &self.schema
    }

    pub(crate) async fn load_literal_row(&self, mut row: Vec<Value>) -> std::io::Result<()> {
        if row.len() != self.schema.column_count() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid Table row",
            ));
        }

        let values = row.split_off(self.schema.key().len());
        let persistent = {
            self.owner
                .state
                .read()
                .expect("state read lock")
                .persistent
                .clone()
        };

        upsert(&persistent, row, values).await
    }

    pub fn finalized(&self) -> Option<TxnId> {
        self.owner.finalized()
    }

    /// Sync the canonical (persistent) state to disk.
    ///
    /// Buffered writeback does not acknowledge durability. Pending workspace
    /// and committed workspace deltas are not included.
    pub async fn sync(&self) -> std::io::Result<()> {
        let persistent = self
            .owner
            .state
            .read()
            .expect("state read lock")
            .persistent
            .clone();
        persistent.sync().await
    }

    /// Durably synchronize materialized canonical storage; pending deltas are excluded.
    pub async fn sync_all(&self) -> std::io::Result<()> {
        let persistent = self
            .owner
            .state
            .read()
            .expect("state read lock")
            .persistent
            .clone();
        persistent.sync_all().await
    }

    /// Stage a native replacement without changing the existing pending delta on failure.
    pub async fn restore_from(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let id = txn.id();
        let _permit = self
            .owner
            .semaphore
            .try_write(id, txn_lock::set::Range::All)?;
        let (canonical, committed) = {
            let state = self.owner.state.read().expect("state read lock");
            state.assert_writable(id)?;
            (
                state.persistent.clone(),
                state
                    .committed
                    .range(..=id)
                    .filter_map(|(_, delta)| delta.clone())
                    .collect::<Vec<_>>(),
            )
        };
        let schema = self.schema.clone();
        if <Value as safecast::CastFrom<TableSchema>>::cast_from(schema.clone())
            != <Value as safecast::CastFrom<TableSchema>>::cast_from(source.schema.clone())
        {
            return Err(tc_error::TCError::bad_request(
                "Table restoration schema mismatch",
            ));
        }
        let workspace = txn.subcontext_unique().context().await?;
        let delta = Delta::for_restore(&canonical, workspace).await?;
        for committed in committed {
            committed.apply_to(&delta.deletes).await?;
        }
        let key_len = self.schema.key().len();
        let mut rows = source.rows(id, Range::default(), vec![], false).await?;
        while let Some(row) = rows.try_next().await? {
            delta
                .upsert(row[..key_len].to_vec(), row[key_len..].to_vec())
                .await?;
        }
        let mut state = self.owner.state.write().expect("state write lock");
        state.assert_writable(id)?;
        state.pending.insert(id, delta);
        Ok(())
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
            .owner
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
            .owner
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))
            .map_err(tc_error::TCError::from)?;

        if self
            .resolve_row(&self.owner.visible_snapshot(txn_id), &key)
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
            .owner
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
            let snapshot = self.owner.visible_snapshot(txn_id);
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
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await;

        let snapshot = self.owner.visible_snapshot(txn_id);
        self.resolve_row(&snapshot, key).await
    }

    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> bool {
        let _permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await;

        let snapshot = self.owner.visible_snapshot(txn_id);
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
            .owner
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
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await;

        let snapshot = self.owner.visible_snapshot(txn_id);
        let collator = RowCollator::for_order(&self.schema, &order, reverse);

        let mut visible: BoxStream<'static, Result<Vec<Value>, std::io::Error>> =
            row_stream(&snapshot.persistent, range.clone(), &order, reverse)
                .await
                .map_err(background_error)?
                .map_ok(|row| row.to_vec())
                .boxed();

        for delta in snapshot.deltas.into_iter() {
            visible = delta
                .merge_into(visible, range.clone(), &order, reverse, collator.clone())
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
            .owner
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn).await?;

        let snapshot = self.owner.visible_snapshot(txn_id);

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
            .owner
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let pending = self.pending_delta_for_txn(txn).await?;

        let snapshot = self.owner.visible_snapshot(txn_id);

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
        let snapshot = self.owner.visible_snapshot(txn_id);

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

    async fn resolve_row(
        &self,
        snapshot: &VisibleSnapshot<Delta<Txn::File>>,
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

    async fn is_row_visible(
        &self,
        snapshot: &VisibleSnapshot<Delta<Txn::File>>,
        key: &[Value],
    ) -> bool {
        self.resolve_row(snapshot, key).await.is_some()
    }

    async fn pending_delta_for_txn(&self, txn: &Txn) -> Result<Delta<Txn::File>, txn_lock::Error> {
        let txn_id = txn.id();
        let schema = {
            let state = self.owner.state.write().expect("state write lock");
            state.assert_writable(txn_id)?;

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

        let delta = Delta::create(schema, txn_dir)
            .await
            .map_err(background_error)?;

        let mut state = self.owner.state.write().expect("state write lock");
        state.assert_writable(txn_id)?;

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
    /// The caller must finish this transaction's operations and release its streams first.
    /// Returns `Outdated` at or before the finalized frontier.
    pub async fn commit(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        self.owner.commit(txn_id)
    }

    pub fn rollback(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        self.owner.rollback(txn_id)
    }

    /// Finalize all committed deltas up to `txn_id` into persistent state.
    ///
    /// Finalize is monotonic. A stale finalize (≤ frontier) is a no-op.
    ///
    /// The caller serializes lifecycle decisions and finishes operations through the
    /// cutoff first. Later transactions may retain read permits; merging uses native
    /// storage locks without acquiring a new semaphore write reservation.
    pub async fn finalize(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        self.owner.finalize(txn_id).await
    }
}

impl<Txn: crate::StorageContext> Transact for PersistentTable<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        PersistentTable::commit(self, txn_id)
            .await
            .map_err(Into::into)
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
