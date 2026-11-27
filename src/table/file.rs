//! Transactional row ownership over native indexed Tables.
//!
//! `Versions` decides visibility and lifecycle; native b-table owners maintain
//! indexes in canonical storage and each delta. Deletes retain the baseline row
//! so scans through any index can suppress its old entry before adding the new
//! row. See TRANSACTIONAL_COLLECTION_CONTRACT.md for caller obligations.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use b_table::{IndexSchema, Range, Row, TableLock};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Transact, TxnId};
use tc_value::{Value, ValueCollator};

use super::Table;
use super::schema::{TableIndexSchema, TableSchema};
use super::stream::Rows;
use super::view::TableSlice;
use crate::persistence::Versions;
use crate::persistence::storage::{Allocation, Store};
use crate::stream::ReadSnapshot;

type RowsResult = std::io::Result<BoxStream<'static, Result<Row<Value>, std::io::Error>>>;
type TableFile<F> = TableLock<TableSchema, TableIndexSchema, ValueCollator, F>;

async fn row_stream<F: crate::CollectionFile>(
    table: &TableFile<F>,
    range: Range<Id, Value>,
    order: &[Id],
    reverse: bool,
) -> RowsResult {
    let view = table.read().await;
    // Keep native scan setup out of enclosing mutation futures. Its inline
    // frame otherwise overflows normal worker stacks through Service and Chain.
    Box::pin(view.select(range, order, reverse, table.schema().primary().columns())).await
}

async fn upsert<F: crate::CollectionFile>(
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
        .map_err(|error| {
            if error.requires_recovery() {
                freqfs::IntegrityError::mark(std::io::Error::other(error))
            } else {
                std::io::Error::other(error)
            }
        })
}

async fn delete_row<F: crate::CollectionFile>(
    table: &TableFile<F>,
    key: &[Value],
) -> std::io::Result<()> {
    table.write().await.delete_row(key).await.map(|_| ())
}

struct Delta<F: crate::CollectionFile> {
    storage: Arc<Allocation<F>>,
    // A lookup distinguishes untouched (None), deleted (Some(None)), and replaced
    // (Some(Some(row))). Inserts take precedence when both tables contain a key.
    inserts: TableFile<F>,
    deletes: TableFile<F>,
}

impl<F: crate::CollectionFile> Delta<F> {
    async fn load(schema: TableSchema, storage: Arc<Allocation<F>>) -> TCResult<Arc<Self>> {
        let (inserts, deletes) = {
            let dir = storage.dir().read().await;
            (
                dir.get_dir("inserts")
                    .cloned()
                    .ok_or_else(|| freqfs::IntegrityError::invalid_data("missing inserts"))?,
                dir.get_dir("deletes")
                    .cloned()
                    .ok_or_else(|| freqfs::IntegrityError::invalid_data("missing deletes"))?,
            )
        };

        let inserts = TableLock::load(schema.clone(), ValueCollator::default(), inserts)?;
        let deletes = TableLock::load(schema, ValueCollator::default(), deletes)?;
        inserts.validate().await?;
        deletes.validate().await?;
        Ok(Arc::new(Self {
            storage,
            inserts,
            deletes,
        }))
    }

    async fn units(&self) -> std::io::Result<u64> {
        let mut units = 0u64;
        for table in [&self.inserts, &self.deletes] {
            let mut rows = row_stream(table, Range::default(), &[], false).await?;
            while rows.try_next().await?.is_some() {
                units = units
                    .checked_add(1)
                    .ok_or_else(|| std::io::Error::other("native version size overflow"))?;
            }
        }

        Ok(units)
    }

    async fn lookup(&self, key: &[Value]) -> std::io::Result<Option<Option<Row<Value>>>> {
        if let Some(row) = self.inserts.get_row(key).await? {
            return Ok(Some(Some(row)));
        }

        if self.deletes.contains(key).await? {
            return Ok(Some(None));
        }

        Ok(None)
    }

    /// Start an ordinary delta with empty insert and delete storage.
    async fn create(
        schema: TableSchema,
        storage: Arc<Allocation<F>>,
    ) -> std::io::Result<Arc<Self>> {
        let (inserts, deletes) = {
            let mut dir = storage.dir().write().await;
            (
                dir.get_or_create_dir("inserts".to_owned())?,
                dir.get_or_create_dir("deletes".to_owned())?,
            )
        };

        Ok(Arc::new(Self {
            storage,
            inserts: TableLock::create(schema.clone(), tc_value::ValueCollator::default(), inserts)
                .await?,
            deletes: TableLock::create(schema, tc_value::ValueCollator::default(), deletes).await?,
        }))
    }

    async fn upsert(&self, key: Vec<Value>, values: Vec<Value>) -> std::io::Result<()> {
        upsert(&self.inserts, key, values).await?;
        Ok(())
    }

    async fn is_changed(&self, key: &[Value]) -> std::io::Result<bool> {
        Ok(self.inserts.contains(key).await? || self.already_deleted(key).await?)
    }

    async fn delete_from_inserts(&self, key: &[Value]) -> std::io::Result<()> {
        delete_row(&self.inserts, key).await
    }

    async fn add_to_deletes(&self, key: Vec<Value>, values: Vec<Value>) -> std::io::Result<()> {
        upsert(&self.deletes, key, values).await
    }

    async fn already_deleted(&self, key: &[Value]) -> std::io::Result<bool> {
        self.deletes.contains(key).await
    }

    // Consume staged changes only after the input snapshot releases its guards.
    async fn apply_to_pending(&self, pending: &Self) -> std::io::Result<()> {
        let key_len = self.inserts.schema().key().len();
        let mut deletes = row_stream(&self.deletes, Range::default(), &[], false).await?;
        while let Some(row) = deletes.try_next().await? {
            let mut key = row.into_vec();
            let values = key.split_off(key_len);
            if !pending.is_changed(&key).await? {
                pending.add_to_deletes(key.clone(), values).await?;
            }

            pending.delete_from_inserts(&key).await?;
        }

        drop(deletes);
        let mut inserts = row_stream(&self.inserts, Range::default(), &[], false).await?;
        while let Some(row) = inserts.try_next().await? {
            let mut key = row.into_vec();
            let values = key.split_off(key_len);
            pending.upsert(key, values).await?;
        }

        Ok(())
    }
}

pub struct PersistentTable<Txn: crate::StorageContext> {
    canonical: Arc<TableFile<Txn::File>>,
    store: Arc<Store<Txn::File>>,
    versions: Versions<Delta<Txn::File>>,
    semaphore: txn_lock::semaphore::Semaphore<
        TxnId,
        b_tree::RangeCollator<ValueCollator>,
        txn_lock::set::Range<Vec<Value>>,
    >,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: crate::StorageContext> Clone for PersistentTable<Txn> {
    fn clone(&self) -> Self {
        Self {
            canonical: self.canonical.clone(),
            store: self.store.clone(),
            versions: self.versions.clone(),
            semaphore: self.semaphore.clone(),
            txn: PhantomData,
        }
    }
}

impl<Txn: crate::StorageContext> fmt::Debug for PersistentTable<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersistentTable")
            .field("schema", &self.schema())
            .finish()
    }
}

impl<Txn: crate::StorageContext> PersistentTable<Txn> {
    /// Acquire ordering capacity and capture visibility in one admission operation.
    async fn read_snapshot(
        &self,
        id: TxnId,
        range: txn_lock::set::Range<Vec<Value>>,
    ) -> TCResult<ReadSnapshot<Delta<Txn::File>>> {
        self.versions.check_readable(id)?;
        let permit = self.semaphore.read(id, range).await?;
        let snapshot = self.versions.visible_snapshot(id)?;
        Ok((permit, snapshot))
    }

    fn try_read_snapshot(
        &self,
        id: TxnId,
        range: txn_lock::set::Range<Vec<Value>>,
    ) -> TCResult<ReadSnapshot<Delta<Txn::File>>> {
        self.versions.check_readable(id)?;
        let permit = self.semaphore.try_read(id, range)?;
        let snapshot = self.versions.visible_snapshot(id)?;
        Ok((permit, snapshot))
    }

    fn try_acquire_write_permit(
        &self,
        id: TxnId,
        range: txn_lock::set::Range<Vec<Value>>,
    ) -> TCResult<txn_lock::semaphore::PermitWrite<txn_lock::set::Range<Vec<Value>>>> {
        self.versions
            .reserve_ordered_write(id, || self.semaphore.try_write(id, range))
    }

    /// Create in empty caller-delegated storage.
    pub async fn try_new(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let owner = Self::unpublished(persistent_dir, schema).await?;
        owner.finish_creation().await?;
        Ok(owner)
    }

    pub(super) async fn unpublished(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let store = Store::create(persistent_dir)
            .await
            .map_err(std::io::Error::other)?;

        let persistent = TableLock::create(
            schema,
            tc_value::ValueCollator::default(),
            store.base().clone(),
        )
        .await?;

        Ok(Self::from_store(persistent, store, Versions::new()))
    }

    pub async fn load(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let store = Store::load(persistent_dir)
            .await
            .map_err(std::io::Error::other)?;

        let persistent = TableLock::load(
            schema.clone(),
            tc_value::ValueCollator::default(),
            store.base().clone(),
        )?;

        persistent.validate().await?;
        let versions = Versions::load(&store, |storage| {
            let schema = schema.clone();
            async move {
                let delta = Delta::load(schema, storage).await?;
                let units = delta.units().await?;
                Ok((delta, units))
            }
        })
        .await
        .map_err(std::io::Error::other)?;

        Ok(Self::from_store(persistent, store, versions))
    }

    fn from_store(
        persistent: TableFile<Txn::File>,
        store: Arc<Store<Txn::File>>,
        versions: Versions<Delta<Txn::File>>,
    ) -> Self {
        let ranges = *persistent.collator();
        Self {
            canonical: Arc::new(persistent),
            store,
            versions,
            semaphore: txn_lock::semaphore::Semaphore::new(ranges),
            txn: PhantomData,
        }
    }

    pub(super) async fn finish_creation(&self) -> std::io::Result<()> {
        self.store.initialize().await.map_err(std::io::Error::other)
    }

    pub(super) async fn recover<Replay, Replayed>(&self, id: TxnId, replay: Replay) -> TCResult<()>
    where
        Replay: FnOnce() -> Replayed + Send,
        Replayed: std::future::Future<Output = TCResult<()>> + Send,
    {
        self.versions
            .recover(id, replay, || tc_ir::Transact::commit(self, id))
            .await
    }

    pub fn schema(&self) -> &TableSchema {
        self.canonical.schema()
    }

    pub(super) async fn load_literal_row(&self, mut row: Vec<Value>) -> std::io::Result<()> {
        if row.len() != self.schema().column_count() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid Table row",
            ));
        }

        let values = row.split_off(self.schema().key().len());
        upsert(&self.canonical, row, values).await
    }

    /// Inspect an unpublished copy target without registering transaction reads.
    pub(super) async fn contains_literal_key(&self, key: &[Value]) -> std::io::Result<bool> {
        self.canonical.contains(key).await
    }

    /// Sync the canonical (persistent) state to disk.
    ///
    pub async fn upsert_row(&self, txn: &Txn, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(self.schema(), key)?;
        let values = b_table::Schema::validate_values(self.schema(), values)?;

        let _permit = self
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_for_key(txn, &key).await?;
        pending.upsert(key, values).await?;

        Ok(())
    }

    pub async fn insert_row(
        &self,
        txn: &Txn,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> tc_error::TCResult<()> {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(self.schema(), key)?;
        let values = b_table::Schema::validate_values(self.schema(), values)?;

        let _permit = self
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        if self.resolve_row(txn_id, &key, true).await?.is_some() {
            return Err(tc_error::TCError::bad_request(format!(
                "cannot insert Table row: key {key:?} already exists"
            )));
        }

        self.pending_delta_for_txn(txn)
            .await?
            .upsert(key, values)
            .await?;

        Ok(())
    }

    pub async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(self.schema(), key)?;

        let _permit = self
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_for_key(txn, &key).await?;
        pending.delete_from_inserts(&key).await?;

        Ok(())
    }

    pub async fn read_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<Option<Row<Value>>> {
        self.versions.check_readable(txn_id)?;
        let _permit = self
            .semaphore
            .read(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await?;

        self.resolve_row(txn_id, key, true).await
    }

    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        Ok(self.read_row(txn_id, key).await?.is_some())
    }

    pub async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.count_in(txn_id, Range::default()).await
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.is_empty_in(txn_id, Range::default()).await
    }

    pub async fn for_each_row_in_order<F>(
        &self,
        txn_id: TxnId,
        range: Range<Id, Value>,
        order: &[Id],
        reverse: bool,
        mut on_row: F,
    ) -> TCResult<()>
    where
        F: FnMut(Row<Value>),
    {
        let mut rows = self.rows(txn_id, range, order.to_vec(), reverse).await?;
        while let Some(row) = rows.try_next().await? {
            on_row(row);
        }

        Ok(())
    }

    pub async fn count_in(&self, txn_id: TxnId, range: Range<Id, Value>) -> TCResult<u64> {
        Ok(self
            .rows(txn_id, range, Vec::new(), false)
            .await?
            .try_count()
            .await?)
    }

    pub async fn is_empty_in(&self, txn_id: TxnId, range: Range<Id, Value>) -> TCResult<bool> {
        Ok(self
            .rows(txn_id, range, Vec::new(), false)
            .await?
            .try_is_empty()
            .await?)
    }

    /// Construct a permit-bound row stream over the visible state at `txn_id`.
    ///
    /// Asynchronous setup acquires the read permit, snapshot, and native streams.
    /// Subsequent rows are produced on demand by polling the returned [`Rows`].
    /// The permit is released on EOF, the first error, or drop.
    ///
    /// If no index supports `range`, the native I/O error is converted to `TCError`.
    pub async fn rows(
        &self,
        txn_id: TxnId,
        range: Range<Id, Value>,
        order: Vec<Id>,
        reverse: bool,
    ) -> TCResult<Rows> {
        let (permit, snapshot) = self
            .read_snapshot(txn_id, txn_lock::set::Range::All)
            .await?;

        let stream = self.visible_rows(snapshot, range, &order, reverse).await?;

        Ok(Rows::new(stream, permit))
    }

    /// Create a range + order + reverse view over this table.
    ///
    /// The view is structural — it holds no row data. Row streaming, count,
    /// and containment checks delegate to this table with the view's bounds.
    pub fn slice(&self, range: Range<Id, Value>, order: &[Id], reverse: bool) -> Table<Txn> {
        Table::Slice(TableSlice::new(
            self.clone(),
            range,
            order.to_vec(),
            reverse,
        ))
    }

    /// Create an ordered view over this table using the given `columns`.
    ///
    /// Equivalent to `slice(Range::default(), columns, reverse)`.
    pub fn order_by(&self, columns: &[Id], reverse: bool) -> Table<Txn> {
        self.slice(Range::default(), columns, reverse)
    }

    /// Create a row-cap view that yields at most `n` rows.
    pub fn limit(&self, n: u64) -> Table<Txn> {
        self.slice(Range::default(), &[], false).limit(n)
    }

    /// Create a column-projection view that yields only `columns`.
    pub fn select(&self, columns: &[Id]) -> tc_error::TCResult<Table<Txn>> {
        self.slice(Range::default(), &[], false).select(columns)
    }

    /// Update all visible rows in `range` at `txn_id` with the given column
    /// `values`.
    ///
    /// Only value columns may be updated. Changes stream into delegated staging
    /// storage and are applied after the input guards are released. A single
    /// whole-table write permit covers the operation; rows are not collected
    /// in memory.
    pub async fn update(
        &self,
        txn: &Txn,
        range: Range<Id, Value>,
        values: tc_ir::Map<Value>,
    ) -> TCResult<()> {
        let txn_id = txn.id();
        let value_columns = self.schema().values();
        for (name, value) in &values {
            let position = value_columns
                .iter()
                .position(|column| column == name)
                .ok_or_else(|| TCError::bad_request(format!("cannot update key column {name}")))?;

            let expected = &b_table::Schema::primary(self.schema()).column_types()
                [self.schema().key().len() + position];

            if !expected.accepts(&value.class()) {
                return Err(TCError::bad_request(format!(
                    "value column {name} expected {expected:?}"
                )));
            }
        }

        let key_len = self.schema().key().len();

        let _permit = self.try_acquire_write_permit(txn_id, txn_lock::set::Range::All)?;

        let snapshot = self.versions.visible_snapshot(txn_id)?;
        let staged = Box::pin(Delta::create(
            self.schema().clone(),
            self.store.allocate().await?,
        ))
        .await?;

        let mut visible = self.visible_rows(snapshot, range, &[], false).await?;

        while let Some(mut row) = visible.try_next().await? {
            staged
                .add_to_deletes(row[..key_len].to_vec(), row[key_len..].to_vec())
                .await?;

            for (i, name) in value_columns.iter().enumerate() {
                if let Some(value) = values.get(name) {
                    row[key_len + i] = value.clone();
                }
            }

            let mut key = row.into_vec();
            let updated_values = key.split_off(key_len);

            staged.upsert(key, updated_values).await?;
        }

        drop(visible);
        let pending = self.pending_delta_for_txn(txn).await?;
        // Mutating pending while scanning its inserts would cross native lock
        // domains and could invalidate traversal. Merge only after the scan drops.
        Box::pin(staged.apply_to_pending(&pending)).await?;
        Ok(())
    }

    /// Delete all visible rows in `range` at `txn_id`.
    ///
    /// Rows stream into delegated staging storage before removal from pending
    /// state, so the input guards are released before publication. A single
    /// whole-table write permit covers the operation without collecting rows
    /// in memory.
    pub async fn truncate(&self, txn: &Txn, range: Range<Id, Value>) -> TCResult<()> {
        let txn_id = txn.id();
        let key_len = self.schema().key().len();

        let _permit = self.try_acquire_write_permit(txn_id, txn_lock::set::Range::All)?;

        let snapshot = self.versions.visible_snapshot(txn_id)?;
        let staged = Box::pin(Delta::create(
            self.schema().clone(),
            self.store.allocate().await?,
        ))
        .await?;

        let mut visible = self.visible_rows(snapshot, range, &[], false).await?;

        while let Some(row) = visible.try_next().await? {
            let mut key = row.into_vec();
            let values = key.split_off(key_len);

            staged.add_to_deletes(key, values).await?;
        }

        drop(visible);
        let pending = self.pending_delta_for_txn(txn).await?;
        Box::pin(staged.apply_to_pending(&pending)).await?;
        Ok(())
    }

    async fn visible_rows(
        &self,
        snapshot: Vec<Arc<Delta<Txn::File>>>,
        range: Range<Id, Value>,
        order: &[Id],
        reverse: bool,
    ) -> RowsResult {
        let (base, collator) = {
            let native = self.canonical.read().await;
            Box::pin(native.rows(range.clone(), order, reverse)).await?
        };

        let mut visible = crate::stream::Overlay::new(collator, base, |current, incoming, _| {
            *current = incoming;
        });

        // Immutable layers are pinned until native streams have been dropped.
        for delta in &snapshot {
            let deletes = row_stream(&delta.deletes, range.clone(), order, reverse).await?;
            let inserts = row_stream(&delta.inserts, range.clone(), order, reverse).await?;
            visible.difference(deletes);
            visible.merge(inserts);
        }

        Ok(visible
            .into_stream()
            .map(move |item| {
                let _ = &snapshot;
                item
            })
            .boxed())
    }

    /// Preserve the pre-transaction row once, using the ordinary visibility owner.
    /// An absent baseline needs no marker: a pending insertion identifies later writes.
    async fn pending_for_key(&self, txn: &Txn, key: &[Value]) -> TCResult<Arc<Delta<Txn::File>>> {
        let pending = self.pending_delta_for_txn(txn).await?;
        if pending.is_changed(key).await? {
            return Ok(pending);
        }

        // Preserve the original indexed row only on the first change. Later
        // upserts/deletes edit pending inserts while keeping this baseline marker.
        if let Some(mut row) = self.resolve_row(txn.id(), key, false).await? {
            let values = row.drain(self.schema().key().len()..).collect();
            pending.add_to_deletes(row.into_vec(), values).await?;
        }

        Ok(pending)
    }

    async fn resolve_row(
        &self,
        txn_id: TxnId,
        key: &[Value],
        include_pending: bool,
    ) -> TCResult<Option<Row<Value>>> {
        if let Some(row) = self
            .versions
            .lookup(txn_id, include_pending, |delta| async move {
                delta.lookup(key).await
            })
            .await?
        {
            return Ok(row);
        }

        Ok(self.canonical.get_row(key).await?)
    }

    async fn pending_delta_for_txn(&self, txn: &Txn) -> TCResult<Arc<Delta<Txn::File>>> {
        self.versions
            .pending(txn.id(), || {
                let schema = self.canonical.schema().clone();
                async move {
                    let storage = self.store.allocate().await?;
                    Ok(Delta::create(schema, storage).await?)
                }
            })
            .await
    }
}

impl<Txn: crate::StorageContext> Transact for PersistentTable<Txn> {
    async fn commit(&self, txn_id: TxnId) -> TCResult<()> {
        // Persist the native delta and receipt before publishing visibility and
        // releasing this transaction's reservations.
        self.versions
            .commit_durable(
                &self.store,
                txn_id,
                |delta| async move { Ok((delta.storage.clone(), delta.units().await?)) },
                || self.semaphore.finalize(&txn_id, false).map_err(Into::into),
            )
            .await
    }

    async fn rollback(&self, txn_id: &TxnId) -> TCResult<()> {
        let discarded = self.versions.rollback(*txn_id, || {
            self.semaphore.finalize(txn_id, false).map_err(Into::into)
        })?;

        if let Some(delta) = discarded {
            let allocation = delta.storage.id().clone();
            drop(delta);
            self.store.discard(&allocation).await?;
        }

        Ok(())
    }

    async fn finalize(&self, txn_id: &TxnId) -> TCResult<()> {
        // The caller serializes cutoffs. Merge immutable native deltas before
        // atomically publishing the new frontier and releasing old reservations.
        self.versions
            .finalize_durable(
                &self.store,
                *txn_id,
                |older, newer, _complete_prefix| async move {
                    let storage = self.store.allocate().await?;
                    let output = Delta::create(self.schema().clone(), storage.clone()).await?;
                    older.apply_to_pending(&output).await?;
                    newer.apply_to_pending(&output).await?;
                    let units = output.units().await?;
                    Ok((output, Some((storage, units))))
                },
                || self.semaphore.finalize(txn_id, true).map_err(Into::into),
            )
            .await
    }
}

impl<Txn: crate::StorageContext> crate::Restore for PersistentTable<Txn> {
    type Txn = Txn;

    /// Restore through a staged indexed-row delta; see [`crate::Restore`].
    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let id = txn.id();
        self.versions.check_writable(id)?;
        if Arc::ptr_eq(&self.canonical, &source.canonical) {
            return Ok(());
        }

        if !self.schema().same_semantics(source.schema()) {
            return Err(tc_error::TCError::bad_request(
                "Table restoration schema mismatch",
            ));
        }

        let _permit = self.try_acquire_write_permit(id, txn_lock::set::Range::All)?;

        let base = self.versions.committed_snapshot(id)?;

        // Build a complete replacement relative to committed state. Excluding
        // the old pending delta lets publication replace it instead of composing
        // the restored rows with earlier writes in this transaction.

        let (source_permit, source_snapshot) =
            source.try_read_snapshot(id, txn_lock::set::Range::All)?;

        let stream = source
            .visible_rows(source_snapshot, Range::default(), &[], false)
            .await?;

        let mut rows = Rows::new(stream, source_permit);
        let storage = self.store.allocate().await?;
        let delta = Delta::create(self.schema().clone(), storage).await?;
        let key_len = self.schema().key().len();
        let mut baseline = self
            .visible_rows(base, Range::default(), &[], false)
            .await?;

        while let Some(row) = baseline.try_next().await? {
            let mut key = row.into_vec();
            let values = key.split_off(key_len);
            delta.add_to_deletes(key, values).await?;
        }

        drop(baseline);

        while let Some(row) = rows.try_next().await? {
            let mut key = row.into_vec();
            let values = key.split_off(key_len);
            delta.upsert(key, values).await?;
        }

        drop(rows);
        // Publish only after both input streams have released their native guards.
        // A decision made during staging must still reject this replacement.
        self.versions.replace_pending(id, delta, |_| Ok(()))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/table/failure.rs"]
mod failure_tests;
