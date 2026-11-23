//! Transactional row ownership over native indexed Tables.
//!
//! `KeyOwner` decides visibility and lifecycle; native b-table owners maintain
//! indexes in canonical storage and each delta. Deletes retain the baseline row
//! so scans through any index can suppress its old entry before adding the new
//! row. See TRANSACTIONAL_COLLECTION_CONTRACT.md for caller obligations.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use b_table::{IndexSchema, Range, Row, TableLock};
use freqfs::DirLock;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Transact, TxnId};
use tc_value::{Value, ValueCollator};

use super::schema::{TableIndexSchema, TableSchema};
use super::stream::Rows;
use super::view::{Limited, Selection, TableSlice};
use crate::persistence::{CollectionOwner, KeyOwner, PersistentDelta, PointRead, VisibleSnapshot};

type RowsResult = std::io::Result<BoxStream<'static, Result<Row<Value>, std::io::Error>>>;
pub(crate) type TableFile<F> = TableLock<TableSchema, TableIndexSchema, ValueCollator, F>;

pub(crate) async fn row_stream<F: crate::CollectionFile>(
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
        .map_err(|error| {
            if error.requires_recovery() {
                freqfs::IntegrityError::mark(std::io::Error::other(error))
            } else {
                std::io::Error::other(error)
            }
        })
}

pub(crate) async fn delete_row<F: crate::CollectionFile>(
    table: &TableFile<F>,
    key: &[Value],
) -> std::io::Result<()> {
    table.write().await.delete_row(key).await.map(|_| ())
}

struct Delta<F: crate::CollectionFile> {
    // A lookup distinguishes untouched (None), deleted (Some(None)), and replaced
    // (Some(Some(row))). Inserts take precedence when both tables contain a key.
    inserts: TableFile<F>,
    deletes: TableFile<F>,
}

impl<F: crate::CollectionFile> Delta<F> {
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
    async fn create(schema: TableSchema, dir: DirLock<F>) -> std::io::Result<Arc<Self>> {
        let (inserts, deletes) = crate::persistence::create_delta_dirs(&dir).await?;
        Ok(Arc::new(Self {
            inserts: TableLock::create(schema.clone(), crate::persistence::value_order(), inserts)
                .await?,
            deletes: TableLock::create(schema, crate::persistence::value_order(), deletes).await?,
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

impl<F: crate::CollectionFile> PersistentDelta for Delta<F> {
    type Native = TableFile<F>;

    async fn apply_to(&self, persistent: &TableFile<F>) -> std::io::Result<()> {
        let key_len = persistent.schema().key().len();
        let mut deletes = row_stream(&self.deletes, Range::default(), &[], false).await?;
        while let Some(row) = deletes.try_next().await? {
            delete_row(persistent, &row[..key_len]).await?;
        }

        drop(deletes);
        let mut inserts = row_stream(&self.inserts, Range::default(), &[], false).await?;
        while let Some(row) = inserts.try_next().await? {
            let mut key = row.into_vec();
            let values = key.split_off(key_len);
            upsert(persistent, key, values).await?;
        }

        Ok(())
    }
}

pub struct PersistentTable<Txn: crate::StorageContext> {
    owner: Arc<KeyOwner<Delta<Txn::File>>>,
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
        let state = self.owner.read_state();

        f.debug_struct("PersistentTable")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl<Txn: crate::StorageContext> PersistentTable<Txn> {
    /// Create in empty caller-delegated storage.
    pub async fn try_new(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let persistent = TableLock::create(
            schema.clone(),
            crate::persistence::value_order(),
            persistent_dir,
        )
        .await?;

        Ok(Self::from_store(schema, persistent))
    }

    pub async fn load(
        persistent_dir: DirLock<Txn::File>,
        schema: TableSchema,
    ) -> std::io::Result<Self> {
        let persistent = TableLock::load(
            schema.clone(),
            crate::persistence::value_order(),
            persistent_dir,
        )?;

        persistent.validate().await?;
        Ok(Self::from_store(schema, persistent))
    }

    fn from_store(schema: TableSchema, persistent: TableFile<Txn::File>) -> Self {
        let ranges = *persistent.ordering().range_collator();
        Self {
            owner: Arc::new(CollectionOwner::new(persistent, ranges)),
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
        let persistent = { self.owner.read_state().persistent.clone() };

        upsert(&persistent, row, values).await
    }

    /// Inspect an unpublished copy target without registering transaction reads.
    pub(super) async fn contains_literal_key(&self, key: &[Value]) -> std::io::Result<bool> {
        let persistent = self.owner.read_state().persistent.clone();
        persistent.contains(key).await
    }

    pub fn finalized(&self) -> TCResult<Option<TxnId>> {
        self.owner.finalized()
    }

    /// Sync the canonical (persistent) state to disk.
    ///
    /// Buffered writeback does not acknowledge durability. Pending workspace
    /// and committed workspace deltas are not included.
    pub async fn sync(&self) -> std::io::Result<()> {
        let persistent = self.owner.read_state().persistent.clone();
        persistent.sync().await?;
        Ok(())
    }

    /// Durably synchronize materialized canonical storage; pending deltas are excluded.
    pub async fn sync_all(&self) -> std::io::Result<()> {
        let persistent = self.owner.read_state().persistent.clone();
        persistent.sync_all().await?;
        Ok(())
    }

    pub async fn upsert_row(&self, txn: &Txn, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        let key = b_table::Schema::validate_key(&self.schema, key)?;
        let values = b_table::Schema::validate_values(&self.schema, values)?;

        let _permit = self
            .owner
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
        let key = b_table::Schema::validate_key(&self.schema, key)?;
        let values = b_table::Schema::validate_values(&self.schema, values)?;

        let _permit = self
            .owner
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        if Self::resolve_row(self.owner.point_read(txn_id)?, &key)
            .await?
            .is_some()
        {
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
        let key = b_table::Schema::validate_key(&self.schema, key)?;

        let _permit = self
            .owner
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_for_key(txn, &key).await?;
        pending.delete_from_inserts(&key).await?;

        Ok(())
    }

    pub async fn read_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<Option<Row<Value>>> {
        let _permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await?;

        Self::resolve_row(self.owner.point_read(txn_id)?, key).await
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
        let permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await?;

        let snapshot = self.owner.visible_snapshot(txn_id)?;
        let stream = self.visible_rows(snapshot, range, &order, reverse).await?;

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
        let value_columns = self.schema.values();
        for (name, value) in &values {
            let position = value_columns
                .iter()
                .position(|column| column == name)
                .ok_or_else(|| TCError::bad_request(format!("cannot update key column {name}")))?;

            let expected = &b_table::Schema::primary(&self.schema).column_types()
                [self.schema.key().len() + position];

            if !expected.accepts(&value.class()) {
                return Err(TCError::bad_request(format!(
                    "value column {name} expected {expected:?}"
                )));
            }
        }

        let key_len = self.schema.key().len();

        let _permit = self
            .owner
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::All)?;

        let snapshot = self.owner.visible_snapshot(txn_id)?;
        let staged = Box::pin(Delta::create(
            self.schema.clone(),
            txn.subcontext_unique().context().await?,
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
        let key_len = self.schema.key().len();

        let _permit = self
            .owner
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::All)?;

        let snapshot = self.owner.visible_snapshot(txn_id)?;
        let staged = Box::pin(Delta::create(
            self.schema.clone(),
            txn.subcontext_unique().context().await?,
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
        snapshot: VisibleSnapshot<Delta<Txn::File>>,
        range: Range<Id, Value>,
        order: &[Id],
        reverse: bool,
    ) -> RowsResult {
        let (base, collator) = {
            let native = snapshot.persistent.read().await;
            Box::pin(native.rows(range.clone(), order, reverse)).await?
        };

        let mut visible = crate::stream::Overlay::new(collator, base, |current, incoming, _| {
            *current = incoming;
        });

        // Oldest committed layer first, then this transaction's pending layer:
        // delete its indexed baseline entries before merging replacement rows.
        // Check retirement on both sides of I/O because finalization can move a
        // layer into canonical storage while native streams are being opened.
        for (id, delta) in snapshot.deltas {
            if snapshot.visibility.retired(id) {
                continue;
            }

            let sources = async {
                let deletes = row_stream(&delta.deletes, range.clone(), order, reverse).await?;
                let inserts = row_stream(&delta.inserts, range.clone(), order, reverse).await?;
                Ok::<_, std::io::Error>((deletes, inserts))
            }
            .await;

            if !snapshot.visibility.retired(id) {
                let (deletes, inserts) = sources?;
                visible.difference(deletes);
                visible.merge(inserts);
            }
        }

        Ok(visible.into_stream())
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
        if let Some(mut row) =
            Self::resolve_row(self.owner.point_read(txn.id())?.committed(), key).await?
        {
            let values = row.drain(self.schema.key().len()..).collect();
            pending.add_to_deletes(row.into_vec(), values).await?;
        }

        Ok(pending)
    }

    async fn resolve_row(
        mut read: PointRead<Delta<Txn::File>>,
        key: &[Value],
    ) -> TCResult<Option<Row<Value>>> {
        if let Some(row) = read
            .resolve(|delta| async move { delta.lookup(key).await })
            .await?
        {
            return Ok(row);
        }

        Ok(read.canonical()?.get_row(key).await?)
    }

    async fn pending_delta_for_txn(&self, txn: &Txn) -> TCResult<Arc<Delta<Txn::File>>> {
        self.owner
            .pending(txn.id(), |native| {
                let schema = native.schema().clone();
                async move {
                    let dir = txn.subcontext_unique().context().await?;
                    Ok(Delta::create(schema, dir).await?)
                }
            })
            .await
    }
}

impl<Txn: crate::StorageContext> PersistentTable<Txn> {
    /// Commit the pending delta at `txn_id`.
    ///
    /// The caller must finish this transaction's mutations first. Immutable result
    /// streams may retain their snapshots across visibility publication.
    /// Returns `Outdated` at or before the finalized frontier.
    pub async fn commit(&self, txn_id: TxnId) -> TCResult<()> {
        self.owner.commit(txn_id)
    }

    pub fn rollback(&self, txn_id: TxnId) -> TCResult<()> {
        self.owner.rollback(txn_id)
    }

    /// Finalize all committed deltas up to `txn_id` into persistent state.
    ///
    /// Finalize is monotonic. A stale finalize (≤ frontier) is a no-op.
    ///
    /// The caller serializes lifecycle decisions and finishes operations through the
    /// cutoff first. Later transactions may retain read permits; merging uses native
    /// storage locks without acquiring a new semaphore write reservation.
    pub async fn finalize(&self, txn_id: TxnId) -> TCResult<()> {
        self.owner.finalize(txn_id).await
    }
}

impl<Txn: crate::StorageContext> Transact for PersistentTable<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        PersistentTable::commit(self, txn_id).await
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { PersistentTable::rollback(self, txn_id) }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { PersistentTable::finalize(self, txn_id).await }
    }
}

impl<Txn: crate::StorageContext> crate::Restore for PersistentTable<Txn> {
    type Txn = Txn;

    /// Restore through a staged indexed-row delta; see [`crate::Restore`].
    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let id = txn.id();
        self.owner.read_state().assert_writable(id)?;
        if Arc::ptr_eq(&self.owner, &source.owner) {
            return Ok(());
        }

        if !self.schema.same_semantics(&source.schema) {
            return Err(tc_error::TCError::bad_request(
                "Table restoration schema mismatch",
            ));
        }

        let _permit = self
            .owner
            .try_acquire_write_permit(id, txn_lock::set::Range::All)?;

        let base = self.owner.committed_snapshot(id)?;

        // Build a complete replacement relative to committed state. Excluding
        // the old pending delta lets publication replace it instead of composing
        // the restored rows with earlier writes in this transaction.

        let source_permit = source
            .owner
            .try_acquire_read_permit(id, txn_lock::set::Range::All)?;

        let source_snapshot = source.owner.visible_snapshot(id)?;
        let stream = source
            .visible_rows(source_snapshot, Range::default(), &[], false)
            .await?;

        let mut rows = Rows::new(stream, source_permit);
        let workspace = txn.subcontext_unique().context().await?;
        let delta = Delta::create(self.schema.clone(), workspace).await?;
        let key_len = self.schema.key().len();
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
        let mut state = self.owner.write_state();
        state.assert_writable(id)?;
        state.pending.insert(id, delta);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/table/failure.rs"]
mod failure_tests;
