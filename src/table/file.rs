use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use b_table::{Range, Row, TableLock};
use collate::{try_diff, try_merge, Collate, Collator as TxnCollator};
use collate::Collator;
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_ir::{Transact, TxnId};
use tc_value::Value;

use super::schema::{TableIndexSchema, TableSchema};
use crate::btree::{BTreeStorageConfig, PersistentFile};

fn background_error(err: impl fmt::Display) -> txn_lock::Error {
    txn_lock::Error::Background(err.to_string())
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct RowCollator {
    key_len: usize,
    reverse: bool,
}

impl Collate for RowCollator {
    type Value = Vec<Value>;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> std::cmp::Ordering {
        let lk = &left[..self.key_len.min(left.len())];
        let rk = &right[..self.key_len.min(right.len())];
        let ord = lk.cmp(rk);
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
        _storage: BTreeStorageConfig,
    ) -> std::io::Result<Self> {
        let collator = Collator::<Value>::default();
        let table = TableLock::load(schema, collator, dir)?;
        Ok(Self { table })
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
        let view = self.table.read().await;
        view.get_row(key).await
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
        Ok(self.deletes.get_row(key).await?.is_some())
    }

    async fn merge_into<'a>(
        &'a self,
        rows: BoxStream<'a, Result<Vec<Value>, std::io::Error>>,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
        key_len: usize,
    ) -> BoxStream<'a, Result<Vec<Value>, std::io::Error>> {
        let collator = RowCollator { key_len, reverse };

        let inserted = self
            .inserts
            .row_stream(range.clone(), order, reverse)
            .await
            .expect("stream insert delta rows")
            .map_ok(|row| row.to_vec())
            .boxed();

        let merged = try_merge(collator, inserted, rows).boxed();

        let deleted = self
            .deletes
            .row_stream(range, order, reverse)
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

        let key_len = self.schema.key().len();

        self.for_each_visible_row_until(txn_id, range, order, reverse, key_len, |row| {
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

    pub async fn finalize(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::All)?;

        let (persistent, committed_to_apply) = {
            let state = self.state.write().expect("state write lock");

            if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
                drop(state);
                self.release_txn_reservation(txn_id);
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

    async fn any_row_in(
        &self,
        txn_id: TxnId,
        range: Range<tc_ir::Id, Value>,
        order: &[tc_ir::Id],
        reverse: bool,
    ) -> bool {
        let key_len = self.schema.key().len();

        let mut found = false;
        self.for_each_visible_row_until(txn_id, range, order, reverse, key_len, |_| {
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
        key_len: usize,
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
                .merge_into(visible, range.clone(), order, reverse, key_len)
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
        for delta in &snapshot.deltas {
            if let Some(row) = delta.inserts.get_row(key).await.expect("check insert delta") {
                return Some(row);
            }
            if delta.deletes.get_row(key).await.expect("check delete delta").is_some() {
                return None;
            }
        }

        snapshot
            .persistent
            .get_row(key)
            .await
            .expect("check persistent visibility")
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
