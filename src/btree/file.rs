//! Core transactional BTree behavior and range/snapshot query logic.
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Bound, RangeBounds};
use std::sync::{Arc, RwLock};

use b_tree::{BTreeLock, Range, Schema};
use collate::{Collate, Collator as TxnCollator, try_diff, try_merge};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::TCError;
use tc_ir::{Transact, TxnId};
use tc_value::{Value, ValueType};

use super::stream::PersistentFile;

const UNARY_KEY_ARITY: usize = 1;

fn invalid_input_error(message: impl fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        tc_error::bad_request!("{}", message),
    )
}

fn background_error(err: impl fmt::Display) -> txn_lock::Error {
    txn_lock::Error::Background(err.to_string())
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct KeyStreamCollator {
    reverse: bool,
}

impl Collate for KeyStreamCollator {
    type Value = Vec<Value>;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> std::cmp::Ordering {
        if self.reverse {
            right.cmp(left)
        } else {
            left.cmp(right)
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StorageConfig {
    pub block_size: usize,
    pub order: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            block_size: 4_096,
            order: 64,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KeySchema {
    storage: StorageConfig,
    key_arity: usize,
    key_types: Option<Vec<ValueType>>,
}

impl KeySchema {
    fn new(
        storage: StorageConfig,
        key_arity: usize,
        key_types: Option<Vec<ValueType>>,
    ) -> Self {
        assert!(key_arity > 0, "BTree key arity must be >= 1");
        if let Some(types) = &key_types {
            assert!(
                types.len() == key_arity,
                "BTree key type list length {} must match key arity {}",
                types.len(),
                key_arity
            );
        }

        Self {
            storage,
            key_arity,
            key_types,
        }
    }
}

impl Default for KeySchema {
    fn default() -> Self {
        Self::new(StorageConfig::default(), UNARY_KEY_ARITY, None)
    }
}

impl Schema for KeySchema {
    type Error = TCError;
    type Value = Value;

    fn block_size(&self) -> usize {
        self.storage.block_size
    }

    fn len(&self) -> usize {
        self.key_arity
    }

    fn order(&self) -> usize {
        self.storage.order
    }

    fn validate_key(&self, key: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        if key.len() != self.key_arity {
            return Err(tc_error::bad_request!(
                "tc-collection BTree keys must have arity {}",
                self.key_arity
            ));
        }

        if let Some(key_types) = &self.key_types {
            for (i, (value, expected)) in key.iter().zip(key_types.iter()).enumerate() {
                let actual = value.class();
                if &actual != expected {
                    return Err(tc_error::bad_request!(
                        "tc-collection BTree key column {i} expected {:?} but got {:?}",
                        expected,
                        actual
                    ));
                }
            }
        }

        Ok(key)
    }
}

#[derive(Clone)]
struct PersistentStore {
    tree: BTreeLock<KeySchema, b_tree::collate::Collator<Value>, PersistentFile>,
}

impl PersistentStore {
    fn key_schema(&self) -> KeySchema {
        self.tree.schema().clone()
    }

    fn from_dir(
        dir: DirLock<PersistentFile>,
        storage: StorageConfig,
        key_arity: usize,
        key_types: Option<Vec<ValueType>>,
    ) -> std::io::Result<Self> {
        if let Some(types) = &key_types {
            if types.len() != key_arity {
                return Err(invalid_input_error(format!(
                    "BTree key type list length {} must match key arity {}",
                    types.len(),
                    key_arity
                )));
            }
        }

        let schema = KeySchema::new(storage, key_arity, key_types);
        let tree = BTreeLock::load(schema, b_tree::collate::Collator::default(), dir)?;
        Ok(Self { tree })
    }

    async fn key_stream_in(
        &self,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> std::io::Result<b_tree::Keys<Value>> {
        let view = self.tree.read().await;
        let range = Range::with_bounds(Vec::<Value>::new(), bounds);

        if reverse {
            view.keys_rev(range).await
        } else {
            view.keys(range).await
        }
    }

    async fn contains_key(&self, key: &[Value]) -> std::io::Result<bool> {
        let view = self.tree.read().await;
        view.contains(key).await
    }

    async fn insert_key(&self, key: Vec<Value>) -> std::io::Result<()> {
        let mut view = self.tree.write().await;
        let _ = view.insert(key).await?;
        Ok(())
    }

    async fn delete_key(&self, key: &[Value]) -> std::io::Result<()> {
        let mut view = self.tree.write().await;
        let _ = view.delete(key).await?;
        Ok(())
    }

    async fn apply_delta(&self, delta: &Delta) -> std::io::Result<()> {
        let mut view = self.tree.write().await;

        {
            let insert_view = delta.inserts.tree.read().await;
            let mut stream = insert_view.keys(Range::<Value>::default()).await?;
            while let Some(key) = stream.try_next().await? {
                let _ = view.insert(key.to_vec()).await?;
            }
        }

        {
            let delete_view = delta.deletes.tree.read().await;
            let mut stream = delete_view.keys(Range::<Value>::default()).await?;
            while let Some(key) = stream.try_next().await? {
                let _ = view.delete(&key).await?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
struct Delta {
    inserts: PersistentStore,
    deletes: PersistentStore,
}

impl Delta {
    async fn insert(&self, key: Vec<Value>) -> std::io::Result<()> {
        self.deletes.delete_key(&key).await?;

        self.inserts.insert_key(key).await
    }

    async fn delete(&self, key: Vec<Value>) -> std::io::Result<()> {
        self.inserts.delete_key(&key).await?;

        self.deletes.insert_key(key).await
    }
}

#[derive(Clone)]
struct State {
    persistent: PersistentStore,
    committed: BTreeMap<TxnId, Delta>,
    pending: BTreeMap<TxnId, Delta>,
    finalized: Option<TxnId>,
    txn_root: DirLock<PersistentFile>,
}

#[derive(Clone)]
struct VisibleSnapshot {
    persistent: PersistentStore,
    deltas: Vec<Delta>,
}

#[derive(Clone)]
pub struct BTree {
    state: Arc<RwLock<State>>,
    semaphore: txn_lock::semaphore::Semaphore<
        TxnId,
        TxnCollator<Vec<Value>>,
        txn_lock::set::Range<Vec<Value>>,
    >,
}

#[derive(Debug, Clone)]
pub struct BTreeSlice {
    btree: BTree,
    lower: Bound<Value>,
    upper: Bound<Value>,
    reverse: bool,
}

impl fmt::Debug for BTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read().expect("state read lock");
        f.debug_struct("BTree")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl BTree {
    /// Construct a transactional BTree with default unary-key schema.
    pub fn new(persistent_dir: DirLock<PersistentFile>, txn_root: DirLock<PersistentFile>) -> Self {
        Self::with_storage_and_key_types(
            persistent_dir,
            txn_root,
            StorageConfig::default(),
            UNARY_KEY_ARITY,
            None,
        )
    }

    pub fn with_key_types(
        persistent_dir: DirLock<PersistentFile>,
        txn_root: DirLock<PersistentFile>,
        key_types: Vec<ValueType>,
    ) -> Self {
        let key_arity = key_types.len();
        Self::with_storage_and_key_types(
            persistent_dir,
            txn_root,
            StorageConfig::default(),
            key_arity,
            Some(key_types),
        )
    }

    pub fn with_storage_and_key_types(
        persistent_dir: DirLock<PersistentFile>,
        txn_root: DirLock<PersistentFile>,
        storage: StorageConfig,
        key_arity: usize,
        key_types: Option<Vec<ValueType>>,
    ) -> Self {
        let persistent = Self::load_store(persistent_dir, storage, key_arity, key_types);

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
        }
    }

    pub fn finalized(&self) -> Option<TxnId> {
        self.state.read().expect("state read lock").finalized
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        let persistent = {
            let state = self.state.read().expect("state read lock");
            state.persistent.clone()
        };

        persistent
            .key_stream_in((Bound::Unbounded, Bound::Unbounded), false)
            .await
    }

    pub fn slice<R>(&self, range: R, reverse: bool) -> BTreeSlice
    where
        R: RangeBounds<Value>,
    {
        BTreeSlice {
            btree: self.clone(),
            lower: Self::clone_bound(range.start_bound()),
            upper: Self::clone_bound(range.end_bound()),
            reverse,
        }
    }

    pub async fn insert_row(&self, txn_id: TxnId, key: Vec<Value>) -> Result<(), txn_lock::Error> {
        // Reserve a write lock for this exact key range at txn_id.
        // This is the canonical ordering gate which enforces conflict semantics across txns.
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn_id).await?;
        pending.insert(key).await.map_err(background_error)?;

        Ok(())
    }

    pub async fn delete_row(&self, txn_id: TxnId, key: Vec<Value>) -> Result<(), txn_lock::Error> {
        // Deletions take the same key-scoped write reservation as inserts.
        let _permit = self
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn_id).await?;
        pending.delete(key).await.map_err(background_error)?;

        Ok(())
    }

    /// Commit the pending delta at `txn_id`.
    ///
    /// This may return `Conflict` or `Outdated` when lifecycle ordering rules are violated.
    pub fn commit(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        // Commit mutates txn lifecycle state globally, so reserve the full range.
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
        // Release waiters blocked on this txn's reservations.
        self.release_txn_reservation(txn_id);

        Ok(())
    }

    /// Roll back the pending delta at `txn_id`.
    ///
    /// This may return `Conflict` or `Outdated` when lifecycle ordering rules are violated.
    pub fn rollback(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        // Rollback also mutates txn lifecycle state globally.
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
        // Rollback completion unblocks later readers/writers on overlapping ranges.
        self.release_txn_reservation(txn_id);

        Ok(())
    }

    /// Finalize all committed deltas up to `txn_id` into persistent state.
    ///
    /// Finalize is monotonic. A stale finalize is a no-op.
    pub async fn finalize(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        // Finalize advances the global visibility frontier; reserve full range.
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

        // drop_past=true clears semaphore versions up to txn_id, matching finalize frontier pruning.
        self.release_txn_frontier(txn_id);

        Ok(())
    }

    /// Return row visibility at `txn_id` for `key`.
    ///
    /// If an earlier overlapping pending write exists, this call waits until that
    /// transaction resolves (commit/rollback/finalize) before reading.
    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> bool {
        // Canonical transactional read behavior: later reads wait behind earlier overlapping
        // pending writes until the earlier txn is finalized.
        let _permit = self
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await;

        let snapshot = self.visible_snapshot(txn_id);
        self.is_row_visible(&snapshot, key).await
    }

    pub async fn count(&self, txn_id: TxnId) -> u64 {
        self.count_in(
            txn_id,
            (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
        )
        .await
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> bool {
        !self
            .any_row_in(
                txn_id,
                (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                false,
            )
            .await
    }

    /// Iterate rows visible at `txn_id` in range order.
    ///
    /// This holds an `All`-range read reservation for the full scan to keep a
    /// transactionally coherent view across stream composition.
    pub async fn for_each_row_in_order<R, F>(
        &self,
        txn_id: TxnId,
        range: R,
        reverse: bool,
        mut on_key: F,
    ) where
        R: RangeBounds<Value>,
        F: FnMut(Vec<Value>),
    {
        // Range scans use an All-range read reservation so the snapshot and stream composition
        // are transactionally coherent with pending/commit/finalize transitions.
        let _permit = self
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await;

        let bounds = (
            Self::clone_bound(range.start_bound()),
            Self::clone_bound(range.end_bound()),
        );

        self.for_each_visible_key_in_order_until(txn_id, bounds, reverse, |key| {
            on_key(key);
            true
        })
        .await;
    }

    pub async fn count_in<R>(&self, txn_id: TxnId, range: R) -> u64
    where
        R: RangeBounds<Value>,
    {
        let mut count = 0_u64;
        self.for_each_row_in_order(txn_id, range, false, |_| {
            count += 1;
        })
        .await;

        count
    }

    async fn any_row_in<R>(&self, txn_id: TxnId, range: R, reverse: bool) -> bool
    where
        R: RangeBounds<Value>,
    {
        let bounds = (
            Self::clone_bound(range.start_bound()),
            Self::clone_bound(range.end_bound()),
        );

        let mut found = false;
        self.for_each_visible_key_in_order_until(txn_id, bounds, reverse, |_| {
            found = true;
            false
        })
        .await;

        found
    }

    async fn for_each_visible_key_in_order_until<F>(
        &self,
        txn_id: TxnId,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
        mut on_key: F,
    ) where
        F: FnMut(Vec<Value>) -> bool,
    {
        let snapshot = self.visible_snapshot(txn_id);
        let collator = KeyStreamCollator { reverse };

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> = snapshot
            .persistent
            .key_stream_in(bounds.clone(), reverse)
            .await
            .expect("stream persistent keys")
            .map_ok(|row| row.to_vec())
            .boxed();

        for delta in &snapshot.deltas {
            let deletes = delta
                .deletes
                .key_stream_in(bounds.clone(), reverse)
                .await
                .expect("stream delete delta keys")
                .map_ok(|row| row.to_vec())
                .boxed();

            visible = try_diff(collator, visible, deletes).boxed();

            let inserts = delta
                .inserts
                .key_stream_in(bounds.clone(), reverse)
                .await
                .expect("stream insert delta keys")
                .map_ok(|row| row.to_vec())
                .boxed();

            visible = try_merge(collator, visible, inserts).boxed();
        }

        while let Some(key) = visible.try_next().await.expect("read visible key stream") {
            if !on_key(key) {
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
        // Use blocking semaphore::read (not try_read) to preserve canonical txn semantics:
        // later overlapping reads wait for earlier pending writes to commit/rollback/finalize.
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

    async fn is_row_visible(&self, snapshot: &VisibleSnapshot, key: &[Value]) -> bool {
        let mut visible = snapshot
            .persistent
            .contains_key(key)
            .await
            .expect("check persistent visibility");

        for delta in &snapshot.deltas {
            if delta
                .deletes
                .contains_key(key)
                .await
                .expect("check delete delta visibility")
            {
                visible = false;
            }

            if delta
                .inserts
                .contains_key(key)
                .await
                .expect("check insert delta visibility")
            {
                visible = true;
            }
        }

        visible
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
        storage: StorageConfig,
        key_arity: usize,
        key_types: Option<Vec<ValueType>>,
    ) -> PersistentStore {
        PersistentStore::from_dir(persistent_dir, storage, key_arity, key_types)
            .expect("load persistent BTree store")
    }

    async fn pending_delta_for_txn(&self, txn_id: TxnId) -> Result<Delta, txn_lock::Error> {
        let (txn_root, key_schema) = {
            let state = self.state.write().expect("state write lock");
            Self::assert_writable_state(&state, txn_id)?;

            if let Some(pending) = state.pending.get(&txn_id).cloned() {
                return Ok(pending);
            }

            (state.txn_root.clone(), state.persistent.key_schema())
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

        let (inserts, deletes) = {
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
            inserts: PersistentStore::from_dir(
                inserts,
                key_schema.storage,
                key_schema.key_arity,
                key_schema.key_types.clone(),
            )
            .map_err(background_error)?,
            deletes: PersistentStore::from_dir(
                deletes,
                key_schema.storage,
                key_schema.key_arity,
                key_schema.key_types,
            )
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

    fn clone_bound(bound: Bound<&Value>) -> Bound<Value> {
        match bound {
            Bound::Included(value) => Bound::Included(value.clone()),
            Bound::Excluded(value) => Bound::Excluded(value.clone()),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
}

impl BTreeSlice {
    pub async fn count(&self, txn_id: TxnId) -> u64 {
        self.btree
            .count_in(txn_id, (self.lower.clone(), self.upper.clone()))
            .await
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> bool {
        !self
            .btree
            .any_row_in(
                txn_id,
                (self.lower.clone(), self.upper.clone()),
                self.reverse,
            )
            .await
    }

    pub async fn for_each_row_in_order<F>(&self, txn_id: TxnId, on_row: F)
    where
        F: FnMut(Vec<Value>),
    {
        self.btree
            .for_each_row_in_order(
                txn_id,
                (self.lower.clone(), self.upper.clone()),
                self.reverse,
                on_row,
            )
            .await;
    }
}

impl Transact for BTree {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        BTree::commit(self, txn_id).expect("BTree commit failed");
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            BTree::rollback(self, txn_id).expect("BTree rollback failed");
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            BTree::finalize(self, txn_id)
                .await
                .expect("BTree finalize failed");
        }
    }
}
