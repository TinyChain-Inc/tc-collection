//! Core transactional BTree behavior and range/snapshot query logic.
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use b_tree::{BTreeLock, Range, Schema};
use collate::{Collate, try_diff, try_merge};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::TCError;
use tc_ir::{Transact, TxnId};
use tc_value::{Value, ValueCollator, ValueType};

use crate::persistence::{CollectionOwner, PersistentDelta, VisibleSnapshot, background_error};

const UNARY_KEY_ARITY: usize = 1;

fn invalid_input_error(message: impl fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        tc_error::bad_request!("{}", message),
    )
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct KeyStreamCollator {
    values: b_tree::Collator<ValueCollator>,
    reverse: bool,
}

impl KeyStreamCollator {
    fn new(reverse: bool) -> Self {
        Self {
            values: b_tree::Collator::new(ValueCollator::default()),
            reverse,
        }
    }
}

impl Collate for KeyStreamCollator {
    type Value = Vec<Value>;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> std::cmp::Ordering {
        let ordering = self.values.cmp(left, right);
        if self.reverse {
            ordering.reverse()
        } else {
            ordering
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
pub struct BTreeSchema {
    storage: StorageConfig,
    key_arity: usize,
    key_types: Option<Vec<ValueType>>,
}

impl BTreeSchema {
    pub fn new(
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

    pub(crate) fn from_key_types(key_types: Vec<ValueType>) -> Self {
        let key_arity = key_types.len();
        Self::new(StorageConfig::default(), key_arity, Some(key_types))
    }

    fn normalize_row(&self, row: Value) -> Result<Vec<Value>, TCError> {
        let key = if self.key_arity == UNARY_KEY_ARITY {
            vec![row]
        } else {
            match row {
                Value::Tuple(values) if values.len() == self.key_arity => values,
                Value::Tuple(values) => {
                    return Err(tc_error::bad_request!(
                        "tc-collection BTree key arity {} does not match schema arity {}",
                        values.len(),
                        self.key_arity
                    ));
                }
                value => {
                    return Err(tc_error::bad_request!(
                        "tc-collection BTree key must be a tuple of length {} but got {value:?}",
                        self.key_arity
                    ));
                }
            }
        };

        self.validate_key(key)
    }
}

impl Default for BTreeSchema {
    fn default() -> Self {
        Self::new(StorageConfig::default(), UNARY_KEY_ARITY, None)
    }
}

impl Schema for BTreeSchema {
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

async fn key_stream_in<F: crate::CollectionFile>(
    tree: &BTreeLock<BTreeSchema, ValueCollator, F>,
    bounds: (Bound<Value>, Bound<Value>),
    reverse: bool,
) -> std::io::Result<b_tree::Keys<Value>> {
    let view = tree.read().await;
    let range = Range::with_bounds(Vec::<Value>::new(), bounds);
    if reverse {
        view.keys_rev(range).await
    } else {
        view.keys(range).await
    }
}

async fn contains_key<F: crate::CollectionFile>(
    tree: &BTreeLock<BTreeSchema, ValueCollator, F>,
    key: &[Value],
) -> std::io::Result<bool> {
    tree.read().await.contains(key).await
}

async fn insert_key<F: crate::CollectionFile>(
    tree: &BTreeLock<BTreeSchema, ValueCollator, F>,
    key: Vec<Value>,
) -> std::io::Result<()> {
    tree.write().await.insert(key).await.map(|_| ())
}

async fn delete_key<F: crate::CollectionFile>(
    tree: &BTreeLock<BTreeSchema, ValueCollator, F>,
    key: &[Value],
) -> std::io::Result<()> {
    tree.write().await.delete(key).await.map(|_| ())
}

#[derive(Clone)]
struct Delta<F: crate::CollectionFile> {
    inserts: BTreeLock<BTreeSchema, ValueCollator, F>,
    deletes: BTreeLock<BTreeSchema, ValueCollator, F>,
}

impl<F: crate::CollectionFile> Delta<F> {
    async fn create(schema: BTreeSchema, dir: DirLock<F>) -> std::io::Result<Self> {
        let (inserts, deletes) = crate::persistence::create_delta_dirs(&dir).await?;
        Ok(Self {
            inserts: BTreeLock::create(schema.clone(), ValueCollator::default(), inserts)?,
            deletes: BTreeLock::create(schema, ValueCollator::default(), deletes)?,
        })
    }

    async fn replacement(
        canonical: &BTreeLock<BTreeSchema, ValueCollator, F>,
        dir: DirLock<F>,
    ) -> std::io::Result<Self> {
        let (inserts, deletes) = crate::persistence::create_delta_dirs(&dir).await?;
        Ok(Self {
            inserts: BTreeLock::create(
                canonical.schema().clone(),
                ValueCollator::default(),
                inserts,
            )?,
            deletes: canonical.copy_into(deletes).await?,
        })
    }

    async fn insert(&self, key: Vec<Value>) -> std::io::Result<()> {
        delete_key(&self.deletes, &key).await?;

        insert_key(&self.inserts, key).await
    }

    async fn delete(&self, key: Vec<Value>) -> std::io::Result<()> {
        delete_key(&self.inserts, &key).await?;

        insert_key(&self.deletes, key).await
    }
}

impl<F: crate::CollectionFile> PersistentDelta for Delta<F> {
    type Native = BTreeLock<BTreeSchema, ValueCollator, F>;

    async fn apply_to(
        &self,
        persistent: &BTreeLock<BTreeSchema, ValueCollator, F>,
    ) -> std::io::Result<()> {
        let mut stream =
            key_stream_in(&self.inserts, (Bound::Unbounded, Bound::Unbounded), false).await?;
        while let Some(key) = stream.try_next().await? {
            insert_key(persistent, key.to_vec()).await?;
        }

        let mut stream =
            key_stream_in(&self.deletes, (Bound::Unbounded, Bound::Unbounded), false).await?;
        while let Some(key) = stream.try_next().await? {
            delete_key(persistent, &key).await?;
        }

        Ok(())
    }
}

pub struct BTree<Txn: crate::StorageContext> {
    owner: Arc<CollectionOwner<Delta<Txn::File>>>,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: crate::StorageContext> Clone for BTree<Txn> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            txn: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BTreeSlice<Txn: crate::StorageContext> {
    btree: BTree<Txn>,
    lower: Bound<Value>,
    upper: Bound<Value>,
    reverse: bool,
}

impl<Txn: crate::StorageContext> fmt::Debug for BTree<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.owner.state.read().expect("state read lock");
        f.debug_struct("BTree")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl<Txn: crate::StorageContext> BTree<Txn> {
    /// Construct a transactional BTree with default unary-key schema.
    pub fn new(persistent_dir: DirLock<Txn::File>) -> Self {
        Self::with_schema(persistent_dir, BTreeSchema::default())
    }

    pub fn with_schema(persistent_dir: DirLock<Txn::File>, schema: BTreeSchema) -> Self {
        Self::try_with_schema(persistent_dir, schema).expect("create persistent BTree store")
    }

    /// Create in empty caller-delegated storage.
    pub fn try_with_schema(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let persistent =
            BTreeLock::create(schema.clone(), ValueCollator::default(), persistent_dir)?;
        Ok(Self::from_store(persistent))
    }

    pub async fn load(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let persistent = BTreeLock::load(schema.clone(), ValueCollator::default(), persistent_dir)?;
        persistent.validate().await?;
        Ok(Self::from_store(persistent))
    }

    fn from_store(persistent: BTreeLock<BTreeSchema, ValueCollator, Txn::File>) -> Self {
        Self {
            owner: Arc::new(CollectionOwner::new(persistent)),
            txn: PhantomData,
        }
    }

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
        let schema = canonical.schema().clone();
        let source_schema = source
            .owner
            .state
            .read()
            .expect("state read lock")
            .persistent
            .schema()
            .clone();
        if schema.key_arity != source_schema.key_arity
            || schema.key_types != source_schema.key_types
        {
            return Err(TCError::bad_request("BTree restoration schema mismatch"));
        }
        let workspace = txn.subcontext_unique().context().await?;
        let delta = Delta::replacement(&canonical, workspace).await?;
        for committed in committed {
            committed.apply_to(&delta.deletes).await?;
        }
        let mut keys = source
            .keys(id, (Bound::Unbounded, Bound::Unbounded), false)
            .await?;
        while let Some(key) = keys.try_next().await? {
            delta.insert(key).await?;
        }
        let mut state = self.owner.state.write().expect("state write lock");
        state.assert_writable(id)?;
        state.pending.insert(id, delta);
        Ok(())
    }

    pub fn finalized(&self) -> Option<TxnId> {
        self.owner.finalized()
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        self.finalized_key_stream_in((Bound::Unbounded, Bound::Unbounded), false)
            .await
    }

    pub async fn finalized_key_stream_in(
        &self,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> std::io::Result<b_tree::Keys<Value>> {
        let persistent = {
            let state = self.owner.state.read().expect("state read lock");
            state.persistent.clone()
        };

        key_stream_in(&persistent, bounds, reverse).await
    }

    /// Stream the keys visible to `txn_id` while retaining the range read permit.
    pub async fn keys(
        &self,
        txn_id: TxnId,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> Result<super::Keys, txn_lock::Error> {
        let permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await;
        let snapshot = self.owner.visible_snapshot(txn_id);
        let collator = KeyStreamCollator::new(reverse);

        let mut visible: BoxStream<'static, Result<Vec<Value>, std::io::Error>> =
            key_stream_in(&snapshot.persistent, bounds.clone(), reverse)
                .await
                .map_err(background_error)?
                .map_ok(|row| row.to_vec())
                .boxed();

        for delta in snapshot.deltas {
            let deletes = key_stream_in(&delta.deletes, bounds.clone(), reverse)
                .await
                .map_err(background_error)?
                .map_ok(|row| row.to_vec())
                .boxed();
            visible = try_diff(collator, visible, deletes).boxed();

            let inserts = key_stream_in(&delta.inserts, bounds.clone(), reverse)
                .await
                .map_err(background_error)?
                .map_ok(|row| row.to_vec())
                .boxed();
            visible = try_merge(collator, visible, inserts).boxed();
        }

        let stream = visible.map_err(TCError::from).boxed();
        Ok(super::Keys::new(stream, permit))
    }

    pub fn slice<R>(&self, range: R, reverse: bool) -> BTreeSlice<Txn>
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

    pub async fn load_literal_row(&self, row: Value) -> std::io::Result<()> {
        let persistent = {
            let state = self.owner.state.read().expect("state read lock");
            state.persistent.clone()
        };

        let row = persistent
            .schema()
            .normalize_row(row)
            .map_err(invalid_input_error)?;
        insert_key(&persistent, row).await
    }

    pub async fn insert_row(&self, txn: &Txn, key: Vec<Value>) -> Result<(), txn_lock::Error>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        // Reserve a write lock for this exact key range at txn_id.
        // This is the canonical ordering gate which enforces conflict semantics across txns.
        let _permit = self
            .owner
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending.insert(key).await.map_err(background_error)?;

        Ok(())
    }

    pub async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> Result<(), txn_lock::Error>
    where
        Txn: crate::StorageContext,
    {
        let txn_id = txn.id();
        // Deletions take the same key-scoped write reservation as inserts.
        let _permit = self
            .owner
            .semaphore
            .try_write(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending.delete(key).await.map_err(background_error)?;

        Ok(())
    }

    /// Commit the pending delta at `txn_id`.
    ///
    /// The caller must finish this transaction's operations and release its streams first.
    /// Returns `Outdated` at or before the finalized frontier.
    pub async fn commit(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        self.owner.commit(txn_id)
    }

    /// Roll back the pending delta at `txn_id`.
    ///
    /// The caller must finish this transaction's operations and release its streams first.
    /// Returns `Conflict` for a committed transaction or `Outdated` at the frontier.
    pub fn rollback(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        self.owner.rollback(txn_id)
    }

    /// Finalize all committed deltas up to `txn_id` into persistent state.
    ///
    /// Finalize is monotonic. A stale finalize is a no-op.
    ///
    /// The caller serializes lifecycle decisions and finishes operations through the
    /// cutoff first. Later transactions may retain read permits; merging uses native
    /// storage locks without acquiring a new semaphore write reservation.
    pub async fn finalize(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        self.owner.finalize(txn_id).await
    }

    /// Return row visibility at `txn_id` for `key`.
    ///
    /// If an earlier overlapping pending write exists, this call waits until that
    /// transaction resolves (commit/rollback/finalize) before reading.
    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> bool {
        // Canonical transactional read behavior: later reads wait behind earlier overlapping
        // pending writes until the earlier txn is finalized.
        let _permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await;

        let snapshot = self.owner.visible_snapshot(txn_id);
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
        // remain coherent through commit and finalize transitions.
        let _permit = self
            .owner
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
        let snapshot = self.owner.visible_snapshot(txn_id);
        let collator = KeyStreamCollator::new(reverse);

        let mut visible: BoxStream<'_, Result<Vec<Value>, std::io::Error>> =
            key_stream_in(&snapshot.persistent, bounds.clone(), reverse)
                .await
                .expect("stream persistent keys")
                .map_ok(|row| row.to_vec())
                .boxed();

        for delta in &snapshot.deltas {
            let deletes = key_stream_in(&delta.deletes, bounds.clone(), reverse)
                .await
                .expect("stream delete delta keys")
                .map_ok(|row| row.to_vec())
                .boxed();

            visible = try_diff(collator, visible, deletes).boxed();

            let inserts = key_stream_in(&delta.inserts, bounds.clone(), reverse)
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

    async fn is_row_visible(
        &self,
        snapshot: &VisibleSnapshot<Delta<Txn::File>>,
        key: &[Value],
    ) -> bool {
        let mut visible = contains_key(&snapshot.persistent, key)
            .await
            .expect("check persistent visibility");

        for delta in &snapshot.deltas {
            if contains_key(&delta.deletes, key)
                .await
                .expect("check delete delta visibility")
            {
                visible = false;
            }

            if contains_key(&delta.inserts, key)
                .await
                .expect("check insert delta visibility")
            {
                visible = true;
            }
        }

        visible
    }

    async fn pending_delta_for_txn(&self, txn: &Txn) -> Result<Delta<Txn::File>, txn_lock::Error> {
        let txn_id = txn.id();
        let key_schema = {
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

        let delta = Delta::create(key_schema, txn_dir)
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

    fn clone_bound(bound: Bound<&Value>) -> Bound<Value> {
        match bound {
            Bound::Included(value) => Bound::Included(value.clone()),
            Bound::Excluded(value) => Bound::Excluded(value.clone()),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
}

impl<Txn: crate::StorageContext> BTreeSlice<Txn> {
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

impl<Txn: crate::StorageContext> Transact for BTree<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        BTree::commit(self, txn_id).await.map_err(Into::into)
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { BTree::rollback(self, txn_id).map_err(Into::into) }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { BTree::finalize(self, txn_id).await.map_err(Into::into) }
    }
}
