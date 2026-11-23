//! Core transactional BTree behavior and range/snapshot query logic.
//!
//! `KeyOwner` owns visibility, reservations, and lifecycle decisions; this module
//! owns how a key delta is read and applied. A read sees canonical storage plus
//! committed versions through its transaction ID and its own pending delta.
//! Native trees hold payloads; the collection state lock only protects handles
//! and decisions. See TRANSACTIONAL_COLLECTION_CONTRACT.md for caller obligations.
use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use b_tree::{BTreeLock, Range, Schema};
use collate::Collate;
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::{TCError, TCResult};
use tc_ir::{Transact, TxnId};
use tc_value::{Value, ValueCollator, ValueType};

use crate::persistence::{CollectionOwner, KeyOwner, PersistentDelta, VisibleSnapshot};

const UNARY_KEY_ARITY: usize = 1;

fn invalid_input_error(message: impl fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        tc_error::bad_request!("{}", message),
    )
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct KeyStreamCollator {
    values: crate::persistence::ValueOrder,
    reverse: bool,
}

impl KeyStreamCollator {
    fn new(reverse: bool) -> Self {
        Self {
            values: crate::persistence::value_order(),
            reverse,
        }
    }
}

impl Collate for KeyStreamCollator {
    type Value = Vec<Value>;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> std::cmp::Ordering {
        let ordering = self.values.key_collator().cmp(left, right);
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
    ) -> TCResult<Self> {
        if key_arity == 0 {
            return Err(TCError::bad_request("BTree key arity must be >= 1"));
        }

        if key_types
            .as_ref()
            .is_some_and(|types| types.len() != key_arity)
        {
            return Err(TCError::bad_request(
                "BTree key type list must match key arity",
            ));
        }

        Ok(Self {
            storage,
            key_arity,
            key_types,
        })
    }

    pub(crate) fn from_key_types(key_types: Vec<ValueType>) -> TCResult<Self> {
        Self::new(StorageConfig::default(), key_types.len(), Some(key_types))
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
        Self {
            storage: StorageConfig::default(),
            key_arity: UNARY_KEY_ARITY,
            key_types: None,
        }
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

    fn validate_prefix(&self, prefix: &[Value]) -> Result<(), Self::Error> {
        if prefix.is_empty() || prefix.len() > self.key_arity {
            return Err(tc_error::bad_request!(
                "tc-collection BTree separator must contain 1..={} columns",
                self.key_arity
            ));
        }

        if let Some(key_types) = &self.key_types {
            for (i, (value, expected)) in prefix.iter().zip(key_types.iter()).enumerate() {
                let actual = value.class();
                if !expected.accepts(&actual) {
                    return Err(tc_error::bad_request!(
                        "tc-collection BTree key column {i} expected {:?} but got {:?}",
                        expected,
                        actual
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_key(&self, key: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        if key.len() != self.key_arity {
            return Err(tc_error::bad_request!(
                "tc-collection BTree keys must have arity {}",
                self.key_arity
            ));
        }

        self.validate_prefix(&key)?;

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

struct Delta<F: crate::CollectionFile> {
    // Inserts win over deletes within this transaction. Keeping a deletion marker
    // after reinsertion is intentional: applying deletes before inserts produces
    // the same membership as lookup and streaming overlays.
    inserts: BTreeLock<BTreeSchema, ValueCollator, F>,
    deletes: BTreeLock<BTreeSchema, ValueCollator, F>,
}

impl<F: crate::CollectionFile> Delta<F> {
    async fn lookup(&self, key: &[Value]) -> std::io::Result<Option<bool>> {
        if contains_key(&self.inserts, key).await? {
            return Ok(Some(true));
        }

        if contains_key(&self.deletes, key).await? {
            return Ok(Some(false));
        }

        Ok(None)
    }

    /// Start an ordinary delta with empty insert and delete storage.
    async fn create(schema: BTreeSchema, dir: DirLock<F>) -> std::io::Result<Arc<Self>> {
        let (inserts, deletes) = crate::persistence::create_delta_dirs(&dir).await?;
        Ok(Arc::new(Self {
            inserts: BTreeLock::create(schema.clone(), crate::persistence::value_order(), inserts)
                .await?,
            deletes: BTreeLock::create(schema, crate::persistence::value_order(), deletes).await?,
        }))
    }

    async fn insert(&self, key: Vec<Value>) -> std::io::Result<()> {
        insert_key(&self.inserts, key).await?;
        Ok(())
    }

    async fn delete(&self, key: Vec<Value>) -> std::io::Result<()> {
        // Remove a same-transaction insertion, then hide any older occurrence.
        delete_key(&self.inserts, &key).await?;
        insert_key(&self.deletes, key).await?;
        Ok(())
    }
}

impl<F: crate::CollectionFile> PersistentDelta for Delta<F> {
    type Native = BTreeLock<BTreeSchema, ValueCollator, F>;

    async fn apply_to(
        &self,
        persistent: &BTreeLock<BTreeSchema, ValueCollator, F>,
    ) -> std::io::Result<()> {
        let mut stream = key_stream_in(
            &self.deletes,
            (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
            false,
        )
        .await?;

        while let Some(key) = stream.try_next().await? {
            delete_key(persistent, &key).await?;
        }

        drop(stream);
        let mut stream = key_stream_in(
            &self.inserts,
            (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
            false,
        )
        .await?;

        while let Some(key) = stream.try_next().await? {
            insert_key(persistent, key.into_vec()).await?;
        }

        Ok(())
    }
}

pub struct BTree<Txn: crate::StorageContext> {
    owner: Arc<KeyOwner<Delta<Txn::File>>>,
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
        let state = self.owner.read_state();

        f.debug_struct("BTree")
            .field("committed_len", &state.committed.len())
            .field("pending_len", &state.pending.len())
            .field("finalized", &state.finalized)
            .finish()
    }
}

impl<Txn: crate::StorageContext> BTree<Txn> {
    /// Construct a transactional BTree with default unary-key schema.
    pub async fn new(persistent_dir: DirLock<Txn::File>) -> std::io::Result<Self> {
        Self::try_with_schema(persistent_dir, BTreeSchema::default()).await
    }

    /// Create in empty caller-delegated storage.
    pub async fn try_with_schema(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let persistent = BTreeLock::create(
            schema.clone(),
            crate::persistence::value_order(),
            persistent_dir,
        )
        .await?;

        Ok(Self::from_store(persistent))
    }

    pub async fn load(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let persistent = BTreeLock::load(
            schema.clone(),
            crate::persistence::value_order(),
            persistent_dir,
        )?;

        persistent.validate().await?;
        Ok(Self::from_store(persistent))
    }

    fn from_store(persistent: BTreeLock<BTreeSchema, ValueCollator, Txn::File>) -> Self {
        let ranges = *persistent.ordering().range_collator();
        Self {
            owner: Arc::new(CollectionOwner::new(persistent, ranges)),
            txn: PhantomData,
        }
    }

    pub(crate) fn validate_columns(&self, columns: &[super::BTreeColumnSchema]) -> TCResult<()> {
        let state = self.owner.read_state();
        let schema = state.persistent.schema();
        if columns.is_empty()
            || columns.len() != schema.key_arity
            || schema.key_types.as_ref().is_none_or(|types| {
                types
                    .iter()
                    .zip(columns)
                    .any(|(dtype, column)| dtype != &column.dtype)
            })
        {
            return Err(TCError::bad_request(
                "BTree view requires a matching typed schema",
            ));
        }

        Ok(())
    }

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

    pub fn finalized(&self) -> TCResult<Option<TxnId>> {
        self.owner.finalized()
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        self.finalized_key_stream_in(
            (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
            false,
        )
        .await
    }

    pub async fn finalized_key_stream_in(
        &self,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> std::io::Result<b_tree::Keys<Value>> {
        let persistent = {
            let state = self.owner.read_state();
            state.persistent.clone()
        };

        let stream = key_stream_in(&persistent, bounds, reverse).await?;
        // Release native stream ownership on error or EOF, before outer drop.
        Ok(crate::stream::GuardedStream::new(stream, ()).boxed())
    }

    /// Stream the keys visible to `txn_id` while retaining the range read permit.
    pub async fn keys(
        &self,
        txn_id: TxnId,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> TCResult<super::Keys> {
        let permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::All)
            .await?;

        let snapshot = self.owner.visible_snapshot(txn_id)?;
        let stream = Self::visible_keys(snapshot, bounds, reverse).await?;
        Ok(super::Keys::new(stream, permit))
    }

    async fn visible_keys(
        snapshot: VisibleSnapshot<Delta<Txn::File>>,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> TCResult<BoxStream<'static, TCResult<Vec<Value>>>> {
        let collator = KeyStreamCollator::new(reverse);

        let visible: BoxStream<'static, Result<Vec<Value>, std::io::Error>> =
            key_stream_in(&snapshot.persistent, bounds.clone(), reverse)
                .await?
                .map_ok(|row| row.into_vec())
                .boxed();

        let mut visible =
            crate::stream::Overlay::new(collator, visible, |current, incoming, present| {
                if !present {
                    *current = incoming;
                }
            });

        // Layers arrive oldest first, with this transaction's pending delta last.
        // Subtract each layer's deletes before merging its inserts. Rechecking
        // retirement after I/O avoids retaining a workspace already materialized
        // into canonical storage, including obsolete workspace-open errors.
        for (id, delta) in snapshot.deltas {
            if snapshot.visibility.retired(id) {
                continue;
            }

            let sources = async {
                let deletes = key_stream_in(&delta.deletes, bounds.clone(), reverse).await?;
                let inserts = key_stream_in(&delta.inserts, bounds.clone(), reverse).await?;
                Ok::<_, std::io::Error>((deletes, inserts))
            }
            .await;

            if !snapshot.visibility.retired(id) {
                let (deletes, inserts) = sources?;
                visible.difference(deletes.map_ok(|row| row.into_vec()).boxed());
                visible.merge(inserts.map_ok(|row| row.into_vec()).boxed());
            }
        }

        Ok(visible.into_stream().map_err(TCError::from).boxed())
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
            let state = self.owner.read_state();
            state.persistent.clone()
        };

        let row = persistent
            .schema()
            .normalize_row(row)
            .map_err(invalid_input_error)?;

        insert_key(&persistent, row).await?;
        Ok(())
    }

    pub async fn insert_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        let key = self
            .owner
            .read_state()
            .persistent
            .schema()
            .validate_key(key)?;

        // Reserve a write lock for this exact key range at txn_id.
        // This is the canonical ordering gate which enforces conflict semantics across txns.
        let _permit = self
            .owner
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending.insert(key).await?;

        Ok(())
    }

    pub async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        let key = self
            .owner
            .read_state()
            .persistent
            .schema()
            .validate_key(key)?;

        // Deletions take the same key-scoped write reservation as inserts.
        let _permit = self
            .owner
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending.delete(key).await?;

        Ok(())
    }

    /// Commit the pending delta at `txn_id`.
    ///
    /// The caller must finish this transaction's mutations first. Immutable result
    /// streams may retain their snapshots across visibility publication.
    /// Returns `Outdated` at or before the finalized frontier.
    pub async fn commit(&self, txn_id: TxnId) -> TCResult<()> {
        self.owner.commit(txn_id)
    }

    /// Roll back the pending delta at `txn_id`.
    ///
    /// The caller must finish this transaction's operations and release its streams first.
    /// Returns `Conflict` for a committed transaction or `Outdated` at the frontier.
    pub fn rollback(&self, txn_id: TxnId) -> TCResult<()> {
        self.owner.rollback(txn_id)
    }

    /// Finalize all committed deltas up to `txn_id` into persistent state.
    ///
    /// Finalize is monotonic. A stale finalize is a no-op.
    ///
    /// The caller serializes lifecycle decisions and finishes operations through the
    /// cutoff first. Later transactions may retain read permits; merging uses native
    /// storage locks without acquiring a new semaphore write reservation.
    pub async fn finalize(&self, txn_id: TxnId) -> TCResult<()> {
        self.owner.finalize(txn_id).await
    }

    /// Return row visibility at `txn_id` for `key`.
    ///
    /// If an earlier overlapping pending write exists, this call waits until that
    /// transaction resolves (commit/rollback/finalize) before reading.
    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        let _permit = self
            .owner
            .acquire_read_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await?;

        let mut read = self.owner.point_read(txn_id)?;
        if let Some(found) = read
            .resolve(|delta| async move { delta.lookup(key).await })
            .await?
        {
            return Ok(found);
        }

        Ok(contains_key(&read.canonical()?, key).await?)
    }

    pub async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.count_in(
            txn_id,
            (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
        )
        .await
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        Ok(!self
            .any_row_in(
                txn_id,
                (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                false,
            )
            .await?)
    }

    /// Iterate visible rows while retaining the stream's read reservation.
    pub async fn for_each_row_in_order<R, F>(
        &self,
        txn_id: TxnId,
        range: R,
        reverse: bool,
        mut on_key: F,
    ) -> TCResult<()>
    where
        R: RangeBounds<Value>,
        F: FnMut(Vec<Value>),
    {
        let bounds = (
            Self::clone_bound(range.start_bound()),
            Self::clone_bound(range.end_bound()),
        );

        let mut keys = self.keys(txn_id, bounds, reverse).await?;
        while let Some(key) = keys.try_next().await? {
            on_key(key);
        }

        Ok(())
    }

    pub async fn count_in<R>(&self, txn_id: TxnId, range: R) -> TCResult<u64>
    where
        R: RangeBounds<Value>,
    {
        let mut count = 0_u64;
        self.for_each_row_in_order(txn_id, range, false, |_| {
            count += 1;
        })
        .await?;

        Ok(count)
    }

    async fn any_row_in<R>(&self, txn_id: TxnId, range: R, reverse: bool) -> TCResult<bool>
    where
        R: RangeBounds<Value>,
    {
        let bounds = (
            Self::clone_bound(range.start_bound()),
            Self::clone_bound(range.end_bound()),
        );

        Ok(self
            .keys(txn_id, bounds, reverse)
            .await?
            .try_next()
            .await?
            .is_some())
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

    fn clone_bound(bound: Bound<&Value>) -> Bound<Value> {
        match bound {
            Bound::Included(value) => Bound::Included(value.clone()),
            Bound::Excluded(value) => Bound::Excluded(value.clone()),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
}

fn max_lower_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => Bound::Included(max_value(left, right)),
        (Bound::Included(left), Bound::Excluded(right)) => {
            if compare_values(&left, &right) == Ordering::Greater {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if compare_values(&left, &right) == Ordering::Less {
                Bound::Included(right)
            } else {
                Bound::Excluded(left)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => Bound::Excluded(max_value(left, right)),
    }
}

fn min_upper_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => Bound::Included(min_value(left, right)),
        (Bound::Included(left), Bound::Excluded(right)) => {
            if compare_values(&left, &right) == Ordering::Less {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if compare_values(&left, &right) != Ordering::Greater {
                Bound::Excluded(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => Bound::Excluded(min_value(left, right)),
    }
}

fn compare_values(left: &Value, right: &Value) -> Ordering {
    ValueCollator::default().cmp(left, right)
}

fn max_value(left: Value, right: Value) -> Value {
    if compare_values(&left, &right) == Ordering::Less {
        right
    } else {
        left
    }
}

fn min_value(left: Value, right: Value) -> Value {
    if compare_values(&left, &right) == Ordering::Greater {
        right
    } else {
        left
    }
}

impl<Txn: crate::StorageContext> BTreeSlice<Txn> {
    pub(crate) fn is_reversed(&self) -> bool {
        self.reverse
    }

    pub fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        Self {
            btree: self.btree.clone(),
            lower: max_lower_bound(self.lower.clone(), bounds.0),
            upper: min_upper_bound(self.upper.clone(), bounds.1),
            reverse,
        }
    }

    pub async fn keys(&self, txn_id: TxnId) -> TCResult<super::Keys> {
        self.btree
            .keys(
                txn_id,
                (self.lower.clone(), self.upper.clone()),
                self.reverse,
            )
            .await
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        self.btree
            .finalized_key_stream_in((self.lower.clone(), self.upper.clone()), self.reverse)
            .await
    }

    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        if !self.btree.contains_row(txn_id, key).await? {
            return Ok(false);
        }

        let value = &key[0];
        let after_lower = match &self.lower {
            Bound::Included(lower) => compare_values(value, lower) != Ordering::Less,
            Bound::Excluded(lower) => compare_values(value, lower) == Ordering::Greater,
            Bound::Unbounded => true,
        };

        let before_upper = match &self.upper {
            Bound::Included(upper) => compare_values(value, upper) != Ordering::Greater,
            Bound::Excluded(upper) => compare_values(value, upper) == Ordering::Less,
            Bound::Unbounded => true,
        };

        Ok(after_lower && before_upper)
    }

    pub async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.btree
            .count_in(txn_id, (self.lower.clone(), self.upper.clone()))
            .await
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        Ok(!self
            .btree
            .any_row_in(
                txn_id,
                (self.lower.clone(), self.upper.clone()),
                self.reverse,
            )
            .await?)
    }

    pub async fn for_each_row_in_order<F>(&self, txn_id: TxnId, on_row: F) -> TCResult<()>
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
            .await
    }
}

impl<Txn: crate::StorageContext> Transact for BTree<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        BTree::commit(self, txn_id).await
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { BTree::rollback(self, txn_id) }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { BTree::finalize(self, txn_id).await }
    }
}

impl<Txn: crate::StorageContext> crate::Restore for BTree<Txn> {
    type Txn = Txn;

    /// Restore through a staged key delta; see [`crate::Restore`].
    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let id = txn.id();
        self.owner.read_state().assert_writable(id)?;
        if Arc::ptr_eq(&self.owner, &source.owner) {
            return Ok(());
        }

        let schema = self.owner.read_state().persistent.schema().clone();
        let source_schema = source.owner.read_state().persistent.schema().clone();
        if schema.key_arity != source_schema.key_arity
            || schema.key_types != source_schema.key_types
        {
            return Err(TCError::bad_request("BTree restoration schema mismatch"));
        }

        let _permit = self
            .owner
            .try_acquire_write_permit(id, txn_lock::set::Range::All)?;

        let base = self.owner.committed_snapshot(id)?;

        // Replacement is relative to committed state, not the old pending delta:
        // successful publication below replaces that delta in its entirety.

        let source_permit = source
            .owner
            .try_acquire_read_permit(id, txn_lock::set::Range::All)?;

        let source_snapshot = source.owner.visible_snapshot(id)?;
        let stream =
            Self::visible_keys(source_snapshot, (Bound::Unbounded, Bound::Unbounded), false)
                .await?;

        let mut keys = super::Keys::new(stream, source_permit);
        let workspace = txn.subcontext_unique().context().await?;
        let delta = Delta::create(schema, workspace).await?;
        let mut baseline =
            Self::visible_keys(base, (Bound::Unbounded, Bound::Unbounded), false).await?;

        while let Some(key) = baseline.try_next().await? {
            insert_key(&delta.deletes, key).await?;
        }

        drop(baseline);

        while let Some(key) = keys.try_next().await? {
            delta.insert(key).await?;
        }

        drop(keys);
        // Staging has completed without changing pending visibility. Recheck the
        // decision after I/O before publishing; the caller owns workspace cleanup.
        let mut state = self.owner.write_state();
        state.assert_writable(id)?;
        state.pending.insert(id, delta);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/btree/failure.rs"]
mod failure_tests;
