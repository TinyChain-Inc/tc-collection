//! Core transactional BTreeBase behavior and range/snapshot query logic.
//!
//! `Versions` owns visibility and lifecycle ordering; this module
//! owns how a key delta is read and applied. A read sees canonical storage plus
//! committed versions through its transaction ID and its own pending delta.
//! Native trees hold payloads; the version-state lock only protects versions
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
use tc_value::{Value, ValueCollator};

use super::BTree;
use super::schema::BTreeSchema;
use crate::persistence::Versions;
use crate::persistence::storage::{Allocation, Store};
use crate::stream::ReadSnapshot;

fn invalid_input_error(message: impl fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        tc_error::bad_request!("{}", message),
    )
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct KeyStreamCollator {
    values: b_tree::RangeCollator<ValueCollator>,
    reverse: bool,
}

impl KeyStreamCollator {
    fn new(reverse: bool) -> Self {
        Self {
            values: b_tree::RangeCollator::new(ValueCollator::default()),
            reverse,
        }
    }
}

impl Collate for KeyStreamCollator {
    type Value = Vec<Value>;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> std::cmp::Ordering {
        let ordering = self.values.cmp_keys(left, right);
        if self.reverse {
            ordering.reverse()
        } else {
            ordering
        }
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
    storage: Arc<Allocation<F>>,
    // Inserts win over deletes within this transaction. Keeping a deletion marker
    // after reinsertion is intentional: applying deletes before inserts produces
    // the same membership as lookup and streaming overlays.
    inserts: BTreeLock<BTreeSchema, ValueCollator, F>,
    deletes: BTreeLock<BTreeSchema, ValueCollator, F>,
}

impl<F: crate::CollectionFile> Delta<F> {
    async fn load(schema: BTreeSchema, storage: Arc<Allocation<F>>) -> TCResult<Arc<Self>> {
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

        let inserts = BTreeLock::load(schema.clone(), ValueCollator::default(), inserts)?;
        let deletes = BTreeLock::load(schema, ValueCollator::default(), deletes)?;
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
        for tree in [&self.inserts, &self.deletes] {
            let mut keys = key_stream_in(tree, (Bound::Unbounded, Bound::Unbounded), false).await?;
            while keys.try_next().await?.is_some() {
                units = units
                    .checked_add(1)
                    .ok_or_else(|| std::io::Error::other("native version size overflow"))?;
            }
        }

        Ok(units)
    }

    async fn merge_into(
        &self,
        output: &Self,
        base: Option<&BTreeLock<BTreeSchema, ValueCollator, F>>,
    ) -> std::io::Result<()> {
        let mut deletes =
            key_stream_in(&self.deletes, (Bound::Unbounded, Bound::Unbounded), false).await?;

        while let Some(key) = deletes.try_next().await? {
            // A complete prefix needs markers only for keys in the initial base.
            // Later layers still need markers to suppress earlier unmerged runs.
            let needs_tombstone = match base {
                Some(base) => contains_key(base, &key).await?,
                None => true,
            };

            if needs_tombstone {
                output.delete(key.into_vec()).await?;
            } else {
                delete_key(&output.inserts, &key).await?;
            }
        }

        drop(deletes);
        let mut inserts =
            key_stream_in(&self.inserts, (Bound::Unbounded, Bound::Unbounded), false).await?;

        while let Some(key) = inserts.try_next().await? {
            output.insert(key.into_vec()).await?;
        }

        Ok(())
    }

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
    async fn create(
        schema: BTreeSchema,
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
            inserts: BTreeLock::create(schema.clone(), tc_value::ValueCollator::default(), inserts)
                .await?,
            deletes: BTreeLock::create(schema, tc_value::ValueCollator::default(), deletes).await?,
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

pub struct BTreeBase<Txn: crate::StorageContext> {
    canonical: Arc<BTreeLock<BTreeSchema, ValueCollator, Txn::File>>,
    store: Arc<Store<Txn::File>>,
    versions: Versions<Delta<Txn::File>>,
    semaphore: txn_lock::semaphore::Semaphore<
        TxnId,
        b_tree::RangeCollator<ValueCollator>,
        txn_lock::set::Range<Vec<Value>>,
    >,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: crate::StorageContext> Clone for BTreeBase<Txn> {
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

#[derive(Debug, Clone)]
pub struct BTreeSlice<Txn: crate::StorageContext> {
    btree: BTreeBase<Txn>,
    lower: Bound<Value>,
    upper: Bound<Value>,
    reverse: bool,
}

impl<Txn: crate::StorageContext> fmt::Debug for BTreeBase<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BTreeBase").finish_non_exhaustive()
    }
}

impl<Txn: crate::StorageContext> BTreeBase<Txn> {
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
    pub(super) async fn try_with_schema(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let owner = Self::unpublished(persistent_dir, schema).await?;
        owner.finish_creation().await?;
        Ok(owner)
    }

    pub(super) async fn unpublished(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let store = Store::create(persistent_dir)
            .await
            .map_err(std::io::Error::other)?;

        let persistent = BTreeLock::create(
            schema,
            tc_value::ValueCollator::default(),
            store.base().clone(),
        )
        .await?;

        Ok(Self::from_store(persistent, store, Versions::new()))
    }

    pub(super) async fn load(
        persistent_dir: DirLock<Txn::File>,
        schema: BTreeSchema,
    ) -> std::io::Result<Self> {
        let store = Store::load(persistent_dir)
            .await
            .map_err(std::io::Error::other)?;

        let persistent = BTreeLock::load(
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
        persistent: BTreeLock<BTreeSchema, ValueCollator, Txn::File>,
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

    pub(super) fn schema(&self) -> &BTreeSchema {
        self.canonical.schema()
    }

    /// Stream the keys visible to `txn_id` while retaining the range read permit.
    pub(super) async fn keys(
        &self,
        txn_id: TxnId,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> TCResult<super::Keys> {
        let (permit, snapshot) = self
            .read_snapshot(txn_id, txn_lock::set::Range::All)
            .await?;

        let stream = Self::visible_keys(self.canonical.clone(), snapshot, bounds, reverse).await?;
        Ok(super::Keys::new(stream, permit))
    }

    async fn visible_keys(
        canonical: Arc<BTreeLock<BTreeSchema, ValueCollator, Txn::File>>,
        snapshot: Vec<Arc<Delta<Txn::File>>>,
        bounds: (Bound<Value>, Bound<Value>),
        reverse: bool,
    ) -> TCResult<BoxStream<'static, TCResult<Vec<Value>>>> {
        let collator = KeyStreamCollator::new(reverse);

        let visible: BoxStream<'static, Result<Vec<Value>, std::io::Error>> =
            key_stream_in(&canonical, bounds.clone(), reverse)
                .await?
                .map_ok(|row| row.into_vec())
                .boxed();

        let mut visible =
            crate::stream::Overlay::new(collator, visible, |current, incoming, present| {
                if !present {
                    *current = incoming;
                }
            });

        // Immutable layers are pinned until native streams have been dropped.
        for delta in &snapshot {
            let deletes = key_stream_in(&delta.deletes, bounds.clone(), reverse).await?;
            let inserts = key_stream_in(&delta.inserts, bounds.clone(), reverse).await?;
            visible.difference(deletes.map_ok(|row| row.into_vec()).boxed());
            visible.merge(inserts.map_ok(|row| row.into_vec()).boxed());
        }

        Ok(visible
            .into_stream()
            .map(move |item| {
                let _ = &snapshot;
                item.map_err(TCError::from)
            })
            .boxed())
    }

    pub(super) fn slice<R>(&self, range: R, reverse: bool) -> BTreeSlice<Txn>
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

    pub(super) async fn load_literal_row(&self, row: Value) -> std::io::Result<()> {
        let arity = self.schema().columns().len();
        let key = if arity == 1 {
            vec![row]
        } else {
            match row {
                Value::Tuple(values) if values.len() == arity => values,
                value => {
                    return Err(invalid_input_error(format!(
                        "BTree key must be a tuple of length {arity} but got {value:?}"
                    )));
                }
            }
        };

        self.schema()
            .validate_key(&key)
            .map_err(invalid_input_error)?;

        insert_key(&self.canonical, key).await
    }

    pub(super) async fn insert_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        self.canonical.schema().validate_key(&key)?;

        // Reserve a write lock for this exact key range at txn_id.
        // This is the canonical ordering gate which enforces conflict semantics across txns.
        let _permit = self
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending.insert(key).await?;

        Ok(())
    }

    pub(super) async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        let txn_id = txn.id();
        self.canonical.schema().validate_key(&key)?;

        // Deletions take the same key-scoped write reservation as inserts.
        let _permit = self
            .try_acquire_write_permit(txn_id, txn_lock::set::Range::One(Arc::new(key.clone())))?;

        let pending = self.pending_delta_for_txn(txn).await?;
        pending.delete(key).await?;

        Ok(())
    }

    /// Return row visibility at `txn_id` for `key`.
    ///
    /// If an earlier overlapping pending write exists, this call waits until that
    /// transaction resolves (commit/rollback/finalize) before reading.
    pub(super) async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        self.versions.check_readable(txn_id)?;
        let _permit = self
            .semaphore
            .read(txn_id, txn_lock::set::Range::One(Arc::new(key.to_vec())))
            .await?;

        if let Some(found) = self
            .versions
            .lookup(txn_id, true, |delta| async move { delta.lookup(key).await })
            .await?
        {
            return Ok(found);
        }

        Ok(contains_key(&self.canonical, key).await?)
    }

    pub(super) async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.count_in(
            txn_id,
            (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
        )
        .await
    }

    pub(super) async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        Ok(!self
            .any_row_in(
                txn_id,
                (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                false,
            )
            .await?)
    }

    async fn count_in(&self, txn_id: TxnId, bounds: (Bound<Value>, Bound<Value>)) -> TCResult<u64> {
        let mut keys = self.keys(txn_id, bounds, false).await?;
        let mut count = 0;
        while keys.try_next().await?.is_some() {
            count += 1;
        }

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

    fn clone_bound(bound: Bound<&Value>) -> Bound<Value> {
        match bound {
            Bound::Included(value) => Bound::Included(value.clone()),
            Bound::Excluded(value) => Bound::Excluded(value.clone()),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
}

fn max_lower_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    // max_by keeps its second argument on ties; preserve the existing left value.
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => {
            Bound::Included(std::cmp::max_by(right, left, |a, b| {
                ValueCollator::default().cmp(a, b)
            }))
        }
        (Bound::Included(left), Bound::Excluded(right)) => {
            if ValueCollator::default().cmp(&left, &right) == Ordering::Greater {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if ValueCollator::default().cmp(&left, &right) == Ordering::Less {
                Bound::Included(right)
            } else {
                Bound::Excluded(left)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => {
            Bound::Excluded(std::cmp::max_by(right, left, |a, b| {
                ValueCollator::default().cmp(a, b)
            }))
        }
    }
}

fn min_upper_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => {
            Bound::Included(std::cmp::min_by(left, right, |a, b| {
                ValueCollator::default().cmp(a, b)
            }))
        }
        (Bound::Included(left), Bound::Excluded(right)) => {
            if ValueCollator::default().cmp(&left, &right) == Ordering::Less {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if ValueCollator::default().cmp(&left, &right) != Ordering::Greater {
                Bound::Excluded(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => {
            Bound::Excluded(std::cmp::min_by(left, right, |a, b| {
                ValueCollator::default().cmp(a, b)
            }))
        }
    }
}

impl<Txn: crate::StorageContext> BTreeSlice<Txn> {
    pub(super) fn schema(&self) -> &BTreeSchema {
        self.btree.schema()
    }

    pub(super) fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        Self {
            btree: self.btree.clone(),
            lower: max_lower_bound(self.lower.clone(), bounds.0),
            upper: min_upper_bound(self.upper.clone(), bounds.1),
            reverse,
        }
    }

    pub(super) async fn keys(&self, txn_id: TxnId) -> TCResult<super::Keys> {
        self.btree
            .keys(
                txn_id,
                (self.lower.clone(), self.upper.clone()),
                self.reverse,
            )
            .await
    }

    pub(super) async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        if !self.btree.contains_row(txn_id, key).await? {
            return Ok(false);
        }

        let value = &key[0];
        let after_lower = match &self.lower {
            Bound::Included(lower) => ValueCollator::default().cmp(value, lower) != Ordering::Less,
            Bound::Excluded(lower) => {
                ValueCollator::default().cmp(value, lower) == Ordering::Greater
            }
            Bound::Unbounded => true,
        };

        let before_upper = match &self.upper {
            Bound::Included(upper) => {
                ValueCollator::default().cmp(value, upper) != Ordering::Greater
            }
            Bound::Excluded(upper) => ValueCollator::default().cmp(value, upper) == Ordering::Less,
            Bound::Unbounded => true,
        };

        Ok(after_lower && before_upper)
    }

    pub(super) async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.btree
            .count_in(txn_id, (self.lower.clone(), self.upper.clone()))
            .await
    }

    pub(super) async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        Ok(!self
            .btree
            .any_row_in(
                txn_id,
                (self.lower.clone(), self.upper.clone()),
                self.reverse,
            )
            .await?)
    }
}

impl<Txn: crate::StorageContext> Transact for BTreeBase<Txn> {
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
                |older, newer, complete_prefix| async move {
                    let storage = self.store.allocate().await?;
                    let output = Delta::create(self.schema().clone(), storage.clone()).await?;
                    let base = complete_prefix.then_some(self.canonical.as_ref());
                    older.merge_into(&output, base).await?;
                    newer.merge_into(&output, base).await?;
                    let units = output.units().await?;
                    Ok((output, Some((storage, units))))
                },
                || self.semaphore.finalize(txn_id, true).map_err(Into::into),
            )
            .await
    }
}

impl<Txn: crate::StorageContext> crate::CopyFrom<&BTree<Txn>> for BTree<Txn> {
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
        source: &BTree<Txn>,
    ) -> tc_error::TCResult<Self> {
        if matches!(source, Self::Slice(slice) if slice.reverse) {
            return Err(tc_error::TCError::bad_request(
                "cannot materialize a reversed BTree view in canonical key order",
            ));
        }

        let schema = source.schema().clone();

        let target = BTreeBase::unpublished(dir, schema).await?;
        let mut keys = source.keys(txn.id()).await?;
        while let Some(key) = keys.try_next().await? {
            insert_key(&target.canonical, key).await?;
        }

        target.finish_creation().await?;
        Ok(Self::File(target))
    }
}

impl<Txn: crate::StorageContext> crate::Restore for BTreeBase<Txn> {
    type Txn = Txn;

    /// Restore through a staged key delta; see [`crate::Restore`].
    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let id = txn.id();
        self.versions.check_writable(id)?;
        if Arc::ptr_eq(&self.canonical, &source.canonical) {
            return Ok(());
        }

        let schema = self.canonical.schema().clone();
        let source_schema = source.canonical.schema();
        if schema.columns() != source_schema.columns() {
            return Err(TCError::bad_request(
                "BTreeBase restoration schema mismatch",
            ));
        }

        let _permit = self.try_acquire_write_permit(id, txn_lock::set::Range::All)?;

        let base = self.versions.committed_snapshot(id)?;

        // Replacement is relative to committed state, not the old pending delta:
        // successful publication below replaces that delta in its entirety.

        let (source_permit, source_snapshot) =
            source.try_read_snapshot(id, txn_lock::set::Range::All)?;

        let stream = Self::visible_keys(
            source.canonical.clone(),
            source_snapshot,
            (Bound::Unbounded, Bound::Unbounded),
            false,
        )
        .await?;

        let mut keys = super::Keys::new(stream, source_permit);
        let storage = self.store.allocate().await?;
        let delta = Delta::create(schema, storage).await?;
        let mut baseline = Self::visible_keys(
            self.canonical.clone(),
            base,
            (Bound::Unbounded, Bound::Unbounded),
            false,
        )
        .await?;

        while let Some(key) = baseline.try_next().await? {
            insert_key(&delta.deletes, key).await?;
        }

        drop(baseline);

        while let Some(key) = keys.try_next().await? {
            delta.insert(key).await?;
        }

        drop(keys);
        // Staging has completed without changing pending visibility. Recheck the
        // decision after I/O before publishing; the owner reclaims unreachable staging.
        self.versions.replace_pending(id, delta, |_| Ok(()))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/btree/failure.rs"]
mod failure_tests;
