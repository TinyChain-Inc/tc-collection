//! Transactional logical-block ownership over native fensor storage.
//!
//! Lock order: reserve the transaction range, briefly inspect collection state,
//! release that guard, then enter one native index or payload domain at a time.
//! Lifecycle callers quiesce covered work; lifecycle methods never reserve ranges.
//! `CollectionOwner` owns transaction decisions; this module translates Tensor
//! mutations into logical-block deltas. Computation and physical payload/index
//! maintenance remain delegated to fensor. Caller obligations are defined in
//! TRANSACTIONAL_COLLECTION_CONTRACT.md.

use std::io;
use std::marker::PhantomData;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use b_tree::collate::Collator;
use b_tree::{BTreeLock, Range, Schema};
use fensor::{
    BoxFuture, Layout, StorageGeometry, TensorArray, TensorElement, TensorFileEntry,
    TensorGeometry, TensorRead, TensorSource,
};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::{TCError, TCResult};
use tc_ir::TxnId;

use crate::persistence::{CollectionOwner, PersistentDelta, Visibility};
use crate::{CollectionFile, StorageContext};

pub(super) type BlockRange = std::ops::Range<u64>;
type BlockOwner<F, T> = CollectionOwner<Delta<F, T>, Collator<u64>, BlockRange>;
type Layer<F, T> = (TxnId, Arc<Delta<F, T>>);
type ReadPermit = txn_lock::semaphore::PermitRead<BlockRange>;
type Touched<F> = BTreeLock<TouchedSchema, Collator<u64>, F>;

const TOUCHED_PAGE_ENTRIES: usize = 4096;

#[derive(Debug, Eq, PartialEq)]
struct TouchedSchema;

impl Schema for TouchedSchema {
    type Error = io::Error;
    type Value = u64;

    fn block_size(&self) -> usize {
        4096
    }

    fn len(&self) -> usize {
        1
    }

    fn order(&self) -> usize {
        // Even order preserves native split/merge occupancy; this single-column
        // node stays below 16 KiB including row containers and index children.
        256
    }

    fn validate_prefix(&self, prefix: &[u64]) -> io::Result<()> {
        touched_id(prefix).map(|_| ())
    }

    fn validate_key(&self, key: Vec<u64>) -> io::Result<Vec<u64>> {
        touched_id(&key)?;
        Ok(key)
    }
}

fn touched_id(key: &[u64]) -> io::Result<u64> {
    match key {
        [id] => Ok(*id),
        _ => Err(freqfs::IntegrityError::invalid_data(
            "invalid Tensor touched-block key",
        )),
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn native_error(error: fensor::Error) -> io::Error {
    match error {
        fensor::Error::Io(error) => error,
        error => io::Error::new(io::ErrorKind::InvalidData, error),
    }
}

/// All payloads are owned by this workspace; no delta points into another version.
pub(super) struct Delta<F: CollectionFile, T: TensorElement> {
    _workspace: DirLock<F>,
    payload: fensor::SparseBlocks<F, T>,
    // Membership distinguishes an explicit zero replacement from an untouched block.
    touched: Touched<F>,
    geometry: StorageGeometry,
    // Assignment stamps retain this identity without retaining the workspace.
    generation: Arc<AtomicU64>,
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> Delta<F, T> {
    pub async fn create(workspace: DirLock<F>, geometry: StorageGeometry) -> io::Result<Arc<Self>> {
        let (payload, touched) = {
            let mut dir = workspace.write().await;
            if !dir.is_empty() {
                return Err(invalid("Tensor delta requires empty delegated storage"));
            }

            (
                dir.create_dir("payload".into())?,
                dir.create_dir("touched".into())?,
            )
        };

        Ok(Arc::new(Self {
            _workspace: workspace,
            payload: fensor::SparseBlocks::create(payload, geometry.clone())
                .await
                .map_err(native_error)?,
            touched: BTreeLock::create(
                TouchedSchema,
                b_tree::IndexOrder::new(Collator::default()),
                touched,
            )
            .await?,
            geometry,

            generation: Arc::new(AtomicU64::new(0)),
        }))
    }

    pub async fn read(&self, id: u64) -> io::Result<Option<Vec<T>>> {
        self.geometry.validate_id(id).map_err(native_error)?;
        if self.touches(id).await? {
            self.read_payload(id).await.map(Some)
        } else {
            Ok(None)
        }
    }

    async fn read_payload(&self, id: u64) -> io::Result<Vec<T>> {
        #[cfg(test)]
        observation::record(&self.generation, |counts| counts.payload_reads += 1);

        let values = self.payload.read(id).await.map_err(native_error)?;

        Ok(values)
    }

    async fn touches(&self, id: u64) -> io::Result<bool> {
        #[cfg(test)]
        observation::record(&self.generation, |counts| counts.touched_lookups += 1);

        let view = self.touched.read().await;
        match view.first(Range::from_prefix(vec![id])).await? {
            Some(key) if touched_id(&key)? == id => Ok(true),
            Some(_) => Err(freqfs::IntegrityError::invalid_data(
                "invalid Tensor touched-block key",
            )),
            None => Ok(false),
        }
    }

    // Return only a bounded page, releasing the native stream and its guards
    // before replacement traversal enters payload I/O.
    async fn touched_page(&self, after: Option<u64>) -> io::Result<Vec<u64>> {
        let range = Range::with_bounds(
            Vec::<u64>::new(),
            (after.map_or(Unbounded, Excluded), Unbounded),
        );

        let mut keys = self.touched.read().await.keys(range).await?;
        let mut page = Vec::new();
        while let Some(key) = keys.try_next().await? {
            let id = touched_id(&key)?;
            self.geometry
                .validate_id(id)
                .map_err(freqfs::IntegrityError::invalid_data)?;

            page.push(id);
            if page.len() == TOUCHED_PAGE_ENTRIES {
                break;
            }
        }

        Ok(page)
    }

    /// Apply one bounded batch to the unpublished patch, preserving earlier batches.
    pub async fn apply_updates(
        &self,
        base: &Snapshot<F, T>,
        updates: fensor::BlockUpdates<T>,
    ) -> TCResult<()> {
        for (block, updates) in updates {
            let mut values = match self.read(block).await? {
                Some(values) => values,
                None => base.read_logical_block(block).await?,
            };

            for (offset, value) in updates {
                values[offset] = value;
            }

            self.write(block, values).await?;
        }

        Ok(())
    }

    pub async fn write(&self, id: u64, values: Vec<T>) -> io::Result<()> {
        self.geometry
            .validate_block(id, &values)
            .map_err(native_error)?;

        // Store the replacement before declaring the block touched. Sparse zero
        // payloads can be absent, so touched membership is the authoritative
        // override marker. These are fallible in-place writes: errors require the
        // caller's rollback/recovery policy, not an assumption of atomicity here.
        self.payload
            .replace(id, values)
            .await
            .map_err(native_error)?;

        {
            let mut index = self.touched.write().await;
            index.insert(vec![id]).await?;
        }

        self.generation
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| invalid("Tensor mutation generation exhausted"))?;

        Ok(())
    }

    fn replacements(&self) -> impl futures::Stream<Item = io::Result<(u64, Vec<T>)>> + '_ {
        futures::stream::try_unfold(
            (None, Vec::new().into_iter()),
            move |(mut after, mut rows)| async move {
                if rows.len() == 0 {
                    rows = self.touched_page(after).await?.into_iter();
                }

                let Some(id) = rows.next() else {
                    return Ok(None);
                };

                let values = self.read_payload(id).await?;
                after = Some(id);
                Ok(Some(((id, values), (after, rows))))
            },
        )
    }

    pub async fn apply_to_delta(&self, output: &Self) -> io::Result<()> {
        let blocks = self.replacements();
        futures::pin_mut!(blocks);
        while let Some((id, values)) = blocks.try_next().await? {
            output.write(id, values).await?;
        }

        Ok(())
    }
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> PersistentDelta for Delta<F, T> {
    type Native = fensor::Tensor<F, T>;

    async fn apply_to(&self, native: &Self::Native) -> io::Result<()> {
        let blocks = self.replacements();
        futures::pin_mut!(blocks);
        while let Some((id, values)) = blocks.try_next().await? {
            native
                .replace_logical_block(id, values)
                .await
                .map_err(native_error)?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct Snapshot<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> {
    canonical: fensor::Tensor<F, T>,

    deltas: Arc<[Layer<F, T>]>,
    visibility: Visibility<Delta<F, T>>,
    _permit: Option<Arc<ReadPermit>>,
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> Snapshot<F, T> {
    // Resolve the bounded request before payload I/O. A touched zero block still
    // selects its delta, whose absent sparse payload leaves the output zero.
    async fn resolve_sources(
        &self,
        ids: &[u64],
        sources: &mut [Option<usize>],
    ) -> fensor::Result<()> {
        sources.fill(None);
        for (index, (id, delta)) in self.deltas.iter().enumerate().rev() {
            if sources.iter().all(Option::is_some) {
                break;
            }

            if self.visibility.retired(*id) {
                continue;
            }

            // Assignments remain tentative until this delta's streams have been
            // released and its materialization handoff has been checked.
            let result: fensor::Result<()> = async {
                let mut start = 0;
                while start < ids.len() {
                    tokio::task::coop::consume_budget().await;
                    if self.visibility.retired(*id) {
                        return Ok(());
                    }

                    if sources[start].is_some() {
                        start += 1;
                        continue;
                    }

                    let mut end = start + 1;
                    while end < ids.len()
                        && sources[end].is_none()
                        && ids[end - 1].checked_add(1) == Some(ids[end])
                    {
                        end += 1;
                    }

                    let first = ids[start];
                    let last = ids[end - 1];
                    let range =
                        Range::with_bounds(Vec::<u64>::new(), (Included(first), Included(last)));

                    #[cfg(test)]
                    observation::record(&delta.generation, |counts| counts.membership_scans += 1);

                    let view = delta.touched.read().await;

                    let mut previous = None;
                    let result: fensor::Result<()> = view
                        .visit(range, false, |key| {
                            let id = touched_id(key)?;
                            if id < first
                                || id > last
                                || previous.is_some_and(|previous| id <= previous)
                            {
                                return Err(freqfs::IntegrityError::invalid_data(
                                    "invalid Tensor touched-block order",
                                )
                                .into());
                            }

                            sources[start + (id - first) as usize] = Some(index);
                            previous = Some(id);
                            Ok(ControlFlow::Continue(()))
                        })
                        .await;

                    drop(view);
                    if self.visibility.retired(*id) {
                        return Ok(());
                    }

                    result?;
                    #[cfg(test)]
                    observation::after_membership(&delta.generation)?;

                    start = end;
                }

                Ok(())
            }
            .await;

            if self.visibility.retired(*id) {
                // Earlier runs from this delta are obsolete too. Newer selections
                // remain intact while unresolved IDs fall through to older storage.
                for source in sources.iter_mut().filter(|source| **source == Some(index)) {
                    *source = None;
                }
            } else {
                result?;
            }
        }

        Ok(())
    }
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> TensorGeometry for Snapshot<F, T> {
    type DType = T;

    fn dtype(&self) -> fensor::NumberType {
        self.canonical.dtype()
    }

    fn layout(&self) -> Layout {
        self.canonical.layout()
    }

    fn shape(&self) -> &[u64] {
        self.canonical.shape()
    }
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> TensorArray for Snapshot<F, T> {
    fn schema(&self) -> &fensor::TensorSchema {
        self.canonical.schema()
    }

    fn strides(&self) -> &[u64] {
        self.canonical.strides()
    }
}

// Each frame owns one native stream and bounded lookahead. The driver visits
// lower frames by index, so polling and destruction do not nest with history.
// Recursive stream composition would make stack use grow with visible versions.
enum Overlay<F: CollectionFile, T: TensorElement> {
    Native(futures::stream::Fuse<BoxStream<'static, fensor::Result<u64>>>),
    Delta {
        id: TxnId,
        delta: Arc<Delta<F, T>>,
        upper: futures::stream::Fuse<BoxStream<'static, fensor::Result<u64>>>,
        left: Option<u64>,
        right: Option<u64>,
        after: u64,
        lower_done: bool,
    },
}

// Bound ready traversal across layers and output keys, independently of native I/O.

enum OverlayStep {
    Lower,
    Retry,
    Ready(Option<u64>),
}

fn overlay_keys<F: CollectionFile + TensorFileEntry<T>, T: TensorElement>(
    lower: BoxStream<'static, fensor::Result<u64>>,
    deltas: impl IntoIterator<Item = (TxnId, Arc<Delta<F, T>>)>,
    visibility: Visibility<Delta<F, T>>,
    canonical: fensor::Tensor<F, T>,
    range: BlockRange,
) -> BoxStream<'static, fensor::Result<u64>> {
    let mut frames = vec![Some(Overlay::Native(lower.fuse()))];
    for (id, delta) in deltas {
        #[cfg(test)]
        observation::record(&delta.generation, |counts| counts.overlay_starts += 1);

        let upper = delta.payload.occupied_blocks(range.clone()).fuse();
        frames.push(Some(Overlay::Delta {
            id,
            delta,
            upper,
            left: None,
            right: None,
            after: range.start,
            lower_done: false,
        }));
    }

    futures::stream::try_unfold(
        (frames, canonical, visibility),
        move |(mut frames, canonical, visibility)| async move {
            let mut index = frames.len() - 1;
            let mut lower_result: Option<fensor::Result<Option<u64>>> = None;
            loop {
                tokio::task::coop::consume_budget().await;
                match &frames[index] {
                    Some(Overlay::Delta { id, after, .. }) if visibility.retired(*id) => {
                        // Drop covered workspace streams before entering canonical storage.
                        // Discard a lower result fetched while finalization was in flight.
                        let after = *after;
                        for frame in &mut frames[..=index] {
                            *frame = None;
                        }

                        frames[index] = Some(Overlay::Native(
                            canonical.occupied_blocks(after..range.end).fuse(),
                        ));

                        lower_result = None;
                    }
                    _ => {}
                }

                let step = match frames[index].as_mut().expect("active overlay frame") {
                    Overlay::Native(stream) => stream.try_next().await.map(OverlayStep::Ready),
                    Overlay::Delta {
                        id,
                        delta,
                        upper,
                        left,
                        right,
                        after,
                        lower_done,
                    } => {
                        let step = async {
                            if let Some(result) = lower_result.take() {
                                *left = result?;
                                *lower_done = left.is_none();
                            }

                            if left.is_none() && !*lower_done {
                                return Ok(OverlayStep::Lower);
                            }

                            if right.is_none() {
                                *right = upper.try_next().await?;
                            }

                            let key = match (*left, *right) {
                                (_, Some(key)) if left.is_none_or(|old| key <= old) => {
                                    *right = None;
                                    if *left == Some(key) {
                                        *left = None;
                                    }

                                    key
                                }
                                (Some(key), _) => {
                                    let suppressed = delta.touches(key).await?;
                                    *left = None;
                                    if suppressed {
                                        return Ok(OverlayStep::Retry);
                                    }

                                    key
                                }
                                (None, None) => return Ok(OverlayStep::Ready(None)),
                                (None, Some(_)) => unreachable!("upper key was selected"),
                            };

                            Ok::<_, fensor::Error>(OverlayStep::Ready(Some(key)))
                        }
                        .await;

                        // Cleanup may race either native read or a touched lookup.
                        // Handoff resumes from the last delivered key, even on error.
                        if visibility.retired(*id) {
                            continue;
                        }

                        if let Ok(OverlayStep::Ready(Some(key))) = step {
                            *after = key + 1;
                        }

                        step
                    }
                };

                match step {
                    Ok(OverlayStep::Lower) => index -= 1,
                    Ok(OverlayStep::Retry) => {}
                    step => {
                        let result = step.map(|step| match step {
                            OverlayStep::Ready(key) => key,
                            _ => unreachable!("completed overlay step"),
                        });

                        if index == frames.len() - 1 {
                            return result
                                .map(|key| key.map(|key| (key, (frames, canonical, visibility))));
                        }

                        lower_result = Some(result);
                        index += 1;
                    }
                }
            }
        },
    )
    .boxed()
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> TensorSource for Snapshot<F, T> {
    fn storage_geometry(&self) -> StorageGeometry {
        self.canonical.storage_geometry()
    }

    fn read_storage<'a>(
        &'a self,
        read: fensor::StorageRead<'a>,
    ) -> BoxFuture<'a, fensor::Result<Vec<T>>> {
        Box::pin(async move {
            let geometry = self.storage_geometry();
            let plan = read.plan(&geometry)?;
            let ids: Vec<_> = plan.block_ids().collect();
            let mut sources = vec![None; ids.len()];
            self.resolve_sources(&ids, &mut sources).await?;

            let mut values = vec![T::ZERO; plan.len()];
            let mut start = 0;
            while start < ids.len() {
                let source = sources[start];
                let end = sources[start..]
                    .iter()
                    .position(|next| *next != source)
                    .map_or(ids.len(), |len| start + len);

                let blocks = &ids[start..end];

                if let Some(index) = source {
                    let (id, delta) = &self.deltas[index];
                    let result = if self.visibility.retired(*id) {
                        Ok(())
                    } else {
                        delta
                            .payload
                            .visit_blocks(blocks, |id, block| {
                                plan.scatter(id, block, &mut values)?;
                                #[cfg(test)]
                                observation::after_scatter(&delta.generation)?;

                                Ok(())
                            })
                            .await
                    };

                    if self.visibility.retired(*id) {
                        // Discard partial output as well as obsolete workspace errors.
                        // Native consumption has released its guards before retrying.
                        let zero = vec![T::ZERO; geometry.block_capacity()];
                        for &id in blocks {
                            plan.scatter(id, &zero[..geometry.block_len(id)?], &mut values)?;
                        }

                        self.resolve_sources(blocks, &mut sources[start..end])
                            .await?;

                        continue;
                    }

                    result?;
                } else {
                    self.canonical
                        .visit_blocks(blocks, |id, block| plan.scatter(id, block, &mut values))
                        .await?;
                }

                start = end;
            }

            Ok(values)
        })
    }

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, fensor::Result<Vec<T>>> {
        Box::pin(async move {
            for (txn_id, delta) in self.deltas.iter().rev() {
                if self.visibility.retired(*txn_id) {
                    continue;
                }

                let result = delta.read(id).await;
                if self.visibility.retired(*txn_id) {
                    continue;
                }

                if let Some(values) = result? {
                    return Ok(values);
                }
            }

            self.canonical.read_logical_block(id).await
        })
    }

    fn occupied_blocks(&self, range: BlockRange) -> BoxStream<'static, fensor::Result<u64>> {
        let keys = overlay_keys(
            self.canonical.occupied_blocks(range.clone()),
            self.deltas.iter().cloned(),
            self.visibility.clone(),
            self.canonical.clone(),
            range,
        );

        crate::stream::GuardedStream::new(keys, self._permit.clone()).boxed()
    }
}

#[derive(Clone)]
pub(super) struct Owner<Txn: StorageContext, T: TensorElement>
where
    Txn::File: TensorFileEntry<T>,
{
    pub core: Arc<BlockOwner<Txn::File, T>>,
    geometry: StorageGeometry,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: StorageContext, T: TensorElement> TensorGeometry for Owner<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    type DType = T;

    fn dtype(&self) -> fensor::NumberType {
        self.geometry.schema().dtype()
    }

    fn layout(&self) -> Layout {
        self.geometry.layout()
    }

    fn shape(&self) -> &[u64] {
        self.geometry.schema().shape()
    }
}

impl<Txn: StorageContext, T: TensorElement> TensorArray for Owner<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    fn schema(&self) -> &fensor::TensorSchema {
        self.geometry.schema()
    }

    fn strides(&self) -> &[u64] {
        self.geometry.schema().strides()
    }
}

impl<Txn: StorageContext, T: TensorElement> Owner<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    pub fn new(native: fensor::Tensor<Txn::File, T>) -> Self {
        let geometry = native.storage_geometry();
        Self {
            core: Arc::new(CollectionOwner::new(native, Collator::default())),
            geometry,
            txn: PhantomData,
        }
    }

    pub fn geometry(&self) -> &StorageGeometry {
        &self.geometry
    }

    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.core, &other.core)
    }

    pub fn canonical(&self) -> TCResult<fensor::Tensor<Txn::File, T>> {
        Ok(self.core.read_state().persistent.clone())
    }

    pub fn snapshot(
        &self,
        id: TxnId,
        permit: Option<ReadPermit>,
    ) -> TCResult<Snapshot<Txn::File, T>> {
        // Retain native/delta handles, not copied payloads. Read snapshots own a
        // range permit; a writer already holding its reservation passes None.
        // Visibility tracks the shared frontier so later reads can hand retired
        // workspace layers off to canonical storage after materialization.
        let visible = self.core.visible_snapshot(id)?;

        Ok(Snapshot {
            canonical: visible.persistent,
            deltas: visible.deltas.into(),
            visibility: visible.visibility,
            _permit: permit.map(Arc::new),
        })
    }

    pub async fn read_snapshot(
        &self,
        id: TxnId,
        range: BlockRange,
    ) -> TCResult<Snapshot<Txn::File, T>> {
        let permit = self.core.acquire_read_permit(id, range).await?;
        self.snapshot(id, Some(permit))
    }

    pub fn try_snapshot(&self, id: TxnId, range: BlockRange) -> TCResult<Snapshot<Txn::File, T>> {
        let permit = self.core.try_acquire_read_permit(id, range)?;
        self.snapshot(id, Some(permit))
    }

    async fn pending(&self, txn: &Txn) -> TCResult<Arc<Delta<Txn::File, T>>> {
        self.core
            .pending(txn.id(), |_| {
                let geometry = self.geometry.clone();
                async move {
                    let dir = txn.subcontext_unique().context().await?;
                    Ok(Delta::create(dir, geometry).await?)
                }
            })
            .await
    }

    pub async fn read_value(&self, id: TxnId, coord: &[u64]) -> TCResult<T> {
        let (block, _) = self.geometry.block_position(coord)?;
        let snapshot = self.read_snapshot(id, block..block + 1).await?;
        Ok(fensor::TensorView::new(snapshot).read_value(coord).await?)
    }

    pub async fn write_value(&self, txn: &Txn, coord: &[u64], value: T) -> TCResult<()> {
        let (block, offset) = self.geometry.block_position(coord)?;
        let _permit = self
            .core
            .try_acquire_write_permit(txn.id(), block..block + 1)?;

        let pending = self.pending(txn).await?;
        // A point mutation replaces a whole logical block. Read our own pending
        // changes as well as committed history, preserving neighboring values;
        // the block write reservation covers this entire read/modify/write.
        let snapshot = self.snapshot(txn.id(), None)?;
        let mut values = snapshot.read_logical_block(block).await?;
        values[offset] = value;
        pending.write(block, values).await?;
        Ok(())
    }

    pub async fn stage(&self, txn: &Txn) -> TCResult<Arc<Delta<Txn::File, T>>> {
        // An unpublished patch has independent storage. Its caller retains the
        // operation's reservations and finishes source reads before publication.
        self.assert_writable(txn.id())?;
        Ok(Delta::create(
            txn.subcontext_unique().context().await?,
            self.geometry.clone(),
        )
        .await?)
    }

    /// Merge only staged blocks, preserving concurrent disjoint pending writes.
    pub async fn publish(&self, txn: &Txn, patch: Arc<Delta<Txn::File, T>>) -> TCResult<()> {
        let pending = {
            let mut state = self.core.write_state();
            state.assert_writable(txn.id())?;
            match state.pending.entry(txn.id()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(patch);
                    return Ok(());
                }
                std::collections::btree_map::Entry::Occupied(entry) => entry.get().clone(),
            }
        };

        // Release bookkeeping before native I/O. Merge only touched blocks so a
        // disjoint write in the same transaction is not replaced by this patch.
        patch.apply_to_delta(&pending).await?;
        Ok(())
    }

    // Assignment drops source read leases before acquiring destination writes.
    // A generation check prevents publishing a result made stale during that gap.
    pub fn pending_stamp(&self, id: TxnId) -> TCResult<Option<(Arc<AtomicU64>, u64)>> {
        let state = self.core.read_state();
        state.assert_writable(id)?;
        Ok(state.pending.get(&id).map(|delta| {
            (
                delta.generation.clone(),
                delta.generation.load(Ordering::Acquire),
            )
        }))
    }

    pub async fn publish_if_unchanged(
        &self,
        txn: &Txn,
        delta: Arc<Delta<Txn::File, T>>,
        stamp: Option<(Arc<AtomicU64>, u64)>,
    ) -> TCResult<()> {
        {
            let id = txn.id();
            let mut state = self.core.write_state();
            state.assert_writable(id)?;
            let unchanged = match (state.pending.get(&id), stamp) {
                (None, None) => true,
                (Some(current), Some((generation, count))) => {
                    Arc::ptr_eq(&current.generation, &generation)
                        && current.generation.load(Ordering::Acquire) == count
                }
                _ => false,
            };

            if !unchanged {
                return Err(TCError::conflict(
                    "Tensor changed while converting assignment reservations",
                ));
            }

            if let std::collections::btree_map::Entry::Vacant(entry) = state.pending.entry(id) {
                entry.insert(delta);
                return Ok(());
            }
        }

        self.publish(txn, delta).await
    }

    pub fn install(&self, id: TxnId, delta: Arc<Delta<Txn::File, T>>) -> TCResult<()> {
        // Full restoration replaces pending ownership after staging succeeds;
        // unlike publish, it deliberately discards earlier pending block writes.
        let replaced = {
            let mut state = self.core.write_state();
            state.assert_writable(id)?;
            state.pending.insert(id, delta)
        };

        drop(replaced);
        Ok(())
    }

    pub fn assert_writable(&self, id: TxnId) -> TCResult<()> {
        self.core.read_state().assert_writable(id)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/storage/observation.rs"]
mod observation;
#[cfg(test)]
#[path = "../../tests/unit/tensor/storage/tests.rs"]
mod tests;
