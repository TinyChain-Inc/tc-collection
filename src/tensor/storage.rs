//! Transactional logical-block ownership over native fensor storage.
//!
//! Lock order: reserve the transaction range, briefly inspect collection state,
//! release that guard, then enter one native index or payload domain at a time.
//! Lifecycle callers quiesce covered work; lifecycle methods never reserve ranges.
//! `Versions` owns transaction visibility and lifecycle ordering; this module translates Tensor
//! mutations into logical-block deltas. Computation and physical payload/index
//! maintenance remain delegated to fensor. Caller obligations are defined in
//! TRANSACTIONAL_COLLECTION_CONTRACT.md.

use std::io;
use std::marker::PhantomData;
use std::ops::Bound::{Excluded, Included};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use b_tree::collate::Collator;
use b_tree::{BTreeLock, BTreeReadGuard, Range, Schema};
use fensor::{
    BoxFuture, Layout, StorageGeometry, TensorArray, TensorElement, TensorFileEntry,
    TensorGeometry, TensorSource,
};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::{TCError, TCResult};
use tc_ir::{Transact, TxnId};

use crate::persistence::Versions;
use crate::persistence::storage::{Allocation, Store};
use crate::stream::{GuardedStream, Overlay};
use crate::{CollectionFile, StorageContext};

pub(super) type BlockRange = std::ops::Range<u64>;
type ReadPermit = txn_lock::semaphore::PermitRead<BlockRange>;
type Zeros<F> = BTreeLock<ZeroSchema, Collator<u64>, F>;

const ZERO_BATCH_SIZE: usize = 4096;

#[derive(Debug, Eq, PartialEq)]
struct ZeroSchema(u64);

impl Schema for ZeroSchema {
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
        if zero_id(prefix)? >= self.0 {
            return Err(freqfs::IntegrityError::invalid_data(
                "Tensor zero block is out of bounds",
            ));
        }

        Ok(())
    }

    fn validate_key(&self, key: &[u64]) -> io::Result<()> {
        self.validate_prefix(key)
    }
}

fn zero_id(key: &[u64]) -> io::Result<u64> {
    match key {
        [id] => Ok(*id),
        _ => Err(freqfs::IntegrityError::invalid_data(
            "invalid Tensor zero block key",
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

/// All payloads are owned by this native allocation; no delta points into another version.
pub(super) struct Delta<F: CollectionFile, T: TensorElement> {
    storage: Arc<Allocation<F>>,
    payload: fensor::SparseBlocks<F, T>,
    // A whole replacement covers every block. A patch records explicit zeros here.
    zeros: Option<Zeros<F>>,
    // Assignment stamps retain this identity without retaining the allocation.
    generation: Arc<AtomicU64>,
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> Delta<F, T> {
    async fn load(
        storage: Arc<Allocation<F>>,
        geometry: StorageGeometry,
    ) -> TCResult<(Arc<Self>, u64)> {
        let (payload, zeros) = {
            let dir = storage.dir().read().await;
            match (
                dir.len(),
                dir.get_dir("replacement"),
                dir.get_dir("payload"),
                dir.get_dir("zeros"),
            ) {
                (1, Some(payload), None, None) => (payload.clone(), None),
                (2, None, Some(payload), Some(zeros)) => (payload.clone(), Some(zeros.clone())),
                _ => {
                    return Err(freqfs::IntegrityError::invalid_data(
                        "invalid Tensor delta layout",
                    )
                    .into());
                }
            }
        };

        let blocks = geometry.block_count();
        let payload = fensor::SparseBlocks::load(payload, geometry).await?;
        let zeros = if let Some(dir) = zeros {
            let zeros = BTreeLock::load(ZeroSchema(blocks), Collator::default(), dir)?;
            zeros.validate().await?;
            Some(zeros)
        } else {
            None
        };
        let delta = Arc::new(Self {
            storage,
            payload,
            zeros,
            generation: Arc::new(AtomicU64::new(0)),
        });

        let units = delta.units().await?;
        Ok((delta, units))
    }

    async fn units(&self) -> io::Result<u64> {
        if self.is_whole() {
            return Ok(self.payload.geometry().block_count());
        }

        // The two streams contain distinct IDs bounded by the block count.
        self.replacement_ids()
            .try_fold(0, |count, _| async move { Ok(count + 1) })
            .await
    }

    async fn create(
        storage: Arc<Allocation<F>>,
        geometry: StorageGeometry,
        whole: bool,
    ) -> io::Result<Arc<Self>> {
        let (payload, zeros) = {
            let mut dir = storage.dir().write().await;
            if !dir.is_empty() {
                return Err(invalid("Tensor delta requires empty delegated storage"));
            }

            if whole {
                (dir.create_dir("replacement".into())?, None)
            } else {
                (
                    dir.create_dir("payload".into())?,
                    Some(dir.create_dir("zeros".into())?),
                )
            }
        };

        let blocks = geometry.block_count();
        let payload = fensor::SparseBlocks::create(payload, geometry)
            .await
            .map_err(native_error)?;
        let zeros = if let Some(dir) = zeros {
            Some(BTreeLock::create(ZeroSchema(blocks), Collator::default(), dir).await?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            storage,
            payload,
            zeros,
            generation: Arc::new(AtomicU64::new(0)),
        }))
    }

    fn is_whole(&self) -> bool {
        self.zeros.is_none()
    }

    async fn read(&self, id: u64) -> io::Result<Option<Vec<T>>> {
        let len = self
            .payload
            .geometry()
            .block_len(id)
            .map_err(native_error)?;
        if self.is_zero(id).await? {
            return Ok(Some(vec![T::ZERO; len]));
        }

        let values = self.read_payload(id).await?;
        Ok(if self.is_whole() {
            Some(values.unwrap_or_else(|| vec![T::ZERO; len]))
        } else {
            values
        })
    }

    async fn is_zero(&self, id: u64) -> io::Result<bool> {
        let Some(zeros) = &self.zeros else {
            return Ok(false);
        };

        #[cfg(test)]
        observation::record(&self.generation, |counts| counts.zero_lookups += 1);

        zeros.read().await.contains(&[id]).await
    }

    async fn read_payload(&self, id: u64) -> io::Result<Option<Vec<T>>> {
        #[cfg(test)]
        observation::record(&self.generation, |counts| counts.payload_reads += 1);

        self.payload.read(id).await.map_err(native_error)
    }

    async fn visit_payloads(
        &self,
        ids: &[u64],
        mut consume: impl FnMut(u64, &[T]) -> fensor::Result<()> + Send,
    ) -> fensor::Result<()> {
        self.payload
            .visit_blocks(ids, |id, block| {
                consume(id, block)?;
                #[cfg(test)]
                observation::after_scatter(&self.generation)?;

                Ok(())
            })
            .await
    }

    async fn visit_zeros(
        zeros: &BTreeReadGuard<ZeroSchema, Collator<u64>, F>,
        blocks: BlockRange,
        mut visit: impl FnMut(u64) -> ControlFlow<()> + Send,
    ) -> io::Result<()> {
        let range = Range::with_bounds(
            Vec::<u64>::new(),
            (Included(blocks.start), Excluded(blocks.end)),
        );

        let mut previous = None;
        zeros
            .visit(range, false, |key| -> io::Result<_> {
                let id = zero_id(key)?;
                if !blocks.contains(&id) || previous.is_some_and(|last| id <= last) {
                    return Err(freqfs::IntegrityError::invalid_data(
                        "invalid Tensor zero block order",
                    ));
                }

                previous = Some(id);
                Ok(visit(id))
            })
            .await
    }

    // Check one extra key for duplicates before resuming above the last buffered ID.
    async fn zero_batch(zeros: &Zeros<F>, blocks: BlockRange) -> io::Result<Vec<u64>> {
        let mut ids = Vec::new();
        Self::visit_zeros(&zeros.read().await, blocks, |id| {
            if ids.len() == ZERO_BATCH_SIZE {
                ControlFlow::Break(())
            } else {
                ids.push(id);
                ControlFlow::Continue(())
            }
        })
        .await?;

        Ok(ids)
    }

    fn zero_ids(&self, range: BlockRange) -> BoxStream<'static, io::Result<u64>> {
        let Some(zeros) = self.zeros.clone() else {
            return futures::stream::empty().boxed();
        };

        futures::stream::try_unfold((zeros, range), |(zeros, mut range)| async move {
            if range.is_empty() {
                return Ok::<_, io::Error>(None);
            }

            let ids = Self::zero_batch(&zeros, range.clone()).await?;
            let Some(last) = ids.last() else {
                return Ok(None);
            };

            range.start = last + 1;
            if ids.len() < ZERO_BATCH_SIZE {
                range.start = range.end;
            }

            Ok(Some((
                futures::stream::iter(ids.into_iter().map(Ok)),
                (zeros, range),
            )))
        })
        .try_flatten()
        .boxed()
    }

    /// Apply one bounded batch to the unpublished patch, preserving earlier batches.
    pub(super) async fn apply_updates(
        &self,
        base: Option<&Snapshot<F, T>>,
        updates: fensor::BlockUpdates<T>,
    ) -> TCResult<()> {
        for (block, updates) in updates {
            // A complete ordered block has no untouched values to read.
            if updates.len() == self.payload.geometry().block_len(block)?
                && updates
                    .iter()
                    .enumerate()
                    .all(|(i, (offset, _))| i == *offset)
            {
                let values = updates.into_iter().map(|(_, value)| value).collect();
                self.write(block, Some(values)).await?;
                continue;
            }

            let mut values = match self.read(block).await? {
                Some(values) => values,
                None => {
                    base.expect("a partial patch requires a base snapshot")
                        .read_logical_block(block)
                        .await?
                }
            };

            for (offset, value) in updates {
                values[offset] = value;
            }

            self.write(block, Some(values)).await?;
        }

        Ok(())
    }

    async fn write(&self, id: u64, values: Option<Vec<T>>) -> io::Result<()> {
        // Publish the payload before reconciling its zero marker. The range
        // reservation excludes readers; a failed write requires rollback.
        let stored = self
            .payload
            .replace(id, values)
            .await
            .map_err(native_error)?;

        if let Some(zeros) = &self.zeros {
            let mut zeros = zeros.write().await;
            if stored {
                zeros.delete(&[id]).await?;
            } else {
                zeros.insert(vec![id]).await?;
            }
        }

        self.generation
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| invalid("Tensor mutation generation exhausted"))?;

        Ok(())
    }

    // Merge disjoint native keysets, releasing each guard before entering the other tree.
    fn replacement_ids(&self) -> impl futures::Stream<Item = io::Result<(u64, bool)>> + '_ {
        let range = 0..self.payload.geometry().block_count();
        let occupied = self
            .payload
            .occupied_blocks(range.clone())
            .map_err(native_error)
            .fuse();

        if self.is_whole() {
            return occupied.map_ok(|id| (id, true)).left_stream();
        }

        let zeros = self.zero_ids(range).fuse();
        futures::stream::try_unfold(
            (zeros, occupied, None, None),
            |(mut zeros, mut occupied, mut zero, mut payload)| async move {
                tokio::task::coop::consume_budget().await;
                if zero.is_none() {
                    zero = zeros.try_next().await?;
                }

                if payload.is_none() {
                    payload = occupied.try_next().await?;
                }

                let next = match (zero, payload) {
                    (Some(z), Some(p)) if z == p => {
                        return Err(freqfs::IntegrityError::invalid_data(
                            "Tensor delta contains both a payload and a zero override",
                        ));
                    }
                    (Some(z), p) if p.is_none_or(|p| z < p) => {
                        zero = None;
                        (z, false)
                    }
                    (_, Some(p)) => {
                        payload = None;
                        (p, true)
                    }
                    _ => return Ok(None),
                };

                Ok(Some((next, (zeros, occupied, zero, payload))))
            },
        )
        .right_stream()
    }

    async fn read_replacement(&self, (id, occupied): (u64, bool)) -> io::Result<Option<Vec<T>>> {
        if !occupied {
            return Ok(None);
        }

        let Some(values) = self.read_payload(id).await? else {
            return Err(freqfs::IntegrityError::invalid_data(
                "missing occupied Tensor delta payload",
            ));
        };

        Ok(Some(values))
    }

    async fn validate_replacement(&self, (id, occupied): (u64, bool)) -> io::Result<()> {
        if occupied {
            let mut found = false;
            self.payload
                .visit_blocks(&[id], |_, _| {
                    found = true;
                    Ok(())
                })
                .await
                .map_err(native_error)?;

            if !found {
                return Err(freqfs::IntegrityError::invalid_data(
                    "missing occupied Tensor delta payload",
                ));
            }
        }

        Ok(())
    }

    fn replacements(&self) -> impl futures::Stream<Item = io::Result<(u64, Option<Vec<T>>)>> + '_ {
        self.replacement_ids()
            .and_then(move |(id, occupied)| async move {
                Ok((id, self.read_replacement((id, occupied)).await?))
            })
    }

    async fn apply_to_delta(&self, output: &Self) -> io::Result<()> {
        let blocks = self.replacements();
        futures::pin_mut!(blocks);
        while let Some((id, values)) = blocks.try_next().await? {
            output.write(id, values).await?;
        }

        Ok(())
    }

    // Without an output, validate both inputs before retaining the newer whole replacement.
    async fn merge_into(&self, newer: &Self, output: Option<&Self>) -> io::Result<u64> {
        let older_ids = self.replacement_ids();
        let newer_ids = newer.replacement_ids();
        futures::pin_mut!(older_ids, newer_ids);

        let mut older_block = older_ids.try_next().await?;
        let mut newer_block = newer_ids.try_next().await?;
        let mut ids = Vec::new();
        let mut units = 0;
        loop {
            let (source, block, head, keys) = if let Some(block) =
                older_block.filter(|older| newer_block.is_none_or(|newer| older.0 < newer.0))
            {
                (self, block, &mut older_block, &mut older_ids)
            } else if let Some(block) = newer_block {
                if let Some(older) = older_block.filter(|older| older.0 == block.0) {
                    // Validate even the shadowed input before publishing its replacement.
                    self.validate_replacement(older).await?;
                    older_block = older_ids.try_next().await?;
                }

                (newer, block, &mut newer_block, &mut newer_ids)
            } else {
                break;
            };

            let Some(output) = output else {
                source.validate_replacement(block).await?;
                *head = keys.try_next().await?;
                continue;
            };

            let values = source.read_replacement(block).await?;
            *head = keys.try_next().await?;
            let id = block.0;
            if values.is_some() {
                output
                    .payload
                    .replace(id, values)
                    .await
                    .map_err(native_error)?;
            } else if let Some(zeros) = &output.zeros {
                ids.push(id);

                if ids.len() == ZERO_BATCH_SIZE {
                    let keys = futures::stream::iter(ids.drain(..).map(|id| Ok(vec![id])));
                    zeros.write().await.insert_sorted(keys).await?;
                }
            }

            units += 1;
        }

        if let Some(zeros) = output.and_then(|output| output.zeros.as_ref()) {
            if !ids.is_empty() {
                let keys = futures::stream::iter(ids.into_iter().map(|id| Ok(vec![id])));
                zeros.write().await.insert_sorted(keys).await?;
            }

            Ok(units)
        } else {
            Ok(newer.payload.geometry().block_count())
        }
    }
}

#[derive(Clone)]
pub(super) struct Snapshot<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> {
    canonical: Arc<fensor::Tensor<F, T>>,
    deltas: Arc<[Arc<Delta<F, T>>]>,
    _permit: Option<Arc<ReadPermit>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockSource {
    Canonical,
    Zero,
    Delta(usize),
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> Snapshot<F, T> {
    // Resolve only requested blocks that no newer version has supplied.
    async fn resolve_sources(
        &self,
        ids: &[u64],
        sources: &mut [BlockSource],
    ) -> fensor::Result<()> {
        sources.fill(BlockSource::Canonical);
        let mut unresolved = Vec::new();
        for (index, delta) in self.deltas.iter().enumerate().rev() {
            if sources
                .iter()
                .all(|source| *source != BlockSource::Canonical)
            {
                break;
            }

            let Some(zeros) = &delta.zeros else {
                for source in sources.iter_mut() {
                    if *source == BlockSource::Canonical {
                        *source = BlockSource::Delta(index);
                    }
                }

                break;
            };

            {
                let zeros = zeros.read().await;
                let mut start = 0;
                while start < ids.len() {
                    tokio::task::coop::consume_budget().await;
                    if sources[start] != BlockSource::Canonical {
                        start += 1;
                        continue;
                    }

                    let mut end = start + 1;
                    while end < ids.len()
                        && sources[end] == BlockSource::Canonical
                        && ids[end - 1].checked_add(1) == Some(ids[end])
                    {
                        end += 1;
                    }

                    let first = ids[start];
                    #[cfg(test)]
                    observation::record(&delta.generation, |counts| counts.membership_scans += 1);

                    Delta::<F, T>::visit_zeros(&zeros, first..ids[end - 1] + 1, |id| {
                        sources[start + (id - first) as usize] = BlockSource::Zero;
                        ControlFlow::Continue(())
                    })
                    .await?;

                    #[cfg(test)]
                    observation::after_membership(&delta.generation)?;

                    start = end;
                }
            }

            unresolved.clear();
            unresolved.extend(
                ids.iter()
                    .zip(sources.iter())
                    .filter_map(|(&id, &source)| (source == BlockSource::Canonical).then_some(id)),
            );

            delta
                .payload
                .visit_occupied(&unresolved, |id| {
                    let offset = ids.binary_search(&id).expect("selected block ID");
                    sources[offset] = BlockSource::Delta(index);
                })
                .await?;
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
}

impl<F: CollectionFile + TensorFileEntry<T>, T: TensorElement> TensorSource for Snapshot<F, T> {
    fn storage_geometry(&self) -> &StorageGeometry {
        self.canonical.storage_geometry()
    }

    fn visit_blocks<'a, C>(
        &'a self,
        ids: &'a [u64],
        mut consume: C,
    ) -> BoxFuture<'a, fensor::Result<()>>
    where
        C: FnMut(u64, &[T]) -> fensor::Result<()> + Send + 'a,
    {
        Box::pin(async move {
            self.storage_geometry().validate_block_ids(ids)?;
            // A single block can be borrowed as soon as it is found.
            if let [id] = ids {
                for delta in self.deltas.iter().rev() {
                    if delta.is_zero(*id).await? {
                        return Ok(());
                    }

                    let mut covered = delta.is_whole();
                    delta
                        .visit_payloads(ids, |id, block| {
                            covered = true;
                            consume(id, block)
                        })
                        .await?;

                    if covered {
                        return Ok(());
                    }
                }

                return self.canonical.visit_blocks(ids, consume).await;
            }

            let mut sources = vec![BlockSource::Canonical; ids.len()];
            self.resolve_sources(ids, &mut sources).await?;
            let mut start = 0;
            while start < ids.len() {
                let source = sources[start];
                let end = sources[start..]
                    .iter()
                    .position(|next| *next != source)
                    .map_or(ids.len(), |len| start + len);

                let blocks = &ids[start..end];
                match source {
                    BlockSource::Canonical => {
                        self.canonical.visit_blocks(blocks, &mut consume).await?;
                    }
                    BlockSource::Delta(index) => {
                        self.deltas[index]
                            .visit_payloads(blocks, &mut consume)
                            .await?;
                    }
                    BlockSource::Zero => {}
                }

                start = end;
            }

            Ok(())
        })
    }

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, fensor::Result<Vec<T>>> {
        Box::pin(async move {
            for delta in self.deltas.iter().rev() {
                let result = delta.read(id).await;

                if let Some(values) = result? {
                    return Ok(values);
                }
            }

            self.canonical.read_logical_block(id).await
        })
    }

    fn occupied_blocks(&self, range: BlockRange) -> BoxStream<'static, fensor::Result<u64>> {
        if self.layout() == Layout::Dense {
            return self.canonical.occupied_blocks(range);
        }

        // The snapshot starts at its latest whole version, if there is one.
        let (base, patches) = match self.deltas.split_first() {
            Some((base, patches)) if base.is_whole() => {
                (base.payload.occupied_blocks(range.clone()), patches)
            }
            _ => (
                self.canonical.occupied_blocks(range.clone()),
                self.deltas.as_ref(),
            ),
        };

        let keys = if patches.is_empty() {
            base
        } else {
            let mut overlay = Overlay::new(
                Collator::default(),
                base.map_err(native_error).boxed(),
                |_, _, _| {},
            );

            for delta in patches {
                #[cfg(test)]
                observation::record(&delta.generation, |counts| counts.overlay_starts += 1);

                overlay.difference(delta.zero_ids(range.clone()));
                overlay.merge(
                    delta
                        .payload
                        .occupied_blocks(range.clone())
                        .map_err(native_error)
                        .boxed(),
                );
            }

            overlay.into_stream().map_err(fensor::Error::Io).boxed()
        };

        GuardedStream::new(keys, (self.deltas.clone(), self._permit.clone())).boxed()
    }
}

/// Transactional Tensor storage. Native handles and reservations remain private.
#[derive(Clone)]
pub(super) struct TensorBase<Txn: StorageContext, T: TensorElement>
where
    Txn::File: TensorFileEntry<T>,
{
    canonical: Arc<fensor::Tensor<Txn::File, T>>,
    store: Arc<Store<Txn::File>>,
    versions: Versions<Delta<Txn::File, T>>,
    semaphore: txn_lock::semaphore::Semaphore<TxnId, Collator<u64>, BlockRange>,
    txn: PhantomData<fn() -> Txn>,
}

impl<Txn: StorageContext, T: TensorElement> TensorGeometry for TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
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

impl<Txn: StorageContext, T: TensorElement> TensorArray for TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    fn schema(&self) -> &fensor::TensorSchema {
        self.canonical.schema()
    }
}

impl<Txn: StorageContext, T: TensorElement> TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    pub(super) fn try_acquire_write_permit(
        &self,
        id: TxnId,
        range: BlockRange,
    ) -> TCResult<txn_lock::semaphore::PermitWrite<BlockRange>> {
        self.versions
            .reserve_ordered_write(id, || self.semaphore.try_write(id, range))
    }

    pub(super) async fn from_native(
        native: fensor::Tensor<Txn::File, T>,
        store: Arc<Store<Txn::File>>,
    ) -> TCResult<Self> {
        store.initialize().await?;
        Ok(Self::from_store(native, store, Versions::new()))
    }

    fn from_store(
        native: fensor::Tensor<Txn::File, T>,
        store: Arc<Store<Txn::File>>,
        versions: Versions<Delta<Txn::File, T>>,
    ) -> Self {
        Self {
            canonical: Arc::new(native),
            store,
            versions,
            semaphore: txn_lock::semaphore::Semaphore::new(Collator::default()),
            txn: PhantomData,
        }
    }

    pub(super) async fn create(
        dir: DirLock<Txn::File>,
        schema: &super::TensorSchema,
        capacity: usize,
    ) -> TCResult<Self> {
        let store = Store::create(dir).await?;
        let native = fensor::Tensor::create(
            store.base().clone(),
            schema.native().clone(),
            schema.layout(),
            capacity,
        )
        .await?;

        Self::from_native(native, store).await
    }

    pub(super) async fn load(dir: DirLock<Txn::File>) -> TCResult<Self> {
        let store = Store::load(dir).await?;
        let native = fensor::Tensor::load(store.base().clone()).await?;
        let geometry = native.storage_geometry();
        let versions =
            Versions::load(&store, |storage| Delta::load(storage, geometry.clone())).await?;

        Ok(Self::from_store(native, store, versions))
    }

    pub(super) fn geometry(&self) -> &StorageGeometry {
        self.canonical.storage_geometry()
    }

    pub(super) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.canonical, &other.canonical)
    }

    pub(super) fn snapshot(
        &self,
        id: TxnId,
        permit: Option<ReadPermit>,
    ) -> TCResult<Snapshot<Txn::File, T>> {
        // Retain native/delta handles, not copied payloads. Read snapshots own a
        // range permit; a writer already holding its reservation passes None.
        // The snapshot pins immutable layers independently of later compaction.
        let mut visible = self.versions.visible_snapshot(id)?;
        if let Some(base) = visible.iter().rposition(|delta| delta.is_whole()) {
            visible.drain(..base);
        }

        Ok(Snapshot {
            canonical: self.canonical.clone(),
            deltas: visible.into(),
            _permit: permit.map(Arc::new),
        })
    }

    pub(super) async fn read_snapshot(
        &self,
        id: TxnId,
        range: BlockRange,
    ) -> TCResult<Snapshot<Txn::File, T>> {
        self.versions.check_readable(id)?;
        let permit = self.semaphore.read(id, range).await?;
        self.snapshot(id, Some(permit))
    }

    pub(super) fn try_snapshot(
        &self,
        id: TxnId,
        range: BlockRange,
    ) -> TCResult<Snapshot<Txn::File, T>> {
        self.versions.check_readable(id)?;
        let permit = self.semaphore.try_read(id, range)?;
        self.snapshot(id, Some(permit))
    }

    async fn pending(&self, txn: &Txn) -> TCResult<Arc<Delta<Txn::File, T>>> {
        self.versions
            .pending(txn.id(), || {
                let geometry = self.geometry().clone();
                async move {
                    let storage = self.store.allocate().await?;
                    Ok(Delta::create(storage, geometry, false).await?)
                }
            })
            .await
    }

    pub(super) async fn read_value(&self, id: TxnId, coord: &[u64]) -> TCResult<T> {
        let (block, offset) = self.geometry().block_position(coord)?;
        let snapshot = self.read_snapshot(id, block..block + 1).await?;
        Ok(snapshot.read_block_value(block, offset).await?)
    }

    pub(super) async fn write_value(&self, txn: &Txn, coord: &[u64], value: T) -> TCResult<()> {
        let (block, offset) = self.geometry().block_position(coord)?;
        let _permit = self.try_acquire_write_permit(txn.id(), block..block + 1)?;

        let pending = self.pending(txn).await?;
        // A point mutation replaces a whole logical block. Read our own pending
        // changes as well as committed history, preserving neighboring values;
        // the block write reservation covers this entire read/modify/write.
        let snapshot = self.snapshot(txn.id(), None)?;
        let mut values = snapshot.read_logical_block(block).await?;
        values[offset] = value;
        pending.write(block, Some(values)).await?;
        Ok(())
    }

    pub(super) async fn fill(&self, txn: &Txn, value: T) -> TCResult<()> {
        let _permit = self.try_acquire_write_permit(txn.id(), 0..self.geometry().block_count())?;
        let staged = self.stage(txn, true).await?;
        staged.payload.fill_empty(value).await?;

        self.versions.replace_pending(txn.id(), staged, |_| Ok(()))
    }

    pub(super) async fn stage_full_blocks(
        &self,
        txn: &Txn,
        blocks: impl futures::Stream<Item = fensor::Result<(u64, Vec<T>)>> + Send,
    ) -> TCResult<Arc<Delta<Txn::File, T>>> {
        let mut staged = self.stage(txn, true).await?;
        Arc::get_mut(&mut staged)
            .expect("new staged Tensor delta has one owner")
            .payload
            .append_sorted(blocks)
            .await?;

        Ok(staged)
    }

    pub(super) async fn stage(&self, txn: &Txn, whole: bool) -> TCResult<Arc<Delta<Txn::File, T>>> {
        // An unpublished version has independent storage. Its caller retains the
        // operation's reservations and finishes source reads before publication.
        self.versions.check_writable(txn.id())?;
        Ok(Delta::create(self.store.allocate().await?, self.geometry().clone(), whole).await?)
    }

    /// Merge only staged blocks, preserving concurrent disjoint pending writes.
    pub(super) async fn publish(
        &self,
        txn: &Txn,
        patch: Arc<Delta<Txn::File, T>>,
        validate: impl FnOnce(Option<&Delta<Txn::File, T>>) -> TCResult<()> + Send,
    ) -> TCResult<()> {
        self.versions
            .merge_pending(txn.id(), patch, validate, |pending, patch| async move {
                Ok(patch.apply_to_delta(&pending).await?)
            })
            .await
    }

    // Capture the generation that assignment must claim after releasing its read leases.
    pub(super) fn pending_stamp(&self, id: TxnId) -> TCResult<Option<(Arc<AtomicU64>, u64)>> {
        self.versions.inspect_pending_write(id, |pending| {
            pending.map(|delta| {
                (
                    delta.generation.clone(),
                    delta.generation.load(Ordering::Acquire),
                )
            })
        })
    }

    pub(super) async fn publish_if_unchanged(
        &self,
        txn: &Txn,
        delta: Arc<Delta<Txn::File, T>>,
        stamp: Option<(Arc<AtomicU64>, u64)>,
    ) -> TCResult<()> {
        let claim = |pending: Option<&Delta<Txn::File, T>>| {
            match (pending, stamp) {
                (None, None) => return Ok(()),
                (Some(current), Some((generation, count)))
                    if Arc::ptr_eq(&current.generation, &generation) =>
                {
                    let next = count
                        .checked_add(1)
                        .ok_or_else(|| invalid("Tensor mutation generation exhausted"))?;
                    // Another assignment must see the claim even while the merge awaits I/O.
                    if generation
                        .compare_exchange(count, next, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return Ok(());
                    }
                }
                _ => {}
            }

            Err(TCError::conflict(
                "Tensor changed while converting assignment reservations",
            ))
        };

        if delta.is_whole() {
            self.versions.replace_pending(txn.id(), delta, claim)
        } else {
            self.publish(txn, delta, claim).await
        }
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
}

impl<Txn: StorageContext, T: TensorElement> crate::Restore for TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    type Txn = Txn;

    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let sparse = self.geometry().sparse_axis().is_some();
        if self.schema() != source.schema() || sparse != source.geometry().sparse_axis().is_some() {
            return Err(TCError::bad_request("Tensor restoration schema mismatch"));
        }

        let id = txn.id();
        self.versions.check_writable(id)?;
        if self.same(source) {
            return Ok(());
        }

        let block_count = self.geometry().block_count();
        let _write = self.try_acquire_write_permit(id, 0..block_count)?;
        let staged = {
            let snapshot = source.try_snapshot(id, 0..source.geometry().block_count())?;
            let blocks = self.geometry().read_blocks(&snapshot)?;
            self.stage_full_blocks(txn, blocks).await?
        };

        self.versions.replace_pending(id, staged, |_| Ok(()))
    }
}

impl<Txn: StorageContext, T: TensorElement> Transact for TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    async fn commit(&self, id: TxnId) -> TCResult<()> {
        self.versions
            .commit_durable(
                &self.store,
                id,
                |delta| async move { Ok((delta.storage.clone(), delta.units().await?)) },
                || self.semaphore.finalize(&id, false).map_err(Into::into),
            )
            .await
    }

    async fn rollback(&self, id: &TxnId) -> TCResult<()> {
        let discarded = self.versions.rollback(*id, || {
            self.semaphore.finalize(id, false).map_err(Into::into)
        })?;

        if let Some(delta) = discarded {
            let allocation = delta.storage.id().clone();
            drop(delta);
            self.store.discard(&allocation).await?;
        }

        Ok(())
    }

    async fn finalize(&self, id: &TxnId) -> TCResult<()> {
        self.versions
            .finalize_durable(
                &self.store,
                *id,
                |older, newer, _complete_prefix| async move {
                    if newer.is_whole() {
                        older.merge_into(&newer, None).await?;
                        return Ok((newer, None));
                    }

                    let storage = self.store.allocate().await?;
                    let output =
                        Delta::create(storage.clone(), self.geometry().clone(), older.is_whole())
                            .await?;
                    let units = older.merge_into(&newer, Some(&output)).await?;
                    Ok((output, Some((storage, units))))
                },
                || self.semaphore.finalize(id, true).map_err(Into::into),
            )
            .await
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/storage/observation.rs"]
mod observation;
#[cfg(test)]
#[path = "../../tests/unit/tensor/storage/tests.rs"]
pub(super) mod tests;
