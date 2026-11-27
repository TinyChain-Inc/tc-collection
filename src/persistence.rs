//! Shared version visibility and lifecycle ordering; concrete collections own storage and reservations.

use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use tc_error::TCResult;
use tc_ir::TxnId;

pub(crate) mod storage;

struct State<D> {
    // None records a committed decision without allocating an empty delta. Its
    // presence still prevents late writes and makes repeated commit idempotent.
    committed: BTreeMap<TxnId, Option<Arc<D>>>,
    pending: BTreeMap<TxnId, Arc<D>>,
    finalized: Option<TxnId>,
    compacted: Vec<(storage::Run, Arc<D>)>,
    replaying: Option<TxnId>,
    failed: bool,
}

impl<D> State<D> {
    fn check_readable(&self, txn_id: TxnId) -> TCResult<()> {
        if self.failed {
            return Err(freqfs::IntegrityError::invalid_data(
                "collection publication interrupted; reopen storage",
            )
            .into());
        }

        if self.finalized.is_some_and(|cutoff| txn_id <= cutoff) {
            return Err(txn_lock::Error::Outdated.into());
        }

        Ok(())
    }

    fn check_writable(&self, txn_id: TxnId) -> TCResult<()> {
        self.check_readable(txn_id)?;
        if self.committed.contains_key(&txn_id) {
            return Err(txn_lock::Error::Committed.into());
        }

        Ok(())
    }

    fn check_ordered_write(&self, id: TxnId) -> TCResult<()> {
        self.check_writable(id)?;
        if self.replaying != Some(id)
            && self
                .committed
                .range((Excluded(id), Unbounded))
                .any(|(_, delta)| delta.is_some())
        {
            return Err(txn_lock::Error::Conflict.into());
        }

        Ok(())
    }
}

/// Shared mutable versions, with no storage handles or reservation policy.
pub(crate) struct Versions<D> {
    state: Arc<RwLock<State<D>>>,
}

// A cancelled durable operation may have already published its descriptor.
// Mark the live owner failed unless the entire transition completes.
struct Publication<'a, D> {
    versions: &'a Versions<D>,
    complete: bool,
}

impl<D> Drop for Publication<'_, D> {
    fn drop(&mut self) {
        if !self.complete {
            self.versions.write_state().failed = true;
        }
    }
}

struct ReplayScope<'a, D>(&'a Versions<D>);

impl<D> Drop for ReplayScope<'_, D> {
    fn drop(&mut self) {
        self.0.write_state().replaying = None;
    }
}

impl<D> Clone for Versions<D> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<D> Versions<D> {
    pub async fn load<F, Load, Loaded>(store: &storage::Store<F>, mut load: Load) -> TCResult<Self>
    where
        F: crate::CollectionFile,
        Load: FnMut(Arc<storage::Allocation<F>>) -> Loaded,
        Loaded: Future<Output = TCResult<(Arc<D>, u64)>>,
    {
        let (finalized, runs) = store.frontier().await?;
        let mut compacted = Vec::new();
        let mut ids = std::collections::BTreeSet::new();
        for run in runs {
            if !ids.insert(run.version.id.clone()) {
                return Err(
                    freqfs::IntegrityError::invalid_data("duplicate native version").into(),
                );
            }

            let allocation = store.open(&run.version).await?;
            let (delta, units) = load(allocation).await?;
            if units != run.version.units {
                return Err(
                    freqfs::IntegrityError::invalid_data("native version size mismatch").into(),
                );
            }

            compacted.push((run, delta));
        }

        let mut committed = BTreeMap::new();
        for (id, version) in store.receipts(finalized).await? {
            if !ids.insert(version.id.clone()) {
                return Err(
                    freqfs::IntegrityError::invalid_data("duplicate native version").into(),
                );
            }

            let (delta, units) = load(store.open(&version).await?).await?;
            if units != version.units {
                return Err(
                    freqfs::IntegrityError::invalid_data("native version size mismatch").into(),
                );
            }

            committed.insert(id, Some(delta));
        }

        let versions = Self {
            state: Arc::new(RwLock::new(State {
                committed,
                pending: BTreeMap::new(),
                finalized,
                compacted,
                replaying: None,
                failed: false,
            })),
        };

        store.reclaim().await?;
        Ok(versions)
    }

    pub async fn recover<Replay, Replayed, Commit, Committed>(
        &self,
        id: TxnId,
        replay: Replay,
        commit: Commit,
    ) -> TCResult<()>
    where
        Replay: FnOnce() -> Replayed,
        Replayed: Future<Output = TCResult<()>>,
        Commit: FnOnce() -> Committed,
        Committed: Future<Output = TCResult<()>>,
    {
        {
            let mut state = self.write_state();
            if state.failed {
                return Err(freqfs::IntegrityError::invalid_data(
                    "collection publication interrupted",
                )
                .into());
            }

            if state.finalized.is_some_and(|cutoff| id <= cutoff)
                || state.committed.contains_key(&id)
            {
                return Ok(());
            }

            if state.replaying.is_some() {
                return Err(txn_lock::Error::Conflict.into());
            }

            state.replaying = Some(id);
        }

        {
            // An accepted older transaction can still need replay after a newer
            // disjoint mutation published its native receipt.
            let _replay = ReplayScope(self);
            replay().await?;
        }

        commit().await
    }

    pub async fn commit_durable<F, Freeze, Frozen>(
        &self,
        store: &storage::Store<F>,
        id: TxnId,
        freeze: Freeze,
        release: impl FnOnce() -> TCResult<()>,
    ) -> TCResult<()>
    where
        F: crate::CollectionFile,
        Freeze: FnOnce(Arc<D>) -> Frozen,
        Frozen: Future<Output = TCResult<(Arc<storage::Allocation<F>>, u64)>>,
    {
        let (committed, delta) = {
            let state = self.read_state();
            state.check_readable(id)?;
            (
                state.committed.contains_key(&id),
                state.pending.get(&id).cloned(),
            )
        };

        if committed {
            return release();
        }

        let mut publication = Publication {
            versions: self,
            complete: false,
        };

        if let Some(delta) = delta {
            let (allocation, units) = freeze(delta).await?;
            store.commit(id, &allocation, units).await?;
        }

        self.commit(id, release)?;
        publication.complete = true;
        Ok(())
    }

    pub async fn finalize_durable<F, Merge, Merged>(
        &self,
        store: &storage::Store<F>,
        cutoff: TxnId,
        mut merge: Merge,
        release: impl FnOnce() -> TCResult<()>,
    ) -> TCResult<()>
    where
        F: crate::CollectionFile,
        Merge: FnMut(Arc<D>, Arc<D>, bool) -> Merged,
        Merged: Future<Output = TCResult<(Arc<D>, Option<(Arc<storage::Allocation<F>>, u64)>)>>,
    {
        let (mut runs, committed) = {
            let state = self.read_state();
            if state.failed {
                return Err(freqfs::IntegrityError::invalid_data(
                    "collection publication interrupted",
                )
                .into());
            }

            if state.finalized.is_some_and(|id| cutoff <= id) {
                return Ok(());
            }

            (
                state.compacted.clone(),
                state
                    .committed
                    .range(..=cutoff)
                    .filter_map(|(id, delta)| delta.as_ref().map(|delta| (*id, delta.clone())))
                    .collect::<BTreeMap<_, _>>(),
            )
        };

        let mut publication = Publication {
            versions: self,
            complete: false,
        };

        // Only unpublished outputs are disposable before the descriptor switch.
        // Decreasing size classes keep at most u64::BITS output leases here.
        let mut temporary = BTreeMap::new();
        for (id, delta) in committed {
            let version = store.receipt(id).await?;
            if version.units == 0 {
                continue;
            }

            runs.push((
                storage::Run {
                    through: id,
                    version,
                },
                delta,
            ));

            while runs.len() >= 2 {
                let newer = &runs[runs.len() - 1].0;
                let older = &runs[runs.len() - 2].0;
                if older.version.units.ilog2() > newer.version.units.ilog2() {
                    break;
                }

                let (newer, right) = runs.pop().expect("newer run");
                let (older, left) = runs.pop().expect("older run");
                let (output, fresh) = merge(left, right, runs.is_empty()).await?;

                if temporary.remove(&older.version.id).is_some() {
                    store.discard(&older.version.id).await?;
                }

                // Without a fresh allocation, the callback returns the newer run's delta.
                let Some((allocation, units)) = fresh else {
                    runs.push((newer, output));
                    continue;
                };

                if temporary.remove(&newer.version.id).is_some() {
                    store.discard(&newer.version.id).await?;
                }

                if units != 0 {
                    temporary.insert(allocation.id().clone(), allocation.clone());
                    runs.push((
                        storage::Run {
                            through: newer.through,
                            version: storage::Version {
                                id: allocation.id().clone(),
                                units,
                            },
                        },
                        output,
                    ));
                } else {
                    drop(output);
                    let id = allocation.id().clone();
                    drop(allocation);
                    store.discard(&id).await?;
                }
            }
        }
        // Intermediate merges are reconstructible from the old published inputs.
        // Synchronize each surviving output once, before publishing its descriptor.
        for allocation in temporary.values() {
            allocation.dir().sync_all().await?;
        }

        drop(temporary);
        store
            .publish(
                Some(cutoff),
                runs.iter().map(|(run, _)| run.clone()).collect(),
            )
            .await?;

        let discarded = {
            let mut state = self.write_state();
            let mut committed = state.committed.split_off(&cutoff);
            let at_cutoff = committed.remove(&cutoff);
            let mut pending = state.pending.split_off(&cutoff);
            let pending_cutoff = pending.remove(&cutoff);
            let discarded = (
                std::mem::replace(&mut state.compacted, runs),
                std::mem::replace(&mut state.committed, committed),
                at_cutoff,
                std::mem::replace(&mut state.pending, pending),
                pending_cutoff,
            );

            state.finalized = Some(cutoff);
            discarded
        };

        // Native handles release outside the bookkeeping guard, before reclamation.
        drop(discarded);
        release()?;
        store.reclaim().await?;
        publication.complete = true;
        Ok(())
    }

    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(State {
                committed: BTreeMap::new(),
                pending: BTreeMap::new(),
                finalized: None,
                compacted: Vec::new(),
                replaying: None,
                failed: false,
            })),
        }
    }

    fn read_state(&self) -> RwLockReadGuard<'_, State<D>> {
        self.state.read().expect("collection state poisoned")
    }

    fn write_state(&self) -> RwLockWriteGuard<'_, State<D>> {
        self.state.write().expect("collection state poisoned")
    }

    pub(crate) fn check_readable(&self, id: TxnId) -> TCResult<()> {
        self.read_state().check_readable(id)
    }

    pub(crate) fn check_writable(&self, id: TxnId) -> TCResult<()> {
        self.read_state().check_writable(id)
    }

    /// Reserve a collection mutation against the same state observed by
    /// commit publication. The reservation callback must be synchronous and nonblocking.
    pub(crate) fn reserve_ordered_write<P>(
        &self,
        id: TxnId,
        reserve: impl FnOnce() -> Result<P, txn_lock::Error>,
    ) -> TCResult<P> {
        let state = self.read_state();
        state.check_ordered_write(id)?;
        Ok(reserve()?)
    }

    /// Check ordered admission before preparing an unpublished assignment.
    /// Inspect the pending version under the same bookkeeping lock.
    pub(crate) fn inspect_pending_write<R>(
        &self,
        id: TxnId,
        inspect: impl FnOnce(Option<&D>) -> R,
    ) -> TCResult<R> {
        let state = self.read_state();
        state.check_ordered_write(id)?;
        Ok(inspect(state.pending.get(&id).map(Arc::as_ref)))
    }

    /// Publish a complete replacement once construction and source reads finish.
    /// Validate the previous pending version under the publication lock.
    pub(crate) fn replace_pending(
        &self,
        id: TxnId,
        delta: Arc<D>,
        validate: impl FnOnce(Option<&D>) -> TCResult<()>,
    ) -> TCResult<()> {
        let replaced = {
            let mut state = self.write_state();
            state.check_writable(id)?;
            validate(state.pending.get(&id).map(Arc::as_ref))?;
            state.pending.insert(id, delta)
        };

        drop(replaced);
        Ok(())
    }

    /// Adopt an unpublished patch or merge it under the caller's reservation.
    /// Validation and selection share a lock. Native I/O runs after unlocking.
    pub(crate) async fn merge_pending<Fut>(
        &self,
        id: TxnId,
        patch: Arc<D>,
        validate: impl FnOnce(Option<&D>) -> TCResult<()>,
        merge: impl FnOnce(Arc<D>, Arc<D>) -> Fut,
    ) -> TCResult<()>
    where
        Fut: Future<Output = TCResult<()>>,
    {
        let pending = {
            use std::collections::btree_map::Entry;

            let mut state = self.write_state();
            state.check_writable(id)?;
            match state.pending.entry(id) {
                Entry::Vacant(entry) => {
                    validate(None)?;
                    entry.insert(patch);
                    return Ok(());
                }
                Entry::Occupied(entry) => {
                    validate(Some(entry.get()))?;
                    entry.get().clone()
                }
            }
        };

        merge(pending, patch).await
    }

    pub fn visible_snapshot(&self, txn_id: TxnId) -> TCResult<Vec<Arc<D>>> {
        let state = self.read_state();
        state.check_readable(txn_id)?;

        // Append this transaction's pending delta last for newest-first precedence.
        Ok(Self::committed_deltas(&state, txn_id)
            .chain(state.pending.get(&txn_id))
            .cloned()
            .collect())
    }

    pub fn committed_snapshot(&self, txn_id: TxnId) -> TCResult<Vec<Arc<D>>> {
        let state = self.read_state();
        state.check_readable(txn_id)?;
        Ok(Self::committed_deltas(&state, txn_id).cloned().collect())
    }

    fn committed_deltas(state: &State<D>, txn_id: TxnId) -> impl Iterator<Item = &Arc<D>> {
        state.compacted.iter().map(|(_, delta)| delta).chain(
            state
                .committed
                .range(..=txn_id)
                .filter_map(|(_, delta)| delta.as_ref()),
        )
    }

    /// Find the newest delta which determines a point result. An explicit
    /// deletion is a result, so callers can distinguish it from falling through
    /// to the immutable base. The caller holds the key's reservation.
    pub async fn lookup<R, F, Fut>(
        &self,
        txn_id: TxnId,
        mut include_pending: bool,
        mut lookup: F,
    ) -> TCResult<Option<R>>
    where
        F: FnMut(Arc<D>) -> Fut,
        Fut: Future<Output = io::Result<Option<R>>>,
    {
        let mut before = Included(txn_id);
        let mut frontier = None;
        loop {
            let next = {
                let state = self.read_state();
                state.check_readable(txn_id)?;

                // Compaction can replace layers while the lookup awaits I/O.
                // Restart selection against the new frontier after a miss.
                if frontier != state.finalized {
                    before = Included(txn_id);
                    frontier = state.finalized;
                }

                let pending = if std::mem::take(&mut include_pending) {
                    state.pending.get(&txn_id).map(|delta| (txn_id, delta))
                } else {
                    None
                };

                pending
                    .or_else(|| {
                        state
                            .committed
                            .range((Unbounded, before))
                            .rev()
                            .find_map(|(id, delta)| delta.as_ref().map(|delta| (*id, delta)))
                    })
                    .or_else(|| {
                        state
                            .compacted
                            .iter()
                            .rev()
                            .find(|(run, _)| match before {
                                Excluded(id) => run.through < id,
                                Included(id) => run.through <= id,
                                Unbounded => true,
                            })
                            .map(|(run, delta)| (run.through, delta))
                    })
                    .map(|(id, delta)| (id, delta.clone()))
            };

            let Some((id, delta)) = next else {
                return Ok(None);
            };

            before = Excluded(id);
            tokio::task::coop::consume_budget().await;
            if let Some(result) = lookup(delta).await? {
                return Ok(Some(result));
            }
        }
    }

    /// Prepare without a bookkeeping guard, then recheck before publication.
    pub async fn pending<F: Future<Output = TCResult<Arc<D>>>>(
        &self,
        txn_id: TxnId,
        create: impl FnOnce() -> F,
    ) -> TCResult<Arc<D>> {
        {
            let state = self.read_state();
            state.check_writable(txn_id)?;
            if let Some(delta) = state.pending.get(&txn_id) {
                return Ok(delta.clone());
            }
        }

        let delta = create().await?;
        // Concurrent disjoint writes may have prepared another allocation. Keep
        // the published one, and never publish after a lifecycle decision.
        let pending = {
            let mut state = self.write_state();
            state.check_writable(txn_id)?;
            match state.pending.entry(txn_id) {
                std::collections::btree_map::Entry::Vacant(entry) => entry.insert(delta).clone(),
                std::collections::btree_map::Entry::Occupied(entry) => entry.get().clone(),
            }
        };

        // A losing construction is released after the bookkeeping guard.
        Ok(pending)
    }

    fn commit(&self, txn_id: TxnId, release: impl FnOnce() -> TCResult<()>) -> TCResult<()> {
        // Callers have sealed mutations. Moving the handle publishes visibility
        // without changing payloads, so retained result snapshots remain valid.
        {
            let mut state = self.write_state();
            if state.finalized.is_some_and(|cutoff| txn_id <= cutoff) {
                return Err(txn_lock::Error::Outdated.into());
            }

            if !state.committed.contains_key(&txn_id) {
                let delta = state.pending.remove(&txn_id);
                state.committed.insert(txn_id, delta);
            }
        }

        // The durable receipt precedes this visibility switch. Release reservations
        // only after the decision is visible, with no bookkeeping guard held.
        release()
    }

    pub fn rollback(
        &self,
        txn_id: TxnId,
        release: impl FnOnce() -> TCResult<()>,
    ) -> TCResult<Option<Arc<D>>> {
        // Callers have stopped operations and released this transaction's streams.
        // Reject an already committed decision rather than reversing visibility.
        let discarded = {
            let mut state = self.write_state();
            state.check_readable(txn_id)?;
            if state.committed.contains_key(&txn_id) {
                return Err(txn_lock::Error::Conflict.into());
            }

            state.pending.remove(&txn_id)
        };

        release()?;
        Ok(discarded)
    }
}

#[cfg(test)]
#[path = "../tests/unit/persistence/tests.rs"]
mod tests;
