//! Shared transactional visibility and lifecycle. Durability belongs to the caller.

use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use collate::{Collate, OverlapsRange};
use freqfs::DirLock;
use tc_error::TCResult;
use tc_ir::TxnId;

/// Native work needed by the shared Collection lifecycle. No transaction policy.
pub(crate) trait PersistentDelta: Send + Sync + 'static {
    type Native: Clone + Send + Sync;

    fn apply_to(&self, persistent: &Self::Native) -> impl Future<Output = io::Result<()>> + Send;
}

pub(crate) struct State<D: PersistentDelta> {
    pub persistent: D::Native,
    // None records a committed decision without allocating an empty delta. Its
    // presence still prevents late writes and makes repeated commit idempotent.
    pub committed: BTreeMap<TxnId, Option<Arc<D>>>,
    pub pending: BTreeMap<TxnId, Arc<D>>,
    pub finalized: Option<TxnId>,
}

impl<D: PersistentDelta> State<D> {
    pub fn assert_readable(&self, txn_id: TxnId) -> TCResult<()> {
        if self.finalized.is_some_and(|cutoff| txn_id <= cutoff) {
            return Err(txn_lock::Error::Outdated.into());
        }

        Ok(())
    }

    pub fn assert_writable(&self, txn_id: TxnId) -> TCResult<()> {
        if self.finalized.is_some_and(|finalized| txn_id <= finalized) {
            return Err(txn_lock::Error::Outdated.into());
        }

        if self.committed.contains_key(&txn_id) {
            return Err(txn_lock::Error::Committed.into());
        }

        Ok(())
    }
}

pub(crate) struct VisibleSnapshot<D: PersistentDelta> {
    pub persistent: D::Native,
    pub deltas: Vec<(TxnId, Arc<D>)>,
    pub visibility: Visibility<D>,
}

/// Snapshots consult the same frontier that authorizes caller workspace cleanup.
pub(crate) struct Visibility<D: PersistentDelta>(pub(crate) Arc<RwLock<State<D>>>);

impl<D: PersistentDelta> Clone for Visibility<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: PersistentDelta> Visibility<D> {
    pub fn retired(&self, id: TxnId) -> bool {
        self.0
            .read()
            .expect("collection state poisoned")
            .finalized
            .is_some_and(|cutoff| id <= cutoff)
    }
}

/// Resolve one key without retaining a cloned history or repeating a layer probe.
pub(crate) struct PointRead<D: PersistentDelta> {
    visibility: Visibility<D>,
    txn_id: TxnId,
    pending: bool,
    before: Bound<TxnId>,
}

impl<D: PersistentDelta> PointRead<D> {
    /// Resolve a baseline after the caller has excluded its pending version.
    pub fn committed(mut self) -> Self {
        self.pending = false;
        self
    }

    pub async fn resolve<R, F, Fut>(&mut self, mut lookup: F) -> TCResult<Option<R>>
    where
        F: FnMut(Arc<D>) -> Fut,
        Fut: Future<Output = io::Result<Option<R>>>,
    {
        loop {
            let next = {
                let state = self.visibility.0.read().expect("collection state poisoned");
                state.assert_readable(self.txn_id)?;
                let pending = if self.pending {
                    self.pending = false;
                    state
                        .pending
                        .get(&self.txn_id)
                        .map(|delta| (self.txn_id, delta.clone()))
                } else {
                    None
                };

                pending.or_else(|| {
                    state
                        .committed
                        .range((Unbounded, self.before))
                        .rev()
                        .find_map(|(id, delta)| delta.as_ref().map(|delta| (*id, delta.clone())))
                })
            };

            let Some((id, delta)) = next else {
                return Ok(None);
            };

            self.before = Excluded(id);
            tokio::task::coop::consume_budget().await;
            if self.visibility.retired(id) {
                continue;
            }

            let result = lookup(delta).await;
            if self.visibility.retired(id) {
                continue;
            }

            if let Some(value) = result? {
                return Ok(Some(value));
            }
        }
    }

    pub fn canonical(&self) -> TCResult<D::Native> {
        let state = self.visibility.0.read().expect("collection state poisoned");
        state.assert_readable(self.txn_id)?;
        Ok(state.persistent.clone())
    }
}

pub(crate) type ValueOrder = b_tree::IndexOrder<tc_value::ValueCollator>;

pub(crate) fn value_order() -> ValueOrder {
    b_tree::IndexOrder::new(tc_value::ValueCollator::default())
}

pub(crate) type KeyOwner<D> = CollectionOwner<
    D,
    b_tree::RangeCollator<tc_value::ValueCollator>,
    txn_lock::set::Range<Vec<tc_value::Value>>,
>;

/// One live state and lifecycle owner, shared by concrete Collection handles.
/// Callers sequence decisions and finish affected operations/streams first.
pub(crate) struct CollectionOwner<D: PersistentDelta, C, R> {
    pub state: Arc<RwLock<State<D>>>,

    pub semaphore: txn_lock::semaphore::Semaphore<TxnId, C, R>,
}

impl<D: PersistentDelta, C, R> CollectionOwner<D, C, R>
where
    C: Collate + Clone + Send + Sync,
    R: OverlapsRange<R, C> + std::fmt::Debug + Send + Sync,
{
    pub fn new(persistent: D::Native, collator: C) -> Self {
        Self {
            state: Arc::new(RwLock::new(State {
                persistent,
                committed: BTreeMap::new(),
                pending: BTreeMap::new(),
                finalized: None,
            })),
            semaphore: txn_lock::semaphore::Semaphore::new(collator),
        }
    }

    pub fn read_state(&self) -> RwLockReadGuard<'_, State<D>> {
        self.state.read().expect("collection state poisoned")
    }

    pub fn write_state(&self) -> RwLockWriteGuard<'_, State<D>> {
        self.state.write().expect("collection state poisoned")
    }

    pub fn visible_snapshot(&self, txn_id: TxnId) -> TCResult<VisibleSnapshot<D>> {
        let state = self.read_state();
        state.assert_readable(txn_id)?;
        let mut snapshot = self.snapshot(&state, txn_id);

        // Other transactions' pending deltas are never visible. Appending our own
        // last gives scan overlays the same precedence as newest-first point reads.
        if let Some(delta) = state.pending.get(&txn_id).cloned() {
            snapshot.deltas.push((txn_id, delta));
        }

        Ok(snapshot)
    }

    pub fn committed_snapshot(&self, txn_id: TxnId) -> TCResult<VisibleSnapshot<D>> {
        let state = self.read_state();
        state.assert_readable(txn_id)?;
        Ok(self.snapshot(&state, txn_id))
    }

    fn snapshot(&self, state: &State<D>, txn_id: TxnId) -> VisibleSnapshot<D> {
        VisibleSnapshot {
            persistent: state.persistent.clone(),
            deltas: state
                .committed
                .range(..=txn_id)
                .filter_map(|(id, delta)| delta.as_ref().map(|delta| (*id, delta.clone())))
                .collect(),
            visibility: Visibility(self.state.clone()),
        }
    }

    pub fn point_read(&self, txn_id: TxnId) -> TCResult<PointRead<D>> {
        self.read_state().assert_readable(txn_id)?;
        Ok(PointRead {
            visibility: Visibility(self.state.clone()),
            txn_id,
            pending: true,
            before: Included(txn_id),
        })
    }

    /// Prepare outside native I/O, then recheck decisions before publishing a delta.
    pub async fn pending<F: Future<Output = TCResult<Arc<D>>>>(
        &self,
        txn_id: TxnId,
        create: impl FnOnce(&D::Native) -> F,
    ) -> TCResult<Arc<D>> {
        let construction = {
            let state = self.read_state();
            state.assert_writable(txn_id)?;
            if let Some(delta) = state.pending.get(&txn_id) {
                return Ok(delta.clone());
            }

            create(&state.persistent)
        };

        let delta = construction.await?;
        // Concurrent disjoint writes may have prepared another workspace. Keep
        // the one already published, and never publish after commit/finalization.
        let mut state = self.write_state();
        state.assert_writable(txn_id)?;
        Ok(state.pending.entry(txn_id).or_insert(delta).clone())
    }

    pub async fn acquire_read_permit(
        &self,
        txn_id: TxnId,
        range: R,
    ) -> TCResult<txn_lock::semaphore::PermitRead<R>> {
        // Use blocking semaphore::read (not try_read) to preserve canonical txn semantics:
        // later overlapping reads wait for earlier pending writes to commit/rollback/finalize.
        self.read_state().assert_readable(txn_id)?;
        let permit = self.semaphore.read(txn_id, range).await?;
        self.read_state().assert_readable(txn_id)?;
        Ok(permit)
    }

    /// Fail without waiting when another owner or an aliased source is already reserved.
    pub fn try_acquire_read_permit(
        &self,
        txn_id: TxnId,
        range: R,
    ) -> TCResult<txn_lock::semaphore::PermitRead<R>> {
        self.read_state().assert_readable(txn_id)?;
        let permit = self.semaphore.try_read(txn_id, range)?;
        self.read_state().assert_readable(txn_id)?;
        Ok(permit)
    }

    /// Reject decided transactions before registering a historical write reservation.
    /// Lifecycle callers still seal mutations before making a decision.
    pub fn try_acquire_write_permit(
        &self,
        txn_id: TxnId,
        range: R,
    ) -> TCResult<txn_lock::semaphore::PermitWrite<R>> {
        self.read_state().assert_writable(txn_id)?;
        Ok(self.semaphore.try_write(txn_id, range)?)
    }

    pub fn commit(&self, txn_id: TxnId) -> TCResult<()> {
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

        // Release reservations only after the decision is visible. No payload is
        // applied or synchronized here; durability belongs to the enclosing owner.
        self.semaphore.finalize(&txn_id, false)?;
        Ok(())
    }

    pub fn rollback(&self, txn_id: TxnId) -> TCResult<()> {
        // Callers have stopped operations and released this transaction's streams.
        // Reject an already committed decision rather than reversing visibility.
        {
            let mut state = self.write_state();
            if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
                return Err(txn_lock::Error::Outdated.into());
            } else if state.committed.contains_key(&txn_id) {
                return Err(txn_lock::Error::Conflict.into());
            } else {
                state.pending.remove(&txn_id);
            }
        }

        self.semaphore.finalize(&txn_id, false)?;
        Ok(())
    }

    pub async fn finalize(&self, txn_id: TxnId) -> TCResult<()> {
        // Callers serialize lifecycle work and quiesce operations through cutoff.
        // Capture handles briefly, then apply in transaction order without holding
        // bookkeeping across native I/O or acquiring another range reservation.
        let (persistent, committed_to_apply) = {
            let state = self.read_state();
            if state.finalized.is_some_and(|cutoff| txn_id <= cutoff) {
                return Ok(());
            }

            let committed_to_apply = state
                .committed
                .range(..=txn_id)
                .filter_map(|(_, delta)| delta.clone())
                .collect::<Vec<_>>();

            (state.persistent.clone(), committed_to_apply)
        };

        for delta in &committed_to_apply {
            delta.apply_to(&persistent).await?;
        }

        // Advance only after every application succeeds. A failure above may have
        // changed canonical payloads: retaining the old frontier is not rollback.
        // Propagate the error for caller-coordinated recovery. On success, prune
        // both committed history and abandoned pending work through the cutoff.
        {
            let mut state = self.write_state();
            state.committed.retain(|id, _| *id > txn_id);
            state.pending.retain(|id, _| *id > txn_id);
            state.finalized = Some(txn_id);
        }

        // drop_past=true clears semaphore versions up to txn_id, matching finalize frontier pruning.
        self.semaphore.finalize(&txn_id, true)?;

        Ok(())
    }
}

pub(crate) async fn create_delta_dirs<F: crate::CollectionFile>(
    dir: &DirLock<F>,
) -> io::Result<(DirLock<F>, DirLock<F>)> {
    let mut dir = dir.write().await;
    Ok((
        dir.get_or_create_dir("inserts".into())?,
        dir.get_or_create_dir("deletes".into())?,
    ))
}

#[cfg(test)]
#[path = "../tests/unit/persistence/tests.rs"]
mod tests;
