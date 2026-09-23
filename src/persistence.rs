//! Shared transactional visibility and lifecycle. Durability belongs to the caller.

use std::{collections::BTreeMap, future::Future, io, sync::RwLock};

use freqfs::DirLock;
use tc_ir::TxnId;
use tc_value::{Value, ValueCollator};

pub(crate) fn background_error(err: impl std::fmt::Display) -> txn_lock::Error {
    txn_lock::Error::Background(err.to_string())
}

/// Native work needed by the shared Collection lifecycle. No transaction policy.
pub(crate) trait PersistentDelta: Clone + Send + Sync + 'static {
    type Native: Clone + Send + Sync;

    fn apply_to(&self, persistent: &Self::Native) -> impl Future<Output = io::Result<()>> + Send;
}

pub(crate) struct State<D: PersistentDelta> {
    pub persistent: D::Native,
    pub committed: BTreeMap<TxnId, Option<D>>,
    pub pending: BTreeMap<TxnId, D>,
    pub finalized: Option<TxnId>,
}

impl<D: PersistentDelta> State<D> {
    pub fn assert_writable(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        if self.finalized.is_some_and(|finalized| txn_id <= finalized) {
            return Err(txn_lock::Error::Outdated);
        }
        if self.committed.contains_key(&txn_id) {
            return Err(txn_lock::Error::Committed);
        }
        Ok(())
    }
}

pub(crate) struct VisibleSnapshot<D: PersistentDelta> {
    pub persistent: D::Native,
    pub deltas: Vec<D>,
}

/// One live state and lifecycle owner, shared by concrete Collection handles.
/// Callers sequence decisions and finish affected operations/streams first.
pub(crate) struct CollectionOwner<D: PersistentDelta> {
    pub state: RwLock<State<D>>,
    pub semaphore: txn_lock::semaphore::Semaphore<
        TxnId,
        b_tree::Collator<ValueCollator>,
        txn_lock::set::Range<Vec<Value>>,
    >,
}

impl<D: PersistentDelta> CollectionOwner<D> {
    pub fn new(persistent: D::Native) -> Self {
        Self {
            state: RwLock::new(State {
                persistent,
                committed: BTreeMap::new(),
                pending: BTreeMap::new(),
                finalized: None,
            }),
            semaphore: txn_lock::semaphore::Semaphore::new(b_tree::Collator::new(
                ValueCollator::default(),
            )),
        }
    }

    pub fn finalized(&self) -> Option<TxnId> {
        self.state.read().expect("state read lock").finalized
    }

    pub fn visible_snapshot(&self, txn_id: TxnId) -> VisibleSnapshot<D> {
        let state = self.state.read().expect("state read lock");
        let mut deltas = state
            .committed
            .range(..=txn_id)
            .filter_map(|(_, delta)| delta.clone())
            .collect::<Vec<_>>();

        if let Some(delta) = state.pending.get(&txn_id).cloned() {
            deltas.push(delta);
        }

        VisibleSnapshot {
            persistent: state.persistent.clone(),
            deltas,
        }
    }

    pub async fn acquire_read_permit(
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

    pub fn commit(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        {
            let mut state = self.state.write().expect("state write lock");
            if state.finalized.is_some_and(|cutoff| txn_id <= cutoff) {
                return Err(txn_lock::Error::Outdated);
            }
            if state.committed.contains_key(&txn_id) {
                return Ok(());
            }
            let delta = state.pending.remove(&txn_id);
            state.committed.insert(txn_id, delta);
        }
        self.semaphore.finalize(&txn_id, false);
        Ok(())
    }

    pub fn rollback(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        {
            let mut state = self.state.write().expect("state write lock");
            if state.finalized.is_some_and(|finalized| txn_id <= finalized) {
                return Err(txn_lock::Error::Outdated);
            } else if state.committed.contains_key(&txn_id) {
                return Err(txn_lock::Error::Conflict);
            } else {
                state.pending.remove(&txn_id);
            }
        }

        self.semaphore.finalize(&txn_id, false);
        Ok(())
    }

    pub async fn finalize(&self, txn_id: TxnId) -> Result<(), txn_lock::Error> {
        let (persistent, committed_to_apply) = {
            let state = self.state.read().expect("state read lock");
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
            delta
                .apply_to(&persistent)
                .await
                .map_err(background_error)?;
        }

        {
            let mut state = self.state.write().expect("state write lock");
            state.committed.retain(|id, _| *id > txn_id);
            state.pending.retain(|id, _| *id > txn_id);
            state.finalized = Some(txn_id);
        }

        // drop_past=true clears semaphore versions up to txn_id, matching finalize frontier pruning.
        self.semaphore.finalize(&txn_id, true);

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
