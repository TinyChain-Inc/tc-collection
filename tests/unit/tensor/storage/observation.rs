//! Task-local observations watch existing delta ownership, never fields on Delta.

use std::cell::RefCell;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

type Action = Box<dyn FnOnce() -> fensor::Result<()> + Send>;

#[derive(Clone, Copy, Default)]
pub(super) struct Counts {
    pub zero_lookups: usize,
    pub payload_reads: usize,
    pub membership_scans: usize,
    pub overlay_starts: usize,
}

#[derive(Default)]
struct Observation {
    target: Option<Arc<AtomicU64>>,
    counts: Counts,
    after_scatter: Option<Action>,
    after_membership: Option<(usize, Action)>,
}

tokio::task_local! {
    static CURRENT: RefCell<Observation>;
}

pub(super) async fn scope<F: Future>(work: F) -> F::Output {
    CURRENT.scope(RefCell::default(), work).await
}

pub(super) fn watch(target: &Arc<AtomicU64>) {
    CURRENT.with(|current| {
        *current.borrow_mut() = Observation {
            target: Some(Arc::clone(target)),
            counts: Counts::default(),
            after_scatter: None,
            after_membership: None,
        };
    });
}

pub(super) fn counts() -> Counts {
    CURRENT.with(|current| current.borrow().counts)
}

pub(super) fn record(target: &Arc<AtomicU64>, update: impl FnOnce(&mut Counts)) {
    let _ = CURRENT.try_with(|current| {
        let mut current = current.borrow_mut();
        if current
            .target
            .as_ref()
            .is_some_and(|watched| Arc::ptr_eq(watched, target))
        {
            update(&mut current.counts);
        }
    });
}

// A one-shot, task-local callback isolates publication at the payload-consumption
// boundary without adding state or mutation policy to production Delta values.
pub(super) fn on_scatter(
    target: &Arc<AtomicU64>,
    action: impl FnOnce() -> fensor::Result<()> + Send + 'static,
) {
    watch(target);
    CURRENT.with(|current| current.borrow_mut().after_scatter = Some(Box::new(action)));
}

pub(super) fn after_scatter(target: &Arc<AtomicU64>) -> fensor::Result<()> {
    let action = CURRENT
        .try_with(|current| {
            let mut current = current.borrow_mut();
            if current
                .target
                .as_ref()
                .is_some_and(|watched| Arc::ptr_eq(watched, target))
            {
                current.after_scatter.take()
            } else {
                None
            }
        })
        .ok()
        .flatten();

    match action {
        Some(action) => action(),
        None => Ok(()),
    }
}

// Observe one completed run after its node guards drop and before the next run.
pub(super) fn on_membership(
    target: &Arc<AtomicU64>,
    completed_run: usize,
    action: impl FnOnce() -> fensor::Result<()> + Send + 'static,
) {
    watch(target);
    CURRENT.with(|current| {
        current.borrow_mut().after_membership = Some((completed_run, Box::new(action)));
    });
}

pub(super) fn after_membership(target: &Arc<AtomicU64>) -> fensor::Result<()> {
    let action = CURRENT
        .try_with(|current| {
            let mut current = current.borrow_mut();
            if current
                .target
                .as_ref()
                .is_some_and(|watched| Arc::ptr_eq(watched, target))
                && current
                    .after_membership
                    .as_ref()
                    .is_some_and(|(run, _)| *run == current.counts.membership_scans)
            {
                current.after_membership.take().map(|(_, action)| action)
            } else {
                None
            }
        })
        .ok()
        .flatten();

    match action {
        Some(action) => action(),
        None => Ok(()),
    }
}
