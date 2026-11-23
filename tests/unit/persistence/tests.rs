use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::*;

#[tokio::test]
async fn poisoned_state_fails_closed() {
    let owner: CollectionOwner<FailingDelta, collate::Collator<u64>, std::ops::Range<u64>> =
        CollectionOwner::new((), collate::Collator::default());

    let id = TxnId::from_parts("1".parse().unwrap(), 0);
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _state = owner.state.write().unwrap();
        panic!("injected state failure");
    }));

    assert!(poisoned.is_err());
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use futures::FutureExt;

    assert!(catch_unwind(AssertUnwindSafe(|| owner.read_state())).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| owner.visible_snapshot(id))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| owner.try_acquire_read_permit(id, 0..1))).is_err());
    assert!(
        AssertUnwindSafe(owner.acquire_read_permit(id, 0..1))
            .catch_unwind()
            .await
            .is_err()
    );

    assert!(catch_unwind(AssertUnwindSafe(|| owner.commit(id))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| owner.rollback(id))).is_err());
    assert!(
        AssertUnwindSafe(owner.finalize(id))
            .catch_unwind()
            .await
            .is_err()
    );

    assert!(owner.state.is_poisoned());
}

#[derive(Clone)]
struct FailingDelta(io::ErrorKind);

impl PersistentDelta for FailingDelta {
    type Native = ();

    async fn apply_to(&self, _: &()) -> io::Result<()> {
        Err(io::Error::new(self.0, "native materialization failed"))
    }
}

#[tokio::test]
async fn finalized_reads_revalidate_admission_and_snapshot_capture() {
    let owner: CollectionOwner<FailingDelta, collate::Collator<u64>, std::ops::Range<u64>> =
        CollectionOwner::new((), collate::Collator::default());

    let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
    let writer = owner.semaphore.try_write(id(9), 0..1).unwrap();
    let mut waiting = Box::pin(owner.acquire_read_permit(id(10), 0..1));
    assert!(futures::poll!(&mut waiting).is_pending());
    assert!(
        owner.state.try_write().is_ok(),
        "admission holds no state lock"
    );

    owner.finalize(id(10)).await.unwrap();
    drop(writer);
    let error = waiting.await.expect_err("cutoff advanced during admission");

    assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
    drop(owner.semaphore.try_write(id(10), 0..1).unwrap());
    owner.semaphore.finalize(&id(10), true).unwrap();

    for stale in [id(9), id(10)] {
        let error = owner.acquire_read_permit(stale, 0..1).await.err().unwrap();
        assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
        assert_eq!(
            owner
                .try_acquire_read_permit(stale, 0..1)
                .unwrap_err()
                .message(),
            txn_lock::Error::Outdated.to_string()
        );

        assert_eq!(
            owner.visible_snapshot(stale).err().unwrap().message(),
            txn_lock::Error::Outdated.to_string()
        );

        drop(owner.semaphore.try_write(stale, 0..1).unwrap());
        // The probe itself registers a write reservation until its decision.
        owner.semaphore.finalize(&stale, false).unwrap();
    }

    // Admission alone cannot certify a snapshot captured after the cutoff moves.
    let permit = owner.acquire_read_permit(id(11), 0..1).await.unwrap();
    owner.finalize(id(11)).await.unwrap();
    assert_eq!(
        owner.visible_snapshot(id(11)).err().unwrap().message(),
        txn_lock::Error::Outdated.to_string()
    );

    drop(permit);
    drop(owner.acquire_read_permit(id(12), 0..1).await.unwrap());
    assert!(owner.visible_snapshot(id(12)).is_ok());
}

#[tokio::test]
async fn write_admission_rejects_decisions_without_reservations() {
    use futures::FutureExt;

    let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
    for finalized in [false, true] {
        let owner: CollectionOwner<FailingDelta, collate::Collator<u64>, std::ops::Range<u64>> =
            CollectionOwner::new((), collate::Collator::default());

        let write = owner.try_acquire_write_permit(id(10), 0..1).unwrap();
        assert!(owner.try_acquire_write_permit(id(10), 0..1).is_err());
        drop(write);
        drop(owner.try_acquire_write_permit(id(10), 0..1).unwrap());
        owner.commit(id(10)).unwrap();
        if finalized {
            owner.finalize(id(10)).await.unwrap();
        }

        let error = owner.try_acquire_write_permit(id(10), 0..1).unwrap_err();
        assert_eq!(
            error.message(),
            if finalized {
                txn_lock::Error::Outdated.to_string()
            } else {
                txn_lock::Error::Committed.to_string()
            }
        );

        drop(
            owner
                .acquire_read_permit(id(11), 0..1)
                .now_or_never()
                .expect("rejected write must not block later reads")
                .unwrap(),
        );

        assert!(owner.state.try_write().is_ok());
        assert!(owner.read_state().pending.is_empty());
    }
}

#[tokio::test]
async fn materialization_preserves_native_error_classification_and_pending_decision() {
    use tc_error::ErrorKind;

    for (kind, expected) in [
        (io::ErrorKind::StorageFull, ErrorKind::Unavailable),
        (io::ErrorKind::PermissionDenied, ErrorKind::BadGateway),
        (io::ErrorKind::InvalidData, ErrorKind::Internal),
    ] {
        let owner: CollectionOwner<_, collate::Collator<u64>, std::ops::Range<u64>> =
            CollectionOwner::new((), collate::Collator::default());

        let id = TxnId::from_parts("1".parse().unwrap(), 0);
        owner
            .state
            .write()
            .unwrap()
            .pending
            .insert(id, Arc::new(FailingDelta(kind)));

        owner.commit(id).unwrap();
        let error = owner.finalize(id).await.unwrap_err();
        assert_eq!(error.code(), expected);
        let state = owner.state.read().unwrap();
        assert!(state.committed.contains_key(&id));
        assert!(state.finalized.is_none());
    }
}

#[derive(Clone)]
struct PartialDelta {
    stop: bool,
    cancel: bool,
}

impl PersistentDelta for PartialDelta {
    type Native = Arc<std::sync::atomic::AtomicUsize>;

    async fn apply_to(&self, native: &Self::Native) -> io::Result<()> {
        native.fetch_add(1, Ordering::SeqCst);
        if self.stop {
            if self.cancel {
                futures::future::pending::<()>().await;
            }

            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "after canonical mutation",
            ))
        } else {
            Ok(())
        }
    }
}

#[tokio::test]
async fn partial_finalization_retains_recovery_evidence() {
    for cancel in [false, true] {
        let native = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let owner: CollectionOwner<_, collate::Collator<u64>, std::ops::Range<u64>> =
            CollectionOwner::new(native.clone(), collate::Collator::default());

        let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
        for n in [1, 2] {
            owner.write_state().pending.insert(
                id(n),
                Arc::new(PartialDelta {
                    stop: n == 2,
                    cancel,
                }),
            );

            owner.commit(id(n)).unwrap();
        }

        if cancel {
            let mut finalize = Box::pin(owner.finalize(id(2)));
            assert!(futures::poll!(&mut finalize).is_pending());
            drop(finalize);
        } else {
            let error = owner.finalize(id(2)).await.unwrap_err();
            assert_eq!(error.code(), tc_error::ErrorKind::Unavailable);
        }

        assert_eq!(native.load(Ordering::SeqCst), 2);
        // The caller must discard this owner after an interrupted canonical write.
        // No retry occurs; the unchanged frontier retains recovery evidence.
        let state = owner.state.read().unwrap();
        assert_eq!(state.committed.len(), 2);
        assert!(state.finalized.is_none());
    }
}

#[tokio::test]
async fn guarded_stream_termination_releases_source_and_permit() {
    use std::task::Poll;

    use futures::StreamExt;

    for case in ["native", "eof", "drop", "pending"] {
        let owner: CollectionOwner<FailingDelta, collate::Collator<u64>, std::ops::Range<u64>> =
            CollectionOwner::new((), collate::Collator::default());

        let id = TxnId::from_parts("1".parse().unwrap(), 0);
        let permit = owner.acquire_read_permit(id, 0..1).await.unwrap();
        let source = Arc::new(());
        let released = Arc::downgrade(&source);
        let input = futures::stream::poll_fn(move |_| {
            let _retain = &source;
            match case {
                "pending" => Poll::Pending,
                "native" => Poll::Ready(Some(Err(io::Error::other("native")))),
                "eof" => Poll::Ready(None),
                _ => Poll::Ready(Some(Ok(1))),
            }
        })
        .boxed();

        let mut stream = crate::stream::GuardedStream::new(input, permit);
        assert!(released.upgrade().is_some());
        assert!(owner.semaphore.try_write(id, 0..1).is_err());

        if case == "pending" {
            assert!(futures::poll!(stream.next()).is_pending());
            assert!(released.upgrade().is_some());
        }

        if !matches!(case, "drop" | "pending") {
            let result = futures::poll!(stream.next());
            if case == "eof" {
                assert!(matches!(result, Poll::Ready(None)));
            } else {
                let Poll::Ready(Some(Err(error))) = result else {
                    panic!("{case}: failure must terminate immediately");
                };

                assert_eq!(error.kind(), io::ErrorKind::Other, "{case}");
            }

            assert!(released.upgrade().is_none(), "{case}");
            assert!(stream.next().await.is_none(), "{case}");
            assert!(owner.semaphore.try_write(id, 0..1).is_ok(), "{case}");
        }

        drop(stream);
        assert!(released.upgrade().is_none(), "{case}");
        assert!(owner.semaphore.try_write(id, 0..1).is_ok(), "{case}");
    }
}

#[tokio::test]
async fn pending_creation_rechecks_decisions_and_releases_cancelled_work() {
    use futures::FutureExt;

    let owner: CollectionOwner<FailingDelta, collate::Collator<u64>, std::ops::Range<u64>> =
        CollectionOwner::new((), collate::Collator::default());

    let id = TxnId::from_parts("1".parse().unwrap(), 0);
    {
        let mut pending = Box::pin(owner.pending(id, |_| async {
            assert!(
                owner.state.try_write().is_ok(),
                "construction runs outside state lock"
            );

            std::future::pending::<()>().await;
            Ok(Arc::new(FailingDelta(io::ErrorKind::Other)))
        }));

        assert!(futures::poll!(&mut pending).is_pending());
    }

    assert!(owner.read_state().pending.is_empty());
    assert!(
        owner
            .pending(id, |_| async {
                Err(io::Error::other("construction failed").into())
            })
            .await
            .is_err()
    );

    assert!(owner.read_state().pending.is_empty());

    let barrier = tokio::sync::Barrier::new(2);
    let (left, right) = tokio::join!(
        owner.pending(id, |_| async {
            barrier.wait().await;
            Ok(Arc::new(FailingDelta(io::ErrorKind::Other)))
        }),
        owner.pending(id, |_| async {
            barrier.wait().await;
            Ok(Arc::new(FailingDelta(io::ErrorKind::InvalidData)))
        })
    );

    assert_eq!(left.unwrap().0, right.unwrap().0);
    owner
        .pending(id, |_| async { panic!("existing delta must be reused") })
        .await
        .unwrap();

    let later = TxnId::from_parts("1".parse().unwrap(), 1);
    let (send, receive) = tokio::sync::oneshot::channel();
    let mut pending = Box::pin(owner.pending(later, |_| async {
        receive.await.unwrap();
        Ok(Arc::new(FailingDelta(io::ErrorKind::Other)))
    }));

    assert!(pending.as_mut().now_or_never().is_none());
    owner.commit(later).unwrap();
    send.send(()).unwrap();
    assert!(pending.await.is_err());
    assert!(!owner.read_state().pending.contains_key(&later));
}

#[tokio::test]
async fn synchronous_read_admission_preserves_conflicts_and_release() {
    let owner: CollectionOwner<FailingDelta, collate::Collator<u64>, std::ops::Range<u64>> =
        CollectionOwner::new((), collate::Collator::default());

    let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
    let writer = owner.try_acquire_write_permit(id(10), 0..1).unwrap();
    assert_eq!(
        owner
            .try_acquire_read_permit(id(10), 0..1)
            .unwrap_err()
            .message(),
        txn_lock::Error::WouldBlock.to_string()
    );

    assert!(owner.state.try_write().is_ok());
    drop(writer);
    let read = owner.try_acquire_read_permit(id(10), 0..1).unwrap();
    assert!(owner.try_acquire_write_permit(id(10), 0..1).is_err());
    drop(read);
    drop(owner.try_acquire_write_permit(id(10), 0..1).unwrap());
    assert!(owner.try_acquire_read_permit(id(11), 0..1).is_err());
    owner.rollback(id(10)).unwrap();
    drop(owner.try_acquire_read_permit(id(11), 0..1).unwrap());
    owner.finalize(id(11)).await.unwrap();
    assert!(owner.try_acquire_read_permit(id(11), 0..1).is_err());
    drop(owner.try_acquire_read_permit(id(12), 0..1).unwrap());
}
