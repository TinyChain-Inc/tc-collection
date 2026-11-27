use std::io;
use std::sync::Arc;

use tc_error::TCResult;
use tc_ir::{Transact, Transaction, TxnId};

use super::{BTreeBase as BTree, BTreeSchema, contains_key};
use crate::btree::BTreeColumnSchema;
use crate::test::tensor::{Txn, setup};
use crate::{Restore, StorageContext};

impl<Txn: StorageContext> BTree<Txn> {
    // Storage tests choose an explicit key type.
    pub(crate) async fn new(persistent_dir: freqfs::DirLock<Txn::File>) -> std::io::Result<Self> {
        let schema = BTreeSchema::try_from(vec![BTreeColumnSchema {
            name: "key".into(),
            dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
            max_size: None,
        }])
        .unwrap();

        Self::try_with_schema(persistent_dir, schema).await
    }

    pub(in crate::btree) fn finalized(&self) -> TCResult<Option<TxnId>> {
        Ok(self.versions.finalized_for_test())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn immutable_access_does_not_wait_for_transaction_state() {
    let (_root, txn) = setup().await;
    let tree = BTree::<Txn>::try_with_schema(
        txn.subcontext_unique().context().await.unwrap(),
        BTreeSchema::try_from(vec![BTreeColumnSchema {
            name: "id".into(),
            dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
            max_size: None,
        }])
        .unwrap(),
    )
    .await
    .unwrap();

    let alias = tree.clone();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let (worker, result) = {
        let _state = tree.versions.hold_state_for_test();
        let worker = tokio::spawn(async move {
            let _ = format!("{alias:?}");
            assert_eq!(alias.schema().columns()[0].name, "id");
            sender.send(()).unwrap();
        });

        let result = receiver.recv_timeout(std::time::Duration::from_secs(3));
        (worker, result)
    };

    worker.await.unwrap();
    assert!(result.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn point_miss_and_scan_setup_survive_compaction() {
    use std::ops::Bound;

    use futures::{FutureExt, TryStreamExt};

    Box::pin(async {
        let (_, txn) = setup().await;
        let tree = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();

        tree.insert_row(&txn, vec![1u64.into()]).await.unwrap();
        tree.commit(txn.id()).await.unwrap();
        let newer = txn.at(11);
        tree.insert_row(&newer, vec![2u64.into()]).await.unwrap();
        tree.commit(newer.id()).await.unwrap();
        let later = txn.at(12);
        let snapshot = tree.versions.visible_snapshot(later.id()).unwrap();
        let (entered, mut observed) = tokio::sync::oneshot::channel();
        let (resume, paused) = tokio::sync::oneshot::channel();
        let mut pause = Some((entered, paused));
        let mut read = Box::pin(tokio::task::unconstrained(tree.versions.lookup(
            later.id(),
            true,
            |delta| {
                let pause = pause.take();
                async move {
                    if let Some((entered, paused)) = pause {
                        entered.send(()).unwrap();
                        paused.await.unwrap();
                    }

                    delta.lookup(&[1u64.into()]).await
                }
            },
        )));

        assert!(read.as_mut().now_or_never().is_none());
        observed.try_recv().unwrap();
        // The first probe misses in the newest input. Compaction publishes the
        // merged delta at that same ID, so resuming below it would miss key 1.
        tree.finalize(&newer.id()).await.unwrap();

        resume.send(()).unwrap();
        assert_eq!(read.await.unwrap(), Some(true));
        assert!(tree.contains_row(later.id(), &[1u64.into()]).await.unwrap());

        let keys: Vec<_> = BTree::<Txn>::visible_keys(
            tree.canonical.clone(),
            snapshot,
            (Bound::Unbounded, Bound::Unbounded),
            false,
        )
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();

        assert_eq!(keys, vec![vec![1u64.into()], vec![2u64.into()]]);
    })
    .await;
}

#[tokio::test]
async fn restoration_and_reinsertion_preserve_numeric_representation() {
    use tc_value::{NumberType, Value, ValueType};

    use crate::Collection;
    use crate::btree::{BTreeColumnSchema, BTreeSchema};

    let (_, txn) = setup().await;
    let columns = vec![BTreeColumnSchema {
        name: "key".into(),
        dtype: ValueType::Number(NumberType::Number),
        max_size: None,
    }];

    let schema = BTreeSchema::try_from(columns).unwrap();
    let target = BTree::<Txn>::try_with_schema(
        txn.subcontext_unique().context().await.unwrap(),
        schema.clone(),
    )
    .await
    .unwrap();

    let source =
        BTree::<Txn>::try_with_schema(txn.subcontext_unique().context().await.unwrap(), schema)
            .await
            .unwrap();

    target
        .insert_row(&txn, vec![Value::from(0u64)])
        .await
        .unwrap();

    target.commit(txn.id()).await.unwrap();
    target.finalize(&txn.id()).await.unwrap();
    let writer = txn.at(11);
    source
        .insert_row(&writer, vec![Value::from(-0f64)])
        .await
        .unwrap();

    let expected = Collection::BTree(Box::new(crate::btree::BTree::File(source.clone())))
        .hash(writer.id())
        .await
        .unwrap();

    target.restore(&writer, &source).await.unwrap();
    let collection = Collection::BTree(Box::new(crate::btree::BTree::File(target.clone())));
    assert_eq!(collection.hash(writer.id()).await.unwrap(), expected);
    target
        .insert_row(&writer, vec![Value::from(0u64)])
        .await
        .unwrap();

    assert_eq!(collection.hash(writer.id()).await.unwrap(), expected);
    target.commit(writer.id()).await.unwrap();
    target.finalize(&writer.id()).await.unwrap();
    assert_eq!(collection.hash(txn.at(12).id()).await.unwrap(), expected);
    let writer = txn.at(12);
    target
        .delete_row(&writer, vec![Value::from(0u64)])
        .await
        .unwrap();

    target
        .insert_row(&writer, vec![Value::from(0f64)])
        .await
        .unwrap();

    let replaced = collection.hash(writer.id()).await.unwrap();
    assert_ne!(replaced, expected);
    target.commit(writer.id()).await.unwrap();
    target.finalize(&writer.id()).await.unwrap();
    assert_eq!(collection.hash(txn.at(13).id()).await.unwrap(), replaced);
}

#[tokio::test]
async fn equal_norm_keys_preserve_identity_and_hash_through_finalization() {
    use futures::TryStreamExt;
    use number_general::{Complex, Number};
    use tc_ir::Transact;
    use tc_value::{NumberType, Value, ValueType};

    use crate::Collection;
    use crate::btree::{BTreeColumnSchema, BTreeSchema};

    let (_, txn) = setup().await;
    let schema = vec![BTreeColumnSchema {
        name: "key".into(),
        dtype: ValueType::Number(NumberType::Number),
        max_size: None,
    }];

    let tree = BTree::<Txn>::try_with_schema(
        txn.subcontext_unique().context().await.unwrap(),
        BTreeSchema::try_from(schema).unwrap(),
    )
    .await
    .unwrap();

    let collection = Collection::BTree(Box::new(crate::btree::BTree::File(tree.clone())));
    let a = Value::from(Number::from(Complex::from([4f64, 3.])));
    let b = Value::from(Number::from(Complex::from([3f64, 4.])));
    for (at, key) in [(txn.clone(), a.clone()), (txn.at(11), b.clone())] {
        tree.insert_row(&at, vec![key]).await.unwrap();
        tree.commit(at.id()).await.unwrap();
    }

    let reader = txn.at(12);
    let hash = collection.hash(reader.id()).await.unwrap();
    for reverse in [false, true] {
        let keys = tree
            .keys(
                reader.id(),
                (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
                reverse,
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            keys,
            if reverse {
                vec![vec![a.clone()], vec![b.clone()]]
            } else {
                vec![vec![b.clone()], vec![a.clone()]]
            }
        );
    }

    collection.finalize(&txn.at(11).id()).await.unwrap();
    assert_eq!(collection.hash(reader.id()).await.unwrap(), hash);
    tree.delete_row(&reader, vec![a.clone()]).await.unwrap();
    let keys = tree
        .keys(
            reader.id(),
            (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
            false,
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(keys, vec![vec![b.clone()]]);
    assert!(tree.contains_row(reader.id(), &[b]).await.unwrap());
    assert!(!tree.contains_row(reader.id(), &[a]).await.unwrap());
    let nan = Value::from(f64::NAN);
    // Native Number equality normalizes NaNs.
    tree.insert_row(&reader, vec![nan.clone()]).await.unwrap();
    assert!(tree.contains_row(reader.id(), &[nan]).await.unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn interrupted_tree_mutation_is_discarded_by_rollback() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let tree = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();

        let delta = tree.pending_delta_for_txn(&txn).await.unwrap();
        delta.delete(vec![1u64.into()]).await.unwrap();
        let snapshot = tree.versions.visible_snapshot(txn.id()).unwrap();
        let blocked = delta.inserts.write().await;
        let mut insert = Box::pin(tree.insert_row(&txn, vec![1u64.into()]));
        assert!(futures::poll!(&mut insert).is_pending());
        assert!(contains_key(&delta.deletes, &[1u64.into()]).await.unwrap());

        drop(insert);
        drop(blocked);
        drop(snapshot); // Retained results must be released before caller-owned rollback.
        assert!(delta.inserts.try_write().is_ok());
        tree.rollback(&txn.id()).await.unwrap();
        assert!(!tree.contains_row(txn.id(), &[1u64.into()]).await.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn deep_visible_versions_use_normal_worker_stacks() {
    use futures::TryStreamExt;
    use tc_ir::{NetworkTime, TxnId};

    for phase in ["drop", "poll"] {
        if !crate::test::depth::child(
            concat!(
                module_path!(),
                "::deep_visible_versions_use_normal_worker_stacks"
            )
            .strip_prefix("tc_collection::")
            .unwrap(),
            phase,
        )
        .await
        {
            continue;
        }

        tokio::spawn(async move {
            let root = crate::test::TempRoot::new("overlay-depth");
            // Two streams per retained version; admission must not mask stack depth.
            let cache = freqfs::Cache::<crate::PersistentFile>::new(
                64 * 1024 * 1024,
                Some(4096),
                0,
                std::time::Duration::from_secs(3),
            );

            let txn = Txn::new(
                TxnId::from_parts(NetworkTime::from_nanos(1), 10),
                cache.load(root.to_path_buf()).unwrap(),
            );

            let collection = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
                .await
                .unwrap();

            let delta = collection.pending_delta_for_txn(&txn).await.unwrap();
            // Reuse native fixtures to isolate version depth from directory setup.
            collection
                .versions
                .seed_committed_for_test((0..1024).map(|nonce| {
                    (
                        TxnId::from_parts(NetworkTime::from_nanos(1), nonce),
                        delta.clone(),
                    )
                }));

            let read_id = TxnId::from_parts(NetworkTime::from_nanos(1), 4096);
            eprintln!("{phase}: constructing visible stream");
            let mut stream = collection
                .keys(
                    read_id,
                    (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
                    false,
                )
                .await
                .unwrap();

            eprintln!("{phase}: stream constructed");
            if phase == "poll" {
                assert!(stream.try_next().await.unwrap().is_none());
            }

            drop(stream);
            eprintln!("{phase}: stream released");
        })
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn point_lookup_skips_shadowed_corruption() {
    let (_root, txn) = setup().await;
    let dir = txn.subcontext_unique().context().await.unwrap();
    let tree = BTree::<Txn>::new(dir.clone()).await.unwrap();
    let canonical = tree.canonical.clone();
    super::insert_key(&canonical, vec![0u64.into()])
        .await
        .unwrap();

    assert!(tree.contains_row(txn.id(), &[0u64.into()]).await.unwrap());
    assert!(!tree.contains_row(txn.id(), &[9u64.into()]).await.unwrap());
    tree.delete_row(&txn, vec![0u64.into()]).await.unwrap();
    tree.insert_row(&txn, vec![1u64.into()]).await.unwrap();
    tree.commit(txn.id()).await.unwrap();
    let later = txn.at(11);
    tree.insert_row(&later, vec![0u64.into()]).await.unwrap();

    assert!(
        dir.read()
            .await
            .get_dir("base")
            .cloned()
            .unwrap()
            .write()
            .await
            .delete("00000000-0000-0000-0000-000000000000")
            .await
    );

    assert!(tree.contains_row(later.id(), &[0u64.into()]).await.unwrap());
    assert!(tree.contains_row(later.id(), &[1u64.into()]).await.unwrap());
    assert!(
        tree.contains_row(later.id(), &[2u64.into()])
            .await
            .unwrap_err()
            .requires_recovery()
    );

    tree.rollback(&later.id()).await.unwrap();
    assert!(!tree.contains_row(later.id(), &[0u64.into()]).await.unwrap());
    assert!(tree.contains_row(later.id(), &[1u64.into()]).await.unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restoration_self_and_conflicting_sources_are_bounded() {
    use futures::FutureExt;

    Box::pin(async {
        let (_, txn) = setup().await;
        let make = |dir| async move { BTree::<Txn>::new(dir).await.unwrap() };
        let seed = |store: BTree<Txn>, txn: Txn, key: u64| async move {
            store.insert_row(&txn, vec![key.into()]).await.unwrap();
        };

        let has = |store: BTree<Txn>, txn: Txn, key: u64| async move {
            store.contains_row(txn.id(), &[key.into()]).await.unwrap()
        };

        for reverse in [false, true] {
            let left = make(txn.subcontext_unique().context().await.unwrap()).await;
            let right = make(txn.subcontext_unique().context().await.unwrap()).await;
            seed(left.clone(), txn.clone(), 1).await;
            seed(right.clone(), txn.clone(), 2).await;
            let (target, source) = if reverse {
                (&right, &left)
            } else {
                (&left, &right)
            };

            let key = if reverse { 2 } else { 1 };

            let held = target
                .keys(
                    txn.id(),
                    (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
                    false,
                )
                .await
                .unwrap();

            target
                .restore(&txn, &target.clone())
                .now_or_never()
                .expect("self-restore does not wait")
                .unwrap();

            drop(held);
            assert!(has(target.clone(), txn.clone(), key).await);

            let context = source.store.base().clone();
            let context_guard = context.write().await;
            let mut restore = Box::pin(target.restore(&txn, source));
            assert!(
                tokio::task::unconstrained(restore.as_mut())
                    .now_or_never()
                    .is_none()
            );

            assert!(
                source
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .is_err(),
                "source snapshot is admitted before staging"
            );

            let reverse_result = source
                .restore(&txn, target)
                .now_or_never()
                .expect("reciprocal restore rejects without waiting");

            assert!(reverse_result.is_err());
            drop(restore);
            drop(context_guard);
            assert!(
                has(target.clone(), txn.clone(), key).await,
                "cancelled staging preserves pending data"
            );

            drop(
                target
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );

            drop(
                source
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );

            target.rollback(&txn.id()).await.unwrap();
            source.rollback(&txn.id()).await.unwrap();
        }

        let dir = txn.subcontext_unique().context().await.unwrap();
        let target = make(dir.clone()).await;
        let source = make(txn.subcontext_unique().context().await.unwrap()).await;
        seed(target.clone(), txn.at(9), 1).await;
        target.commit(txn.at(9).id()).await.unwrap();
        target.finalize(&txn.at(9).id()).await.unwrap();
        seed(target.clone(), txn.clone(), 3).await;
        seed(source.clone(), txn.clone(), 2).await;
        target.restore(&txn, &source).await.unwrap();
        assert!(has(target.clone(), txn.clone(), 2).await);
        assert!(!has(target.clone(), txn.clone(), 3).await);
        target.rollback(&txn.id()).await.unwrap();
        assert!(has(target.clone(), txn.clone(), 1).await);
        assert!(!has(target.clone(), txn.clone(), 2).await);
        source.commit(txn.id()).await.unwrap();
        let later = txn.at(11);
        target.restore(&later, &source).await.unwrap();
        target.commit(later.id()).await.unwrap();
        assert!(target.restore(&later, &target).await.is_err());
        target.finalize(&later.id()).await.unwrap();
        assert!(target.restore(&later, &target).await.is_err());
        let path = dir.read().await.path().to_owned();
        drop(target);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            256 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let schema = BTreeSchema::try_from(vec![BTreeColumnSchema {
            name: "key".into(),
            dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
            max_size: None,
        }])
        .unwrap();

        let reopened = BTree::<Txn>::load(cache.load(path).unwrap(), schema)
            .await
            .unwrap();

        assert!(has(reopened, txn.at(12), 2).await);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn empty_restore_rejects_late_older_inserts_after_reopen() {
    Box::pin(async {
        let (_root, txn) = setup().await;
        let dir = txn.subcontext_unique().context().await.unwrap();
        let target = BTree::<Txn>::new(dir.clone()).await.unwrap();
        let source = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();
        let restored = txn.at(12);
        target.commit(txn.at(14).id()).await.unwrap();
        target.restore(&restored, &source).await.unwrap();
        target.commit(restored.id()).await.unwrap();
        let path = dir.read().await.path().to_owned();
        let schema = target.schema().clone();
        drop(target);

        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            256 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );
        let target = BTree::<Txn>::load(cache.load(path).unwrap(), schema)
            .await
            .unwrap();
        let error = target
            .insert_row(&txn.at(11), vec![1u64.into()])
            .await
            .unwrap_err();
        assert_eq!(error.code(), tc_error::ErrorKind::Conflict);
        assert_eq!(target.count(restored.id()).await.unwrap(), 0);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restoration_source_failure_preserves_pending_data() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let make = |dir| async move { BTree::<Txn>::new(dir).await.unwrap() };
        let target = make(txn.subcontext_unique().context().await.unwrap()).await;
        let source_dir = txn.subcontext_unique().context().await.unwrap();
        let source = make(source_dir.clone()).await;
        target.insert_row(&txn, vec![3_u64.into()]).await.unwrap();
        let native = source_dir.read().await.get_dir("base").cloned().unwrap();
        let root = native
            .read()
            .await
            .get_file("00000000-0000-0000-0000-000000000000")
            .unwrap()
            .clone();

        *root
            .write::<b_table::Node<tc_value::Value>>()
            .await
            .unwrap() = b_tree::Node::Leaf(vec![Default::default()]);

        let error = target.restore(&txn, &source).await.unwrap_err();
        assert!(error.requires_recovery());
        assert!(
            target
                .contains_row(txn.id(), &[3_u64.into()])
                .await
                .unwrap()
        );

        drop(root.try_write::<b_table::Node<tc_value::Value>>().unwrap());
        drop(
            source
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        drop(
            target
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        target.rollback(&txn.id()).await.unwrap();
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn finalized_reads_revalidate_admission_and_snapshot_capture() {
    let (_root, txn) = setup().await;
    let owner = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
        .await
        .unwrap();

    let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
    let writer = owner
        .semaphore
        .try_write(id(9), txn_lock::set::Range::All)
        .unwrap();

    let mut waiting = Box::pin(owner.read_snapshot(id(10), txn_lock::set::Range::All));
    assert!(futures::poll!(&mut waiting).is_pending());
    assert!(
        owner.versions.state_unlocked_for_test(),
        "admission holds no state lock"
    );

    owner.finalize(&id(10)).await.unwrap();
    drop(writer);
    let error = waiting
        .await
        .err()
        .expect("cutoff advanced during admission");

    assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
    drop(
        owner
            .semaphore
            .try_write(id(10), txn_lock::set::Range::All)
            .unwrap(),
    );

    owner.semaphore.finalize(&id(10), true).unwrap();

    for stale in [id(9), id(10)] {
        let error = owner
            .read_snapshot(stale, txn_lock::set::Range::All)
            .await
            .err()
            .unwrap();

        assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
        assert_eq!(
            owner
                .try_read_snapshot(stale, txn_lock::set::Range::All)
                .err()
                .unwrap()
                .message(),
            txn_lock::Error::Outdated.to_string()
        );

        assert_eq!(
            owner
                .versions
                .visible_snapshot(stale)
                .err()
                .unwrap()
                .message(),
            txn_lock::Error::Outdated.to_string()
        );

        drop(
            owner
                .semaphore
                .try_write(stale, txn_lock::set::Range::All)
                .unwrap(),
        );
        // The probe itself registers a write reservation until its decision.
        owner.semaphore.finalize(&stale, false).unwrap();
    }

    // A retained read state cannot authorize a fresh capture after cutoff advances.
    let permit = owner
        .read_snapshot(id(11), txn_lock::set::Range::All)
        .await
        .unwrap();

    owner.finalize(&id(11)).await.unwrap();
    assert_eq!(
        owner
            .versions
            .visible_snapshot(id(11))
            .err()
            .unwrap()
            .message(),
        txn_lock::Error::Outdated.to_string()
    );

    drop(permit);
    drop(
        owner
            .read_snapshot(id(12), txn_lock::set::Range::All)
            .await
            .unwrap(),
    );

    assert!(owner.versions.visible_snapshot(id(12)).is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn write_admission_rejects_decisions_without_reservations() {
    use futures::FutureExt;

    let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
    for finalized in [false, true] {
        let (_root, txn) = setup().await;
        let owner = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();

        let write = owner
            .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
            .unwrap();

        assert!(
            owner
                .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
                .is_err()
        );

        drop(write);
        drop(
            owner
                .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
                .unwrap(),
        );

        owner.commit(id(10)).await.unwrap();
        if finalized {
            owner.finalize(&id(10)).await.unwrap();
        }

        let error = owner
            .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
            .unwrap_err();

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
                .read_snapshot(id(11), txn_lock::set::Range::All)
                .now_or_never()
                .expect("rejected write must not block later reads")
                .unwrap(),
        );

        assert!(owner.versions.state_unlocked_for_test());
        assert!(!owner.versions.has_pending_for_test());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn guarded_stream_termination_releases_source_and_permit() {
    use std::task::Poll;

    use futures::StreamExt;

    for case in ["native", "eof", "drop", "pending"] {
        let (_root, txn) = setup().await;
        let owner = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();

        let id = TxnId::from_parts("1".parse().unwrap(), 0);
        let permit = owner
            .read_snapshot(id, txn_lock::set::Range::All)
            .await
            .unwrap();

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
        assert!(
            owner
                .semaphore
                .try_write(id, txn_lock::set::Range::All)
                .is_err()
        );

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
            assert!(
                owner
                    .semaphore
                    .try_write(id, txn_lock::set::Range::All)
                    .is_ok(),
                "{case}"
            );
        }

        drop(stream);
        assert!(released.upgrade().is_none(), "{case}");
        assert!(
            owner
                .semaphore
                .try_write(id, txn_lock::set::Range::All)
                .is_ok(),
            "{case}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn synchronous_read_admission_preserves_conflicts_and_release() {
    let (_root, txn) = setup().await;
    let owner = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
        .await
        .unwrap();

    let id = |n| TxnId::from_parts("1".parse().unwrap(), n);
    let writer = owner
        .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
        .unwrap();

    assert_eq!(
        owner
            .try_read_snapshot(id(10), txn_lock::set::Range::All)
            .err()
            .unwrap()
            .message(),
        txn_lock::Error::WouldBlock.to_string()
    );

    assert!(owner.versions.state_unlocked_for_test());
    drop(writer);
    let read = owner
        .try_read_snapshot(id(10), txn_lock::set::Range::All)
        .unwrap();

    assert!(
        owner
            .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
            .is_err()
    );

    drop(read);
    drop(
        owner
            .try_acquire_write_permit(id(10), txn_lock::set::Range::All)
            .unwrap(),
    );

    assert!(
        owner
            .try_read_snapshot(id(11), txn_lock::set::Range::All)
            .is_err()
    );

    owner.rollback(&id(10)).await.unwrap();
    drop(
        owner
            .try_read_snapshot(id(11), txn_lock::set::Range::All)
            .unwrap(),
    );

    owner.finalize(&id(11)).await.unwrap();
    assert!(
        owner
            .try_read_snapshot(id(11), txn_lock::set::Range::All)
            .is_err()
    );

    drop(
        owner
            .try_read_snapshot(id(12), txn_lock::set::Range::All)
            .unwrap(),
    );
}
