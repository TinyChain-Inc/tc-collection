use super::{BTree, contains_key};
use crate::StorageContext;
use crate::test::tensor::{Txn, setup};
use tc_ir::Transaction;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn selected_delta_and_scan_setup_survive_finalized_workspace_cleanup() {
    use std::ops::Bound;

    use futures::{FutureExt, TryStreamExt};

    Box::pin(async {
        let (_, txn) = setup().await;
        let tree = BTree::<Txn>::new(txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();

        let writes = txn.subcontext("writes");
        tree.insert_row(&writes, vec![1u64.into()]).await.unwrap();
        tree.commit(txn.id()).await.unwrap();
        let later = txn.at(11);
        let snapshot = tree.owner.visible_snapshot(later.id()).unwrap();
        let mut point = tree.owner.point_read(later.id()).unwrap();
        let (entered, mut observed) = tokio::sync::oneshot::channel();
        let (resume, paused) = tokio::sync::oneshot::channel();
        let mut pause = Some((entered, paused));
        let mut read = Box::pin(tokio::task::unconstrained(point.resolve(|delta| {
            let pause = pause.take();
            async move {
                if let Some((entered, paused)) = pause {
                    entered.send(()).unwrap();
                    paused.await.unwrap();
                }

                delta.lookup(&[1u64.into()]).await
            }
        })));

        assert!(read.as_mut().now_or_never().is_none());
        observed.try_recv().unwrap();
        tree.finalize(txn.id()).await.unwrap();
        writes
            .context()
            .await
            .unwrap()
            .write()
            .await
            .truncate_and_sync()
            .await
            .unwrap();

        resume.send(()).unwrap();
        assert!(read.await.unwrap().is_none());
        assert!(
            contains_key(&point.canonical().unwrap(), &[1u64.into()])
                .await
                .unwrap()
        );

        let keys: Vec<_> =
            BTree::<Txn>::visible_keys(snapshot, (Bound::Unbounded, Bound::Unbounded), false)
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();

        assert_eq!(keys, vec![vec![1u64.into()]]);
    })
    .await;
}

#[tokio::test]
async fn restoration_and_reinsertion_preserve_numeric_representation() {
    use crate::Collection;
    use crate::btree::{BTreeColumnSchema, BTreeSchema};
    use crate::collection::BTreeView;
    use tc_value::{NumberType, Value, ValueType};

    let (_, txn) = setup().await;
    let columns = vec![BTreeColumnSchema {
        name: "key".into(),
        dtype: ValueType::Number(NumberType::Number),
        max_size: None,
    }];

    let schema = BTreeSchema::try_from(columns.as_slice()).unwrap();
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
    target.finalize(txn.id()).await.unwrap();
    let writer = txn.at(11);
    source
        .insert_row(&writer, vec![Value::from(-0f64)])
        .await
        .unwrap();

    let expected = Collection::BTree(Box::new(
        BTreeView::new(columns.clone(), source.clone()).unwrap(),
    ))
    .hash(writer.id())
    .await
    .unwrap();

    target.restore_from(&writer, &source).await.unwrap();
    let collection = Collection::BTree(Box::new(BTreeView::new(columns, target.clone()).unwrap()));
    assert_eq!(collection.hash(writer.id()).await.unwrap(), expected);
    target
        .insert_row(&writer, vec![Value::from(0u64)])
        .await
        .unwrap();

    assert_eq!(collection.hash(writer.id()).await.unwrap(), expected);
    target.commit(writer.id()).await.unwrap();
    target.finalize(writer.id()).await.unwrap();
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
    target.finalize(writer.id()).await.unwrap();
    assert_eq!(collection.hash(txn.at(13).id()).await.unwrap(), replaced);
}

#[tokio::test]
async fn equal_norm_keys_preserve_identity_and_hash_through_finalization() {
    use futures::TryStreamExt;
    use number_general::{Complex, Number};

    use crate::Collection;
    use crate::btree::{BTreeColumnSchema, BTreeSchema};
    use crate::collection::BTreeView;
    use tc_ir::Transact;
    use tc_value::{NumberType, Value, ValueType};

    let (_, txn) = setup().await;
    let schema = vec![BTreeColumnSchema {
        name: "key".into(),
        dtype: ValueType::Number(NumberType::Number),
        max_size: None,
    }];

    let tree = BTree::<Txn>::try_with_schema(
        txn.subcontext_unique().context().await.unwrap(),
        BTreeSchema::try_from(schema.as_slice()).unwrap(),
    )
    .await
    .unwrap();

    let collection = Collection::BTree(Box::new(BTreeView::new(schema, tree.clone()).unwrap()));
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
        let snapshot = tree.owner.visible_snapshot(txn.id()).unwrap();
        let blocked = delta.inserts.write().await;
        let mut insert = Box::pin(tree.insert_row(&txn, vec![1u64.into()]));
        assert!(futures::poll!(&mut insert).is_pending());
        assert!(contains_key(&delta.deletes, &[1u64.into()]).await.unwrap());

        drop(insert);
        drop(blocked);
        drop(snapshot); // Retained results must be released before caller-owned rollback.
        assert!(delta.inserts.try_write().is_ok());
        tree.rollback(txn.id()).unwrap();
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
            {
                let mut state = collection.owner.write_state();
                state.pending.clear();
                // Reuse native fixtures to measure version depth rather than directory setup.
                for nonce in 0..1024 {
                    state.committed.insert(
                        TxnId::from_parts(NetworkTime::from_nanos(1), nonce),
                        Some(delta.clone()),
                    );
                }
            }

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
    let canonical = tree.owner.read_state().persistent.clone();
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
        dir.write()
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

    tree.rollback(later.id()).unwrap();
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
                .restore_from(&txn, &target.clone())
                .now_or_never()
                .expect("self-restore does not wait")
                .unwrap();

            drop(held);
            assert!(has(target.clone(), txn.clone(), key).await);

            let context = txn.context().await.unwrap();
            let context_guard = context.write().await;
            let mut restore = Box::pin(target.restore_from(&txn, source));
            assert!(
                tokio::task::unconstrained(restore.as_mut())
                    .now_or_never()
                    .is_none()
            );

            assert!(
                source
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .is_err(),
                "source snapshot is admitted before staging"
            );

            let reverse_result = source
                .restore_from(&txn, target)
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
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );

            drop(
                source
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );

            target.rollback(txn.id()).unwrap();
            source.rollback(txn.id()).unwrap();
        }

        let dir = txn.subcontext_unique().context().await.unwrap();
        let target = make(dir.clone()).await;
        let source = make(txn.subcontext_unique().context().await.unwrap()).await;
        seed(target.clone(), txn.at(9), 1).await;
        target.commit(txn.at(9).id()).await.unwrap();
        target.finalize(txn.at(9).id()).await.unwrap();
        seed(target.clone(), txn.clone(), 3).await;
        seed(source.clone(), txn.clone(), 2).await;
        target.restore_from(&txn, &source).await.unwrap();
        assert!(has(target.clone(), txn.clone(), 2).await);
        assert!(!has(target.clone(), txn.clone(), 3).await);
        target.rollback(txn.id()).unwrap();
        assert!(has(target.clone(), txn.clone(), 1).await);
        assert!(!has(target.clone(), txn.clone(), 2).await);
        source.commit(txn.id()).await.unwrap();
        let later = txn.at(11);
        target.restore_from(&later, &source).await.unwrap();
        target.commit(later.id()).await.unwrap();
        assert!(target.restore_from(&later, &target).await.is_err());
        target.finalize(later.id()).await.unwrap();
        assert!(target.restore_from(&later, &target).await.is_err());
        target.sync_all().await.unwrap();
        let path = dir.read().await.path().to_owned();
        drop(target);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            256 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let reopened = BTree::<Txn>::load(cache.load(path).unwrap(), super::BTreeSchema::default())
            .await
            .unwrap();

        assert!(has(reopened, txn.at(12), 2).await);
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
        let native = source_dir.clone();
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

        let error = target.restore_from(&txn, &source).await.unwrap_err();
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
                .owner
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        drop(
            target
                .owner
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        target.rollback(txn.id()).unwrap();
    })
    .await;
}
