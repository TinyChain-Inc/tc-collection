use std::sync::Arc;

use futures::{FutureExt, TryStreamExt};
use tc_ir::Transact;
use tc_value::{NumberType, Value, ValueType};

use super::*;
use crate::btree::{BTree, BTreeColumnSchema, BTreeSchema};
use crate::test::{TestTxn, init_root, load_roots, tx};

#[tokio::test]
async fn cancelled_replay_releases_its_exact_id_exemption() {
    let versions = Versions::new();
    let older = tx(1);
    versions
        .pending(tx(3), || async { Ok(Arc::new(())) })
        .await
        .unwrap();
    versions.commit(tx(3), || Ok(())).unwrap();

    {
        let replay = versions.recover(
            older,
            || async {
                versions.reserve_ordered_write(older, || Ok(()))?;
                let error = versions
                    .reserve_ordered_write(tx(2), || Ok(()))
                    .unwrap_err();
                assert_eq!(error.code(), tc_error::ErrorKind::Conflict);
                std::future::pending().await
            },
            || async { panic!("cancelled replay cannot commit") },
        );
        assert!(std::pin::pin!(replay).now_or_never().is_none());
    }

    let error = versions
        .reserve_ordered_write(older, || Ok(()))
        .unwrap_err();
    assert_eq!(error.code(), tc_error::ErrorKind::Conflict);
}

#[tokio::test]
async fn publication_metadata_preserves_identity_and_rejects_ambiguous_roots() {
    let id = tx(10);
    let earlier = tx(9);
    for (input, valid) in [
        ("[1,0,null,[]]".to_string(), true),
        (format!(r#"[1,1,"{id}","version",0]"#), true),
        (format!(r#"[1,0,"{id}",[["{earlier}","version",8]]]"#), true),
        ("[2,0,null,[]]".to_string(), false),
        ("[1,0,null]".to_string(), false),
        ("[1,0,null,[],0]".to_string(), false),
        (format!(r#"[1,0,null,[["{id}","version",1]]]"#), false),
        (
            format!(r#"[1,0,"{earlier}",[["{id}","version",1]]]"#),
            false,
        ),
        (
            format!(r#"[1,0,"{id}",[["{earlier}","left",1],["{id}","right",1]]]"#),
            false,
        ),
        (
            format!(r#"[1,0,"{id}",[["{id}","left",8],["{earlier}","right",1]]]"#),
            false,
        ),
        (format!(r#"[1,0,"{id}",[["{id}","version",0]]]"#), false),
        (format!(r#"[1,1,"{id}","version",1,0]"#), false),
    ] {
        let result = destream_json::try_decode::<_, _, crate::CollectionMetadata>(
            (),
            futures::stream::iter([Ok::<_, io::Error>(input.into_bytes().into())]),
        )
        .await;

        assert_eq!(result.is_ok(), valid);
        if let Ok(metadata) = result {
            let decoded: crate::CollectionMetadata =
                tbon::de::try_decode((), tbon::en::encode(&metadata).unwrap())
                    .await
                    .unwrap();

            let original = tbon::en::encode(&metadata)
                .unwrap()
                .try_fold(Vec::new(), |mut bytes, chunk| async move {
                    bytes.extend_from_slice(&chunk);
                    Ok(bytes)
                })
                .await
                .unwrap();

            let encoded = tbon::en::encode(&decoded)
                .unwrap()
                .try_fold(Vec::new(), |mut bytes, chunk| async move {
                    bytes.extend_from_slice(&chunk);
                    Ok(bytes)
                })
                .await
                .unwrap();

            assert_eq!(encoded, original);
        }
    }
}

#[tokio::test]
async fn lifecycle_callbacks_run_after_publication_without_state_guards() {
    let versions = Versions::new();
    let id = tx(10);
    versions
        .pending(id, || async { Ok(Arc::new(1u64)) })
        .await
        .unwrap();

    for _ in 0..2 {
        versions
            .commit(id, || {
                let state = versions.state.try_write().unwrap();
                assert!(state.committed.contains_key(&id));
                assert!(!state.pending.contains_key(&id));
                Ok(())
            })
            .unwrap();
    }

    versions
        .pending(tx(11), || async { Ok(Arc::new(2)) })
        .await
        .unwrap();

    versions
        .rollback(tx(11), || {
            assert!(versions.state.try_write().unwrap().pending.is_empty());
            Ok(())
        })
        .unwrap();
}

#[tokio::test]
async fn poisoned_state_fails_closed() {
    let owner = Versions::<()>::new();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _state = owner.state.write().unwrap();
        panic!("injected state failure");
    }));

    assert!(poisoned.is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || owner.visible_snapshot(tx(10))
        ))
        .is_err()
    );

    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || owner.commit(tx(10), || Ok(()))
        ))
        .is_err()
    );

    assert!(owner.state.is_poisoned());
}

#[tokio::test]
async fn interrupted_publication_requires_reopen_and_discards_unaccepted_versions() {
    let root = init_root("interrupted-native-publication");
    let (dir, _) = load_roots(&root);
    let store = storage::Store::<crate::PersistentFile>::create(dir)
        .await
        .unwrap();

    store.initialize().await.unwrap();
    let allocation = store.allocate().await.unwrap();
    let owner = Versions::new();
    owner
        .pending(tx(10), || async { Ok(allocation.clone()) })
        .await
        .unwrap();

    {
        let held = allocation.dir().write().await;
        let mut commit = Box::pin(owner.commit_durable(
            &store,
            tx(10),
            |allocation| async { Ok((allocation, 1)) },
            || Ok(()),
        ));

        assert!(commit.as_mut().now_or_never().is_none());
        drop(commit);
        drop(held);
    }

    assert!(
        owner
            .visible_snapshot(tx(11))
            .err()
            .unwrap()
            .requires_recovery()
    );

    drop(owner);
    drop(allocation);
    drop(store);
    let cache = freqfs::Cache::<crate::PersistentFile>::new(
        16 * 1024 * 1024,
        None,
        0,
        std::time::Duration::from_secs(3),
    );

    let store = storage::Store::load(cache.load(root.join("persistent")).unwrap())
        .await
        .unwrap();

    assert!(store.receipts(None).await.unwrap().is_empty());
    store.reclaim().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn durable_versions_reopen_and_compact_without_invalidating_readers() {
    Box::pin(async {
        let root = init_root("native-durable-versions");
        let (dir, txn) = load_roots(&root);
        let schema = BTreeSchema::try_from(vec![BTreeColumnSchema {
            name: "key".into(),
            dtype: ValueType::Number(NumberType::Number),
            max_size: None,
        }])
        .unwrap();

        let tree = BTree::<TestTxn>::create(dir, schema.clone()).await.unwrap();
        for n in 10..13 {
            tree.insert_row(&txn.with_id(tx(n)), vec![Value::from(u64::from(n))])
                .await
                .unwrap();

            tree.commit(tx(n)).await.unwrap();
        }

        let keys = tree.keys(tx(13)).await.unwrap();
        tree.finalize(&tx(12)).await.unwrap();
        assert_eq!(
            keys.try_collect::<Vec<_>>().await.unwrap(),
            vec![vec![10u64.into()], vec![11u64.into()], vec![12u64.into()]]
        );

        tree.finalize(&tx(13)).await.unwrap();
        drop(tree);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            16 * 1024 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let tree = BTree::<TestTxn>::load(cache.load(root.join("persistent")).unwrap(), schema)
            .await
            .unwrap();

        assert_eq!(tree.count(tx(14)).await.unwrap(), 3);
        let collection = crate::Collection::BTree(Box::new(tree));
        collection
            .recover(tx(12), || async {
                panic!("a durable decision must not replay")
            })
            .await
            .unwrap();

        // Fully cancelled insertions must not accumulate historical deletion keys.
        let crate::Collection::BTree(tree) = collection else {
            unreachable!()
        };

        for key in 10u64..13 {
            tree.delete_row(&txn.with_id(tx(15)), vec![key.into()])
                .await
                .unwrap();
        }

        tree.commit(tx(15)).await.unwrap();
        tree.finalize(&tx(15)).await.unwrap();
        assert_eq!(tree.count(tx(16)).await.unwrap(), 0);
        drop(tree);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            16 * 1024 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let store = storage::Store::load(cache.load(root.join("persistent")).unwrap())
            .await
            .unwrap();

        assert!(store.frontier().await.unwrap().1.is_empty());
    })
    .await;
}

// Fault and stack-depth fixtures inspect the lock without exposing it in production.
impl<D> Versions<D> {
    pub(crate) fn hold_state_for_test(&self) -> impl Drop + '_ {
        self.write_state()
    }

    pub(crate) fn state_unlocked_for_test(&self) -> bool {
        self.state.try_write().is_ok()
    }

    pub(crate) fn finalized_for_test(&self) -> Option<TxnId> {
        self.read_state().finalized
    }

    pub(crate) fn pending_for_test(&self, id: TxnId) -> Arc<D> {
        self.read_state().pending[&id].clone()
    }

    pub(crate) fn has_pending_for_test(&self) -> bool {
        !self.read_state().pending.is_empty()
    }

    pub(crate) fn seed_committed_for_test(&self, versions: impl Iterator<Item = (TxnId, Arc<D>)>) {
        let mut state = self.write_state();
        state.pending.clear();
        state
            .committed
            .extend(versions.map(|(id, delta)| (id, Some(delta))));
    }
}
