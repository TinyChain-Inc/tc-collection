//! Transactional visibility and ordering regression tests for `BTree`.
use std::ops::Bound;
use std::sync::Arc;

use freqfs::Cache;
use futures::TryStreamExt;
use futures::future::join_all;
use tc_value::Value;
use tokio::sync::Barrier;
use tokio::time::{Duration, sleep, timeout};

use super::{BTree, BTreeSlice};
use crate::PersistentFile;
use crate::test::{TestTxn, init_root, load_roots, tx};

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn pending_is_visible_only_to_its_txn() {
    Box::pin(async {
        let root = init_root("pending-visible");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();
        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from(1_u64)])
            .await
            .expect("insert pending");

        assert!(
            btree
                .contains_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
        );

        assert!(
            !btree
                .contains_row(tx(9), &[Value::from(1_u64)])
                .await
                .unwrap()
        );

        assert!(
            timeout(
                Duration::from_millis(50),
                btree.contains_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .is_err(),
            "later txn read should block while earlier overlapping write is pending"
        );

        btree.commit(tx(10)).await.expect("commit 10");
        btree.finalize(tx(10)).await.expect("finalize 10");

        let visible = timeout(
            Duration::from_secs(1),
            btree.contains_row(tx(11), &[Value::from(1_u64)]),
        )
        .await
        .expect("later txn read should complete after finalize");

        assert!(visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn committed_is_visible_in_txn_order() {
    Box::pin(async {
        let root = init_root("committed-visible");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();
        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from(1_u64)])
            .await
            .expect("insert key");

        let mut result = btree
            .keys(tx(10), (Bound::Unbounded, Bound::Unbounded), false)
            .await
            .unwrap();

        btree.commit(tx(10)).await.expect("commit");

        btree
            .insert_row(&txn.with_id(tx(12)), vec![Value::from(2_u64)])
            .await
            .unwrap();

        btree.commit(tx(12)).await.unwrap();
        assert_eq!(result.try_next().await.unwrap(), Some(vec![1_u64.into()]));
        assert!(result.try_next().await.unwrap().is_none());
        drop(result);

        assert!(
            !btree
                .contains_row(tx(9), &[Value::from(1_u64)])
                .await
                .unwrap()
        );

        assert!(
            btree
                .contains_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
        );

        btree.finalize(tx(10)).await.expect("finalize 10");
        for stale in [tx(9), tx(10)] {
            let error = btree
                .contains_row(stale, &[1_u64.into()])
                .await
                .unwrap_err();

            assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
            let error = btree
                .keys(stale, (Bound::Unbounded, Bound::Unbounded), false)
                .await
                .err()
                .expect("finalized history is unavailable");

            assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
        }

        assert!(
            btree
                .contains_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn direct_mutation_flow_from_chain_state() {
    Box::pin(async {
        let root = init_root("direct-mutation");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from(1_u64)])
            .await
            .expect("insert 1");

        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from(2_u64)])
            .await
            .expect("insert 2");

        btree
            .delete_row(&txn.with_id(tx(10)), vec![Value::from(2_u64)])
            .await
            .expect("delete 2");

        btree.commit(tx(10)).await.expect("commit 10");
        btree.finalize(tx(10)).await.expect("finalize 10");

        let mut keys = Vec::new();
        btree
            .for_each_row_in_order(
                tx(11),
                (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                false,
                |row| keys.push(row.into_iter().next().expect("expected unary row")),
            )
            .await
            .unwrap();

        assert_eq!(keys, vec![Value::from(1_u64)]);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cannot_write_after_commit_or_finalize() {
    Box::pin(async {
        let root = init_root("write-after-finalize");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from(1_u64)])
            .await
            .expect("insert key");

        btree.commit(tx(10)).await.expect("commit");

        assert_eq!(
            btree
                .insert_row(&txn.with_id(tx(10)), vec![Value::from(2_u64)])
                .await
                .unwrap_err()
                .message(),
            txn_lock::Error::Committed.to_string()
        );

        assert!(
            !timeout(
                Duration::from_millis(200),
                btree.contains_row(tx(11), &[2_u64.into()])
            )
            .await
            .expect("rejected committed write must not reserve its key")
            .unwrap()
        );

        btree.finalize(tx(10)).await.expect("finalize");
        assert_eq!(
            btree
                .insert_row(&txn.with_id(tx(10)), vec![Value::from(3_u64)])
                .await
                .unwrap_err()
                .message(),
            txn_lock::Error::Outdated.to_string()
        );

        assert!(
            !timeout(
                Duration::from_millis(200),
                btree.contains_row(tx(12), &[3_u64.into()])
            )
            .await
            .expect("rejected finalized write must not reserve its key")
            .unwrap()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn streamed_keys_match_materialized_keys() {
    Box::pin(async {
        let root = init_root("streamed-vs-materialized");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for key in [1, 2, 3, 4, 5] {
            btree
                .insert_row(&txn.with_id(tx(10)), vec![Value::from(key as u64)])
                .await
                .expect("insert key");
        }

        btree
            .delete_row(&txn.with_id(tx(10)), vec![Value::from(2_u64)])
            .await
            .expect("delete key");

        btree.commit(tx(10)).await.expect("commit 10");

        btree
            .insert_row(&txn.with_id(tx(11)), vec![Value::from(7_u64)])
            .await
            .expect("insert key 7");

        btree
            .delete_row(&txn.with_id(tx(11)), vec![Value::from(3_u64)])
            .await
            .expect("delete key 3");

        let mut streamed = Vec::new();
        btree
            .for_each_row_in_order(
                tx(11),
                (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                false,
                |row| streamed.push(row.into_iter().next().expect("expected unary row")),
            )
            .await
            .unwrap();

        assert_eq!(
            streamed,
            vec![
                Value::from(1_u64),
                Value::from(4_u64),
                Value::from(5_u64),
                Value::from(7_u64),
            ]
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slice_keys_match_range_view() {
    Box::pin(async {
        let root = init_root("slice-keys");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for key in [1, 3, 5, 7, 9] {
            btree
                .insert_row(&txn.with_id(tx(40)), vec![Value::from(key as u64)])
                .await
                .expect("insert key");
        }

        btree.commit(tx(40)).await.expect("commit");

        let slice = btree.slice(Value::from(3_u64)..=Value::from(8_u64), false);
        let mut keys = Vec::new();
        slice
            .for_each_row_in_order(tx(40), |row| {
                keys.push(row.into_iter().next().expect("expected unary row"))
            })
            .await
            .unwrap();

        assert_eq!(
            keys,
            vec![Value::from(3_u64), Value::from(5_u64), Value::from(7_u64)]
        );

        assert_eq!(slice.count(tx(40)).await.unwrap(), 3);
        assert!(!slice.is_empty(tx(40)).await.unwrap());
    })
    .await;
}

#[test]
fn btree_is_send_and_sync_when_key_is_send_and_sync() {
    fn assert_send<T: Send>() {}

    fn assert_sync<T: Sync>() {}

    assert_send::<BTree<TestTxn>>();
    assert_sync::<BTree<TestTxn>>();
    assert_send::<BTreeSlice<TestTxn>>();
    assert_sync::<BTreeSlice<TestTxn>>();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn overlapping_write_in_past_txn_fails_closed() {
    Box::pin(async {
        let root = init_root("overlapping-write");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(2)), vec![Value::from("k")])
            .await
            .expect("reserve write at txn 2");

        let err = btree
            .insert_row(&txn.with_id(tx(1)), vec![Value::from("k")])
            .await
            .expect_err("expected conflict for out-of-order overlapping write");

        assert_eq!(err.code(), tc_error::ErrorKind::Conflict);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn rollback_unblocks_later_read_and_discards_pending() {
    Box::pin(async {
        let root = init_root("rollback-unblocks-read");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from("k")])
            .await
            .expect("insert pending key");

        assert!(
            timeout(
                Duration::from_millis(50),
                btree.contains_row(tx(11), &[Value::from("k")]),
            )
            .await
            .is_err(),
            "later txn read should block while earlier overlapping write is pending"
        );

        btree.rollback(tx(10)).expect("rollback 10");

        let visible = timeout(
            Duration::from_secs(1),
            btree.contains_row(tx(11), &[Value::from("k")]),
        )
        .await
        .expect("later txn read should complete after rollback");

        assert!(!visible.unwrap(), "rolled-back key must not be visible");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn duplicate_commit_is_idempotent() {
    Box::pin(async {
        let root = init_root("duplicate-commit");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from(42_u64)])
            .await
            .expect("insert key");

        btree.commit(tx(10)).await.expect("first commit");
        btree
            .commit(tx(10))
            .await
            .expect("second commit should be no-op");

        btree.finalize(tx(10)).await.expect("finalize 10");

        let visible = timeout(
            Duration::from_secs(1),
            btree.contains_row(tx(11), &[Value::from(42_u64)]),
        )
        .await
        .expect("later txn read should complete after duplicate commit path");

        assert!(visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stale_finalize_is_noop() {
    Box::pin(async {
        let root = init_root("stale-finalize");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(10)), vec![Value::from("x")])
            .await
            .expect("insert key");

        btree.commit(tx(10)).await.expect("commit 10");
        btree.finalize(tx(10)).await.expect("finalize 10");

        // Stale finalize must not regress the frontier or alter visibility.
        btree
            .finalize(tx(9))
            .await
            .expect("stale finalize should be no-op");

        assert_eq!(btree.finalized().unwrap(), Some(tx(10)));
        let visible = timeout(
            Duration::from_secs(1),
            btree.contains_row(tx(11), &[Value::from("x")]),
        )
        .await
        .expect("post-stale-finalize read should complete");

        assert!(visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn large_scan_completes_under_timeout() {
    Box::pin(async {
        let root = init_root("large-scan");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for i in 0_u64..2_000_u64 {
            btree
                .insert_row(&txn.with_id(tx(20)), vec![Value::from(i)])
                .await
                .expect("insert large keyset");
        }

        btree.commit(tx(20)).await.expect("commit 20");
        btree.finalize(tx(20)).await.expect("finalize 20");

        let mut seen = 0_u64;
        timeout(Duration::from_secs(5), async {
            btree
                .for_each_row_in_order(
                    tx(21),
                    (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                    false,
                    |_| {
                        seen += 1;
                    },
                )
                .await
                .unwrap();
        })
        .await
        .expect("large scan should finish in bounded time");

        assert_eq!(seen, 2_000_u64);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn many_later_readers_unblock_after_commit() {
    Box::pin(async {
        let root = init_root("many-readers-unblock");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(50)), vec![Value::from("hot")])
            .await
            .expect("insert pending key");

        let mut readers = Vec::new();
        for _ in 0..64 {
            let btree = btree.clone();
            readers.push(tokio::spawn(async move {
                btree
                    .contains_row(tx(51), &[Value::from("hot")])
                    .await
                    .unwrap()
            }));
        }

        sleep(Duration::from_millis(50)).await;
        assert!(
            readers.iter().all(|reader| !reader.is_finished()),
            "later overlapping readers should still be blocked before finalize"
        );

        btree.commit(tx(50)).await.expect("commit 50");

        let results = timeout(Duration::from_secs(2), async { join_all(readers).await })
            .await
            .expect("all readers should complete after commit");

        for result in results {
            assert!(result.expect("reader task should join successfully"));
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concurrent_writer_conflict_matrix() {
    Box::pin(async {
        let root = init_root("writer-conflict-matrix");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();
        let barrier = Arc::new(Barrier::new(4));

        let later_same = {
            let btree = btree.clone();
            let barrier = barrier.clone();
            let txn = txn.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                btree
                    .insert_row(&txn.with_id(tx(20)), vec![Value::from("same")])
                    .await
            })
        };

        let earlier_same = {
            let btree = btree.clone();
            let barrier = barrier.clone();
            let txn = txn.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                sleep(Duration::from_millis(10)).await;
                btree
                    .insert_row(&txn.with_id(tx(19)), vec![Value::from("same")])
                    .await
            })
        };

        let earlier_disjoint = {
            let btree = btree.clone();
            let barrier = barrier.clone();
            let txn = txn.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                sleep(Duration::from_millis(10)).await;
                btree
                    .insert_row(&txn.with_id(tx(19)), vec![Value::from("other")])
                    .await
            })
        };

        barrier.wait().await;

        let later_same = later_same.await.expect("later_same join");
        let earlier_same = earlier_same.await.expect("earlier_same join");
        let earlier_disjoint = earlier_disjoint.await.expect("earlier_disjoint join");

        assert!(
            later_same.is_ok(),
            "later same-key writer should reserve first"
        );

        assert_eq!(
            earlier_same.unwrap_err().code(),
            tc_error::ErrorKind::Conflict
        );

        assert!(
            earlier_disjoint.is_ok(),
            "disjoint earlier writer should succeed"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn snapshot_scan_is_coherent_under_concurrent_commits() {
    Box::pin(async {
        let root = init_root("snapshot-coherence");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for i in 0_u64..100_u64 {
            btree
                .insert_row(&txn.with_id(tx(30)), vec![Value::from(i)])
                .await
                .expect("seed baseline key");
        }

        btree.commit(tx(30)).await.expect("commit 30");
        btree.finalize(tx(30)).await.expect("finalize 30");

        let barrier = Arc::new(Barrier::new(3));

        let scan_task = {
            let btree = btree.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;

                let mut saw_future_key = false;
                btree
                    .for_each_row_in_order(
                        tx(31),
                        (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                        false,
                        |row| {
                            if row.as_slice() == [Value::from(100_u64)] {
                                saw_future_key = true;
                            }
                        },
                    )
                    .await
                    .unwrap();

                saw_future_key
            })
        };

        let writer_task = {
            let btree = btree.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                sleep(Duration::from_millis(10)).await;

                btree
                    .insert_row(&txn.with_id(tx(32)), vec![Value::from(100_u64)])
                    .await
                    .expect("insert future key");

                btree.commit(tx(32)).await.expect("commit 32");
            })
        };

        barrier.wait().await;

        let saw_future_key = scan_task.await.expect("scan task join");
        writer_task.await.expect("writer task join");
        btree.finalize(tx(32)).await.expect("finalize after scan");

        assert!(
            !saw_future_key,
            "txn 31 snapshot must not include key committed/finalized at txn 32"
        );

        assert!(
            btree
                .contains_row(tx(33), &[Value::from(100_u64)])
                .await
                .unwrap()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn lifecycle_noop_paths_do_not_leak_reservations_soak() {
    Box::pin(async {
        let root = init_root("reservation-leak-soak");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for i in 0_u16..200_u16 {
            let txn_id = tx(1_000 + i);
            let next = tx(1_001 + i);

            btree
                .insert_row(&txn.with_id(txn_id), vec![Value::from(i as u64)])
                .await
                .expect("insert key");

            btree.commit(txn_id).await.expect("commit");
            btree.commit(txn_id).await.expect("duplicate commit no-op");
            btree.finalize(txn_id).await.expect("finalize");

            if i > 0 {
                btree
                    .finalize(tx(999 + i))
                    .await
                    .expect("stale finalize no-op");
            }

            let visible = timeout(
                Duration::from_millis(200),
                btree.contains_row(next, &[Value::from(i as u64)]),
            )
            .await
            .expect("later read should not starve");

            assert!(visible.unwrap());
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn load_smoke_mixed_mutation_and_scan() {
    Box::pin(async {
        let root = init_root("load-smoke");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for i in 0_u64..10_000_u64 {
            btree
                .insert_row(&txn.with_id(tx(2_000)), vec![Value::from(i)])
                .await
                .expect("seed key");
        }

        btree.commit(tx(2_000)).await.expect("commit 2000");
        btree.finalize(tx(2_000)).await.expect("finalize 2000");

        for i in (0_u64..10_000_u64).step_by(2) {
            btree
                .delete_row(&txn.with_id(tx(2_001)), vec![Value::from(i)])
                .await
                .expect("delete even key");
        }

        for i in 10_000_u64..15_000_u64 {
            btree
                .insert_row(&txn.with_id(tx(2_001)), vec![Value::from(i)])
                .await
                .expect("insert extension key");
        }

        btree.commit(tx(2_001)).await.expect("commit 2001");
        btree.finalize(tx(2_001)).await.expect("finalize 2001");

        let total = timeout(Duration::from_secs(10), btree.count(tx(2_002)))
            .await
            .expect("count should complete");

        assert_eq!(total.unwrap(), 10_000);

        assert!(!btree.is_empty(tx(2_002)).await.unwrap());

        let mut range_count = 0_u64;
        timeout(Duration::from_secs(10), async {
            btree
                .for_each_row_in_order(
                    tx(2_002),
                    (
                        Bound::Included(Value::from(9_000_u64)),
                        Bound::Included(Value::from(11_000_u64)),
                    ),
                    false,
                    |_| {
                        range_count += 1;
                    },
                )
                .await
                .unwrap();
        })
        .await
        .expect("range scan should complete");

        assert_eq!(range_count, 1_501);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn finalize_conflicts_with_future_read() {
    Box::pin(async {
        let root = init_root("finalize-future-read-conflict");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(9)), vec![Value::from("k")])
            .await
            .expect("insert seed key");

        btree.commit(tx(9)).await.expect("commit 9");
        btree.finalize(tx(9)).await.expect("finalize 9");

        // Register an overlapping future read at txn 11.
        assert!(
            btree
                .contains_row(tx(11), &[Value::from("k")])
                .await
                .unwrap()
        );

        // Finalize at txn 10 should succeed even though txn 11 has an active
        // read reservation. Finalize is a lifecycle operation, not a write —
        // it merges already-committed data into canon. Future reads are
        // protected by their own permits and by the DirLock on persistent storage.
        btree
            .finalize(tx(10))
            .await
            .expect("finalize should succeed with future read");

        // After finalize, the data should still be visible to later reads.
        assert!(
            btree
                .contains_row(tx(12), &[Value::from("k")])
                .await
                .unwrap()
        );

        for (n, consume) in [(13, false), (15, true)] {
            btree
                .insert_row(&txn.with_id(tx(n)), vec![format!("key-{n}").into()])
                .await
                .unwrap();

            btree.commit(tx(n)).await.unwrap();
            let stream = btree
                .keys(tx(n + 1), (Bound::Unbounded, Bound::Unbounded), false)
                .await
                .unwrap();

            let mut finalize = Box::pin(btree.finalize(tx(n)));
            assert!(futures::poll!(&mut finalize).is_pending());
            if consume {
                assert!(!stream.try_collect::<Vec<_>>().await.unwrap().is_empty());
            } else {
                drop(stream);
            }

            timeout(Duration::from_secs(2), finalize)
                .await
                .unwrap()
                .unwrap();

            assert!(btree.contains_row(tx(n + 1), &["k".into()]).await.unwrap());
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn blocked_reader_cancellation_does_not_poison_lock_state() {
    Box::pin(async {
        let root = init_root("blocked-reader-cancel");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(60)), vec![Value::from("hot")])
            .await
            .expect("insert pending key");

        let blocked_reader = {
            let btree = btree.clone();
            tokio::spawn(async move {
                btree
                    .contains_row(tx(61), &[Value::from("hot")])
                    .await
                    .unwrap()
            })
        };

        sleep(Duration::from_millis(50)).await;
        assert!(
            !blocked_reader.is_finished(),
            "reader should still be blocked before lifecycle resolution"
        );

        blocked_reader.abort();
        let aborted = blocked_reader.await;
        assert!(aborted.is_err(), "blocked reader should abort cleanly");

        // Simulate timeout cleanup using rollback and verify no lock poisoning remains.
        btree.rollback(tx(60)).expect("rollback 60");

        let visible = timeout(
            Duration::from_secs(1),
            btree.contains_row(tx(61), &[Value::from("hot")]),
        )
        .await
        .expect("later read should complete after rollback cleanup");

        assert!(!visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn multi_column_partial_overlap_blocking_behavior() {
    Box::pin(async {
        let root = init_root("multi-column-overlap");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::try_with_schema(
            persistent,
            super::BTreeSchema::new(super::StorageConfig::default(), 2, None).unwrap(),
        )
        .await
        .unwrap();

        btree
            .insert_row(
                &txn.with_id(tx(80)),
                vec![Value::from("a"), Value::from(1_u64)],
            )
            .await
            .expect("insert pending composite key");

        assert!(
            timeout(
                Duration::from_millis(50),
                btree.contains_row(tx(81), &[Value::from("a"), Value::from(1_u64)]),
            )
            .await
            .is_err(),
            "overlapping composite-key read should block"
        );

        let disjoint = timeout(
            Duration::from_secs(1),
            btree.contains_row(tx(81), &[Value::from("a"), Value::from(2_u64)]),
        )
        .await
        .expect("disjoint composite-key read should not block");

        assert!(!disjoint.unwrap());

        btree
            .commit(tx(80))
            .await
            .expect("commit preserves the disjoint future read");

        assert!(
            btree
                .contains_row(tx(81), &[Value::from("a"), Value::from(1_u64)])
                .await
                .unwrap()
        );

        assert_eq!(
            btree.rollback(tx(80)).unwrap_err().code(),
            tc_error::ErrorKind::Conflict
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn reservation_fuzz_no_read_starvation() {
    Box::pin(async {
        let root = init_root("reservation-fuzz");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        for i in 0_u16..300_u16 {
            let txn_id = tx(3_000 + i);
            let key = Value::from((i % 32) as u64);

            if i % 3 != 0 {
                btree
                    .insert_row(&txn.with_id(txn_id), vec![key.clone()])
                    .await
                    .expect("fuzz insert");
            }

            if i % 2 == 0 {
                btree.commit(txn_id).await.expect("fuzz commit");
                if i % 4 == 0 {
                    btree.commit(txn_id).await.expect("fuzz duplicate commit");
                }
            } else {
                btree.rollback(txn_id).expect("fuzz rollback");
            }

            if i > 0 {
                btree
                    .finalize(tx(2_999 + i))
                    .await
                    .expect("fuzz finalize frontier");
            }

            let probe_txn = tx(3_001 + i);
            timeout(
                Duration::from_millis(500),
                btree.contains_row(probe_txn, &[key]),
            )
            .await
            .expect("probe read should not starve")
            .expect("probe read");
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn same_txn_read_your_own_write_is_non_blocking() {
    Box::pin(async {
        let root = init_root("same-txn-read-own-write");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(90)), vec![Value::from("own")])
            .await
            .expect("insert own key");

        let visible = timeout(
            Duration::from_millis(200),
            btree.contains_row(tx(90), &[Value::from("own")]),
        )
        .await
        .expect("same-txn read-your-own-write should not block");

        assert!(visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn same_txn_read_your_own_delete_is_non_blocking() {
    Box::pin(async {
        let root = init_root("same-txn-read-own-delete");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(91)), vec![Value::from("gone")])
            .await
            .expect("insert key");

        btree
            .delete_row(&txn.with_id(tx(91)), vec![Value::from("gone")])
            .await
            .expect("delete key in same txn");

        let visible = timeout(
            Duration::from_millis(200),
            btree.contains_row(tx(91), &[Value::from("gone")]),
        )
        .await
        .expect("same-txn read-your-own-delete should not block");

        assert!(!visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn repeated_rollback_and_finalize_are_idempotent() {
    Box::pin(async {
        let root = init_root("repeated-lifecycle-idempotency");
        let (persistent, txn) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        btree
            .insert_row(&txn.with_id(tx(92)), vec![Value::from("rollback")])
            .await
            .expect("insert key for rollback");

        btree.rollback(tx(92)).expect("first rollback");
        btree
            .rollback(tx(92))
            .expect("second rollback should be no-op");

        btree
            .insert_row(&txn.with_id(tx(93)), vec![Value::from("finalize")])
            .await
            .expect("insert key for finalize");

        btree.commit(tx(93)).await.expect("commit 93");

        btree.finalize(tx(93)).await.expect("first finalize");
        btree
            .finalize(tx(93))
            .await
            .expect("second finalize should be no-op");

        assert_eq!(btree.finalized().unwrap(), Some(tx(93)));
        assert!(
            btree
                .contains_row(tx(94), &[Value::from("finalize")])
                .await
                .unwrap()
        );

        assert!(
            !btree
                .contains_row(tx(94), &[Value::from("rollback")])
                .await
                .unwrap()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn empty_tree_semantics_across_lifecycle() {
    Box::pin(async {
        let root = init_root("empty-tree-lifecycle");
        let (persistent, _) = load_roots(&root);
        let btree = BTree::<TestTxn>::new(persistent).await.unwrap();

        assert!(btree.is_empty(tx(95)).await.unwrap());
        assert_eq!(btree.count(tx(95)).await.unwrap(), 0);
        assert!(
            !btree
                .contains_row(tx(95), &[Value::from("none")])
                .await
                .unwrap()
        );

        btree
            .commit(tx(95))
            .await
            .expect("empty commit should be allowed");

        btree
            .finalize(tx(95))
            .await
            .expect("empty finalize should be allowed");

        assert!(btree.is_empty(tx(96)).await.unwrap());
        assert_eq!(btree.count(tx(96)).await.unwrap(), 0);
        assert!(
            !btree
                .contains_row(tx(96), &[Value::from("none")])
                .await
                .unwrap()
        );
    })
    .await;
}

#[tokio::test]
async fn decoding_reports_root_admission_failure() {
    let root = init_root("decode-root-admission");
    let cache = Cache::<PersistentFile>::new(1, None, 0, Duration::from_millis(10));
    let dir = cache.load(root.join("txn")).expect("delegated directory");
    let txn = TestTxn::new(tx(10), dir);
    let schema = vec![super::BTreeColumnSchema {
        name: "id".to_owned(),
        dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
        max_size: None,
    }];

    let encoded =
        destream_json::encode((schema, Vec::<Vec<Value>>::new())).expect("encode empty BTree");

    let decoded: Result<super::DecodedBTreePayload<TestTxn>, _> =
        destream_json::try_decode(txn, encoded).await;

    let error = decoded.expect_err("root exceeds delegated cache budget");
    assert!(error.to_string().contains("capacity"), "{error}");
    std::fs::remove_dir_all(&*root).expect("remove test root");
}

#[tokio::test]
async fn schema_creation_and_storage_failures_are_errors() {
    use safecast::{CastFrom, TryCastFrom};

    use super::{BTreeColumnSchema, BTreeSchema, StorageConfig};

    let name = String::from("column");
    let pointer = name.as_ptr();
    let dtype = Value::cast_from(tc_value::ValueType::String);
    let column = BTreeColumnSchema::opt_cast_from(Value::Tuple(vec![
        Value::String(name),
        dtype.clone(),
        Value::from(8_u64),
    ]))
    .unwrap();

    assert_eq!(column.name.as_ptr(), pointer);
    assert_eq!(column.dtype, tc_value::ValueType::String);
    assert_eq!(column.max_size, Some(8_u64.into()));

    for value in [
        Value::Tuple(vec![Value::from("column"), dtype.clone()]),
        Value::Tuple(vec![
            Value::from("column"),
            dtype.clone(),
            Value::from(8_u64),
        ]),
        Value::Tuple(vec![Value::from(7_u64), dtype.clone()]),
        Value::Tuple(vec![Value::from("column"), Value::from("invalid type")]),
        Value::Tuple(vec![Value::from("column"), dtype, Value::None]),
        Value::Tuple(vec![Value::from("column")]),
    ] {
        assert_eq!(
            BTreeColumnSchema::can_cast_from(&value),
            BTreeColumnSchema::opt_cast_from(value).is_some()
        );
    }

    assert!(BTreeSchema::new(StorageConfig::default(), 0, None).is_err());
    assert!(BTreeSchema::new(StorageConfig::default(), 2, Some(vec![])).is_err());
    let root = init_root("fallible-btree");
    let (dir, _) = load_roots(&root);
    let tree = BTree::<TestTxn>::new(dir.clone()).await.unwrap();
    assert!(BTree::<TestTxn>::new(dir.clone()).await.is_err());
    let root_name = dir.read().await.iter().next().unwrap().0.clone();
    assert!(dir.write().await.delete(&root_name).await);
    assert!(
        tree.contains_row(tx(1), &[Value::from(1_u64)])
            .await
            .is_err()
    );

    assert!(tree.count(tx(1)).await.is_err());
    assert!(tree.is_empty(tx(1)).await.is_err());
    assert!(
        tree.for_each_row_in_order(tx(1), .., false, |_| panic!("missing root yielded a row"))
            .await
            .is_err()
    );

    drop(tree);
    drop(dir);
    std::fs::remove_dir_all(&*root).unwrap();
}
