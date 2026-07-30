//! Transactional visibility and ordering regression tests for `BTree`.
use super::{BTree, BTreeSlice, PersistentFile};
use freqfs::Cache;
use futures::future::join_all;
use std::ops::Bound;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tc_ir::{NetworkTime, TxnId};
use tc_value::Value;
use tokio::sync::Barrier;
use tokio::time::{Duration, sleep, timeout};

fn tx(nonce: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), nonce)
}

fn test_root(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();

    PathBuf::from(format!(
        "/tmp/tc-collection-{name}-{nanos}-{}",
        std::process::id()
    ))
}

async fn init_root(name: &str) -> PathBuf {
    let root = test_root(name);
    std::fs::create_dir_all(root.join("persistent")).expect("create persistent root");
    std::fs::create_dir_all(root.join("txn")).expect("create txn root");
    root
}

fn run_async_test(
    name: &str,
    test_fn: impl FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    + Send
    + 'static,
) {
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_stack_size(16 * 1024 * 1024)
                .enable_all()
                .build()
                .expect("create test runtime");

            runtime.block_on(test_fn());
        })
        .expect("spawn test thread")
        .join()
        .expect("join test thread");
}

fn load_roots(
    root: &Path,
) -> (
    freqfs::DirLock<PersistentFile>,
    freqfs::DirLock<PersistentFile>,
) {
    let cache = Cache::<PersistentFile>::new(16 * 1024 * 1024, None);
    let persistent = Arc::clone(&cache)
        .load(root.join("persistent"))
        .expect("load persistent root");
    let txn = Arc::clone(&cache)
        .load(root.join("txn"))
        .expect("load txn root");
    (persistent, txn)
}

#[test]
fn pending_is_visible_only_to_its_txn() {
    run_async_test("pending_is_visible_only_to_its_txn", || {
        Box::pin(async {
            let root = init_root("pending-visible").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);
            btree
                .insert_row(tx(10), vec![Value::from(1_u64)])
                .await
                .expect("insert pending");

            assert!(btree.contains_row(tx(10), &[Value::from(1_u64)]).await);
            assert!(!btree.contains_row(tx(9), &[Value::from(1_u64)]).await);

            assert!(
                timeout(
                    Duration::from_millis(50),
                    btree.contains_row(tx(11), &[Value::from(1_u64)]),
                )
                .await
                .is_err(),
                "later txn read should block while earlier overlapping write is pending"
            );

            btree.commit(tx(10)).expect("commit 10");
            btree.finalize(tx(10)).await.expect("finalize 10");

            let visible = timeout(
                Duration::from_secs(1),
                btree.contains_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .expect("later txn read should complete after finalize");

            assert!(visible);
        })
    });
}

#[test]
fn committed_is_visible_in_txn_order() {
    run_async_test("committed_is_visible_in_txn_order", || {
        Box::pin(async {
            let root = init_root("committed-visible").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);
            btree
                .insert_row(tx(10), vec![Value::from(1_u64)])
                .await
                .expect("insert key");
            btree.commit(tx(10)).expect("commit");

            assert!(!btree.contains_row(tx(9), &[Value::from(1_u64)]).await);
            assert!(btree.contains_row(tx(10), &[Value::from(1_u64)]).await);
            btree.finalize(tx(10)).await.expect("finalize 10");
            assert!(btree.contains_row(tx(11), &[Value::from(1_u64)]).await);
        })
    });
}

#[test]
fn direct_mutation_flow_from_chain_state() {
    run_async_test("direct_mutation_flow_from_chain_state", || {
        Box::pin(async {
            let root = init_root("direct-mutation").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(10), vec![Value::from(1_u64)])
                .await
                .expect("insert 1");
            btree
                .insert_row(tx(10), vec![Value::from(2_u64)])
                .await
                .expect("insert 2");
            btree
                .delete_row(tx(10), vec![Value::from(2_u64)])
                .await
                .expect("delete 2");
            btree.commit(tx(10)).expect("commit 10");
            btree.finalize(tx(10)).await.expect("finalize 10");

            let mut keys = Vec::new();
            btree
                .for_each_row_in_order(
                    tx(10),
                    (Bound::<Value>::Unbounded, Bound::<Value>::Unbounded),
                    false,
                    |row| keys.push(row.into_iter().next().expect("expected unary row")),
                )
                .await;
            assert_eq!(keys, vec![Value::from(1_u64)]);
        })
    });
}

#[test]
fn cannot_write_after_commit_or_finalize() {
    run_async_test("cannot_write_after_commit_or_finalize", || {
        Box::pin(async {
            let root = init_root("write-after-finalize").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(10), vec![Value::from(1_u64)])
                .await
                .expect("insert key");
            btree.commit(tx(10)).expect("commit");

            assert_eq!(
                btree.insert_row(tx(10), vec![Value::from(2_u64)]).await,
                Err(txn_lock::Error::Committed)
            );

            btree.finalize(tx(10)).await.expect("finalize");
            assert_eq!(
                btree.insert_row(tx(10), vec![Value::from(3_u64)]).await,
                Err(txn_lock::Error::Outdated)
            );
        })
    });
}

#[test]
fn streamed_keys_match_materialized_keys() {
    run_async_test("streamed_keys_match_materialized_keys", || {
        Box::pin(async {
            let root = init_root("streamed-vs-materialized").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for key in [1, 2, 3, 4, 5] {
                btree
                    .insert_row(tx(10), vec![Value::from(key as u64)])
                    .await
                    .expect("insert key");
            }

            btree
                .delete_row(tx(10), vec![Value::from(2_u64)])
                .await
                .expect("delete key");
            btree.commit(tx(10)).expect("commit 10");

            btree
                .insert_row(tx(11), vec![Value::from(7_u64)])
                .await
                .expect("insert key 7");
            btree
                .delete_row(tx(11), vec![Value::from(3_u64)])
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
                .await;
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
    });
}

#[test]
fn slice_keys_match_range_view() {
    run_async_test("slice_keys_match_range_view", || {
        Box::pin(async {
            let root = init_root("slice-keys").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for key in [1, 3, 5, 7, 9] {
                btree
                    .insert_row(tx(40), vec![Value::from(key as u64)])
                    .await
                    .expect("insert key");
            }

            btree.commit(tx(40)).expect("commit");

            let slice = btree.slice(Value::from(3_u64)..=Value::from(8_u64), false);
            let mut keys = Vec::new();
            slice
                .for_each_row_in_order(tx(40), |row| {
                    keys.push(row.into_iter().next().expect("expected unary row"))
                })
                .await;
            assert_eq!(
                keys,
                vec![Value::from(3_u64), Value::from(5_u64), Value::from(7_u64)]
            );
            assert_eq!(slice.count(tx(40)).await, 3);
            assert!(!slice.is_empty(tx(40)).await);
        })
    });
}

#[test]
fn btree_is_send_and_sync_when_key_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<BTree>();
    assert_sync::<BTree>();
    assert_send::<BTreeSlice>();
    assert_sync::<BTreeSlice>();
}

#[test]
fn overlapping_write_in_past_txn_fails_closed() {
    run_async_test("overlapping_write_in_past_txn_fails_closed", || {
        Box::pin(async {
            let root = init_root("overlapping-write").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(2), vec![Value::from("k")])
                .await
                .expect("reserve write at txn 2");

            let err = btree
                .insert_row(tx(1), vec![Value::from("k")])
                .await
                .expect_err("expected conflict for out-of-order overlapping write");

            assert_eq!(err, txn_lock::Error::Conflict);
        })
    });
}

#[test]
fn rollback_unblocks_later_read_and_discards_pending() {
    run_async_test("rollback_unblocks_later_read_and_discards_pending", || {
        Box::pin(async {
            let root = init_root("rollback-unblocks-read").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(10), vec![Value::from("k")])
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

            assert!(!visible, "rolled-back key must not be visible");
        })
    });
}

#[test]
fn duplicate_commit_is_idempotent() {
    run_async_test("duplicate_commit_is_idempotent", || {
        Box::pin(async {
            let root = init_root("duplicate-commit").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(10), vec![Value::from(42_u64)])
                .await
                .expect("insert key");

            btree.commit(tx(10)).expect("first commit");
            btree.commit(tx(10)).expect("second commit should be no-op");
            btree.finalize(tx(10)).await.expect("finalize 10");

            let visible = timeout(
                Duration::from_secs(1),
                btree.contains_row(tx(11), &[Value::from(42_u64)]),
            )
            .await
            .expect("later txn read should complete after duplicate commit path");

            assert!(visible);
        })
    });
}

#[test]
fn stale_finalize_is_noop() {
    run_async_test("stale_finalize_is_noop", || {
        Box::pin(async {
            let root = init_root("stale-finalize").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(10), vec![Value::from("x")])
                .await
                .expect("insert key");
            btree.commit(tx(10)).expect("commit 10");
            btree.finalize(tx(10)).await.expect("finalize 10");

            // Stale finalize must not regress the frontier or alter visibility.
            btree
                .finalize(tx(9))
                .await
                .expect("stale finalize should be no-op");

            assert_eq!(btree.finalized(), Some(tx(10)));
            let visible = timeout(
                Duration::from_secs(1),
                btree.contains_row(tx(10), &[Value::from("x")]),
            )
            .await
            .expect("post-stale-finalize read should complete");

            assert!(visible);
        })
    });
}

#[test]
fn large_scan_completes_under_timeout() {
    run_async_test("large_scan_completes_under_timeout", || {
        Box::pin(async {
            let root = init_root("large-scan").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for i in 0_u64..2_000_u64 {
                btree
                    .insert_row(tx(20), vec![Value::from(i)])
                    .await
                    .expect("insert large keyset");
            }

            btree.commit(tx(20)).expect("commit 20");
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
                    .await;
            })
            .await
            .expect("large scan should finish in bounded time");

            assert_eq!(seen, 2_000_u64);
        })
    });
}

#[test]
fn many_later_readers_unblock_after_commit() {
    run_async_test("many_later_readers_unblock_after_commit", || {
        Box::pin(async {
            let root = init_root("many-readers-unblock").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(50), vec![Value::from("hot")])
                .await
                .expect("insert pending key");

            let mut readers = Vec::new();
            for _ in 0..64 {
                let btree = btree.clone();
                readers.push(tokio::spawn(async move {
                    btree.contains_row(tx(51), &[Value::from("hot")]).await
                }));
            }

            sleep(Duration::from_millis(50)).await;
            assert!(
                readers.iter().all(|reader| !reader.is_finished()),
                "later overlapping readers should still be blocked before finalize"
            );

            btree.commit(tx(50)).expect("commit 50");

            let results = timeout(Duration::from_secs(2), async { join_all(readers).await })
                .await
                .expect("all readers should complete after commit");

            for result in results {
                assert!(result.expect("reader task should join successfully"));
            }
        })
    });
}

#[test]
fn concurrent_writer_conflict_matrix() {
    run_async_test("concurrent_writer_conflict_matrix", || {
        Box::pin(async {
            let root = init_root("writer-conflict-matrix").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);
            let barrier = Arc::new(Barrier::new(4));

            let later_same = {
                let btree = btree.clone();
                let barrier = barrier.clone();
                tokio::spawn(async move {
                    barrier.wait().await;
                    btree.insert_row(tx(20), vec![Value::from("same")]).await
                })
            };

            let earlier_same = {
                let btree = btree.clone();
                let barrier = barrier.clone();
                tokio::spawn(async move {
                    barrier.wait().await;
                    sleep(Duration::from_millis(10)).await;
                    btree.insert_row(tx(19), vec![Value::from("same")]).await
                })
            };

            let earlier_disjoint = {
                let btree = btree.clone();
                let barrier = barrier.clone();
                tokio::spawn(async move {
                    barrier.wait().await;
                    sleep(Duration::from_millis(10)).await;
                    btree.insert_row(tx(19), vec![Value::from("other")]).await
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
            assert_eq!(earlier_same, Err(txn_lock::Error::Conflict));
            assert!(
                earlier_disjoint.is_ok(),
                "disjoint earlier writer should succeed"
            );
        })
    });
}

#[test]
fn snapshot_scan_is_coherent_under_concurrent_commits() {
    run_async_test("snapshot_scan_is_coherent_under_concurrent_commits", || {
        Box::pin(async {
            let root = init_root("snapshot-coherence").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for i in 0_u64..100_u64 {
                btree
                    .insert_row(tx(30), vec![Value::from(i)])
                    .await
                    .expect("seed baseline key");
            }

            btree.commit(tx(30)).expect("commit 30");
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
                        .await;

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
                        .insert_row(tx(32), vec![Value::from(100_u64)])
                        .await
                        .expect("insert future key");
                    btree.commit(tx(32)).expect("commit 32");
                    btree.finalize(tx(32)).await.expect("finalize 32");
                })
            };

            barrier.wait().await;

            let saw_future_key = scan_task.await.expect("scan task join");
            writer_task.await.expect("writer task join");

            assert!(
                !saw_future_key,
                "txn 31 snapshot must not include key committed/finalized at txn 32"
            );
            assert!(btree.contains_row(tx(33), &[Value::from(100_u64)]).await);
        })
    });
}

#[test]
#[ignore = "long-running soak test"]
fn lifecycle_noop_paths_do_not_leak_reservations_soak() {
    run_async_test("lifecycle_noop_paths_do_not_leak_reservations_soak", || {
        Box::pin(async {
            let root = init_root("reservation-leak-soak").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for i in 0_u16..200_u16 {
                let txn_id = tx(1_000 + i);
                let next = tx(1_001 + i);

                btree
                    .insert_row(txn_id, vec![Value::from(i as u64)])
                    .await
                    .expect("insert key");

                btree.commit(txn_id).expect("commit");
                btree.commit(txn_id).expect("duplicate commit no-op");
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

                assert!(visible);
            }
        })
    });
}

#[test]
#[ignore = "load smoke test"]
fn load_smoke_mixed_mutation_and_scan() {
    run_async_test("load_smoke_mixed_mutation_and_scan", || {
        Box::pin(async {
            let root = init_root("load-smoke").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for i in 0_u64..10_000_u64 {
                btree
                    .insert_row(tx(2_000), vec![Value::from(i)])
                    .await
                    .expect("seed key");
            }

            btree.commit(tx(2_000)).expect("commit 2000");
            btree.finalize(tx(2_000)).await.expect("finalize 2000");

            for i in (0_u64..10_000_u64).step_by(2) {
                btree
                    .delete_row(tx(2_001), vec![Value::from(i)])
                    .await
                    .expect("delete even key");
            }

            for i in 10_000_u64..15_000_u64 {
                btree
                    .insert_row(tx(2_001), vec![Value::from(i)])
                    .await
                    .expect("insert extension key");
            }

            btree.commit(tx(2_001)).expect("commit 2001");
            btree.finalize(tx(2_001)).await.expect("finalize 2001");

            let total = timeout(Duration::from_secs(10), btree.count(tx(2_002)))
                .await
                .expect("count should complete");
            assert_eq!(total, 10_000);

            assert!(!btree.is_empty(tx(2_002)).await);

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
                    .await;
            })
            .await
            .expect("range scan should complete");

            assert_eq!(range_count, 1_501);
        })
    });
}

#[test]
fn finalize_conflicts_with_future_read() {
    run_async_test("finalize_conflicts_with_future_read", || {
        Box::pin(async {
            let root = init_root("finalize-future-read-conflict").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(9), vec![Value::from("k")])
                .await
                .expect("insert seed key");
            btree.commit(tx(9)).expect("commit 9");
            btree.finalize(tx(9)).await.expect("finalize 9");

            // Register an overlapping future read at txn 11.
            assert!(btree.contains_row(tx(11), &[Value::from("k")]).await);

            // Finalize at txn 10 should succeed even though txn 11 has an active
            // read reservation. Finalize is a lifecycle operation, not a write —
            // it merges already-committed data into canon. Future reads are
            // protected by their own permits and by the DirLock on persistent storage.
            btree.finalize(tx(10)).await.expect("finalize should succeed with future read");

            // After finalize, the data should still be visible to later reads.
            assert!(btree.contains_row(tx(12), &[Value::from("k")]).await);
        })
    });
}

#[test]
fn blocked_reader_cancellation_does_not_poison_lock_state() {
    run_async_test(
        "blocked_reader_cancellation_does_not_poison_lock_state",
        || {
            Box::pin(async {
                let root = init_root("blocked-reader-cancel").await;
                let (persistent, txn) = load_roots(&root);
                let btree = BTree::new(persistent, txn);

                btree
                    .insert_row(tx(60), vec![Value::from("hot")])
                    .await
                    .expect("insert pending key");

                let blocked_reader = {
                    let btree = btree.clone();
                    tokio::spawn(
                        async move { btree.contains_row(tx(61), &[Value::from("hot")]).await },
                    )
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

                assert!(!visible);
            })
        },
    );
}

#[test]
fn timeout_cleanup_path_unblocks_later_reads() {
    run_async_test("timeout_cleanup_path_unblocks_later_reads", || {
        Box::pin(async {
            let root = init_root("timeout-cleanup-unblock").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(70), vec![Value::from("pending")])
                .await
                .expect("insert pending key");

            assert!(
                timeout(
                    Duration::from_millis(50),
                    btree.contains_row(tx(71), &[Value::from("pending")]),
                )
                .await
                .is_err(),
                "later read should block while pending write exists"
            );

            // Host timeout/cleanup semantics map to rollback+finalize(release) behavior.
            btree.rollback(tx(70)).expect("rollback timeout txn");

            let visible = timeout(
                Duration::from_secs(1),
                btree.contains_row(tx(71), &[Value::from("pending")]),
            )
            .await
            .expect("later read should complete after timeout cleanup");

            assert!(!visible);
        })
    });
}

#[test]
fn multi_column_partial_overlap_blocking_behavior() {
    run_async_test("multi_column_partial_overlap_blocking_behavior", || {
        Box::pin(async {
            let root = init_root("multi-column-overlap").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::with_storage_and_key_types(
                persistent,
                txn,
                super::StorageConfig::default(),
                2,
                None,
            );

            btree
                .insert_row(tx(80), vec![Value::from("a"), Value::from(1_u64)])
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
            assert!(!disjoint);

            let err = btree
                .commit(tx(80))
                .expect_err("commit should conflict while future overlapping read is active");
            assert_eq!(err, txn_lock::Error::Conflict);

            let err = btree
                .rollback(tx(80))
                .expect_err("rollback should also conflict with active future read version");
            assert_eq!(err, txn_lock::Error::Conflict);
        })
    });
}

#[test]
#[ignore = "long-running randomized reservation stress test"]
fn reservation_fuzz_no_read_starvation() {
    run_async_test("reservation_fuzz_no_read_starvation", || {
        Box::pin(async {
            let root = init_root("reservation-fuzz").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            for i in 0_u16..300_u16 {
                let txn_id = tx(3_000 + i);
                let key = Value::from((i % 32) as u64);

                if i % 3 != 0 {
                    btree
                        .insert_row(txn_id, vec![key.clone()])
                        .await
                        .expect("fuzz insert");
                }

                if i % 2 == 0 {
                    btree.commit(txn_id).expect("fuzz commit");
                    if i % 4 == 0 {
                        btree.commit(txn_id).expect("fuzz duplicate commit");
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
                .expect("probe read should not starve");
            }
        })
    });
}

#[test]
fn same_txn_read_your_own_write_is_non_blocking() {
    run_async_test("same_txn_read_your_own_write_is_non_blocking", || {
        Box::pin(async {
            let root = init_root("same-txn-read-own-write").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(90), vec![Value::from("own")])
                .await
                .expect("insert own key");

            let visible = timeout(
                Duration::from_millis(200),
                btree.contains_row(tx(90), &[Value::from("own")]),
            )
            .await
            .expect("same-txn read-your-own-write should not block");

            assert!(visible);
        })
    });
}

#[test]
fn same_txn_read_your_own_delete_is_non_blocking() {
    run_async_test("same_txn_read_your_own_delete_is_non_blocking", || {
        Box::pin(async {
            let root = init_root("same-txn-read-own-delete").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(91), vec![Value::from("gone")])
                .await
                .expect("insert key");
            btree
                .delete_row(tx(91), vec![Value::from("gone")])
                .await
                .expect("delete key in same txn");

            let visible = timeout(
                Duration::from_millis(200),
                btree.contains_row(tx(91), &[Value::from("gone")]),
            )
            .await
            .expect("same-txn read-your-own-delete should not block");

            assert!(!visible);
        })
    });
}

#[test]
fn repeated_rollback_and_finalize_are_idempotent() {
    run_async_test("repeated_rollback_and_finalize_are_idempotent", || {
        Box::pin(async {
            let root = init_root("repeated-lifecycle-idempotency").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            btree
                .insert_row(tx(92), vec![Value::from("rollback")])
                .await
                .expect("insert key for rollback");

            btree.rollback(tx(92)).expect("first rollback");
            btree
                .rollback(tx(92))
                .expect("second rollback should be no-op");

            btree
                .insert_row(tx(93), vec![Value::from("finalize")])
                .await
                .expect("insert key for finalize");
            btree.commit(tx(93)).expect("commit 93");

            btree.finalize(tx(93)).await.expect("first finalize");
            btree
                .finalize(tx(93))
                .await
                .expect("second finalize should be no-op");

            assert_eq!(btree.finalized(), Some(tx(93)));
            assert!(btree.contains_row(tx(94), &[Value::from("finalize")]).await);
            assert!(!btree.contains_row(tx(94), &[Value::from("rollback")]).await);
        })
    });
}

#[test]
fn empty_tree_semantics_across_lifecycle() {
    run_async_test("empty_tree_semantics_across_lifecycle", || {
        Box::pin(async {
            let root = init_root("empty-tree-lifecycle").await;
            let (persistent, txn) = load_roots(&root);
            let btree = BTree::new(persistent, txn);

            assert!(btree.is_empty(tx(95)).await);
            assert_eq!(btree.count(tx(95)).await, 0);
            assert!(!btree.contains_row(tx(95), &[Value::from("none")]).await);

            btree
                .commit(tx(95))
                .expect("empty commit should be allowed");
            btree
                .finalize(tx(95))
                .await
                .expect("empty finalize should be allowed");

            assert!(btree.is_empty(tx(96)).await);
            assert_eq!(btree.count(tx(96)).await, 0);
            assert!(!btree.contains_row(tx(96), &[Value::from("none")]).await);
        })
    });
}
