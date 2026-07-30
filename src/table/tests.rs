//! Transactional visibility and ordering regression tests for `PersistentTable`.
use super::{Column, PersistentTable, TableSchema};
use crate::btree::{StorageConfig, PersistentFile};
use freqfs::Cache;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tc_ir::{NetworkTime, Transact, TxnId};
use tc_value::{Value, ValueType};
use tokio::time::{Duration, timeout};

fn tx(nonce: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), nonce)
}

fn test_root(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();

    PathBuf::from(format!(
        "/tmp/tc-collection-table-{name}-{nanos}-{}",
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

fn simple_schema() -> TableSchema {
    let key = vec![Column {
        name: "id".parse().expect("Id"),
        dtype: ValueType::Number,
    }];
    let values = vec![Column {
        name: "label".parse().expect("Id"),
        dtype: ValueType::String,
    }];
    TableSchema::new(key, values, Vec::new(), StorageConfig::default())
        .expect("create test schema")
}

fn composite_schema() -> TableSchema {
    let key = vec![
        Column {
            name: "part_a".parse().expect("Id"),
            dtype: ValueType::Number,
        },
        Column {
            name: "part_b".parse().expect("Id"),
            dtype: ValueType::Number,
        },
    ];
    let values = vec![
        Column {
            name: "val1".parse().expect("Id"),
            dtype: ValueType::String,
        },
    ];
    TableSchema::new(key, values, Vec::new(), StorageConfig::default())
        .expect("create composite key schema")
}

fn schema_with_index() -> TableSchema {
    let key = vec![Column {
        name: "id".parse().expect("Id"),
        dtype: ValueType::Number,
    }];
    let values = vec![
        Column {
            name: "ref_id".parse().expect("Id"),
            dtype: ValueType::Number,
        },
        Column {
            name: "label".parse().expect("Id"),
            dtype: ValueType::String,
        },
    ];
    let indices = vec![(
        "by_ref".to_string(),
        vec!["ref_id".parse().expect("Id")],
    )];
    TableSchema::new(key, values, indices, StorageConfig::default())
        .expect("create schema with index")
}

#[test]
fn upsert_inserts_and_updates_pending_row() {
    run_async_test("upsert_inserts_and_updates_pending_row", || {
        Box::pin(async {
            let root = init_root("upsert-pending").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("upsert row");

            let row = table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await;
            assert!(row.is_some());
            assert_eq!(
                row.unwrap().as_ref(),
                &[Value::from(1_u64), Value::from("alpha")]
            );

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("beta")],
                )
                .await
                .expect("update row");

            let row = table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await;
            assert_eq!(
                row.unwrap().as_ref(),
                &[Value::from(1_u64), Value::from("beta")]
            );
        })
    });
}

#[test]
fn delete_moves_row_to_pending_deletes() {
    run_async_test("delete_moves_row_to_pending_deletes", || {
        Box::pin(async {
            let root = init_root("delete-pending").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("upsert row");
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            table
                .delete_row(tx(11), vec![Value::from(1_u64)])
                .await
                .expect("delete row");

            assert!(
                table
                    .read_row(tx(11), &[Value::from(1_u64)])
                    .await
                    .is_none(),
                "deleted row should not be visible"
            );
        })
    });
}

#[test]
fn commit_promotes_pending_to_committed() {
    run_async_test("commit_promotes_pending_to_committed", || {
        Box::pin(async {
            let root = init_root("commit-promotes").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("upsert row");

            assert!(
                table.read_row(tx(9), &[Value::from(1_u64)]).await.is_none(),
                "committed row should not be visible to earlier txn"
            );
            assert!(
                table.read_row(tx(10), &[Value::from(1_u64)]).await.is_some(),
                "pending row should be visible to own txn"
            );

            table.commit(tx(10)).await;

            assert!(
                table.read_row(tx(10), &[Value::from(1_u64)]).await.is_some(),
                "committed row should be visible at commit txn"
            );
            assert!(
                table.read_row(tx(11), &[Value::from(1_u64)]).await.is_some(),
                "committed delta should be visible to later txn (committed <= T)"
            );
        })
    });
}

#[test]
fn rollback_discards_pending_and_unblocks() {
    run_async_test("rollback_discards_pending_and_unblocks", || {
        Box::pin(async {
            let root = init_root("rollback-discards").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("upsert pending");

            assert!(
                timeout(
                    Duration::from_millis(50),
                    table.read_row(tx(11), &[Value::from(1_u64)])
                )
                .await
                .is_err(),
                "later txn read should block while earlier overlapping write is pending"
            );

            table.rollback(&tx(10)).await;

            let row = timeout(
                Duration::from_secs(1),
                table.read_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .expect("later txn read should complete after rollback");

            assert!(row.is_none(), "rolled-back row must not be visible");
        })
    });
}

#[test]
fn finalize_merges_committed_into_canon() {
    run_async_test("finalize_merges_committed_into_canon", || {
        Box::pin(async {
            let root = init_root("finalize-merges").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("upsert row");
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            assert!(
                table
                    .read_row(tx(11), &[Value::from(1_u64)])
                    .await
                    .is_some(),
                "finalized row should be visible to later txn"
            );
            assert_eq!(table.finalized(), Some(tx(10)));
        })
    });
}

#[test]
fn duplicate_commit_is_idempotent_table() {
    run_async_test("duplicate_commit_is_idempotent_table", || {
        Box::pin(async {
            let root = init_root("duplicate-commit-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(42_u64)],
                    vec![Value::from("x")],
                )
                .await
                .expect("upsert row");

            table.commit(tx(10)).await;
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            assert!(
                table
                    .read_row(tx(11), &[Value::from(42_u64)])
                    .await
                    .is_some(),
                "row should be visible after duplicate commit"
            );
        })
    });
}

#[test]
fn stale_finalize_is_noop_table() {
    run_async_test("stale_finalize_is_noop_table", || {
        Box::pin(async {
            let root = init_root("stale-finalize-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("x")],
                )
                .await
                .expect("upsert row");
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            table
                .finalize(&tx(9))
                .await;

            assert_eq!(table.finalized(), Some(tx(10)));
            assert!(
                table
                    .read_row(tx(11), &[Value::from(1_u64)])
                    .await
                    .is_some(),
                "row should still be visible after stale finalize"
            );
        })
    });
}

#[test]
fn cannot_write_after_commit_or_finalize_table() {
    run_async_test("cannot_write_after_commit_or_finalize_table", || {
        Box::pin(async {
            let root = init_root("write-after-finalize-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("a")],
                )
                .await
                .expect("upsert row");
            table.commit(tx(10)).await;

            assert_eq!(
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(1_u64)],
                        vec![Value::from("b")]
                    )
                    .await,
                Err(txn_lock::Error::Committed)
            );

            table.finalize(&tx(10)).await;
            assert_eq!(
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(1_u64)],
                        vec![Value::from("c")]
                    )
                    .await,
                Err(txn_lock::Error::Outdated)
            );
        })
    });
}

#[test]
fn pending_is_visible_only_to_its_txn_table() {
    run_async_test("pending_is_visible_only_to_its_txn_table", || {
        Box::pin(async {
            let root = init_root("pending-visible-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("insert pending");

            assert!(
                table
                    .read_row(tx(10), &[Value::from(1_u64)])
                    .await
                    .is_some(),
                "pending row should be visible to own txn"
            );
            assert!(
                table
                    .read_row(tx(9), &[Value::from(1_u64)])
                    .await
                    .is_none(),
                "pending row should not be visible to earlier txn"
            );

            assert!(
                timeout(
                    Duration::from_millis(50),
                    table.read_row(tx(11), &[Value::from(1_u64)])
                )
                .await
                .is_err(),
                "later txn read should block while earlier overlapping write is pending"
            );

            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let row = timeout(
                Duration::from_secs(1),
                table.read_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .expect("later txn read should complete after finalize");

            assert!(row.is_some());
        })
    });
}

#[test]
fn committed_is_visible_in_txn_order_table() {
    run_async_test("committed_is_visible_in_txn_order_table", || {
        Box::pin(async {
            let root = init_root("committed-visible-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("insert key");
            table.commit(tx(10)).await;

            assert!(
                table
                    .read_row(tx(9), &[Value::from(1_u64)])
                    .await
                    .is_none(),
                "committed row should not be visible to earlier txn"
            );
            assert!(
                table
                    .read_row(tx(10), &[Value::from(1_u64)])
                    .await
                    .is_some(),
                "committed row should be visible at commit txn"
            );
            table.finalize(&tx(10)).await;
            assert!(
                table
                    .read_row(tx(11), &[Value::from(1_u64)])
                    .await
                    .is_some(),
                "finalized row should be visible to later txn"
            );
        })
    });
}

#[test]
fn no_pending_leakage_across_txns_table() {
    run_async_test("no_pending_leakage_across_txns_table", || {
        Box::pin(async {
            let root = init_root("no-leakage-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("a")],
                )
                .await
                .expect("txn 10 insert");

            table
                .upsert_row(
                    tx(12),
                    vec![Value::from(2_u64)],
                    vec![Value::from("b")],
                )
                .await
                .expect("txn 12 insert");

            assert!(
                table.read_row(tx(10), &[Value::from(1_u64)]).await.is_some(),
                "txn 10 should see its own pending write"
            );

            assert!(
                table.read_row(tx(9), &[Value::from(1_u64)]).await.is_none(),
                "earlier txn should not see txn 10's pending write"
            );

            assert!(
                table.read_row(tx(10), &[Value::from(2_u64)]).await.is_none(),
                "txn 10 should not see txn 12's pending write"
            );

            assert!(
                timeout(
                    Duration::from_millis(50),
                    table.read_row(tx(12), &[Value::from(1_u64)])
                )
                .await
                .is_err(),
                "txn 12 read should block while earlier overlapping write at txn 10 is pending"
            );
        })
    });
}

#[test]
fn read_resolves_delta_stack_table() {
    run_async_test("read_resolves_delta_stack_table", || {
        Box::pin(async {
            let root = init_root("delta-stack-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("original")],
                )
                .await
                .expect("insert original");
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            table
                .upsert_row(
                    tx(11),
                    vec![Value::from(1_u64)],
                    vec![Value::from("updated")],
                )
                .await
                .expect("update in txn 11");

            let row = table
                .read_row(tx(11), &[Value::from(1_u64)])
                .await
                .expect("read updated row");
            assert_eq!(
                row.as_ref(),
                &[Value::from(1_u64), Value::from("updated")],
                "should see the delta update, not the canon version"
            );
        })
    });
}

#[test]
fn streamed_rows_match_materialized_table() {
    run_async_test("streamed_rows_match_materialized_table", || {
        Box::pin(async {
            let root = init_root("streamed-materialized-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=5u64 {
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(i)],
                        vec![Value::from(format!("label{i}"))],
                    )
                    .await
                    .expect("insert row");
            }

            table
                .delete_row(tx(10), vec![Value::from(2_u64)])
                .await
                .expect("delete row");
            table.commit(tx(10)).await;

            table
                .upsert_row(
                    tx(11),
                    vec![Value::from(7_u64)],
                    vec![Value::from("label7")],
                )
                .await
                .expect("insert key 7");
            table
                .delete_row(tx(11), vec![Value::from(3_u64)])
                .await
                .expect("delete key 3");

            let mut streamed = Vec::new();
            table
                .for_each_row_in_order(
                    tx(11),
                    super::Range::default(),
                    &[],
                    false,
                    |row| {
                        streamed.push(row.into_vec());
                    },
                )
                .await;

            assert_eq!(
                streamed,
                vec![
                    vec![Value::from(1_u64), Value::from("label1")],
                    vec![Value::from(4_u64), Value::from("label4")],
                    vec![Value::from(5_u64), Value::from("label5")],
                    vec![Value::from(7_u64), Value::from("label7")],
                ]
            );
        })
    });
}

#[test]
fn count_matches_streamed_fold_table() {
    run_async_test("count_matches_streamed_fold_table", || {
        Box::pin(async {
            let root = init_root("count-fold-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=10u64 {
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(i)],
                        vec![Value::from(format!("v{i}"))],
                    )
                    .await
                    .expect("insert row");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            for i in (1..=10u64).step_by(2) {
                table
                    .delete_row(tx(11), vec![Value::from(i)])
                    .await
                    .expect("delete odd row");
            }

            assert_eq!(table.count(tx(11)).await, 5);
        })
    });
}

#[test]
fn contains_all_key_range_table() {
    run_async_test("contains_all_key_range_table", || {
        Box::pin(async {
            let root = init_root("contains-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=5u64 {
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(i)],
                        vec![Value::from(format!("v{i}"))],
                    )
                    .await
                    .expect("insert row");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            assert!(table.contains_row(tx(11), &[Value::from(3_u64)]).await);
            assert!(!table.contains_row(tx(11), &[Value::from(99_u64)]).await);
            assert!(!table.is_empty(tx(11)).await);
        })
    });
}

#[test]
fn empty_table_semantics_across_lifecycle_table() {
    run_async_test("empty_table_semantics_across_lifecycle_table", || {
        Box::pin(async {
            let root = init_root("empty-table-lifecycle-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            assert!(table.is_empty(tx(95)).await);
            assert_eq!(table.count(tx(95)).await, 0);
            assert!(
                table
                    .read_row(tx(95), &[Value::from(1_u64)])
                    .await
                    .is_none()
            );

            table.commit(tx(95)).await;
            table
                .finalize(&tx(95))
                .await;

            assert!(table.is_empty(tx(96)).await);
            assert_eq!(table.count(tx(96)).await, 0);
        })
    });
}

#[test]
fn invalid_key_arity_fails_closed_table() {
    run_async_test("invalid_key_arity_fails_closed_table", || {
        Box::pin(async {
            let root = init_root("invalid-arity-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            let err = table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64), Value::from(2_u64)],
                    vec![Value::from("a")],
                )
                .await
                .expect_err("wrong key arity should fail");

            assert!(matches!(err, txn_lock::Error::Background(_)));

            let err = table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("a"), Value::from("b")],
                )
                .await
                .expect_err("wrong value arity should fail");

            assert!(matches!(err, txn_lock::Error::Background(_)));
        })
    });
}

#[test]
fn invalid_key_type_fails_closed_table() {
    run_async_test("invalid_key_type_fails_closed_table", || {
        Box::pin(async {
            let root = init_root("invalid-key-type-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            let err = table
                .upsert_row(
                    tx(10),
                    vec![Value::from("not_a_number")],
                    vec![Value::from("a")],
                )
                .await
                .expect_err("wrong key type should fail");

            assert!(matches!(err, txn_lock::Error::Background(_)));
        })
    });
}

#[test]
fn composite_key_upsert_and_read() {
    run_async_test("composite_key_upsert_and_read", || {
        Box::pin(async {
            let root = init_root("composite-key-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, composite_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64), Value::from(2_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("upsert composite key");

            let row = table
                .read_row(tx(10), &[Value::from(1_u64), Value::from(2_u64)])
                .await
                .expect("read composite key");

            assert_eq!(
                row.as_ref(),
                &[Value::from(1_u64), Value::from(2_u64), Value::from("alpha")]
            );

            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let mut streamed = Vec::new();
            table
                .for_each_row_in_order(
                    tx(11),
                    super::Range::default(),
                    &[],
                    false,
                    |row| streamed.push(row.into_vec()),
                )
                .await;

            assert_eq!(streamed.len(), 1);
            assert_eq!(
                streamed[0],
                vec![Value::from(1_u64), Value::from(2_u64), Value::from("alpha")]
            );
        })
    });
}

#[test]
fn table_with_auxiliary_index() {
    run_async_test("table_with_auxiliary_index", || {
        Box::pin(async {
            let root = init_root("aux-index-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, schema_with_index());

            for i in 1..=5u64 {
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(i)],
                        vec![Value::from(i * 10), Value::from(format!("item{i}"))],
                    )
                    .await
                    .expect("insert row");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            assert!(table.contains_row(tx(11), &[Value::from(3_u64)]).await);
            assert_eq!(table.count(tx(11)).await, 5);

            let row = table
                .read_row(tx(11), &[Value::from(2_u64)])
                .await
                .expect("read row");

            assert_eq!(
                row.as_ref(),
                &[
                    Value::from(2_u64),
                    Value::from(20_u64),
                    Value::from("item2")
                ]
            );
        })
    });
}

#[test]
fn large_scan_completes_under_timeout_table() {
    run_async_test("large_scan_completes_under_timeout_table", || {
        Box::pin(async {
            let root = init_root("large-scan-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 0_u64..2_000_u64 {
                table
                    .upsert_row(
                        tx(20),
                        vec![Value::from(i)],
                        vec![Value::from(format!("v{i}"))],
                    )
                    .await
                    .expect("insert large keyset");
            }

            table.commit(tx(20)).await;
            table.finalize(&tx(20)).await;

            let mut seen = 0_u64;
            timeout(Duration::from_secs(10), async {
                table
                    .for_each_row_in_order(
                        tx(21),
                        super::Range::default(),
                        &[],
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
fn same_txn_read_your_own_write_is_non_blocking_table() {
    run_async_test("same_txn_read_your_own_write_is_non_blocking_table", || {
        Box::pin(async {
            let root = init_root("same-txn-rw-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(90),
                    vec![Value::from(1_u64)],
                    vec![Value::from("own")],
                )
                .await
                .expect("insert own key");

            let row = timeout(
                Duration::from_millis(200),
                table.read_row(tx(90), &[Value::from(1_u64)]),
            )
            .await
            .expect("same-txn read-your-own-write should not block");

            assert!(row.is_some());
        })
    });
}

#[test]
fn same_txn_read_your_own_delete_is_non_blocking_table() {
    run_async_test("same_txn_read_your_own_delete_is_non_blocking_table", || {
        Box::pin(async {
            let root = init_root("same-txn-del-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(91),
                    vec![Value::from(1_u64)],
                    vec![Value::from("gone")],
                )
                .await
                .expect("insert key");
            table
                .delete_row(tx(91), vec![Value::from(1_u64)])
                .await
                .expect("delete key in same txn");

            let row = timeout(
                Duration::from_millis(200),
                table.read_row(tx(91), &[Value::from(1_u64)]),
            )
            .await
            .expect("same-txn read-your-own-delete should not block");

            assert!(row.is_none());
        })
    });
}

#[test]
fn repeated_rollback_and_finalize_are_idempotent_table() {
    run_async_test("repeated_rollback_and_finalize_are_idempotent_table", || {
        Box::pin(async {
            let root = init_root("repeated-lifecycle-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(92),
                    vec![Value::from(1_u64)],
                    vec![Value::from("rollback")],
                )
                .await
                .expect("insert key for rollback");

            table.rollback(&tx(92)).await;
            table
                .rollback(&tx(92))
                .await;

            table
                .upsert_row(
                    tx(93),
                    vec![Value::from(2_u64)],
                    vec![Value::from("finalize")],
                )
                .await
                .expect("insert key for finalize");
            table.commit(tx(93)).await;

            table.finalize(&tx(93)).await;
            table
                .finalize(&tx(93))
                .await;

            assert_eq!(table.finalized(), Some(tx(93)));
            assert!(
                table
                    .read_row(tx(94), &[Value::from(2_u64)])
                    .await
                    .is_some()
            );
            assert!(
                table
                    .read_row(tx(94), &[Value::from(1_u64)])
                    .await
                    .is_none()
            );
        })
    });
}

#[test]
fn table_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<PersistentTable>();
    assert_sync::<PersistentTable>();
}

#[test]
fn overlapping_write_in_past_txn_fails_closed_table() {
    run_async_test("overlapping_write_in_past_txn_fails_closed_table", || {
        Box::pin(async {
            let root = init_root("overlapping-write-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(2),
                    vec![Value::from(1_u64)],
                    vec![Value::from("first")],
                )
                .await
                .expect("reserve write at txn 2");

            let err = table
                .upsert_row(
                    tx(1),
                    vec![Value::from(1_u64)],
                    vec![Value::from("second")],
                )
                .await
                .expect_err("expected conflict for out-of-order overlapping write");

            assert_eq!(err, txn_lock::Error::Conflict);
        })
    });
}

#[test]
fn rollback_unblocks_later_read_and_discards_pending_table() {
    run_async_test("rollback_unblocks_later_read_and_discards_pending_table", || {
        Box::pin(async {
            let root = init_root("rollback-unblock-table").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(
                    tx(10),
                    vec![Value::from(1_u64)],
                    vec![Value::from("hot")],
                )
                .await
                .expect("insert pending key");

            assert!(
                timeout(
                    Duration::from_millis(50),
                    table.read_row(tx(11), &[Value::from(1_u64)])
                )
                .await
                .is_err(),
                "later txn read should block while earlier overlapping write is pending"
            );

            table.rollback(&tx(10)).await;

            let row = timeout(
                Duration::from_secs(1),
                table.read_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .expect("later txn read should complete after rollback");

            assert!(row.is_none(), "rolled-back key must not be visible");
        })
    });
}
