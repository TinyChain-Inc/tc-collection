//! Transactional visibility and ordering regression tests for `PersistentTable`.
use super::{Column, ColumnRange, Limited, PersistentTable, Range, Rows, Selection, TableSchema, TableSlice};
use crate::btree::{StorageConfig, PersistentFile};
use freqfs::Cache;
use futures::TryStreamExt;
use std::collections::HashMap;
use std::ops::Bound;
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

fn id(name: &str) -> tc_ir::Id {
    name.parse().expect("Id")
}

fn range_in(col: &str, start: Value, end: Value) -> Range<tc_ir::Id, Value> {
    let mut map = HashMap::new();
    map.insert(
        id(col),
        ColumnRange::In((Bound::Included(start), Bound::Excluded(end))),
    );
    map.into()
}

fn range_eq(col: &str, val: Value) -> Range<tc_ir::Id, Value> {
    let mut map = HashMap::new();
    map.insert(id(col), ColumnRange::Eq(val));
    map.into()
}

async fn collect_rows(rows: Rows) -> Vec<Vec<Value>> {
    let mut result = Vec::new();
    let mut rows = std::pin::pin!(rows);
    while let Some(row) = rows.try_next().await.expect("read row") {
        result.push(row.into_vec());
    }
    result
}

// ---------------------------------------------------------------------------
// §7.5 Query correctness
// ---------------------------------------------------------------------------

#[test]
fn slice_returns_in_range_rows_table() {
    run_async_test("slice_returns_in_range_rows_table", || {
        Box::pin(async {
            let root = init_root("slice-in-range").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=5u64 {
                table
                    .upsert_row(tx(10), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("insert");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let slice = table.slice(range_in("id", Value::from(2_u64), Value::from(5_u64)), &[], false);
            let rows = slice.rows(tx(11)).await.expect("slice rows");
            let collected = collect_rows(rows).await;

            let ids: Vec<Value> = collected.iter().map(|r| r[0].clone()).collect();
            assert_eq!(
                ids,
                vec![Value::from(2_u64), Value::from(3_u64), Value::from(4_u64)],
                "slice should return only in-range rows [2, 5)"
            );

            assert_eq!(slice.count(tx(11)).await, 3);
            assert!(!slice.is_empty(tx(11)).await);
        })
    });
}

#[test]
fn select_projects_columns_table() {
    run_async_test("select_projects_columns_table", || {
        Box::pin(async {
            let root = init_root("select-projects").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(tx(10), vec![Value::from(1_u64)], vec![Value::from("alpha")])
                .await
                .expect("insert");
            table
                .upsert_row(tx(10), vec![Value::from(2_u64)], vec![Value::from("beta")])
                .await
                .expect("insert");
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let selection = table.select(&[id("label")]);
            let rows = selection.rows(tx(11)).await.expect("select rows");
            let collected = collect_rows(rows).await;

            assert_eq!(collected.len(), 2, "should have 2 projected rows");
            assert_eq!(collected[0], vec![Value::from("alpha")], "first row should project label only");
            assert_eq!(collected[1], vec![Value::from("beta")], "second row should project label only");

            assert_eq!(selection.count(tx(11)).await, 2);
            assert!(!selection.is_empty(tx(11)).await);
        })
    });
}

#[test]
fn limit_caps_row_stream_table() {
    run_async_test("limit_caps_row_stream_table", || {
        Box::pin(async {
            let root = init_root("limit-caps").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=10u64 {
                table
                    .upsert_row(tx(10), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("insert");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let limited = table.limit(3);
            assert_eq!(limited.count(tx(11)).await, 3, "count should be capped at 3");

            let rows = limited.rows(tx(11)).await.expect("limited rows");
            let collected = collect_rows(rows).await;
            assert_eq!(collected.len(), 3, "stream should yield exactly 3 rows");

            let limited_zero = table.limit(0);
            assert_eq!(limited_zero.count(tx(11)).await, 0);
            assert!(limited_zero.is_empty(tx(11)).await);

            let limited_large = table.limit(100);
            assert_eq!(limited_large.count(tx(11)).await, 10, "limit larger than table should return all");
        })
    });
}

#[test]
fn order_by_uses_supporting_index_table() {
    run_async_test("order_by_uses_supporting_index_table", || {
        Box::pin(async {
            let root = init_root("order-by-index").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, schema_with_index());

            // Use ref_ids where textual and numeric ordering coincide
            // (tc_value::Value compares numbers by their string representation).
            let data = [
                (1u64, 30u64, "a"),
                (2u64, 10u64, "b"),
                (3u64, 50u64, "c"),
                (4u64, 20u64, "d"),
                (5u64, 40u64, "e"),
            ];
            for (id_val, ref_id, label) in data {
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(id_val)],
                        vec![Value::from(ref_id), Value::from(label)],
                    )
                    .await
                    .expect("insert");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let ordered = table.order_by(&[id("ref_id")], false);
            let rows = ordered.rows(tx(11)).await.expect("ordered rows");
            let collected = collect_rows(rows).await;

            let ref_ids: Vec<Value> = collected.iter().map(|r| r[1].clone()).collect();
            assert_eq!(
                ref_ids,
                vec![
                    Value::from(10_u64),
                    Value::from(20_u64),
                    Value::from(30_u64),
                    Value::from(40_u64),
                    Value::from(50_u64),
                ],
                "rows should be ordered by ref_id ascending"
            );

            // Verify reverse ordering
            let ordered_rev = table.order_by(&[id("ref_id")], true);
            let rows = ordered_rev.rows(tx(11)).await.expect("ordered reverse rows");
            let collected = collect_rows(rows).await;
            let ref_ids_rev: Vec<Value> = collected.iter().map(|r| r[1].clone()).collect();
            assert_eq!(
                ref_ids_rev,
                vec![
                    Value::from(50_u64),
                    Value::from(40_u64),
                    Value::from(30_u64),
                    Value::from(20_u64),
                    Value::from(10_u64),
                ],
                "reverse order_by should flip iteration"
            );
        })
    });
}

#[test]
fn reverse_flips_order_table() {
    run_async_test("reverse_flips_order_table", || {
        Box::pin(async {
            let root = init_root("reverse-flips").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=5u64 {
                table
                    .upsert_row(tx(10), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("insert");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            let forward = table.slice(Range::default(), &[], false);
            let rows = forward.rows(tx(11)).await.expect("forward rows");
            let forward_ids: Vec<Value> = collect_rows(rows).await.iter().map(|r| r[0].clone()).collect();
            assert_eq!(
                forward_ids,
                vec![Value::from(1_u64), Value::from(2_u64), Value::from(3_u64), Value::from(4_u64), Value::from(5_u64)],
            );

            let reverse = table.slice(Range::default(), &[], true);
            let rows = reverse.rows(tx(11)).await.expect("reverse rows");
            let reverse_ids: Vec<Value> = collect_rows(rows).await.iter().map(|r| r[0].clone()).collect();
            assert_eq!(
                reverse_ids,
                vec![Value::from(5_u64), Value::from(4_u64), Value::from(3_u64), Value::from(2_u64), Value::from(1_u64)],
                "reverse should flip iteration order"
            );
        })
    });
}

#[test]
fn unsupported_range_fails_closed_table() {
    run_async_test("unsupported_range_fails_closed_table", || {
        Box::pin(async {
            let root = init_root("unsupported-range").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(tx(10), vec![Value::from(1_u64)], vec![Value::from("alpha")])
                .await
                .expect("insert");
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            // A range on a non-existent column cannot be supported by any index
            // and must fail closed.
            let range = range_eq("nonexistent", Value::from(42_u64));
            let result = table.rows(tx(11), range, vec![], false).await;
            assert!(result.is_err(), "unsupported range should fail closed");

            // Similarly, ordering by a non-existent column must fail closed.
            let result = table.rows(tx(11), Range::default(), vec![id("nonexistent")], false).await;
            assert!(result.is_err(), "unsupported order should fail closed");
        })
    });
}

// ---------------------------------------------------------------------------
// §7.6 Streaming / no-materialization
// ---------------------------------------------------------------------------

#[test]
fn truncate_use_scratch_not_buffer_table() {
    run_async_test("truncate_use_scratch_not_buffer_table", || {
        Box::pin(async {
            let root = init_root("truncate-scratch").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=5u64 {
                table
                    .upsert_row(tx(10), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("insert");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            // Truncate rows with id in [2, 4)
            table
                .truncate(tx(11), range_in("id", Value::from(2_u64), Value::from(4_u64)))
                .await
                .expect("truncate");

            assert!(table.read_row(tx(11), &[Value::from(1_u64)]).await.is_some(), "id 1 should remain");
            assert!(table.read_row(tx(11), &[Value::from(2_u64)]).await.is_none(), "id 2 should be deleted");
            assert!(table.read_row(tx(11), &[Value::from(3_u64)]).await.is_none(), "id 3 should be deleted");
            assert!(table.read_row(tx(11), &[Value::from(4_u64)]).await.is_some(), "id 4 should remain");
            assert!(table.read_row(tx(11), &[Value::from(5_u64)]).await.is_some(), "id 5 should remain");

            assert_eq!(table.count(tx(11)).await, 3, "3 rows should remain after truncate");

            // Commit and finalize the truncate
            table.commit(tx(11)).await;
            table.finalize(&tx(11)).await;

            assert_eq!(table.count(tx(12)).await, 3, "count should persist after finalize");
        })
    });
}

#[test]
fn view_composition_is_lazy_table() {
    run_async_test("view_composition_is_lazy_table", || {
        Box::pin(async {
            let root = init_root("view-composition-lazy").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, schema_with_index());

            for i in 1..=8u64 {
                table
                    .upsert_row(
                        tx(10),
                        vec![Value::from(i)],
                        vec![Value::from(i * 10), Value::from(format!("item{i}"))],
                    )
                    .await
                    .expect("insert");
            }
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            // Compose: slice [3, 7) -> limit 2 -> select ["label"]
            let slice = table.slice(range_in("id", Value::from(3_u64), Value::from(7_u64)), &[], false);
            let limited = slice.limit(2);
            let selection = limited.select(vec![id("label")]);

            let rows = selection.rows(tx(11)).await.expect("composed view rows");
            let collected = collect_rows(rows).await;

            assert_eq!(collected.len(), 2, "limit should cap at 2 rows");
            assert_eq!(
                collected[0],
                vec![Value::from("item3")],
                "first row should be id 3 with only label projected"
            );
            assert_eq!(
                collected[1],
                vec![Value::from("item4")],
                "second row should be id 4 with only label projected"
            );

            // Also verify count on composed view
            assert_eq!(limited.count(tx(11)).await, 2, "limited count should be 2");
        })
    });
}

// ---------------------------------------------------------------------------
// §7.1 Lifecycle — no-op paths release reservations
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_noop_paths_release_reservations_table() {
    run_async_test("lifecycle_noop_paths_release_reservations_table", || {
        Box::pin(async {
            let root = init_root("lifecycle-noop-reservations").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(tx(10), vec![Value::from(1_u64)], vec![Value::from("a")])
                .await
                .expect("insert");
            table.commit(tx(10)).await;

            // Duplicate commit (no-op) — should release reservation.
            table.commit(tx(10)).await;

            // Finalize tx(10), then duplicate finalize (stale no-op).
            table.finalize(&tx(10)).await;
            assert_eq!(table.finalized(), Some(tx(10)));
            table.finalize(&tx(10)).await;
            assert_eq!(table.finalized(), Some(tx(10)));

            // After all no-op paths, a later read should not be blocked.
            let row = timeout(
                Duration::from_secs(1),
                table.read_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .expect("later read should not be blocked after no-op lifecycle paths");

            assert!(row.is_some(), "row should be visible");

            // Write at a new txn should succeed (no leaked reservations).
            table
                .upsert_row(tx(12), vec![Value::from(2_u64)], vec![Value::from("b")])
                .await
                .expect("write should succeed after no-op paths");
        })
    });
}

// ---------------------------------------------------------------------------
// §7.2 Visibility — overlapping read blocks until earlier finalize
// ---------------------------------------------------------------------------

#[test]
fn overlapping_read_blocks_until_earlier_finalize_table() {
    run_async_test("overlapping_read_blocks_until_earlier_finalize_table", || {
        Box::pin(async {
            let root = init_root("overlapping-read-finalize").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            table
                .upsert_row(tx(10), vec![Value::from(1_u64)], vec![Value::from("hot")])
                .await
                .expect("insert pending");

            // Later read should block while earlier pending write is active
            assert!(
                timeout(Duration::from_millis(50), table.read_row(tx(11), &[Value::from(1_u64)]))
                    .await
                    .is_err(),
                "later read should block while earlier pending write is active"
            );

            // Commit + finalize the earlier txn
            table.commit(tx(10)).await;
            table.finalize(&tx(10)).await;

            // Now the later read should complete
            let row = timeout(
                Duration::from_secs(1),
                table.read_row(tx(11), &[Value::from(1_u64)]),
            )
            .await
            .expect("later read should complete after earlier finalize");

            assert!(row.is_some(), "row should be visible after finalize");
        })
    });
}

// ---------------------------------------------------------------------------
// §7.8 Concurrency / locking
// ---------------------------------------------------------------------------

#[test]
fn concurrent_read_write_finalize_table() {
    run_async_test("concurrent_read_write_finalize_table", || {
        Box::pin(async {
            let root = init_root("concurrent-rwf").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            // Seed the table
            for i in 1..=20u64 {
                table
                    .upsert_row(tx(1), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("seed insert");
            }
            table.commit(tx(1)).await;
            table.finalize(&tx(1)).await;

            let t_read = table.clone();
            let t_write = table.clone();
            let t_finalize = table.clone();

            // Spawn concurrent operations with a timeout to detect deadlock
            let result = timeout(Duration::from_secs(5), async {
                let read_task = tokio::spawn(async move {
                    for i in 1..=20u64 {
                        let _ = t_read.read_row(tx(100), &[Value::from(i)]).await;
                    }
                });

                let write_task = tokio::spawn(async move {
                    for i in 1..=10u64 {
                        let _ = t_write
                            .upsert_row(tx(50), vec![Value::from(i)], vec![Value::from("updated")])
                            .await;
                    }
                    t_write.commit(tx(50)).await;
                });

                let finalize_task = tokio::spawn(async move {
                    t_finalize.finalize(&tx(1)).await;
                });

                let _ = tokio::join!(read_task, write_task, finalize_task);
            })
            .await;

            assert!(result.is_ok(), "concurrent read/write/finalize should not deadlock");
        })
    });
}

#[test]
fn lock_order_no_deadlock_table() {
    run_async_test("lock_order_no_deadlock_table", || {
        Box::pin(async {
            let root = init_root("lock-order-no-deadlock").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            for i in 1..=10u64 {
                table
                    .upsert_row(tx(1), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("seed");
            }
            table.commit(tx(1)).await;
            table.finalize(&tx(1)).await;

            // Interleave commit, finalize, read, and write at different txns
            // to exercise lock ordering. If lock order is canonical, no deadlock.
            let result = timeout(Duration::from_secs(3), async {
                let t1 = table.clone();
                let t2 = table.clone();

                let w1 = tokio::spawn(async move {
                    t1.upsert_row(tx(10), vec![Value::from(1_u64)], vec![Value::from("w1")])
                        .await
                        .expect("w1");
                    t1.commit(tx(10)).await;
                    t1.finalize(&tx(10)).await;
                });

                let w2 = tokio::spawn(async move {
                    t2.upsert_row(tx(20), vec![Value::from(2_u64)], vec![Value::from("w2")])
                        .await
                        .expect("w2");
                    t2.commit(tx(20)).await;
                    t2.finalize(&tx(20)).await;
                });

                let _ = tokio::join!(w1, w2);
            })
            .await;

            assert!(result.is_ok(), "interleaved operations should not deadlock");
        })
    });
}

#[test]
fn finalize_sync_drops_guard_first_table() {
    run_async_test("finalize_sync_drops_guard_first_table", || {
        Box::pin(async {
            let root = init_root("finalize-drops-guard").await;
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::new(persistent, txn, simple_schema());

            // Insert rows in multiple txns, then finalize in order.
            // If finalize holds the state guard while applying deltas (which
            // acquire table read/write locks), it would deadlock. This test
            // verifies that finalize drops the guard before applying deltas.
            for i in 1..=5u64 {
                table
                    .upsert_row(tx(10), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("insert txn 10");
            }
            table.commit(tx(10)).await;

            for i in 6..=10u64 {
                table
                    .upsert_row(tx(11), vec![Value::from(i)], vec![Value::from(format!("v{i}"))])
                    .await
                    .expect("insert txn 11");
            }
            table.commit(tx(11)).await;

            // Finalize should merge both committed deltas without deadlock
            let result = timeout(Duration::from_secs(2), table.finalize(&tx(11))).await;
            assert!(result.is_ok(), "finalize should complete without deadlock");

            // Verify all rows are visible after finalize
            assert_eq!(table.count(tx(12)).await, 10, "all 10 rows should be visible after finalize");
            assert_eq!(table.finalized(), Some(tx(11)));
        })
    });
}

// ---------------------------------------------------------------------------
// Rows stream — Send + Sync assertions
// ---------------------------------------------------------------------------

#[test]
fn rows_and_views_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<Rows>();
    assert_send::<TableSlice>();
    assert_sync::<TableSlice>();
    assert_send::<Limited>();
    assert_sync::<Limited>();
    assert_send::<Selection>();
    assert_sync::<Selection>();
}
