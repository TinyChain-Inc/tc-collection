//! Transactional visibility and ordering regression tests for `PersistentTable`.
use std::collections::HashMap;
use std::ops::Bound;
use std::sync::Arc;

use b_table::Row;
use futures::TryStreamExt;
use futures::future::join_all;
use tokio::sync::{Barrier, mpsc, oneshot};
use tokio::time::{Duration, timeout};

use super::{
    Column, ColumnRange, Limited, PersistentTable, Range, Rows, Selection, TableSchema, TableSlice,
};
use crate::PersistentFile;
use crate::btree::StorageConfig;
use crate::test::{TestTxn, init_root, load_roots, tx};
use tc_value::{Value, ValueType};

fn simple_schema() -> TableSchema {
    let key = vec![Column {
        name: "id".parse().expect("Id"),
        dtype: ValueType::Number(tc_value::NumberType::Number),
    }];

    let values = vec![Column {
        name: "label".parse().expect("Id"),
        dtype: ValueType::String,
    }];

    TableSchema::new(key, values, Vec::new(), StorageConfig::default()).expect("create test schema")
}

fn composite_schema() -> TableSchema {
    let key = vec![
        Column {
            name: "part_a".parse().expect("Id"),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        },
        Column {
            name: "part_b".parse().expect("Id"),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        },
    ];

    let values = vec![Column {
        name: "val1".parse().expect("Id"),
        dtype: ValueType::String,
    }];

    TableSchema::new(key, values, Vec::new(), StorageConfig::default())
        .expect("create composite key schema")
}

fn schema_with_index() -> TableSchema {
    let key = vec![Column {
        name: "id".parse().expect("Id"),
        dtype: ValueType::Number(tc_value::NumberType::Number),
    }];

    let values = vec![
        Column {
            name: "ref_id".parse().expect("Id"),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        },
        Column {
            name: "label".parse().expect("Id"),
            dtype: ValueType::String,
        },
    ];

    let indices = vec![("by_ref".to_string(), vec!["ref_id".parse().expect("Id")])];
    TableSchema::new(key, values, indices, StorageConfig::default())
        .expect("create schema with index")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn upsert_inserts_and_updates_pending_row() {
    Box::pin(async {
        let root = init_root("upsert-pending");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("upsert row");

        let row = timeout(
            Duration::from_millis(200),
            table.read_row(tx(10), &[Value::from(1_u64)]),
        )
        .await
        .expect("same-txn read-your-own-write should not block")
        .unwrap();

        assert!(row.is_some());
        assert_eq!(
            row.unwrap().as_ref(),
            &[Value::from(1_u64), Value::from("alpha")]
        );

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("beta")],
            )
            .await
            .expect("update row");

        let row = table.read_row(tx(10), &[Value::from(1_u64)]).await.unwrap();
        assert_eq!(
            row.unwrap().as_ref(),
            &[Value::from(1_u64), Value::from("beta")]
        );

        assert!(
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from("not a number")],
                    vec![Value::from("invalid")],
                )
                .await
                .is_err()
        );

        let row = table.read_row(tx(10), &[Value::from(1_u64)]).await.unwrap();
        let row = row.unwrap();
        assert_eq!(row.as_ref(), &[Value::from(1_u64), Value::from("beta")]);
        assert_eq!(row[0].class(), Value::from(1_u64).class());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn delete_moves_row_to_pending_deletes() {
    Box::pin(async {
        let root = init_root("delete-pending");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("upsert row");

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        table
            .delete_row(&txn.with_id(tx(11)), vec![Value::from(1_u64)])
            .await
            .expect("delete row");

        for key in [1_u64, 2] {
            if key == 2 {
                table
                    .upsert_row(
                        &txn.with_id(tx(11)),
                        vec![Value::from(key)],
                        vec![Value::from("pending")],
                    )
                    .await
                    .expect("insert pending row");

                table
                    .delete_row(&txn.with_id(tx(11)), vec![Value::from(key)])
                    .await
                    .expect("delete pending row in the same transaction");
            }

            let row = timeout(
                Duration::from_millis(200),
                table.read_row(tx(11), &[Value::from(key)]),
            )
            .await
            .expect("same-txn read-your-own-delete should not block")
            .unwrap();

            assert!(row.is_none(), "deleted row should not be visible");
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn commit_promotes_pending_to_committed() {
    Box::pin(async {
        let root = init_root("commit-promotes");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("upsert row");

        assert!(
            table
                .read_row(tx(9), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_none(),
            "committed row should not be visible to earlier txn"
        );

        assert!(
            table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "pending row should be visible to own txn"
        );

        let mut result = table
            .rows(tx(10), Range::default(), vec![], false)
            .await
            .unwrap();

        table.commit(tx(10)).await.expect("commit");
        table
            .upsert_row(
                &txn.with_id(tx(12)),
                vec![Value::from(1_u64)],
                vec![Value::from("beta")],
            )
            .await
            .unwrap();

        table.commit(tx(12)).await.unwrap();
        assert_eq!(
            result.try_next().await.unwrap().unwrap().as_slice(),
            &[Value::from(1_u64), Value::from("alpha")]
        );

        assert!(result.try_next().await.unwrap().is_none());

        assert!(
            table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "committed row should be visible at commit txn"
        );

        assert!(
            table
                .read_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "committed delta should be visible to later txn (committed <= T)"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn rollback_discards_pending_and_unblocks() {
    Box::pin(async {
        let root = init_root("rollback-discards");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
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

        table.rollback(tx(10)).expect("rollback");

        let row = timeout(
            Duration::from_secs(1),
            table.read_row(tx(11), &[Value::from(1_u64)]),
        )
        .await
        .expect("later txn read should complete after rollback");

        assert!(
            row.unwrap().is_none(),
            "rolled-back row must not be visible"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn finalize_merges_committed_into_canon() {
    Box::pin(async {
        let root = init_root("finalize-merges");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("upsert row");

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        assert!(
            table
                .read_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "finalized row should be visible to later txn"
        );

        assert_eq!(table.finalized().unwrap(), Some(tx(10)));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn duplicate_commit_is_idempotent_table() {
    Box::pin(async {
        let root = init_root("duplicate-commit-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(42_u64)],
                vec![Value::from("x")],
            )
            .await
            .expect("upsert row");

        table.commit(tx(10)).await.expect("commit");
        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        assert!(
            table
                .read_row(tx(11), &[Value::from(42_u64)])
                .await
                .unwrap()
                .is_some(),
            "row should be visible after duplicate commit"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stale_finalize_is_noop_table() {
    Box::pin(async {
        let root = init_root("stale-finalize-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("x")],
            )
            .await
            .expect("upsert row");

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        table.finalize(tx(9)).await.expect("stale finalize no-op");

        assert_eq!(table.finalized().unwrap(), Some(tx(10)));
        assert!(
            table
                .read_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "row should still be visible after stale finalize"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cannot_write_after_commit_or_finalize_table() {
    Box::pin(async {
        let root = init_root("write-after-finalize-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("a")],
            )
            .await
            .expect("upsert row");

        table.commit(tx(10)).await.expect("commit");

        assert_eq!(
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(1_u64)],
                    vec![Value::from("b")]
                )
                .await
                .unwrap_err()
                .message(),
            txn_lock::Error::Committed.to_string()
        );

        let row = timeout(
            Duration::from_millis(200),
            table.read_row(tx(11), &[1_u64.into()]),
        )
        .await
        .expect("rejected committed write must not reserve its key")
        .unwrap()
        .unwrap();

        assert_eq!(row.as_slice(), &[1_u64.into(), "a".into()]);

        table.finalize(tx(10)).await.expect("finalize");
        assert_eq!(
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(1_u64)],
                    vec![Value::from("c")]
                )
                .await
                .unwrap_err()
                .message(),
            txn_lock::Error::Outdated.to_string()
        );

        let row = timeout(
            Duration::from_millis(200),
            table.read_row(tx(12), &[1_u64.into()]),
        )
        .await
        .expect("rejected finalized write must not reserve its key")
        .unwrap()
        .unwrap();

        assert_eq!(row.as_slice(), &[1_u64.into(), "a".into()]);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn pending_is_visible_only_to_its_txn_table() {
    Box::pin(async {
        let root = init_root("pending-visible-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("insert pending");

        assert!(
            table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "pending row should be visible to own txn"
        );

        assert!(
            table
                .read_row(tx(9), &[Value::from(1_u64)])
                .await
                .unwrap()
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

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let row = timeout(
            Duration::from_secs(1),
            table.read_row(tx(11), &[Value::from(1_u64)]),
        )
        .await
        .expect("later txn read should complete after finalize");

        assert!(row.unwrap().is_some());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn committed_is_visible_in_txn_order_table() {
    Box::pin(async {
        let root = init_root("committed-visible-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("insert key");

        table.commit(tx(10)).await.expect("commit");

        assert!(
            table
                .read_row(tx(9), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_none(),
            "committed row should not be visible to earlier txn"
        );

        assert!(
            table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "committed row should be visible at commit txn"
        );

        table.finalize(tx(10)).await.expect("finalize");
        for stale in [tx(9), tx(10)] {
            let error = table.read_row(stale, &[1_u64.into()]).await.unwrap_err();
            assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
            let error = table
                .rows(stale, Range::default(), Vec::new(), false)
                .await
                .err()
                .expect("finalized history is unavailable");

            assert_eq!(error.message(), txn_lock::Error::Outdated.to_string());
        }

        assert!(
            table
                .read_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "finalized row should be visible to later txn"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn no_pending_leakage_across_txns_table() {
    Box::pin(async {
        let root = init_root("no-leakage-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("a")],
            )
            .await
            .expect("txn 10 insert");

        table
            .upsert_row(
                &txn.with_id(tx(12)),
                vec![Value::from(2_u64)],
                vec![Value::from("b")],
            )
            .await
            .expect("txn 12 insert");

        assert!(
            table
                .read_row(tx(10), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "txn 10 should see its own pending write"
        );

        assert!(
            table
                .read_row(tx(9), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_none(),
            "earlier txn should not see txn 10's pending write"
        );

        assert!(
            table
                .read_row(tx(10), &[Value::from(2_u64)])
                .await
                .unwrap()
                .is_none(),
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
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn read_resolves_delta_stack_table() {
    Box::pin(async {
        let root = init_root("delta-stack-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("original")],
            )
            .await
            .expect("insert original");

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        table
            .upsert_row(
                &txn.with_id(tx(11)),
                vec![Value::from(1_u64)],
                vec![Value::from("updated")],
            )
            .await
            .expect("update in txn 11");

        let row = table
            .read_row(tx(11), &[Value::from(1_u64)])
            .await
            .unwrap()
            .expect("read updated row");

        assert_eq!(
            row.as_ref(),
            &[Value::from(1_u64), Value::from("updated")],
            "should see the delta update, not the canon version"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn streamed_rows_match_materialized_table() {
    Box::pin(async {
        let root = init_root("streamed-materialized-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("label{i}"))],
                )
                .await
                .expect("insert row");
        }

        table
            .delete_row(&txn.with_id(tx(10)), vec![Value::from(2_u64)])
            .await
            .expect("delete row");

        table.commit(tx(10)).await.expect("commit");

        table
            .upsert_row(
                &txn.with_id(tx(11)),
                vec![Value::from(7_u64)],
                vec![Value::from("label7")],
            )
            .await
            .expect("insert key 7");

        table
            .delete_row(&txn.with_id(tx(11)), vec![Value::from(3_u64)])
            .await
            .expect("delete key 3");

        let mut streamed = Vec::new();
        table
            .for_each_row_in_order(tx(11), super::Range::default(), &[], false, |row| {
                streamed.push(row.into_vec());
            })
            .await
            .unwrap();

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
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn count_matches_streamed_fold_table() {
    Box::pin(async {
        let root = init_root("count-fold-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=10u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert row");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        for i in (1..=10u64).step_by(2) {
            table
                .delete_row(&txn.with_id(tx(11)), vec![Value::from(i)])
                .await
                .expect("delete odd row");
        }

        assert_eq!(table.count(tx(11)).await.unwrap(), 5);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn contains_all_key_range_table() {
    Box::pin(async {
        let root = init_root("contains-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert row");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        assert!(
            table
                .contains_row(tx(11), &[Value::from(3_u64)])
                .await
                .unwrap()
        );

        assert!(
            !table
                .contains_row(tx(11), &[Value::from(99_u64)])
                .await
                .unwrap()
        );

        assert!(!table.is_empty(tx(11)).await.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn empty_table_semantics_across_lifecycle_table() {
    Box::pin(async {
        let root = init_root("empty-table-lifecycle-table");
        let (persistent, _) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        assert!(table.is_empty(tx(95)).await.unwrap());
        assert_eq!(table.count(tx(95)).await.unwrap(), 0);
        assert!(
            table
                .read_row(tx(95), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_none()
        );

        table.commit(tx(95)).await.expect("commit");
        table.finalize(tx(95)).await.expect("finalize");

        assert!(table.is_empty(tx(96)).await.unwrap());
        assert_eq!(table.count(tx(96)).await.unwrap(), 0);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn invalid_key_arity_fails_closed_table() {
    Box::pin(async {
        let root = init_root("invalid-arity-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        let err = table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64), Value::from(2_u64)],
                vec![Value::from("a")],
            )
            .await
            .expect_err("wrong key arity should fail");

        assert_eq!(err.code(), tc_error::ErrorKind::BadRequest);

        let err = table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("a"), Value::from("b")],
            )
            .await
            .expect_err("wrong value arity should fail");

        assert_eq!(err.code(), tc_error::ErrorKind::BadRequest);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn invalid_key_type_fails_closed_table() {
    Box::pin(async {
        let root = init_root("invalid-key-type-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        let err = table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from("not_a_number")],
                vec![Value::from("a")],
            )
            .await
            .expect_err("wrong key type should fail");

        assert_eq!(err.code(), tc_error::ErrorKind::BadRequest);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn composite_key_upsert_and_read() {
    Box::pin(async {
        let root = init_root("composite-key-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, composite_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64), Value::from(2_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("upsert composite key");

        let row = table
            .read_row(tx(10), &[Value::from(1_u64), Value::from(2_u64)])
            .await
            .unwrap()
            .expect("read composite key");

        assert_eq!(
            row.as_ref(),
            &[Value::from(1_u64), Value::from(2_u64), Value::from("alpha")]
        );

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let mut streamed = Vec::new();
        table
            .for_each_row_in_order(tx(11), super::Range::default(), &[], false, |row| {
                streamed.push(row.into_vec())
            })
            .await
            .unwrap();

        assert_eq!(streamed.len(), 1);
        assert_eq!(
            streamed[0],
            vec![Value::from(1_u64), Value::from(2_u64), Value::from("alpha")]
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn table_with_auxiliary_index() {
    Box::pin(async {
        let root = init_root("aux-index-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, schema_with_index())
            .await
            .unwrap();

        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(i * 10), Value::from(format!("item{i}"))],
                )
                .await
                .expect("insert row");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        assert!(
            table
                .contains_row(tx(11), &[Value::from(3_u64)])
                .await
                .unwrap()
        );

        assert_eq!(table.count(tx(11)).await.unwrap(), 5);

        let row = table
            .read_row(tx(11), &[Value::from(2_u64)])
            .await
            .unwrap()
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
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn large_scan_completes_under_timeout_table() {
    Box::pin(async {
        let root = init_root("large-scan-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 0_u64..2_000_u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(20)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert large keyset");
        }

        table.commit(tx(20)).await.expect("commit");
        table.finalize(tx(20)).await.expect("finalize");

        let mut seen = 0_u64;
        timeout(Duration::from_secs(10), async {
            table
                .for_each_row_in_order(tx(21), super::Range::default(), &[], false, |_| {
                    seen += 1;
                })
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
async fn repeated_rollback_and_finalize_are_idempotent_table() {
    Box::pin(async {
        let root = init_root("repeated-lifecycle-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(92)),
                vec![Value::from(1_u64)],
                vec![Value::from("rollback")],
            )
            .await
            .expect("insert key for rollback");

        table.rollback(tx(92)).expect("rollback");
        table.rollback(tx(92)).expect("second rollback no-op");

        table
            .upsert_row(
                &txn.with_id(tx(93)),
                vec![Value::from(2_u64)],
                vec![Value::from("finalize")],
            )
            .await
            .expect("insert key for finalize");

        table.commit(tx(93)).await.expect("commit");

        table.finalize(tx(93)).await.expect("finalize");
        table.finalize(tx(93)).await.expect("second finalize no-op");

        assert_eq!(table.finalized().unwrap(), Some(tx(93)));
        assert!(
            table
                .read_row(tx(94), &[Value::from(2_u64)])
                .await
                .unwrap()
                .is_some()
        );

        assert!(
            table
                .read_row(tx(94), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_none()
        );
    })
    .await;
}

#[test]
fn table_is_send_and_sync() {
    fn assert_send<T: Send>() {}

    fn assert_sync<T: Sync>() {}

    assert_send::<PersistentTable<TestTxn>>();
    assert_sync::<PersistentTable<TestTxn>>();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn overlapping_write_in_past_txn_fails_closed_table() {
    Box::pin(async {
        let root = init_root("overlapping-write-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(2)),
                vec![Value::from(1_u64)],
                vec![Value::from("first")],
            )
            .await
            .expect("reserve write at txn 2");

        let err = table
            .upsert_row(
                &txn.with_id(tx(1)),
                vec![Value::from(1_u64)],
                vec![Value::from("second")],
            )
            .await
            .expect_err("expected conflict for out-of-order overlapping write");

        assert_eq!(err.code(), tc_error::ErrorKind::Conflict);
    })
    .await;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slice_returns_in_range_rows_table() {
    Box::pin(async {
        let root = init_root("slice-in-range");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let slice = table.slice(
            range_in("id", Value::from(2_u64), Value::from(5_u64)),
            &[],
            false,
        );

        let rows = slice.rows(tx(11)).await.expect("slice rows");
        let collected = collect_rows(rows).await;

        let ids: Vec<Value> = collected.iter().map(|r| r[0].clone()).collect();
        assert_eq!(
            ids,
            vec![Value::from(2_u64), Value::from(3_u64), Value::from(4_u64)],
            "slice should return only in-range rows [2, 5)"
        );

        assert_eq!(slice.count(tx(11)).await.expect("count slice"), 3);
        assert!(!slice.is_empty(tx(11)).await.expect("test slice"));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn select_projects_columns_table() {
    Box::pin(async {
        // Projection consumes owned fields; only repeated output columns need copies.
        for names in [
            vec!["label"],
            vec!["label", "id", "label"],
            vec![],
            vec!["missing"],
        ] {
            let payload = String::from("owned projection payload");
            let pointer = payload.as_ptr();
            let row: b_table::Row<Value> = vec![Value::from(1_u64), Value::from(payload)].into();
            let rows = Rows::local(Box::pin(futures::stream::iter([Ok(row)])));
            let columns: Vec<_> = names.iter().map(|name| id(name)).collect();
            let mut rows = rows.select(&simple_schema(), &columns);
            let projected = rows.try_next().await.unwrap().unwrap();
            let expected: Vec<_> = names
                .iter()
                .filter_map(|name| match *name {
                    "id" => Some(Value::from(1_u64)),
                    "label" => Some(Value::from("owned projection payload")),
                    _ => None,
                })
                .collect();

            assert_eq!(projected.as_slice(), expected.as_slice());
            if let Some(last) = names.iter().rposition(|name| *name == "label") {
                let Value::String(value) = &projected[last] else {
                    panic!("expected payload")
                };

                assert_eq!(
                    value.as_ptr(),
                    pointer,
                    "last use moves the original payload"
                );
            }

            assert!(rows.try_next().await.unwrap().is_none());
        }

        let root = init_root("select-projects");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("insert");

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(2_u64)],
                vec![Value::from("beta")],
            )
            .await
            .expect("insert");

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let selection = table
            .select(&[id("id"), id("label")])
            .expect("valid selection");

        let rows = selection.rows(tx(11)).await.expect("select rows");
        let collected = collect_rows(rows).await;

        assert_eq!(collected.len(), 2, "should have 2 projected rows");
        assert_eq!(
            collected[0],
            vec![Value::from(1_u64), Value::from("alpha")],
            "first row should preserve its key"
        );

        assert_eq!(
            collected[1],
            vec![Value::from(2_u64), Value::from("beta")],
            "second row should preserve its key"
        );

        assert_eq!(selection.count(tx(11)).await.expect("count selection"), 2);
        assert!(!selection.is_empty(tx(11)).await.expect("test selection"));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn limit_caps_row_stream_table() {
    Box::pin(async {
        let root = init_root("limit-caps");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=10u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let limited = table.limit(3);
        assert_eq!(
            limited.count(tx(11)).await.expect("count limited view"),
            3,
            "count should be capped at 3"
        );

        let rows = limited.rows(tx(11)).await.expect("limited rows");
        let collected = collect_rows(rows).await;
        assert_eq!(collected.len(), 3, "stream should yield exactly 3 rows");

        let limited_zero = table.limit(0);
        assert_eq!(
            limited_zero
                .count(tx(11))
                .await
                .expect("count empty limited view"),
            0
        );

        assert!(
            limited_zero
                .is_empty(tx(11))
                .await
                .expect("test empty limited view")
        );

        let limited_large = table.limit(100);
        assert_eq!(
            limited_large
                .count(tx(11))
                .await
                .expect("count large limited view"),
            10,
            "limit larger than table should return all"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn order_by_uses_supporting_index_table() {
    Box::pin(async {
        let root = init_root("order-by-index");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, schema_with_index())
            .await
            .unwrap();

        let data = [
            (1u64, 11_000u64, "a"),
            (2u64, 9_000u64, "b"),
            (3u64, 20_000u64, "c"),
            (4u64, 10_000u64, "d"),
            (5u64, 12_000u64, "e"),
        ];

        for (id_val, ref_id, label) in data {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(id_val)],
                    vec![Value::from(ref_id), Value::from(label)],
                )
                .await
                .expect("insert");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let ordered = table.order_by(&[id("ref_id")], false);
        let rows = ordered.rows(tx(11)).await.expect("ordered rows");
        let collected = collect_rows(rows).await;

        let ref_ids: Vec<Value> = collected.iter().map(|r| r[1].clone()).collect();
        assert_eq!(
            ref_ids,
            vec![
                Value::from(9_000_u64),
                Value::from(10_000_u64),
                Value::from(11_000_u64),
                Value::from(12_000_u64),
                Value::from(20_000_u64),
            ],
            "rows should be ordered by ref_id ascending"
        );

        // Verify reverse ordering
        let ordered_rev = table.order_by(&[id("ref_id")], true);
        let rows = ordered_rev
            .rows(tx(11))
            .await
            .expect("ordered reverse rows");

        let collected = collect_rows(rows).await;
        let ref_ids_rev: Vec<Value> = collected.iter().map(|r| r[1].clone()).collect();
        assert_eq!(
            ref_ids_rev,
            vec![
                Value::from(20_000_u64),
                Value::from(12_000_u64),
                Value::from(11_000_u64),
                Value::from(10_000_u64),
                Value::from(9_000_u64),
            ],
            "reverse order_by should flip iteration"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn reverse_flips_order_table() {
    Box::pin(async {
        let root = init_root("reverse-flips");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        let forward = table.slice(Range::default(), &[], false);
        let rows = forward.rows(tx(11)).await.expect("forward rows");
        let forward_ids: Vec<Value> = collect_rows(rows)
            .await
            .iter()
            .map(|r| r[0].clone())
            .collect();

        assert_eq!(
            forward_ids,
            vec![
                Value::from(1_u64),
                Value::from(2_u64),
                Value::from(3_u64),
                Value::from(4_u64),
                Value::from(5_u64)
            ],
        );

        let reverse = table.slice(Range::default(), &[], true);
        let rows = reverse.rows(tx(11)).await.expect("reverse rows");
        let reverse_ids: Vec<Value> = collect_rows(rows)
            .await
            .iter()
            .map(|r| r[0].clone())
            .collect();

        assert_eq!(
            reverse_ids,
            vec![
                Value::from(5_u64),
                Value::from(4_u64),
                Value::from(3_u64),
                Value::from(2_u64),
                Value::from(1_u64)
            ],
            "reverse should flip iteration order"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn unsupported_range_fails_closed_table() {
    Box::pin(async {
        let root = init_root("unsupported-range");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("alpha")],
            )
            .await
            .expect("insert");

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        // A range on a non-existent column cannot be supported by any index
        // and must fail closed.
        let range = range_eq("nonexistent", Value::from(42_u64));
        let result = table.rows(tx(11), range, vec![], false).await;
        assert!(result.is_err(), "unsupported range should fail closed");

        // Similarly, ordering by a non-existent column must fail closed.
        let result = table
            .rows(tx(11), Range::default(), vec![id("nonexistent")], false)
            .await;

        assert!(result.is_err(), "unsupported order should fail closed");

        // Both columns exist, but a constraint after a range cannot share one scan.
        let mut range = range_in("id", Value::from(0_u64), Value::from(3_u64)).into_inner();
        range.insert(id("label"), ColumnRange::Eq(Value::from("alpha")));
        let range = Range::from_iter(range);
        assert!(
            table
                .rows(tx(11), range.clone(), vec![], false)
                .await
                .is_err()
        );

        assert!(table.truncate(&txn.with_id(tx(11)), range).await.is_err());
        table
            .rollback(tx(11))
            .expect("release rejected mutation reservations");

        let supported = range_in("id", Value::from(0_u64), Value::from(3_u64));
        let rows = table
            .rows(tx(12), supported, vec![id("id")], false)
            .await
            .unwrap();

        assert_eq!(
            collect_rows(rows).await,
            vec![vec![Value::from(1_u64), Value::from("alpha")]]
        );

        assert!(
            table
                .rows(tx(12), Range::default(), vec![id("label")], false)
                .await
                .is_err()
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// §7.6 Streaming / no-materialization
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn truncate_use_scratch_not_buffer_table() {
    Box::pin(async {
        let root = init_root("truncate-scratch");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        // Truncate rows with id in [2, 4)
        table
            .truncate(
                &txn.with_id(tx(11)),
                range_in("id", Value::from(2_u64), Value::from(4_u64)),
            )
            .await
            .expect("truncate");

        assert!(
            table
                .read_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_some(),
            "id 1 should remain"
        );

        assert!(
            table
                .read_row(tx(11), &[Value::from(2_u64)])
                .await
                .unwrap()
                .is_none(),
            "id 2 should be deleted"
        );

        assert!(
            table
                .read_row(tx(11), &[Value::from(3_u64)])
                .await
                .unwrap()
                .is_none(),
            "id 3 should be deleted"
        );

        assert!(
            table
                .read_row(tx(11), &[Value::from(4_u64)])
                .await
                .unwrap()
                .is_some(),
            "id 4 should remain"
        );

        assert!(
            table
                .read_row(tx(11), &[Value::from(5_u64)])
                .await
                .unwrap()
                .is_some(),
            "id 5 should remain"
        );

        assert_eq!(
            table.count(tx(11)).await.unwrap(),
            3,
            "3 rows should remain after truncate"
        );

        // Commit and finalize the truncate
        table.commit(tx(11)).await.expect("commit");
        table.finalize(tx(11)).await.expect("finalize");

        assert_eq!(
            table.count(tx(12)).await.unwrap(),
            3,
            "count should persist after finalize"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn view_composition_is_lazy_table() {
    Box::pin(async {
        let root = init_root("view-composition-lazy");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, schema_with_index())
            .await
            .unwrap();

        for i in 1..=8u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(i * 10), Value::from(format!("item{i}"))],
                )
                .await
                .expect("insert");
        }

        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        // Compose: slice [3, 7) -> limit 2 -> select the key and label.
        let slice = table.slice(
            range_in("id", Value::from(3_u64), Value::from(7_u64)),
            &[],
            false,
        );

        let limited = slice.limit(2);
        let selection = limited
            .select(vec![id("id"), id("label")])
            .expect("valid selection");

        let rows = selection.rows(tx(11)).await.expect("composed view rows");
        let collected = collect_rows(rows).await;

        assert_eq!(collected.len(), 2, "limit should cap at 2 rows");
        assert_eq!(
            collected[0],
            vec![Value::from(3_u64), Value::from("item3")],
            "first row should be id 3"
        );

        assert_eq!(
            collected[1],
            vec![Value::from(4_u64), Value::from("item4")],
            "second row should be id 4"
        );

        assert_eq!(selection.count(tx(11)).await.unwrap(), 2);
        assert!(!selection.is_empty(tx(11)).await.unwrap());
        let mut visited = Vec::new();
        selection
            .for_each_row_in_order(tx(11), |row| visited.push(row.into_vec()))
            .await
            .unwrap();

        assert_eq!(visited, collected);

        for limit in [0, 1, 4, 20] {
            let limited = slice.limit(limit);
            assert!(
                limited
                    .select(vec![id("id"), id("label"), id("label")])
                    .is_err()
            );

            let selected = limited.select(vec![id("id"), id("label")]).unwrap();

            let expected = collect_rows(selected.rows(tx(11)).await.unwrap()).await;
            assert_eq!(expected.len(), (limit as usize).min(4));
            assert_eq!(selected.count(tx(11)).await.unwrap(), expected.len() as u64);
            assert_eq!(
                selected.is_empty(tx(11)).await.unwrap(),
                expected.is_empty()
            );

            let mut visited = Vec::new();
            selected
                .for_each_row_in_order(tx(11), |row| visited.push(row.into_vec()))
                .await
                .unwrap();

            assert_eq!(visited, expected);
        }

        for limit in [0, 2] {
            let limited = table.limit(limit);
            let selected = limited.select(vec![id("id"), id("label")]).unwrap();
            assert!(limited.count(tx(10)).await.is_err());
            assert!(limited.is_empty(tx(10)).await.is_err());
            assert!(
                limited
                    .for_each_row_in_order(tx(10), |_| panic!("outdated callback"))
                    .await
                    .is_err()
            );

            assert!(selected.count(tx(10)).await.is_err());
            assert!(selected.is_empty(tx(10)).await.is_err());
            assert!(
                selected
                    .for_each_row_in_order(tx(10), |_| panic!("outdated callback"))
                    .await
                    .is_err()
            );
        }

        // Also verify count on composed view
        assert_eq!(
            limited.count(tx(11)).await.expect("count limited view"),
            2,
            "limited count should be 2"
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// §7.1 Lifecycle — no-op paths release reservations
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn lifecycle_noop_paths_release_reservations_table() {
    Box::pin(async {
        let root = init_root("lifecycle-noop-reservations");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("a")],
            )
            .await
            .expect("insert");

        table.commit(tx(10)).await.expect("commit");

        // Duplicate commit (no-op) — should release reservation.
        table.commit(tx(10)).await.expect("commit");

        // Finalize tx(10), then duplicate finalize (stale no-op).
        table.finalize(tx(10)).await.expect("finalize");
        assert_eq!(table.finalized().unwrap(), Some(tx(10)));
        table.finalize(tx(10)).await.expect("finalize");
        assert_eq!(table.finalized().unwrap(), Some(tx(10)));

        // After all no-op paths, a later read should not be blocked.
        let row = timeout(
            Duration::from_secs(1),
            table.read_row(tx(11), &[Value::from(1_u64)]),
        )
        .await
        .expect("later read should not be blocked after no-op lifecycle paths");

        assert!(row.unwrap().is_some(), "row should be visible");

        // Write at a new txn should succeed (no leaked reservations).
        table
            .upsert_row(
                &txn.with_id(tx(12)),
                vec![Value::from(2_u64)],
                vec![Value::from("b")],
            )
            .await
            .expect("write should succeed after no-op paths");
    })
    .await;
}

// ---------------------------------------------------------------------------
// §7.2 Visibility — overlapping read blocks until earlier finalize
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn overlapping_read_blocks_until_earlier_finalize_table() {
    Box::pin(async {
        let root = init_root("overlapping-read-finalize");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("hot")],
            )
            .await
            .expect("insert pending");

        // Later read should block while earlier pending write is active
        assert!(
            timeout(
                Duration::from_millis(50),
                table.read_row(tx(11), &[Value::from(1_u64)])
            )
            .await
            .is_err(),
            "later read should block while earlier pending write is active"
        );

        // Commit + finalize the earlier txn
        table.commit(tx(10)).await.expect("commit");
        table.finalize(tx(10)).await.expect("finalize");

        // Now the later read should complete
        let row = timeout(
            Duration::from_secs(1),
            table.read_row(tx(11), &[Value::from(1_u64)]),
        )
        .await
        .expect("later read should complete after earlier finalize");

        assert!(
            row.unwrap().is_some(),
            "row should be visible after finalize"
        );
    })
    .await;
}

// ---------------------------------------------------------------------------
// §7.8 Concurrency / locking
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concurrent_read_write_finalize_table() {
    Box::pin(async {
        let root = init_root("concurrent-rwf");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        // Seed the table
        for i in 1..=20u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(1)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("seed insert");
        }

        table.commit(tx(1)).await.expect("commit");
        table.finalize(tx(1)).await.expect("finalize");

        let t_read = table.clone();
        let t_write = table.clone();
        let t_finalize = table.clone();

        // Spawn concurrent operations with a timeout to detect deadlock
        let result = timeout(Duration::from_secs(5), async {
            let read_task = tokio::spawn(async move {
                for i in 1..=20u64 {
                    let _ = t_read.read_row(tx(100), &[Value::from(i)]).await.unwrap();
                }
            });

            let write_task = tokio::spawn(async move {
                for i in 1..=10u64 {
                    let _ = t_write
                        .upsert_row(
                            &txn.with_id(tx(50)),
                            vec![Value::from(i)],
                            vec![Value::from("updated")],
                        )
                        .await;
                }
            });

            let finalize_task = tokio::spawn(async move {
                t_finalize.finalize(tx(1)).await.expect("finalize");
            });

            let (read, write, finalize) = tokio::join!(read_task, write_task, finalize_task);
            read.expect("read task");
            write.expect("write task");
            finalize.expect("finalize task");
            table.commit(tx(50)).await.expect("commit");
        })
        .await;

        assert!(
            result.is_ok(),
            "concurrent read/write/finalize should not deadlock"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn lock_order_no_deadlock_table() {
    Box::pin(async {
        let root = init_root("lock-order-no-deadlock");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 1..=10u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(1)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("seed");
        }

        table.commit(tx(1)).await.expect("commit");
        table.finalize(tx(1)).await.expect("finalize");

        // Concurrent mutations precede caller-sequenced lifecycle decisions.
        let result = timeout(Duration::from_secs(3), async {
            let t1 = table.clone();
            let t2 = table.clone();
            let txn1 = txn.clone();
            let txn2 = txn.clone();

            let w1 = tokio::spawn(async move {
                t1.upsert_row(
                    &txn1.with_id(tx(10)),
                    vec![Value::from(1_u64)],
                    vec![Value::from("w1")],
                )
                .await
                .expect("w1");
            });

            let w2 = tokio::spawn(async move {
                t2.upsert_row(
                    &txn2.with_id(tx(20)),
                    vec![Value::from(2_u64)],
                    vec![Value::from("w2")],
                )
                .await
                .expect("w2");
            });

            let (w1, w2) = tokio::join!(w1, w2);
            w1.expect("w1 task");
            w2.expect("w2 task");
            table.commit(tx(10)).await.expect("commit 10");
            table.finalize(tx(10)).await.expect("finalize 10");
            table.commit(tx(20)).await.expect("commit 20");
            table.finalize(tx(20)).await.expect("finalize 20");
        })
        .await;

        assert!(result.is_ok(), "interleaved operations should not deadlock");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn finalize_sync_drops_guard_first_table() {
    Box::pin(async {
        let root = init_root("finalize-drops-guard");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        // Insert rows in multiple txns, then finalize in order.
        // If finalize holds the state guard while applying deltas (which
        // acquire table read/write locks), it would deadlock. This test
        // verifies that finalize drops the guard before applying deltas.
        for i in 1..=5u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert txn 10");
        }

        table.commit(tx(10)).await.expect("commit");

        for i in 6..=10u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(11)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("insert txn 11");
        }

        table.commit(tx(11)).await.expect("commit");

        // Finalize should merge both committed deltas without deadlock
        let result = timeout(Duration::from_secs(2), table.finalize(tx(11))).await;
        assert!(result.is_ok(), "finalize should complete without deadlock");
        result.unwrap().expect("finalize should succeed");

        // Verify all rows are visible after finalize
        assert_eq!(
            table.count(tx(12)).await.unwrap(),
            10,
            "all 10 rows should be visible after finalize"
        );

        assert_eq!(table.finalized().unwrap(), Some(tx(11)));
    })
    .await;
}

// ---------------------------------------------------------------------------
// Rows stream — Send + Sync assertions
// ---------------------------------------------------------------------------

#[test]
fn rows_and_views_are_send_and_sync() {
    fn assert_send<T: Send>() {}

    fn assert_sync<T: Sync>() {}

    assert_send::<Rows>();
    assert_send::<TableSlice<TestTxn>>();
    assert_sync::<TableSlice<TestTxn>>();
    assert_send::<Limited<TestTxn>>();
    assert_sync::<Limited<TestTxn>>();
    assert_send::<Selection<TestTxn>>();
    assert_sync::<Selection<TestTxn>>();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn blocked_reader_cancellation_does_not_poison_lock_state_table() {
    Box::pin(async {
        let root = init_root("blocked-reader-cancel-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(60)),
                vec![Value::from(1_u64)],
                vec![Value::from("hot")],
            )
            .await
            .expect("insert pending key");

        let (waiting, registered) = oneshot::channel();
        let blocked_reader = {
            let table = table.clone();
            tokio::spawn(async move {
                let key = [Value::from(1_u64)];
                let read = table.contains_row(tx(61), &key);
                tokio::pin!(read);
                assert!(futures::poll!(&mut read).is_pending());
                waiting.send(()).unwrap();
                read.await.unwrap()
            })
        };

        timeout(Duration::from_secs(1), registered)
            .await
            .expect("reader must reach its wait")
            .unwrap();

        assert!(
            !blocked_reader.is_finished(),
            "reader should still be blocked before lifecycle resolution"
        );

        blocked_reader.abort();
        let aborted = blocked_reader.await;
        assert!(aborted.is_err(), "blocked reader should abort cleanly");

        // Simulate timeout cleanup using rollback and verify no lock poisoning remains.
        table.rollback(tx(60)).expect("rollback 60");

        let visible = timeout(
            Duration::from_secs(1),
            table.contains_row(tx(61), &[Value::from(1_u64)]),
        )
        .await
        .expect("later read should complete after rollback cleanup");

        assert!(!visible.unwrap());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn commit_then_update_same_key_different_txn_table() {
    Box::pin(async {
        let root = init_root("commit-then-update-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        // txn 10: insert, commit (but don't finalize)
        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("first")],
            )
            .await
            .expect("insert first");

        table.commit(tx(10)).await.expect("commit 10");

        // txn 11: update same key — should succeed after commit released reservation
        table
            .upsert_row(
                &txn.with_id(tx(11)),
                vec![Value::from(1_u64)],
                vec![Value::from("second")],
            )
            .await
            .expect("update same key in later txn");

        // Read at txn 11 should see the later update
        let row = table
            .read_row(tx(11), &[Value::from(1_u64)])
            .await
            .unwrap()
            .expect("read at txn 11");

        assert_eq!(row.as_ref(), &[Value::from(1_u64), Value::from("second")]);

        // Read at txn 10 should see the original committed value
        let row = table
            .read_row(tx(10), &[Value::from(1_u64)])
            .await
            .unwrap()
            .expect("read at txn 10");

        assert_eq!(row.as_ref(), &[Value::from(1_u64), Value::from("first")]);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concurrent_writer_conflict_matrix_table() {
    Box::pin(async {
        let root = init_root("writer-conflict-matrix-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(3));

        let later_same = {
            let table = table.clone();
            let barrier = barrier.clone();
            let txn = txn.clone();
            tokio::spawn(async move {
                let result = table
                    .upsert_row(
                        &txn.with_id(tx(20)),
                        vec![Value::from(1_u64)],
                        vec![Value::from("same")],
                    )
                    .await;

                barrier.wait().await;
                result
            })
        };

        let earlier_same = {
            let table = table.clone();
            let barrier = barrier.clone();
            let txn = txn.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                table
                    .upsert_row(
                        &txn.with_id(tx(19)),
                        vec![Value::from(1_u64)],
                        vec![Value::from("same")],
                    )
                    .await
            })
        };

        let earlier_disjoint = {
            let table = table.clone();
            let barrier = barrier.clone();
            let txn = txn.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                table
                    .upsert_row(
                        &txn.with_id(tx(19)),
                        vec![Value::from(2_u64)],
                        vec![Value::from("other")],
                    )
                    .await
            })
        };

        let (later_same, earlier_same, earlier_disjoint) = timeout(Duration::from_secs(5), async {
            tokio::join!(later_same, earlier_same, earlier_disjoint)
        })
        .await
        .expect("writers must complete");

        let later_same = later_same.expect("later_same join");
        let earlier_same = earlier_same.expect("earlier_same join");
        let earlier_disjoint = earlier_disjoint.expect("earlier_disjoint join");

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
async fn delta_stack_overrides_with_later_committed_delta_table() {
    Box::pin(async {
        let root = init_root("delta-stack-override-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        // txn 10: insert key=1, value="a", commit (but don't finalize yet)
        table
            .upsert_row(
                &txn.with_id(tx(10)),
                vec![Value::from(1_u64)],
                vec![Value::from("a")],
            )
            .await
            .expect("insert at txn 10");

        table.commit(tx(10)).await.expect("commit 10");

        // txn 11: update key=1, value="b" — this acquires a write reservation
        // after txn 10's commit released it.
        table
            .upsert_row(
                &txn.with_id(tx(11)),
                vec![Value::from(1_u64)],
                vec![Value::from("b")],
            )
            .await
            .expect("update at txn 11");

        table.commit(tx(11)).await.expect("commit 11");

        // Reading at txn 11 should see txn 11's value ("b"), not txn 10's ("a").
        // This tests the resolve_row fix: later deltas override earlier ones.
        let row = table
            .read_row(tx(11), &[Value::from(1_u64)])
            .await
            .unwrap()
            .expect("read at txn 11");

        assert_eq!(
            row.as_ref(),
            &[Value::from(1_u64), Value::from("b")],
            "later committed delta should override earlier committed delta"
        );

        // Reading at txn 10 should see txn 10's value ("a").
        let row = table
            .read_row(tx(10), &[Value::from(1_u64)])
            .await
            .unwrap()
            .expect("read at txn 10");

        assert_eq!(
            row.as_ref(),
            &[Value::from(1_u64), Value::from("a")],
            "earlier txn should see its own committed delta, not later ones"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn empty_commit_then_finalize_allows_future_writes_table() {
    Box::pin(async {
        let root = init_root("empty-commit-future-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        // Commit and finalize an empty transaction.
        table.commit(tx(10)).await.expect("empty commit");
        table.finalize(tx(10)).await.expect("empty finalize");

        // Future writes should still work.
        table
            .upsert_row(
                &txn.with_id(tx(11)),
                vec![Value::from(1_u64)],
                vec![Value::from("post")],
            )
            .await
            .expect("write after empty finalize");

        table.commit(tx(11)).await.expect("commit 11");
        table.finalize(tx(11)).await.expect("finalize 11");

        assert!(
            table
                .contains_row(tx(12), &[Value::from(1_u64)])
                .await
                .unwrap()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn finalize_conflicts_with_future_read_table() {
    Box::pin(async {
        let root = init_root("finalize-future-read-conflict-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(9)),
                vec![Value::from(1_u64)],
                vec![Value::from("k")],
            )
            .await
            .expect("insert seed key");

        table.commit(tx(9)).await.expect("commit 9");
        table.finalize(tx(9)).await.expect("finalize 9");

        // Register an overlapping future read at txn 11.
        assert!(
            table
                .contains_row(tx(11), &[Value::from(1_u64)])
                .await
                .unwrap()
        );

        // Finalize at txn 10 should succeed even though txn 11 has an active
        // read reservation. Finalize is a lifecycle operation, not a write —
        // it merges already-committed data into canon. Future reads are
        // protected by their own permits and by the DirLock on persistent storage.
        table
            .finalize(tx(10))
            .await
            .expect("finalize should succeed with future read");

        // After finalize, the data should still be visible to later reads.
        assert!(
            table
                .contains_row(tx(12), &[Value::from(1_u64)])
                .await
                .unwrap()
        );

        for (n, consume) in [(13, false), (15, true)] {
            table
                .upsert_row(&txn.with_id(tx(n)), vec![n.into()], vec!["later".into()])
                .await
                .unwrap();

            table.commit(tx(n)).await.unwrap();
            let stream = table
                .rows(tx(n + 1), Range::default(), Vec::new(), false)
                .await
                .unwrap();

            let mut finalize = Box::pin(table.finalize(tx(n)));
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

            assert!(
                table
                    .contains_row(tx(n + 1), &[1_u64.into()])
                    .await
                    .unwrap()
            );
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn many_later_readers_unblock_after_commit_table() {
    Box::pin(async {
        let root = init_root("many-readers-unblock-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(50)),
                vec![Value::from(1_u64)],
                vec![Value::from("hot")],
            )
            .await
            .expect("insert pending key");

        let (waiting, mut registered) = mpsc::channel(64);
        let mut readers = Vec::new();
        for _ in 0..64 {
            let table = table.clone();
            let waiting = waiting.clone();
            readers.push(tokio::spawn(async move {
                let key = [Value::from(1_u64)];
                let read = table.contains_row(tx(51), &key);
                tokio::pin!(read);
                assert!(futures::poll!(&mut read).is_pending());
                waiting.send(()).await.unwrap();
                read.await.unwrap()
            }));
        }

        drop(waiting);

        timeout(Duration::from_secs(2), async {
            for _ in 0..64 {
                registered.recv().await.expect("reader reached its wait");
            }
        })
        .await
        .expect("all readers must reach their waits");

        assert!(
            readers.iter().all(|reader| !reader.is_finished()),
            "later overlapping readers should still be blocked before finalize"
        );

        table.commit(tx(50)).await.expect("commit 50");
        table.finalize(tx(50)).await.expect("finalize 50");

        let results = timeout(Duration::from_secs(2), async { join_all(readers).await })
            .await
            .expect("all readers should complete after finalize");

        for result in results {
            assert!(result.expect("reader task should join successfully"));
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn multi_column_partial_overlap_blocking_behavior_table() {
    Box::pin(async {
        let root = init_root("multi-column-overlap-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, composite_schema())
            .await
            .unwrap();

        table
            .upsert_row(
                &txn.with_id(tx(80)),
                vec![Value::from(1_u64), Value::from(1_u64)],
                vec![Value::from("a")],
            )
            .await
            .expect("insert pending composite key");

        assert!(
            timeout(
                Duration::from_millis(50),
                table.contains_row(tx(81), &[Value::from(1_u64), Value::from(1_u64)]),
            )
            .await
            .is_err(),
            "overlapping composite-key read should block"
        );

        let disjoint = timeout(
            Duration::from_secs(1),
            table.contains_row(tx(81), &[Value::from(1_u64), Value::from(2_u64)]),
        )
        .await
        .expect("disjoint composite-key read should not block");

        assert!(!disjoint.unwrap());

        table
            .commit(tx(80))
            .await
            .expect("commit preserves the disjoint future read");

        assert!(
            table
                .contains_row(tx(81), &[Value::from(1_u64), Value::from(1_u64)])
                .await
                .unwrap()
        );

        assert_eq!(
            table.rollback(tx(80)).unwrap_err().code(),
            tc_error::ErrorKind::Conflict
        );

        for scan_first in [false, true] {
            let root = init_root("point-and-scan-reservations-table");
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::<TestTxn>::try_new(persistent, composite_schema())
                .await
                .unwrap();

            let txn = txn.with_id(tx(90));
            let key = [Value::from(1_u64), Value::from(1_u64)];

            if !scan_first {
                assert!(table.read_row(tx(90), &key).await.unwrap().is_none());
            }

            // Even a filtered scan currently reserves the whole table.
            let rows = table
                .rows(tx(90), range_eq("part_a", 1_u64.into()), vec![], false)
                .await
                .unwrap();

            if scan_first {
                assert!(table.read_row(tx(90), &key).await.unwrap().is_none());
            }

            assert!(
                table
                    .upsert_row(&txn, key.to_vec(), vec!["blocked".into()])
                    .await
                    .is_err(),
                "point write must respect the scan (scan_first={scan_first})"
            );

            assert!(table.truncate(&txn, Range::default()).await.is_err());
            drop(rows);

            table
                .upsert_row(&txn, key.to_vec(), vec!["released".into()])
                .await
                .unwrap();

            assert_eq!(
                table
                    .read_row(tx(90), &key)
                    .await
                    .unwrap()
                    .unwrap()
                    .as_ref(),
                &[1_u64.into(), 1_u64.into(), Value::from("released")]
            );

            table.truncate(&txn, Range::default()).await.unwrap();
            assert!(table.read_row(tx(90), &key).await.unwrap().is_none());
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restart_drops_uncommitted_pending_table() {
    Box::pin(async {
        let root = init_root("restart-pending-dropped-table");
        {
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
                .await
                .unwrap();

            table
                .sync_all()
                .await
                .expect("publish initial canonical table");

            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(1_u64)],
                    vec![Value::from("pending")],
                )
                .await
                .expect("insert pending (not committed)");
        }

        // Reload — pending delta should be gone (no WAL owned by Table).
        let (persistent, _) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::load(persistent, simple_schema())
            .await
            .unwrap();

        assert!(table.is_empty(tx(20)).await.unwrap());
        assert!(
            table
                .read_row(tx(20), &[Value::from(1_u64)])
                .await
                .unwrap()
                .is_none(),
            "uncommitted pending row must not survive restart"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restart_reconstructs_committed_state_table() {
    Box::pin(async {
        let root = init_root("restart-committed-table");
        {
            let (persistent, txn) = load_roots(&root);
            let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
                .await
                .unwrap();

            for i in 1..=5u64 {
                table
                    .upsert_row(
                        &txn.with_id(tx(10)),
                        vec![Value::from(i)],
                        vec![Value::from(format!("v{i}"))],
                    )
                    .await
                    .expect("insert row");
            }

            table.commit(tx(10)).await.expect("commit 10");
            table
                .finalize(tx(10))
                .await
                .expect("finalize 10 — merges into canon");

            table.sync().await.expect("sync canon to disk");
        }

        // Reload from the same dirs — canon should have the finalized rows.
        let (persistent, _) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::load(persistent, simple_schema())
            .await
            .unwrap();

        assert_eq!(table.count(tx(20)).await.unwrap(), 5);
        assert!(
            table
                .contains_row(tx(20), &[Value::from(3_u64)])
                .await
                .unwrap()
        );

        let row = table
            .read_row(tx(20), &[Value::from(2_u64)])
            .await
            .unwrap()
            .expect("read row after restart");

        assert_eq!(row.as_ref(), &[Value::from(2_u64), Value::from("v2")]);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn schema_mismatch_merge_fails_closed_table() {
    Box::pin(async {
        let root = init_root("schema-mismatch-table");
        let (persistent, txn) = load_roots(&root);
        let schema_a = simple_schema();
        {
            let table_a = PersistentTable::<TestTxn>::try_new(persistent.clone(), schema_a)
                .await
                .unwrap();

            table_a
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![Value::from(1_u64)],
                    vec![Value::from("alpha")],
                )
                .await
                .expect("insert row with schema A");

            table_a.commit(tx(10)).await.expect("commit 10");
            table_a.finalize(tx(10)).await.expect("finalize 10");
        }

        // Attempt to load with a mismatched schema (different value column type).
        let mismatched_key = vec![Column {
            name: "id".parse().expect("Id"),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        }];

        let mismatched_values = vec![Column {
            name: "label".parse().expect("Id"),
            dtype: ValueType::Number(tc_value::NumberType::Number), // was String, now Number
        }];

        let mismatched_schema = TableSchema::new(
            mismatched_key,
            mismatched_values,
            Vec::new(),
            StorageConfig::default(),
        )
        .expect("create mismatched schema");

        assert!(
            PersistentTable::<TestTxn>::load(persistent, mismatched_schema)
                .await
                .is_err()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn snapshot_scan_is_coherent_under_concurrent_commits_table() {
    Box::pin(async {
        let root = init_root("snapshot-coherence-table");
        let (persistent, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        for i in 0_u64..100_u64 {
            table
                .upsert_row(
                    &txn.with_id(tx(30)),
                    vec![Value::from(i)],
                    vec![Value::from(format!("v{i}"))],
                )
                .await
                .expect("seed baseline key");
        }

        table.commit(tx(30)).await.expect("commit 30");
        table.finalize(tx(30)).await.expect("finalize 30");

        let (captured, snapshot_ready) = oneshot::channel();
        let (committed, commit_ready) = oneshot::channel();

        let scan_task = {
            let table = table.clone();
            tokio::spawn(async move {
                let mut rows = table
                    .rows(tx(31), Range::default(), vec![], false)
                    .await
                    .unwrap();

                captured.send(()).unwrap();
                commit_ready.await.unwrap();

                let mut saw_future_key = false;
                while let Some(row) = rows.try_next().await.unwrap() {
                    if row.as_ref()[0] == Value::from(100_u64) {
                        saw_future_key = true;
                    }
                }

                saw_future_key
            })
        };

        let writer_task = {
            let table = table.clone();
            tokio::spawn(async move {
                snapshot_ready.await.unwrap();
                table
                    .upsert_row(
                        &txn.with_id(tx(32)),
                        vec![Value::from(100_u64)],
                        vec![Value::from("future")],
                    )
                    .await
                    .expect("insert future key");

                table.commit(tx(32)).await.expect("commit 32");
                committed.send(()).unwrap();
            })
        };

        let (scan, writer) = timeout(Duration::from_secs(5), async {
            tokio::join!(scan_task, writer_task)
        })
        .await
        .expect("snapshot and writer must complete");

        let saw_future_key = scan.expect("scan task join");
        writer.expect("writer task join");
        table.finalize(tx(32)).await.expect("finalize after scan");

        assert!(
            !saw_future_key,
            "txn 31 snapshot must not include key committed/finalized at txn 32"
        );

        assert!(
            table
                .contains_row(tx(33), &[Value::from(100_u64)])
                .await
                .unwrap()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn materialization_delegates_view_representability_and_retains_independent_rows() {
    Box::pin(async {
        let root = init_root("materialization");
        let (persistent, workspace) = load_roots(&root);
        let txn = workspace.with_id(tx(10));
        let workspace = crate::StorageContext::context(&txn).await.unwrap();
        let table = PersistentTable::<TestTxn>::try_new(persistent, simple_schema())
            .await
            .unwrap();

        table
            .upsert_row(&txn, vec![1u64.into()], vec!["one".into()])
            .await
            .unwrap();

        table
            .upsert_row(&txn, vec![2u64.into()], vec!["two".into()])
            .await
            .unwrap();

        let reversed = table.order_by(&["id".parse().unwrap()], true);
        let limited = reversed.limit(1);
        let selected = limited.select(vec!["id".parse().unwrap()]).unwrap();
        for (i, view) in [
            super::Table::from(reversed),
            limited.into(),
            selected.into(),
        ]
        .into_iter()
        .enumerate()
        {
            let dir = workspace
                .write()
                .await
                .create_dir(format!("rejected-{i}"))
                .unwrap();

            assert!(
                view.materialize(&txn, dir.clone())
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("reordered Table")
            );

            assert!(dir.read().await.is_empty());
        }

        let dir = workspace
            .write()
            .await
            .create_dir("accepted".into())
            .unwrap();

        let copy = super::Table::from(table.clone())
            .materialize(&txn, dir)
            .await
            .unwrap();

        table
            .upsert_row(&txn, vec![1u64.into()], vec!["changed".into()])
            .await
            .unwrap();

        assert_eq!(
            copy.read_row(tx(10), &[1u64.into()])
                .await
                .unwrap()
                .unwrap()
                .as_ref(),
            &[1u64.into(), "one".into()]
        );
    })
    .await;
}

#[tokio::test]
async fn creation_and_storage_failures_are_errors() {
    let root = init_root("fallible-table");
    let (dir, _) = load_roots(&root);
    let table = PersistentTable::<TestTxn>::try_new(dir.clone(), simple_schema())
        .await
        .unwrap();

    assert!(
        PersistentTable::<TestTxn>::try_new(dir.clone(), simple_schema())
            .await
            .is_err()
    );

    let primary = dir.read().await.get_dir("primary").cloned().unwrap();
    let root_name = primary.read().await.iter().next().unwrap().0.clone();
    assert!(primary.write().await.delete(&root_name).await);
    let key = vec![Value::from(1_u64)];
    assert!(table.read_row(tx(1), &key).await.is_err());
    assert!(table.contains_row(tx(1), &key).await.is_err());
    assert!(table.count(tx(1)).await.is_err());
    assert!(table.is_empty(tx(1)).await.is_err());
    assert!(
        table
            .for_each_row_in_order(tx(1), Range::default(), &[], false, |_| panic!(
                "missing root yielded a row"
            ))
            .await
            .is_err()
    );

    drop(table);
    drop(primary);
    drop(dir);
    std::fs::remove_dir_all(&*root).unwrap();
}

#[tokio::test]
async fn rows_release_sources_and_permits_at_termination() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    use futures::{Stream, StreamExt};

    // Projection preserves requested order and duplicates, skipping missing names.
    let source = futures::stream::iter([Ok(b_table::Row::from_vec(vec![
        Value::from(1_u64),
        Value::from("one"),
    ]))])
    .boxed();

    let mut selected = Rows::local(source).select(
        &simple_schema(),
        &[
            "label".parse().unwrap(),
            "id".parse().unwrap(),
            "label".parse().unwrap(),
            "missing".parse().unwrap(),
        ],
    );

    assert_eq!(
        selected.try_next().await.unwrap().unwrap().as_ref(),
        &[Value::from("one"), Value::from(1_u64), Value::from("one")]
    );

    assert!(selected.try_next().await.unwrap().is_none());

    for case in ["eof", "error", "limit", "zero_limit", "unpolled", "pending"] {
        let semaphore = txn_lock::semaphore::Semaphore::new(b_tree::RangeCollator::new(
            tc_value::ValueCollator::default(),
        ));

        let permit = semaphore
            .read(tx(1), txn_lock::set::Range::All)
            .await
            .unwrap();

        let lease = Arc::new(());
        let retained = Arc::downgrade(&lease);
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = polls.clone();
        let mut position = 0;
        let source = futures::stream::poll_fn(move |_| {
            let _lease = &lease;
            observed.fetch_add(1, Ordering::Relaxed);
            if case == "pending" {
                return Poll::Pending;
            }

            position += 1;
            Poll::Ready(match position {
                1 => Some(Ok(b_table::Row::from_vec(vec![
                    Value::from(1_u64),
                    Value::from("one"),
                ]))),
                2 if case == "error" => Some(Err(std::io::Error::other("source failed"))),
                2 => Some(Ok(b_table::Row::from_vec(vec![
                    Value::from(2_u64),
                    Value::from("two"),
                ]))),
                _ => None,
            })
        })
        .boxed();

        let mut rows = Rows::new(source, permit)
            .select(&simple_schema(), &["label".parse().unwrap()])
            .limit(match case {
                "limit" => 1,
                "zero_limit" => 0,
                _ => 10,
            });

        assert!(
            semaphore
                .try_write(tx(1), txn_lock::set::Range::All)
                .is_err()
        );

        match case {
            "unpolled" => drop(rows),
            "pending" => {
                let waker = futures::task::noop_waker();
                assert!(
                    std::pin::Pin::new(&mut rows)
                        .poll_next(&mut Context::from_waker(&waker))
                        .is_pending()
                );

                drop(rows);
            }
            _ => {
                if case != "zero_limit" {
                    assert_eq!(
                        rows.try_next().await.unwrap().unwrap().as_slice(),
                        &[Value::from("one")]
                    );
                }

                if case == "error" {
                    assert!(rows.try_next().await.is_err());
                } else {
                    if case == "eof" {
                        assert!(rows.try_next().await.unwrap().is_some());
                    }

                    assert!(rows.try_next().await.unwrap().is_none());
                }

                // Retaining the terminated wrapper must not retain the source or permit.
                assert!(retained.upgrade().is_none(), "{case}");
                assert!(
                    semaphore
                        .try_write(tx(1), txn_lock::set::Range::All)
                        .is_ok(),
                    "{case}"
                );

                let before = polls.load(Ordering::Relaxed);
                assert!(rows.try_next().await.unwrap().is_none());
                assert_eq!(polls.load(Ordering::Relaxed), before);
            }
        }

        assert!(retained.upgrade().is_none(), "{case}");
        assert!(
            semaphore
                .try_write(tx(1), txn_lock::set::Range::All)
                .is_ok(),
            "{case}"
        );

        if matches!(case, "zero_limit" | "unpolled") {
            assert_eq!(polls.load(Ordering::Relaxed), 0);
        }

        if case == "limit" {
            assert_eq!(polls.load(Ordering::Relaxed), 1);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn indexed_queries_resolve_primary_identity_before_predicates() {
    Box::pin(async {
        let root = init_root("indexed-visible-identity");
        let (persistent, txn) = load_roots(&root);
        let schema = schema_with_index();
        let table = PersistentTable::<TestTxn>::try_new(persistent.clone(), schema.clone())
            .await
            .unwrap();

        for (key, score, label) in [(1_u64, 10_u64, "a"), (2, 10, "b"), (3, 20, "c")] {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![key.into()],
                    vec![score.into(), label.into()],
                )
                .await
                .unwrap();
        }

        table.commit(tx(10)).await.unwrap();
        table.finalize(tx(10)).await.unwrap();
        let current = txn.with_id(tx(11));
        for (key, score, label) in [(4_u64, 10_u64, "d"), (1, 30, "a-new"), (3, 10, "c-new")] {
            table
                .upsert_row(&current, vec![key.into()], vec![score.into(), label.into()])
                .await
                .unwrap();
        }

        table
            .delete_row(&current, vec![2_u64.into()])
            .await
            .unwrap();

        let order = vec![id("ref_id")];
        for reverse in [false, true] {
            let rows = table
                .rows(tx(11), Range::default(), order.clone(), reverse)
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            let mut expected: Vec<Row<Value>> = vec![
                vec![3_u64.into(), 10_u64.into(), "c-new".into()].into(),
                vec![4_u64.into(), 10_u64.into(), "d".into()].into(),
                vec![1_u64.into(), 30_u64.into(), "a-new".into()].into(),
            ];

            if reverse {
                expected.reverse();
            }

            assert_eq!(rows, expected);
            for row in &rows {
                assert_eq!(
                    table.read_row(tx(11), &row[..1]).await.unwrap().as_ref(),
                    Some(row)
                );
            }
        }

        let matching = range_eq("ref_id", 10_u64.into());
        let obsolete = range_eq("ref_id", 20_u64.into());
        let rows = table
            .rows(tx(11), matching.clone(), Vec::new(), false)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            [3_u64.into(), 4_u64.into()]
        );

        assert_eq!(table.count(tx(11)).await.unwrap(), 3);
        assert_eq!(table.count_in(tx(11), matching.clone()).await.unwrap(), 2);
        assert!(table.is_empty_in(tx(11), obsolete.clone()).await.unwrap());
        timeout(Duration::from_secs(2), table.truncate(&current, obsolete))
            .await
            .expect("obsolete predicate must not remove the replacement")
            .unwrap();

        assert_eq!(
            table
                .read_row(tx(11), &[3_u64.into()])
                .await
                .unwrap()
                .unwrap()
                .as_slice(),
            &[3_u64.into(), 10_u64.into(), Value::from("c-new")]
        );

        // An unpolled query releases both native layer capabilities and its permit.
        drop(
            table
                .rows(tx(11), matching.clone(), order.clone(), false)
                .await
                .unwrap(),
        );

        let updates = tc_ir::Map::from_iter([(id("label"), Value::from("bulk"))]);
        timeout(
            Duration::from_secs(2),
            table.update(&current, matching.clone(), updates),
        )
        .await
        .expect("bulk update must not write through its own read guards")
        .unwrap();

        timeout(
            Duration::from_secs(2),
            table.truncate(&current, range_eq("ref_id", 30_u64.into())),
        )
        .await
        .expect("range deletion must release input guards before publication")
        .unwrap();

        assert!(
            table
                .read_row(tx(11), &[1_u64.into()])
                .await
                .unwrap()
                .is_none()
        );

        for key in [3_u64, 4] {
            assert_eq!(
                table
                    .read_row(tx(11), &[key.into()])
                    .await
                    .unwrap()
                    .unwrap()[2],
                Value::from("bulk")
            );
        }

        table.commit(tx(11)).await.unwrap();
        table
            .upsert_row(
                &txn.with_id(tx(12)),
                vec![5_u64.into()],
                vec![10_u64.into(), "e".into()],
            )
            .await
            .unwrap();

        table.commit(tx(12)).await.unwrap();
        table.finalize(tx(11)).await.unwrap();
        let expected: Vec<Row<Value>> = vec![
            vec![3_u64.into(), 10_u64.into(), "bulk".into()].into(),
            vec![4_u64.into(), 10_u64.into(), "bulk".into()].into(),
            vec![5_u64.into(), 10_u64.into(), "e".into()].into(),
        ];

        let rows = table
            .rows(tx(13), matching.clone(), order.clone(), false)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(rows, expected);
        table.finalize(tx(12)).await.unwrap();
        table.sync_all().await.unwrap();
        drop(table);
        drop(persistent);
        let cache =
            freqfs::Cache::<PersistentFile>::new(16 * 1024 * 1024, None, 0, Duration::from_secs(3));

        let table =
            PersistentTable::<TestTxn>::load(cache.load(root.join("persistent")).unwrap(), schema)
                .await
                .unwrap();

        let rows = table
            .rows(tx(13), matching, order, false)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(rows, expected);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn layered_queries_follow_the_complete_supporting_index_order() {
    Box::pin(async {
        let root = init_root("layered-index-order");
        let (persistent, txn) = load_roots(&root);
        let column = |name: &str, dtype| Column {
            name: id(name),
            dtype,
        };

        let number = || ValueType::Number(tc_value::NumberType::Number);
        let schema = TableSchema::new(
            vec![column("id", number())],
            vec![
                column("ref_id", number()),
                column("label", ValueType::String),
            ],
            vec![("by_ref_label".into(), vec![id("ref_id"), id("label")])],
            StorageConfig::default(),
        )
        .unwrap();

        let table = PersistentTable::<TestTxn>::try_new(persistent, schema.clone())
            .await
            .unwrap();

        for (key, label) in [(1_u64, "z"), (2, "b")] {
            table
                .upsert_row(
                    &txn.with_id(tx(10)),
                    vec![key.into()],
                    vec![10_u64.into(), label.into()],
                )
                .await
                .unwrap();
        }

        table.commit(tx(10)).await.unwrap();
        table.finalize(tx(10)).await.unwrap();
        for (key, label) in [(3_u64, "a"), (4, "m"), (1, "c")] {
            table
                .upsert_row(
                    &txn.with_id(tx(11)),
                    vec![key.into()],
                    vec![10_u64.into(), label.into()],
                )
                .await
                .unwrap();
        }

        table.commit(tx(11)).await.unwrap();
        // An identical newer row still replaces its older version exactly once.
        table
            .upsert_row(
                &txn.with_id(tx(12)),
                vec![1_u64.into()],
                vec![10_u64.into(), "c".into()],
            )
            .await
            .unwrap();

        table.commit(tx(12)).await.unwrap();
        for order in [Vec::new(), vec![id("ref_id")]] {
            for reverse in [false, true] {
                let mut expected: Vec<Value> =
                    vec![3_u64.into(), 2_u64.into(), 1_u64.into(), 4_u64.into()];

                if reverse {
                    expected.reverse();
                }

                let rows = table
                    .rows(
                        tx(13),
                        range_eq("ref_id", 10_u64.into()),
                        order.clone(),
                        reverse,
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();

                assert_eq!(
                    rows.iter()
                        .map(|row| row[0].clone())
                        .collect::<Vec<Value>>(),
                    expected
                );
            }
        }

        let mut first = table
            .rows(
                tx(13),
                range_eq("ref_id", 10_u64.into()),
                vec![id("ref_id")],
                false,
            )
            .await
            .unwrap()
            .select(&schema, &[id("label"), id("id"), id("label")])
            .limit(1);

        assert_eq!(
            first.try_next().await.unwrap().unwrap().as_slice(),
            &["a".into(), 3_u64.into(), "a".into()]
        );

        assert!(first.try_next().await.unwrap().is_none());
        drop(first);
        table.finalize(tx(12)).await.unwrap();
        assert_eq!(table.count(tx(13)).await.unwrap(), 4);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn nested_slices_intersect_and_reject_disjoint_constraints() {
    Box::pin(async {
        let root = init_root("intersecting-table-slices");
        let (dir, txn) = load_roots(&root);
        let table = PersistentTable::<TestTxn>::try_new(dir, simple_schema())
            .await
            .unwrap();

        let txn = txn.with_id(tx(10));
        for key in 0..6_u64 {
            table
                .upsert_row(&txn, vec![key.into()], vec!["row".into()])
                .await
                .unwrap();
        }

        let first = table.slice(range_in("id", 1_u64.into(), 4_u64.into()), &[], true);
        let wider = first
            .slice(range_in("id", 0_u64.into(), 6_u64.into()))
            .unwrap();

        assert!(wider.reverse());
        assert_eq!(wider.order(), first.order());
        let rows = collect_rows(wider.rows(tx(10)).await.unwrap()).await;
        assert_eq!(
            rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![3_u64.into(), 2_u64.into(), 1_u64.into()]
        );

        assert_eq!(first.count(tx(10)).await.unwrap(), 3);
        let at_edge: Range<_, _> = [(
            id("id"),
            ColumnRange::In((Bound::Included(3_u64.into()), Bound::Included(5_u64.into()))),
        )]
        .into_iter()
        .collect();

        assert_eq!(
            first.slice(at_edge).unwrap().count(tx(10)).await.unwrap(),
            1
        );

        assert!(
            first
                .slice(range_in("id", 4_u64.into(), 6_u64.into()))
                .is_err()
        );

        assert!(first.slice(range_eq("id", 0_u64.into())).is_err());
        assert!(
            first
                .slice(range_in("id", 2_u64.into(), 2_u64.into()))
                .is_err()
        );

        assert_eq!(table.count(tx(10)).await.unwrap(), 6);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn selection_metadata_matches_requested_value_order() {
    use collate::Collate;

    use crate::StorageContext;
    use tc_ir::IntoView;

    Box::pin(async {
        let root = init_root("ordered-projection");
        let (dir, txn) = load_roots(&root);
        let txn = txn.with_id(tx(10));
        let table = PersistentTable::<TestTxn>::try_new(dir, schema_with_index())
            .await
            .unwrap();

        for key in 1..=2_u64 {
            table
                .upsert_row(
                    &txn,
                    vec![key.into()],
                    vec![(key * 10).into(), format!("item{key}").into()],
                )
                .await
                .unwrap();
        }

        let columns = vec![id("id"), id("label"), id("ref_id")];
        for (reverse, limit) in [(false, None), (true, Some(1)), (false, Some(0))] {
            let source = table.slice(Range::default(), &[], reverse);
            let selection = match limit {
                Some(limit) => source.limit(limit).select(columns.clone()),
                None => source.select(columns.clone()),
            }
            .unwrap();

            assert_eq!(selection.columns(), columns);
            assert_eq!(
                selection.schema().columns().cloned().collect::<Vec<_>>(),
                columns
            );

            let mut expected = vec![
                vec![1_u64.into(), "item1".into(), 10_u64.into()],
                vec![2_u64.into(), "item2".into(), 20_u64.into()],
            ];

            if reverse {
                expected.reverse();
            }

            if let Some(limit) = limit {
                expected.truncate(limit as usize);
            }

            assert_eq!(
                collect_rows(selection.rows(tx(10)).await.unwrap()).await,
                expected
            );

            assert_eq!(
                selection.count(tx(10)).await.unwrap(),
                expected.len() as u64
            );

            assert_eq!(
                selection.is_empty(tx(10)).await.unwrap(),
                expected.is_empty()
            );

            let mut visited = Vec::new();
            selection
                .for_each_row_in_order(tx(10), |row| visited.push(row.into_vec()))
                .await
                .unwrap();

            assert_eq!(visited, expected);

            for binary in [false, true] {
                let view = crate::Collection::from(super::Table::from(selection.clone()))
                    .into_view(txn.clone())
                    .await
                    .unwrap();

                let crate::CollectionView::Table(view) = view else {
                    panic!("Table view")
                };

                let decoded: super::DecodedTablePayload<TestTxn> = if binary {
                    tbon::de::try_decode(txn.clone(), tbon::en::encode(view).unwrap())
                        .await
                        .unwrap()
                } else {
                    destream_json::try_decode(
                        txn.clone(),
                        destream_json::encode(view)
                            .unwrap()
                            .map_err(std::io::Error::other),
                    )
                    .await
                    .unwrap()
                };

                assert_eq!(decoded.table.schema(), selection.schema());
                let rows: Vec<_> = decoded
                    .table
                    .row_stream(tx(10))
                    .await
                    .unwrap()
                    .map_ok(|row| row.into_vec())
                    .try_collect()
                    .await
                    .unwrap();

                // Decoding a Table restores canonical primary order.
                let mut canonical = expected.clone();
                canonical.sort_by(|a, b| tc_value::ValueCollator::default().cmp(&a[0], &b[0]));
                assert_eq!(rows, canonical);
            }
        }

        for names in [
            vec!["label"],
            vec!["label", "id", "ref_id"],
            vec!["id", "label", "label"],
            vec!["id", "missing"],
            vec![],
        ] {
            let names: Vec<_> = names.into_iter().map(id).collect();
            assert!(table.select(&names).is_err());
            assert!(table.limit(1).select(names).is_err());
        }

        assert_eq!(table.count(tx(10)).await.unwrap(), 2);
        let composite = PersistentTable::<TestTxn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            composite_schema(),
        )
        .await
        .unwrap();

        assert!(
            composite
                .select(&[id("part_b"), id("part_a"), id("val1")])
                .is_err()
        );
    })
    .await;
}
