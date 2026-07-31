//! Route-level tests for table public API handlers.

use super::response::TableResponse;
use super::{TableRouter, TableStatic};
use super::selector::KeyOrRange;
use crate::btree::{StorageConfig, PersistentFile};
use crate::table::{Column, TableSchema};
use crate::Collection;
use freqfs::Cache;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tc_ir::{Claim, HandleDelete, HandleGet, HandlePost, HandlePut, NetworkTime, Route, Scalar, Transaction, TxnId};
use tc_value::ValueType;
use umask::Mode;

fn segment(name: &str) -> pathlink::PathSegment {
    pathlink::PathSegment::from_str(name).expect("path segment")
}

fn tx(nonce: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), nonce)
}

struct MockTxn {
    id: TxnId,
    claim: Claim,
}

impl MockTxn {
    fn new(nonce: u16) -> Self {
        Self {
            id: tx(nonce),
            claim: Claim::new(
                pathlink::Link::from_str("/test").expect("link"),
                Mode::all(),
            ),
        }
    }
}

impl Transaction for MockTxn {
    fn id(&self) -> TxnId {
        self.id
    }
    fn timestamp(&self) -> NetworkTime {
        self.id.timestamp()
    }
    fn claim(&self) -> &Claim {
        &self.claim
    }
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

fn test_root(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    PathBuf::from(format!(
        "/tmp/tc-collection-route-{name}-{nanos}-{}",
        std::process::id()
    ))
}

async fn init_root(name: &str) -> PathBuf {
    let root = test_root(name);
    std::fs::create_dir_all(root.join("persistent")).expect("create persistent root");
    std::fs::create_dir_all(root.join("txn")).expect("create txn root");
    root
}

fn load_roots(
    root: &std::path::Path,
) -> (freqfs::DirLock<PersistentFile>, freqfs::DirLock<PersistentFile>) {
    let cache = Cache::<PersistentFile>::new(16 * 1024 * 1024, None);
    let persistent = Arc::clone(&cache)
        .load(root.join("persistent"))
        .expect("load persistent root");
    let txn = Arc::clone(&cache).load(root.join("txn")).expect("load txn root");
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

fn schema_value() -> tc_value::Value {
    simple_schema().to_value()
}

async fn make_table_with_data() -> crate::table::PersistentTable {
    let root = init_root("route-tests").await;
    let (persistent, txn) = load_roots(&root);
    let table = crate::table::PersistentTable::new(persistent, txn, simple_schema());
    table
        .upsert_row(tx(10), vec![tc_value::Value::from(1_u64)], vec![tc_value::Value::from("alpha")])
        .await
        .expect("upsert 1");
    table
        .upsert_row(tx(10), vec![tc_value::Value::from(2_u64)], vec![tc_value::Value::from("beta")])
        .await
        .expect("upsert 2");
    table
        .upsert_row(tx(10), vec![tc_value::Value::from(3_u64)], vec![tc_value::Value::from("gamma")])
        .await
        .expect("upsert 3");
    table.commit(tx(10)).expect("commit");
    table.finalize(tx(10)).await.expect("finalize");
    table
}

// ── Route resolution ──────────────────────────────────────────

#[test]
fn table_router_resolves_all_routes() {
    run_async_test("table_router_resolves_all_routes", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);

            assert!(router.route(&[]).is_some(), "root route");
            assert!(router.route(&[segment("columns")]).is_some());
            assert!(router.route(&[segment("contains")]).is_some());
            assert!(router.route(&[segment("count")]).is_some());
            assert!(router.route(&[segment("key_columns")]).is_some());
            assert!(router.route(&[segment("key_names")]).is_some());
            assert!(router.route(&[segment("limit")]).is_some());
            assert!(router.route(&[segment("order")]).is_some());
            assert!(router.route(&[segment("select")]).is_some());

            assert!(router.route(&[segment("unknown")]).is_none(), "unknown route");
            assert!(router.route(&[segment("count"), segment("extra")]).is_none(), "too deep");
        })
    });
}

#[test]
fn collection_router_resolves_schema_route() {
    run_async_test("collection_router_resolves_schema_route", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let coll = Collection::Table(table);
            let router = coll.router();

            assert!(router.route(&[]).is_some(), "root");
            assert!(router.route(&[segment("schema")]).is_some(), "schema route");
            assert!(router.route(&[segment("count")]).is_some(), "count route");
            assert!(router.route(&[segment("unknown")]).is_none(), "unknown");
        })
    });
}

// ── Schema roundtrip ──────────────────────────────────────────

#[test]
fn schema_roundtrip() {
    let original = simple_schema();
    let encoded = original.to_value();
    let decoded = TableSchema::try_from_value(encoded).expect("decode schema");
    assert_eq!(original.key(), decoded.key());
    assert_eq!(original.values(), decoded.values());
}

// ── GET handlers ──────────────────────────────────────────────

#[test]
fn get_table_all_returns_table() {
    run_async_test("get_table_all_returns_table", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(20);
            let fut = handler
                .get(&txn, Scalar::Value(tc_value::Value::None))
                .expect("get all");
            let resp = fut.await.expect("response");
            assert!(matches!(resp, TableResponse::Table(_)));
        })
    });
}

#[test]
fn get_table_key_returns_row() {
    run_async_test("get_table_key_returns_row", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let fut = handler.get(&txn, req).expect("get key");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Tuple(row)) => {
                    assert_eq!(row, vec![Value::from(2_u64), Value::from("beta")]);
                }
                other => panic!("expected row value, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_columns() {
    run_async_test("get_columns", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("columns")];
            let handler = router.route(&path).expect("columns route");
            let txn = MockTxn::new(20);
            let fut = handler
                .get(&txn, Scalar::Value(Value::None))
                .expect("get columns");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Tuple(cols)) => {
                    assert_eq!(cols.len(), 2);
                    assert_eq!(cols[0], Value::from("id"));
                    assert_eq!(cols[1], Value::from("label"));
                }
                other => panic!("expected value, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_count_all() {
    run_async_test("get_count_all", || {
        Box::pin(async {
            use safecast::CastFrom;
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("count")];
            let handler = router.route(&path).expect("count route");
            let txn = MockTxn::new(20);
            let fut = handler
                .get(&txn, Scalar::Value(Value::None))
                .expect("get count");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Number(n)) => {
                    assert_eq!(u64::cast_from(n), 3);
                }
                other => panic!("expected count value, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_count_key() {
    run_async_test("get_count_key", || {
        Box::pin(async {
            use safecast::CastFrom;
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("count")];
            let handler = router.route(&path).expect("count route");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let fut = handler.get(&txn, req).expect("get count key");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Number(n)) => {
                    assert_eq!(u64::cast_from(n), 1);
                }
                other => panic!("expected count value, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_contains_key() {
    run_async_test("get_contains_key", || {
        Box::pin(async {
            use safecast::CastFrom;
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("contains")];
            let handler = router.route(&path).expect("contains route");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let fut = handler.get(&txn, req).expect("get contains");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Number(n)) => {
                    assert!(bool::cast_from(n));
                }
                other => panic!("expected bool value, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_key_columns() {
    run_async_test("get_key_columns", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("key_columns")];
            let handler = router.route(&path).expect("key_columns route");
            let txn = MockTxn::new(20);
            let fut = handler
                .get(&txn, Scalar::Value(Value::None))
                .expect("get key_columns");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Tuple(cols)) => {
                    assert_eq!(cols.len(), 1);
                    assert_eq!(cols[0], Value::from("id"));
                }
                other => panic!("expected value, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_limit() {
    run_async_test("get_limit", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("limit")];
            let handler = router.route(&path).expect("limit route");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::from(2_u64));
            let fut = handler.get(&txn, req).expect("get limit");
            let resp = fut.await.expect("response");
            assert!(matches!(resp, TableResponse::Limited(_)));
        })
    });
}

#[test]
fn get_order() {
    run_async_test("get_order", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("order")];
            let handler = router.route(&path).expect("order route");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from("id")]));
            let fut = handler.get(&txn, req).expect("get order");
            let resp = fut.await.expect("response");
            assert!(matches!(resp, TableResponse::Slice(_)));
        })
    });
}

#[test]
fn get_select() {
    run_async_test("get_select", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("select")];
            let handler = router.route(&path).expect("select route");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from("label")]));
            let fut = handler.get(&txn, req).expect("get select");
            let resp = fut.await.expect("response");
            assert!(matches!(resp, TableResponse::Selection(_)));
        })
    });
}

// ── PUT handler: upsert ───────────────────────────────────────

#[test]
fn put_upsert_via_key() {
    run_async_test("put_upsert_via_key", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table.clone());
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(30);

            let mut params = tc_ir::Map::new();
            params.insert(
                "key".parse().expect("Id"),
                Scalar::Value(Value::Tuple(vec![Value::from(2_u64)])),
            );
            params.insert(
                "value".parse().expect("Id"),
                Scalar::Value(Value::Tuple(vec![Value::from("updated")])),
            );

            let fut = handler.put(&txn, params).expect("put upsert");
            fut.await.expect("upsert ok");

            let row = table.read_row(tx(30), &[Value::from(2_u64)]).await;
            assert!(row.is_some());
            assert_eq!(
                row.unwrap().as_ref(),
                &[Value::from(2_u64), Value::from("updated")]
            );
        })
    });
}

// ── PUT handler: update ───────────────────────────────────────

#[test]
fn put_update_all() {
    run_async_test("put_update_all", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table.clone());
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(30);

            let mut params = tc_ir::Map::new();
            params.insert("key".parse().expect("Id"), Scalar::Value(Value::None));
            params.insert(
                "value".parse().expect("Id"),
                Scalar::Value(Value::Tuple(vec![
                    Value::Tuple(vec![Value::from("label"), Value::from("updated")]),
                ])),
            );

            let fut = handler.put(&txn, params).expect("put update all");
            fut.await.expect("update ok");

            for id in [1_u64, 2_u64, 3_u64] {
                let row = table.read_row(tx(30), &[Value::from(id)]).await;
                assert!(row.is_some(), "row {id} should exist");
                assert_eq!(
                    row.unwrap().as_ref(),
                    &[Value::from(id), Value::from("updated")]
                );
            }
        })
    });
}

#[test]
fn put_update_range() {
    run_async_test("put_update_range", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table.clone());
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(30);

            let key_selector = Value::Tuple(vec![
                Value::Tuple(vec![
                    Value::from("id"),
                    Value::Tuple(vec![Value::from(1_u64), Value::from(2_u64)]),
                ]),
            ]);

            let mut params = tc_ir::Map::new();
            params.insert("key".parse().expect("Id"), Scalar::Value(key_selector));
            params.insert(
                "value".parse().expect("Id"),
                Scalar::Value(Value::Tuple(vec![
                    Value::Tuple(vec![Value::from("label"), Value::from("range_updated")]),
                ])),
            );

            let fut = handler.put(&txn, params).expect("put update range");
            fut.await.expect("update ok");

            let row1 = table.read_row(tx(30), &[Value::from(1_u64)]).await;
            assert_eq!(
                row1.unwrap().as_ref(),
                &[Value::from(1_u64), Value::from("range_updated")]
            );

            let row2 = table.read_row(tx(30), &[Value::from(2_u64)]).await;
            assert_eq!(
                row2.unwrap().as_ref(),
                &[Value::from(2_u64), Value::from("range_updated")]
            );

            let row3 = table.read_row(tx(30), &[Value::from(3_u64)]).await;
            assert_eq!(
                row3.unwrap().as_ref(),
                &[Value::from(3_u64), Value::from("gamma")]
            );
        })
    });
}

#[test]
fn update_direct_method() {
    run_async_test("update_direct_method", || {
        Box::pin(async {
            use std::collections::HashMap;
            use tc_value::Value;
            let table = make_table_with_data().await;

            let mut updates = HashMap::new();
            updates.insert(
                "label".parse().expect("Id"),
                Value::from("method_updated"),
            );

            table
                .update(tx(30), b_table::Range::default(), updates)
                .await
                .expect("update");

            for id in [1_u64, 2_u64, 3_u64] {
                let row = table.read_row(tx(30), &[Value::from(id)]).await;
                assert_eq!(
                    row.unwrap().as_ref(),
                    &[Value::from(id), Value::from("method_updated")]
                );
            }
        })
    });
}

// ── POST handler ──────────────────────────────────────────────

#[test]
fn post_slice() {
    run_async_test("post_slice", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(20);

            let req = Scalar::Value(Value::Tuple(vec![
                Value::Tuple(vec![
                    Value::from("id"),
                    Value::Tuple(vec![Value::from(1_u64), Value::from(2_u64)]),
                ]),
            ]));

            let fut = handler.post(&txn, req).expect("post slice");
            let resp = fut.await.expect("response");
            assert!(matches!(resp, TableResponse::Slice(_)));
        })
    });
}

// ── DELETE handler ────────────────────────────────────────────

#[test]
fn delete_key() {
    run_async_test("delete_key", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table.clone());
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(30);

            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let fut = handler.delete(&txn, req).expect("delete key");
            fut.await.expect("delete ok");

            let row = table.read_row(tx(30), &[Value::from(2_u64)]).await;
            assert!(row.is_none(), "row should be deleted");
        })
    });
}

#[test]
fn delete_all_truncates() {
    run_async_test("delete_all_truncates", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let router = TableRouter::new(table.clone());
            let handler = router.route(&[]).expect("root route");
            let txn = MockTxn::new(30);

            let req = Scalar::Value(Value::None);
            let fut = handler.delete(&txn, req).expect("delete all");
            fut.await.expect("truncate ok");

            assert!(table.is_empty(tx(30)).await, "table should be empty");
        })
    });
}

// ── Collection schema route ───────────────────────────────────

#[test]
fn collection_schema_route() {
    run_async_test("collection_schema_route", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let coll = Collection::Table(table);
            let router = coll.router();
            let path = [segment("schema")];
            let handler = router.route(&path).expect("schema route");
            let txn = MockTxn::new(20);

            let fut = handler
                .get(&txn, Scalar::Value(Value::None))
                .expect("get schema");
            let resp = fut.await.expect("response");
            match resp {
                TableResponse::Value(Value::Tuple(outer)) => {
                    assert_eq!(outer.len(), 2);
                }
                other => panic!("expected schema value, got {other:?}"),
            }
        })
    });
}

// ── Static routes: create ─────────────────────────────────────

#[test]
fn static_create_route() {
    run_async_test("static_create_route", || {
        Box::pin(async {
            let root = init_root("static-create").await;
            let cache = Cache::<PersistentFile>::new(16 * 1024 * 1024, None);
            let dir = Arc::clone(&cache).load(root).expect("load root");

            let stat = TableStatic::new(dir);
            let handler = stat.route(&[]).expect("create route");
            let txn = MockTxn::new(40);

            let req = Scalar::Value(schema_value());
            let fut = handler.get(&txn, req).expect("create table");
            let resp = fut.await.expect("response");
            assert!(matches!(resp, TableResponse::Table(_)));
        })
    });
}

// ── Static routes: copy_from ──────────────────────────────────

#[test]
fn static_copy_from_inline_rows() {
    run_async_test("static_copy_from_inline_rows", || {
        Box::pin(async {
            use tc_value::Value;
            let root = init_root("static-copy").await;
            let cache = Cache::<PersistentFile>::new(16 * 1024 * 1024, None);
            let dir = Arc::clone(&cache).load(root).expect("load root");

            let stat = TableStatic::new(dir);
            let path = [segment("copy_from")];
            let handler = stat.route(&path).expect("copy_from route");
            let txn = MockTxn::new(40);

            let source_rows = Value::Tuple(vec![
                Value::Tuple(vec![Value::from(10_u64), Value::from("row1")]),
                Value::Tuple(vec![Value::from(20_u64), Value::from("row2")]),
            ]);

            let mut params = tc_ir::Map::new();
            params.insert("schema".parse().expect("Id"), Scalar::Value(schema_value()));
            params.insert("source".parse().expect("Id"), Scalar::Value(source_rows));

            let fut = handler.post(&txn, params).expect("copy_from");
            let resp = fut.await.expect("response");

            match resp {
                TableResponse::Table(table) => {
                    assert_eq!(table.count(tx(40)).await, 2);
                    let row = table.read_row(tx(40), &[Value::from(10_u64)]).await;
                    assert!(row.is_some());
                    assert_eq!(
                        row.unwrap().as_ref(),
                        &[Value::from(10_u64), Value::from("row1")]
                    );
                }
                other => panic!("expected table, got {other:?}"),
            }
        })
    });
}

#[test]
fn copy_from_direct_method() {
    run_async_test("copy_from_direct_method", || {
        Box::pin(async {
            use tc_value::Value;
            let source = make_table_with_data().await;

            let root = init_root("copy-direct").await;
            let (persistent, txn) = load_roots(&root);
            let dest = crate::table::PersistentTable::new(persistent, txn, simple_schema());

            dest.copy_from(tx(10), &source).await.expect("copy");

            assert_eq!(dest.count(tx(10)).await, 3);

            let row = dest.read_row(tx(10), &[Value::from(2_u64)]).await;
            assert!(row.is_some());
            assert_eq!(
                row.unwrap().as_ref(),
                &[Value::from(2_u64), Value::from("beta")]
            );
        })
    });
}

// ── KeyOrRange parsing ────────────────────────────────────────

#[test]
fn key_or_range_all() {
    run_async_test("key_or_range_all", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let kor = KeyOrRange::try_from_value(&table, Value::None).expect("parse None");
            assert!(matches!(kor, KeyOrRange::All));
        })
    });
}

#[test]
fn key_or_range_key() {
    run_async_test("key_or_range_key", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let kor = KeyOrRange::try_from_value(
                &table,
                Value::Tuple(vec![Value::from(1_u64)]),
            )
            .expect("parse key");
            match kor {
                KeyOrRange::Key(key) => assert_eq!(key, vec![Value::from(1_u64)]),
                other => panic!("expected Key, got {other:?}"),
            }
        })
    });
}

#[test]
fn key_or_range_range() {
    run_async_test("key_or_range_range", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let selector = Value::Tuple(vec![
                Value::Tuple(vec![
                    Value::from("id"),
                    Value::Tuple(vec![Value::from(1_u64), Value::from(2_u64)]),
                ]),
            ]);
            let kor = KeyOrRange::try_from_value(&table, selector).expect("parse range");
            assert!(matches!(kor, KeyOrRange::Range(_)));
        })
    });
}

// ── Method-not-supported ──────────────────────────────────────

#[test]
fn put_on_count_route_rejected() {
    run_async_test("put_on_count_route_rejected", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("count")];
            let handler = router.route(&path).expect("count route");
            let txn = MockTxn::new(20);

            let params = tc_ir::Map::new();
            let result = handler.put(&txn, params);
            assert!(result.is_err());
        })
    });
}

#[test]
fn delete_on_columns_route_rejected() {
    run_async_test("delete_on_columns_route_rejected", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let router = TableRouter::new(table);
            let path = [segment("columns")];
            let handler = router.route(&path).expect("columns route");
            let txn = MockTxn::new(20);

            let result = handler.delete(&txn, Scalar::default());
            assert!(result.is_err());
        })
    });
}
