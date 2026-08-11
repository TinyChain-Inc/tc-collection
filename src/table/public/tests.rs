//! Route-level tests for table public API handlers.

use super::route;
use super::selector::KeyOrRange;
use crate::PersistentFile;
use crate::btree::StorageConfig;
use crate::table::{Column, LocalTable, PersistentTable, Table, TableSchema};
use crate::test::run_async_test;
use crate::{CollectionState, StorageContext};
use freqfs::Cache;
use safecast::TryCastInto;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tc_ir::{Claim, Handler, Map, NetworkTime, Scalar, Transact, Transaction, TxnId};
use tc_value::{ValueCollator, ValueType};
use umask::Mode;

fn segment(name: &str) -> pathlink::PathSegment {
    pathlink::PathSegment::from_str(name).expect("path segment")
}

fn tx(nonce: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), nonce)
}

/// Test response type for table handlers.
#[derive(Clone, Debug)]
enum State {
    Collection(Table<MockTxn>),
    Value(tc_value::Value),
    Map(Map<State>),
    Count(u64),
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Count(a), Self::Count(b)) => a == b,
            (Self::Value(a), Self::Value(b)) => a == b,
            (Self::Collection(a), Self::Collection(b)) => {
                matches!(a, Table::File(_)) && matches!(b, Table::File(_))
            }
            (Self::Map(a), Self::Map(b)) => a == b,
            _ => false,
        }
    }
}

impl From<Table<MockTxn>> for State {
    fn from(table: Table<MockTxn>) -> Self {
        Self::Collection(table)
    }
}

impl From<tc_value::Value> for State {
    fn from(v: tc_value::Value) -> Self {
        Self::Value(v)
    }
}

impl From<u64> for State {
    fn from(n: u64) -> Self {
        Self::Count(n)
    }
}

impl tc_ir::StateInstance for State {
    type Transaction = MockTxn;
}

impl crate::CollectionState for State {
    type Txn = MockTxn;

    fn none() -> Self {
        Self::Value(tc_value::Value::None)
    }

    fn from_scalar(scalar: Scalar) -> Self {
        match scalar {
            Scalar::Map(map) => Self::Map(
                map.into_iter()
                    .map(|(id, scalar)| (id, Self::from_scalar(scalar)))
                    .collect(),
            ),
            scalar => Self::Value(
                scalar
                    .try_cast_into(|scalar| panic!("test scalar must be a value: {scalar:?}"))
                    .expect("test scalar"),
            ),
        }
    }

    fn from_value(value: tc_value::Value) -> Self {
        Self::Value(value)
    }

    fn from_collection(collection: crate::Collection<MockTxn>) -> Self {
        match collection {
            crate::Collection::Table(table) => Self::Collection(*table),
            _ => panic!("table route test received a non-table collection"),
        }
    }

    fn into_scalar(self) -> tc_error::TCResult<Scalar> {
        match self {
            Self::Value(value) => Ok(Scalar::Value(value)),
            Self::Map(map) => map
                .into_iter()
                .map(|(id, state)| state.into_scalar().map(|scalar| (id, scalar)))
                .collect::<tc_error::TCResult<Map<_>>>()
                .map(Scalar::Map),
            Self::Count(count) => Ok(Scalar::Value(tc_value::Value::from(count))),
            Self::Collection(_) => Err(tc_error::TCError::bad_request("table is not a scalar")),
        }
    }

    fn into_value(self) -> tc_error::TCResult<tc_value::Value> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Count(count) => Ok(tc_value::Value::from(count)),
            Self::Map(_) | Self::Collection(_) => {
                Err(tc_error::TCError::bad_request("table is not a value"))
            }
        }
    }

    fn into_tuple(self) -> tc_error::TCResult<Vec<Self>> {
        match self.into_value()? {
            tc_value::Value::Tuple(values) => Ok(values.into_iter().map(Self::from).collect()),
            value => Err(tc_error::TCError::bad_request(format!(
                "expected tuple, found {value:?}"
            ))),
        }
    }

    fn into_map(self) -> tc_error::TCResult<Map<Self>> {
        match self {
            Self::Map(map) => Ok(map),
            _ => Err(tc_error::TCError::bad_request(
                "table test state is not a map",
            )),
        }
    }

    fn into_tensor(self) -> tc_error::TCResult<crate::tensor::Tensor> {
        Err(tc_error::TCError::bad_request(
            "table test state is not a tensor",
        ))
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::Value(tc_value::Value::None))
    }
}

#[derive(Clone, Debug)]
struct MockTxn {
    id: TxnId,
    claim: Claim,
    root: freqfs::DirLock<PersistentFile>,
    path: Vec<String>,
}

impl MockTxn {
    fn new(nonce: u16) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tc-collection-table-route-txn-{}-{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&root).expect("transaction root");
        let cache = Cache::<PersistentFile>::new(
            16 * 1024 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );
        let root = cache.load(root).expect("load transaction root");
        Self {
            id: tx(nonce),
            claim: Claim::new(
                pathlink::Link::from_str("/test").expect("link"),
                Mode::all(),
            ),
            root,
            path: Vec::new(),
        }
    }
}

impl crate::StorageContext for MockTxn {
    fn context(
        &self,
    ) -> impl std::future::Future<Output = tc_error::TCResult<freqfs::DirLock<PersistentFile>>> + Send
    {
        let root = self.root.clone();
        let mut path = vec![self.id.to_string()];
        path.extend(self.path.clone());
        async move {
            let mut current = root;
            for name in path {
                let next = {
                    let mut dir = current.write().await;
                    dir.get_or_create_dir(name)
                        .map_err(tc_error::TCError::internal)?
                };
                current = next;
            }
            Ok(current)
        }
    }

    fn subcontext(&self, name: impl Into<String>) -> Self {
        let mut txn = self.clone();
        txn.path.push(name.into());
        txn
    }

    fn subcontext_unique(&self) -> Self {
        self.subcontext(format!("literal-{}", self.id))
    }

    fn materialized_tensor_bytes(&self) -> usize {
        256 * 1024 * 1024
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
) -> (
    freqfs::DirLock<PersistentFile>,
    freqfs::DirLock<PersistentFile>,
) {
    let cache =
        Cache::<PersistentFile>::new(16 * 1024 * 1024, None, 0, std::time::Duration::from_secs(3));
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
    TableSchema::new(key, values, Vec::new(), StorageConfig::default()).expect("create test schema")
}

async fn make_table_with_data() -> PersistentTable<MockTxn> {
    use tc_value::Value;
    let root = init_root("route-tests").await;
    let (persistent, _) = load_roots(&root);
    let table = PersistentTable::new(persistent, simple_schema());
    table
        .upsert_row(
            &MockTxn::new(10),
            vec![Value::from(1_u64)],
            vec![Value::from("alpha")],
        )
        .await
        .expect("upsert 1");
    table
        .upsert_row(
            &MockTxn::new(10),
            vec![Value::from(2_u64)],
            vec![Value::from("beta")],
        )
        .await
        .expect("upsert 2");
    table
        .upsert_row(
            &MockTxn::new(10),
            vec![Value::from(3_u64)],
            vec![Value::from("gamma")],
        )
        .await
        .expect("upsert 3");
    table.commit(tx(10)).expect("commit");
    table.finalize(tx(10)).await.expect("finalize");
    table
}

async fn make_local_table_with_data(txn: &MockTxn) -> Table<MockTxn> {
    use tc_value::Value;

    let dir = crate::StorageContext::subcontext_unique(txn)
        .context()
        .await
        .expect("local table directory");
    let table = LocalTable::create(simple_schema(), ValueCollator::default(), dir)
        .expect("create local table");
    {
        let mut table = table.write().await;
        table
            .upsert(vec![Value::from(1_u64)], vec![Value::from("alpha")])
            .await
            .expect("upsert local row");
        table
            .upsert(vec![Value::from(2_u64)], vec![Value::from("beta")])
            .await
            .expect("upsert local row");
    }
    Table::Local(table)
}

// ── Route resolution ──────────────────────────────────────────

#[test]
fn route_resolves_all_paths() {
    run_async_test("route_resolves_all_paths", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            assert!(route::<State>(&table, &[]).is_some(), "root");
            assert!(route::<State>(&table, &[segment("columns")]).is_some());
            assert!(route::<State>(&table, &[segment("contains")]).is_some());
            assert!(route::<State>(&table, &[segment("count")]).is_some());
            assert!(route::<State>(&table, &[segment("key_columns")]).is_some());
            assert!(route::<State>(&table, &[segment("limit")]).is_some());
            assert!(route::<State>(&table, &[segment("order")]).is_some());
            assert!(route::<State>(&table, &[segment("select")]).is_some());
            assert!(route::<State>(&table, &[segment("unknown")]).is_none());
            assert!(route::<State>(&table, &[segment("count"), segment("x")]).is_none());
        })
    });
}

// ── Schema roundtrip ──────────────────────────────────────────

#[test]
fn schema_roundtrip() {
    let original = simple_schema();
    let encoded: tc_value::Value = safecast::CastFrom::cast_from(original.clone());
    let decoded: TableSchema = encoded
        .try_cast_into(|v| tc_error::bad_request!("invalid schema: {v:?}"))
        .expect("decode schema");
    assert_eq!(original.key(), decoded.key());
    assert_eq!(original.values(), decoded.values());
}

// ── GET handlers ──────────────────────────────────────────────

#[test]
fn get_table_all() {
    run_async_test("get_table_all", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(20);
            let resp = handler
                .get(&txn, Scalar::Value(tc_value::Value::None))
                .await
                .expect("response");
            assert!(matches!(resp, State::Collection(_)));
        })
    });
}

#[test]
fn local_table_uses_native_routes_without_transaction_deltas() {
    run_async_test(
        "local_table_uses_native_routes_without_transaction_deltas",
        || {
            Box::pin(async {
                use tc_value::Value;

                let txn = MockTxn::new(21);
                let table = make_local_table_with_data(&txn).await;
                let collection = crate::Collection::from(table.clone());
                collection
                    .commit(txn.id())
                    .await
                    .expect("local commit is a no-op");
                collection
                    .rollback(&txn.id())
                    .await
                    .expect("local rollback is a no-op");
                collection
                    .finalize(&txn.id())
                    .await
                    .expect("local finalize is a no-op");

                let count = route::<State>(&table, &[segment("count")]).expect("count route");
                assert_eq!(
                    count
                        .get(&txn, Scalar::Value(Value::None))
                        .await
                        .expect("count local rows"),
                    State::Count(2)
                );

                let root = route::<State>(&table, &[]).expect("root route");
                root.put(
                    &txn,
                    Scalar::Value(Value::Tuple(vec![Value::from(3_u64)])),
                    State::from_scalar(Scalar::Value(Value::from("gamma"))),
                )
                .await
                .expect("upsert local row");

                let row = root
                    .get(&txn, Scalar::Value(Value::Tuple(vec![Value::from(3_u64)])))
                    .await
                    .expect("read local row");
                assert_eq!(
                    row,
                    State::Value(Value::Tuple(vec![Value::from(3_u64), Value::from("gamma")]))
                );
            })
        },
    );
}

#[test]
fn decoded_table_is_local_not_persistent() {
    run_async_test("decoded_table_is_local_not_persistent", || {
        Box::pin(async {
            use safecast::CastFrom;
            use tc_value::Value;

            let schema = simple_schema();
            let payload = (
                Value::cast_from(schema),
                vec![
                    Value::Tuple(vec![Value::from(1_u64), Value::from("alpha")]),
                    Value::Tuple(vec![Value::from(2_u64), Value::from("beta")]),
                ],
            );
            let stream = destream_json::encode(payload).expect("encode Table payload");
            let decoded: crate::table::DecodedTablePayload<MockTxn> =
                destream_json::try_decode(MockTxn::new(22), stream)
                    .await
                    .expect("decode Table payload");

            assert!(matches!(decoded.table, Table::Local(_)));
            assert_eq!(decoded.table.count(tx(22)).await.expect("count rows"), 2);
        })
    });
}

#[test]
fn get_table_key() {
    run_async_test("get_table_key", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let resp = handler.get(&txn, req).await.expect("response");
            match resp {
                State::Value(Value::Tuple(row)) => {
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
            let handler = route::<State>(&table, &[segment("columns")]).expect("columns");
            let txn = MockTxn::new(20);
            let resp = handler
                .get(&txn, Scalar::Value(Value::None))
                .await
                .expect("response");
            match resp {
                State::Value(Value::Tuple(cols)) => {
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
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("count")]).expect("count");
            let txn = MockTxn::new(20);
            let resp = handler
                .get(&txn, Scalar::Value(tc_value::Value::None))
                .await
                .expect("response");
            assert_eq!(resp, State::Count(3));
        })
    });
}

#[test]
fn get_count_key() {
    run_async_test("get_count_key", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("count")]).expect("count");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let resp = handler.get(&txn, req).await.expect("response");
            assert_eq!(resp, State::Count(1));
        })
    });
}

#[test]
fn get_count_missing_key() {
    run_async_test("get_count_missing_key", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("count")]).expect("count");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(999_u64)]));
            let resp = handler.get(&txn, req).await.expect("response");
            assert_eq!(resp, State::Count(0));
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
            let handler = route::<State>(&table, &[segment("contains")]).expect("contains");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let resp = handler.get(&txn, req).await.expect("response");
            match resp {
                State::Value(Value::Number(n)) => assert!(bool::cast_from(n)),
                other => panic!("expected bool, got {other:?}"),
            }
        })
    });
}

#[test]
fn get_contains_missing() {
    run_async_test("get_contains_missing", || {
        Box::pin(async {
            use safecast::CastFrom;
            use tc_value::Value;
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("contains")]).expect("contains");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from(999_u64)]));
            let resp = handler.get(&txn, req).await.expect("response");
            match resp {
                State::Value(Value::Number(n)) => assert!(!bool::cast_from(n)),
                other => panic!("expected bool, got {other:?}"),
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
            let handler = route::<State>(&table, &[segment("key_columns")]).expect("key_columns");
            let txn = MockTxn::new(20);
            let resp = handler
                .get(&txn, Scalar::Value(Value::None))
                .await
                .expect("response");
            match resp {
                State::Value(Value::Tuple(cols)) => {
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
            let handler = route::<State>(&table, &[segment("limit")]).expect("limit");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::from(2_u64));
            let resp = handler.get(&txn, req).await.expect("response");
            assert!(matches!(resp, State::Collection(_)));
        })
    });
}

#[test]
fn get_order() {
    run_async_test("get_order", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("order")]).expect("order");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from("id")]));
            let resp = handler.get(&txn, req).await.expect("response");
            assert!(matches!(resp, State::Collection(_)));
        })
    });
}

#[test]
fn get_select() {
    run_async_test("get_select", || {
        Box::pin(async {
            use tc_value::Value;
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("select")]).expect("select");
            let txn = MockTxn::new(20);
            let req = Scalar::Value(Value::Tuple(vec![Value::from("label")]));
            let resp = handler.get(&txn, req).await.expect("response");
            assert!(matches!(resp, State::Collection(_)));
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
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(30);

            let key = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            let value =
                State::from_scalar(Scalar::Value(Value::Tuple(vec![Value::from("updated")])));

            handler.put(&txn, key, value).await.expect("upsert ok");

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
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(30);

            let mut value_map = Map::new();
            value_map.insert(
                "label".parse().expect("Id"),
                Scalar::Value(Value::from("updated")),
            );

            handler
                .put(
                    &txn,
                    Scalar::Value(Value::None),
                    State::from_scalar(Scalar::Map(value_map)),
                )
                .await
                .expect("update ok");

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
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(30);

            let key_selector = Value::Tuple(vec![Value::Tuple(vec![
                Value::from("id"),
                Value::Tuple(vec![Value::from(1_u64), Value::from(2_u64)]),
            ])]);

            let mut value_map = Map::new();
            value_map.insert(
                "label".parse().expect("Id"),
                Scalar::Value(Value::from("range_updated")),
            );

            handler
                .put(
                    &txn,
                    Scalar::Value(key_selector),
                    State::from_scalar(Scalar::Map(value_map)),
                )
                .await
                .expect("update ok");

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
            use tc_value::Value;
            let table = make_table_with_data().await;

            let mut updates = tc_ir::Map::new();
            updates.insert("label".parse().expect("Id"), Value::from("method_updated"));

            table
                .update(&MockTxn::new(30), b_table::Range::default(), updates)
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
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(20);

            let mut req = Map::new();
            req.insert(
                "id".parse().expect("id"),
                State::from_scalar(Scalar::Value(Value::Tuple(vec![
                    Value::from(1_u64),
                    Value::from(2_u64),
                ]))),
            );

            let resp = handler.post(&txn, req).await.expect("response");
            assert!(matches!(resp, State::Collection(_)));
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
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(30);

            let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
            handler.delete(&txn, req).await.expect("delete ok");

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
            let handler = route::<State>(&table, &[]).expect("root");
            let txn = MockTxn::new(30);

            let req = Scalar::Value(Value::None);
            handler.delete(&txn, req).await.expect("truncate ok");

            assert!(table.is_empty(tx(30)).await, "table should be empty");
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
            let kor = KeyOrRange::try_from_value(table.schema(), Value::None).expect("parse None");
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
            let kor =
                KeyOrRange::try_from_value(table.schema(), Value::Tuple(vec![Value::from(1_u64)]))
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
            let selector = Value::Tuple(vec![Value::Tuple(vec![
                Value::from("id"),
                Value::Tuple(vec![Value::from(1_u64), Value::from(2_u64)]),
            ])]);
            let kor = KeyOrRange::try_from_value(table.schema(), selector).expect("parse range");
            assert!(matches!(kor, KeyOrRange::Range(_)));
        })
    });
}

// ── Method-not-supported ──────────────────────────────────────

#[test]
fn put_on_count_rejected() {
    run_async_test("put_on_count_rejected", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("count")]).expect("count");
            let txn = MockTxn::new(20);
            let result = handler
                .put(
                    &txn,
                    Scalar::default(),
                    State::from_scalar(Scalar::default()),
                )
                .await;
            assert!(result.is_err());
        })
    });
}

#[test]
fn delete_on_columns_rejected() {
    run_async_test("delete_on_columns_rejected", || {
        Box::pin(async {
            let table = make_table_with_data().await;
            let handler = route::<State>(&table, &[segment("columns")]).expect("columns");
            let txn = MockTxn::new(20);
            let result = handler.delete(&txn, Scalar::default()).await;
            assert!(result.is_err());
        })
    });
}
