//! Route-level tests for table public API handlers.

use std::str::FromStr;
use std::sync::Arc;

use freqfs::Cache;
use safecast::{CastFrom, TryCastFrom, TryCastInto};

use super::selector::KeyOrRange;
use crate::btree::StorageConfig;
use crate::collection::CollectionSchema;
use crate::table::{Column, PersistentTable, Table, TableSchema};
use crate::test::{TempRoot, TestTxn as MockTxn};
use crate::{PersistentFile, StorageContext};
use tc_ir::{Map, NetworkTime, Scalar, Transact, Transaction, TxnId};
use tc_value::{Value, ValueType};

fn segment(name: &str) -> pathlink::PathSegment {
    pathlink::PathSegment::from_str(name).expect("path segment")
}

fn tx(nonce: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), nonce)
}

/// Test response type for table handlers.
#[derive(Clone, Debug)]
enum State {
    Collection(Box<Table<MockTxn>>),
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
                matches!(a.as_ref(), Table::File(_)) && matches!(b.as_ref(), Table::File(_))
            }
            (Self::Map(a), Self::Map(b)) => a == b,
            _ => false,
        }
    }
}

impl From<Table<MockTxn>> for State {
    fn from(table: Table<MockTxn>) -> Self {
        Self::Collection(Box::new(table))
    }
}

impl From<tc_value::Value> for State {
    fn from(v: tc_value::Value) -> Self {
        Self::Value(v)
    }
}

impl From<Scalar> for State {
    fn from(scalar: Scalar) -> Self {
        match scalar {
            Scalar::Map(map) => Self::Map(
                map.into_iter()
                    .map(|(id, scalar)| (id, Self::from(scalar)))
                    .collect(),
            ),
            kind => Self::Value(
                kind.try_cast_into(|scalar| panic!("test scalar must be a value: {scalar:?}"))
                    .expect("test scalar"),
            ),
        }
    }
}

impl From<crate::Collection<MockTxn>> for State {
    fn from(collection: crate::Collection<MockTxn>) -> Self {
        match collection {
            crate::Collection::Table(table) => Self::Collection(table),
            _ => panic!("table route test received a non-table collection"),
        }
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

    fn into_tensor(self) -> tc_error::TCResult<crate::tensor::Tensor<Self::Txn>> {
        Err(tc_error::TCError::bad_request(
            "table test state is not a tensor",
        ))
    }

    fn as_number(&self) -> Option<number_general::Number> {
        match self {
            Self::Value(Value::Number(number)) => Some(*number),
            Self::Count(count) => Some((*count).into()),
            _ => None,
        }
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::Value(value) if matches!(&value, tc_value::Value::None))
    }
}

async fn init_root(name: &str) -> Arc<TempRoot> {
    let root = TempRoot::new(name);
    std::fs::create_dir(root.join("persistent")).expect("create persistent root");
    std::fs::create_dir(root.join("txn")).expect("create txn root");
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
        dtype: ValueType::Number(tc_value::NumberType::Number),
    }];

    let values = vec![Column {
        name: "label".parse().expect("Id"),
        dtype: ValueType::String,
    }];

    TableSchema::new(key, values, Vec::new(), StorageConfig::default()).expect("create test schema")
}

async fn make_table_with_data() -> (Arc<TempRoot>, Table<MockTxn>) {
    use tc_value::Value;

    let root = init_root("route-tests").await;
    let (persistent, _) = load_roots(&root);
    let txn = MockTxn::fresh(10);
    let table = PersistentTable::try_new(persistent, simple_schema())
        .await
        .unwrap();

    table
        .upsert_row(&txn, vec![Value::from(1_u64)], vec![Value::from("alpha")])
        .await
        .expect("upsert 1");

    table
        .upsert_row(&txn, vec![Value::from(2_u64)], vec![Value::from("beta")])
        .await
        .expect("upsert 2");

    table
        .upsert_row(&txn, vec![Value::from(3_u64)], vec![Value::from("gamma")])
        .await
        .expect("upsert 3");

    table.commit(tx(10)).await.expect("commit");
    table.finalize(tx(10)).await.expect("finalize");
    (root, Table::from(table))
}

async fn make_local_table_with_data(txn: &MockTxn) -> Table<MockTxn> {
    use tc_value::Value;

    let dir = crate::StorageContext::subcontext_unique(txn)
        .context()
        .await
        .expect("local table directory");

    let table = b_table::TableLock::create(simple_schema(), crate::persistence::value_order(), dir)
        .await
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

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn route_resolves_all_paths() {
    Box::pin(async {
        let (_root, table) = make_table_with_data().await;
        assert!(tc_ir::Route::<State>::route(&table, &[]).is_some(), "root");
        assert!(tc_ir::Route::<State>::route(&table, &[segment("columns")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("contains")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("count")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("insert")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("is_empty")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("key_columns")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("key_names")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("limit")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("order")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("select")]).is_some());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("unknown")]).is_none());
        assert!(tc_ir::Route::<State>::route(&table, &[segment("count"), segment("x")]).is_none());
    })
    .await;
}

// ── Schema roundtrip ──────────────────────────────────────────

#[test]
fn schema_roundtrip() {
    let original = simple_schema();
    let encoded: tc_value::Value = safecast::CastFrom::cast_from(original.clone());
    let decoded = TableSchema::try_from(encoded).expect("decode schema");

    assert_eq!(original.key(), decoded.key());
    assert_eq!(original.values(), decoded.values());

    let dtype = Value::cast_from(ValueType::String);
    for fields in [
        vec![Value::from("label"), dtype.clone()],
        vec![Value::from(""), dtype.clone()],
        vec![Value::from(7_u64), dtype.clone()],
        vec![Value::from("label"), Value::from("invalid type")],
        vec![Value::from("label")],
        vec![Value::from("label"), dtype, Value::None],
    ] {
        let value = Value::Tuple(fields);
        let eligible = Column::can_cast_from(&value);
        let column = Column::opt_cast_from(value);
        assert_eq!(eligible, column.is_some());
        if let Some(column) = column {
            assert_eq!(column.name.as_str(), "label");
            assert_eq!(column.dtype, ValueType::String);
        }
    }
}

fn malformed_table_schemas() -> Vec<Value> {
    const INDEX_PATH: pathlink::PathLabel = pathlink::path_label(&["index"]);

    let column = |name: Value, dtype: Value| Value::Tuple(vec![name, dtype]);
    let dtype = Value::cast_from(ValueType::String);
    let key = Value::Tuple(vec![column(Value::from("id"), dtype.clone())]);
    let values = Value::Tuple(vec![column(Value::from("label"), dtype.clone())]);
    let schema =
        |key, values, indices| Value::Tuple(vec![Value::Tuple(vec![key, values]), indices]);

    let index = |name, columns| Value::Tuple(vec![Value::Tuple(vec![name, columns])]);

    vec![
        schema(key.clone(), key.clone(), Value::None),
        schema(
            Value::Tuple(vec![
                column(Value::from("id"), dtype.clone()),
                column(Value::from("id"), dtype.clone()),
            ]),
            values.clone(),
            Value::None,
        ),
        schema(
            key.clone(),
            values.clone(),
            index(
                Value::from("primary"),
                Value::Tuple(vec![Value::from("label")]),
            ),
        ),
        schema(
            key.clone(),
            values.clone(),
            index(Value::from("by_key"), Value::Tuple(vec![Value::from("id")])),
        ),
        schema(
            key.clone(),
            values.clone(),
            index(
                Value::from("duplicate_columns"),
                Value::Tuple(vec![Value::from("label"), Value::from("label")]),
            ),
        ),
        schema(
            key.clone(),
            values.clone(),
            index(Value::from("empty_columns"), Value::Tuple(vec![])),
        ),
        schema(
            key.clone(),
            values.clone(),
            Value::Tuple(vec![
                Value::Tuple(vec![
                    Value::from("same"),
                    Value::Tuple(vec![Value::from("label")]),
                ]),
                Value::Tuple(vec![
                    Value::from("same"),
                    Value::Tuple(vec![Value::from("label")]),
                ]),
            ]),
        ),
        Value::None,
        Value::Tuple(vec![]),
        Value::Tuple(vec![Value::Tuple(vec![key.clone(), values.clone()])]),
        Value::Tuple(vec![Value::None, Value::None]),
        Value::Tuple(vec![Value::Tuple(vec![key.clone()]), Value::None]),
        schema(Value::None, values.clone(), Value::None),
        schema(key.clone(), Value::None, Value::None),
        schema(Value::Tuple(vec![]), values.clone(), Value::None),
        schema(
            Value::Tuple(vec![column(Value::from(7_u64), dtype.clone())]),
            values.clone(),
            Value::None,
        ),
        schema(
            Value::Tuple(vec![column(Value::from(""), dtype)]),
            values.clone(),
            Value::None,
        ),
        schema(
            Value::Tuple(vec![column(Value::from("id"), Value::from("invalid type"))]),
            values.clone(),
            Value::None,
        ),
        schema(key.clone(), values.clone(), Value::from("indices")),
        schema(key.clone(), values.clone(), Value::Tuple(vec![Value::None])),
        schema(
            key.clone(),
            values.clone(),
            Value::Tuple(vec![Value::Tuple(vec![Value::from("by_label")])]),
        ),
        schema(
            key.clone(),
            values.clone(),
            index(Value::from("by_missing"), Value::from("missing")),
        ),
        schema(
            key.clone(),
            values.clone(),
            index(Value::from("by_label"), Value::from(7_u64)),
        ),
        schema(
            key.clone(),
            values.clone(),
            index(Value::from(7_u64), Value::from("label")),
        ),
        schema(
            key,
            values,
            index(
                Value::Link(pathlink::Link::from(INDEX_PATH)),
                Value::from("label"),
            ),
        ),
    ]
}

#[test]
fn malformed_schema_construction_returns_structured_errors() {
    use tc_value::class::NativeClass;

    for value in malformed_table_schemas() {
        let error = TableSchema::try_from(value.clone()).unwrap_err();
        assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);

        let native = (crate::TableType.path(), value);
        let error = CollectionSchema::try_from(native).unwrap_err();
        assert!(!error.requires_recovery());

        assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
    }
}

#[test]
fn schema_conversion_preserves_index_shapes_and_moves_names() {
    use tc_value::class::NativeClass;

    for columns in [
        Value::from("label"),
        Value::Tuple(vec![Value::from("label")]),
    ] {
        let name = String::from("by_label");
        let pointer = name.as_ptr();
        let mut value = Value::cast_from(simple_schema());
        let Value::Tuple(outer) = &mut value else {
            panic!("schema tuple")
        };

        outer[1] = Value::Tuple(vec![Value::Tuple(vec![Value::String(name), columns])]);
        let decoded = TableSchema::try_from(value).unwrap();
        assert_eq!(decoded.indices().len(), 1);
        let (name, index) = &decoded.indices()[0];
        assert_eq!(name, "by_label");
        assert_eq!(name.as_ptr(), pointer);
        assert_eq!(
            b_table::IndexSchema::columns(index),
            &[
                "label".parse::<tc_ir::Id>().unwrap(),
                "id".parse::<tc_ir::Id>().unwrap(),
            ]
        );

        let encoded = Value::cast_from(decoded.clone());
        let decoded_again = TableSchema::try_from(encoded.clone()).unwrap();
        assert_eq!(decoded_again, decoded);
        let native = (crate::TableType.path(), encoded);
        let CollectionSchema::Table(collection) = CollectionSchema::try_from(native).unwrap()
        else {
            panic!("Table schema")
        };

        assert_eq!(collection, decoded);
    }

    for indices in [Value::None, Value::Tuple(vec![])] {
        let mut value = Value::cast_from(simple_schema());
        let Value::Tuple(outer) = &mut value else {
            panic!("schema tuple")
        };

        outer[1] = indices;
        let decoded = TableSchema::try_from(value).unwrap();
        assert!(decoded.indices().is_empty());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn malformed_schema_decoding_precedes_native_table_creation() {
    Box::pin(async {
        for binary in [false, true] {
            for schema in malformed_table_schemas() {
                let txn = MockTxn::fresh(23);
                let payload = (schema, vec![vec![Value::from(7_u64), Value::from("row")]]);
                let result: Result<crate::table::DecodedTablePayload<MockTxn>, _> = if binary {
                    tbon::de::try_decode(txn.clone(), tbon::en::encode(payload).unwrap())
                        .await
                        .map_err(|error| error.to_string())
                } else {
                    destream_json::try_decode(txn.clone(), destream_json::encode(payload).unwrap())
                        .await
                        .map_err(|error| error.to_string())
                };

                assert!(result.is_err());
                let owner = txn.context().await.unwrap();
                let workspace = {
                    let owner = owner.read().await;
                    assert_eq!(owner.iter().count(), 1);
                    owner.iter().next().unwrap().1.as_dir().unwrap().clone()
                };

                assert!(workspace.read().await.is_empty());
            }
        }
    })
    .await;
}

// ── GET handlers ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_table_all() {
    Box::pin(async {
        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let resp = {
            let receiver = table.clone();
            let handler = tc_ir::Route::<State>::route(&receiver, &[]).expect("root");
            handler.get().expect("GET handler")(&txn, Scalar::Value(tc_value::Value::None))
                .await
                .expect("response")
        }; // The returned view owns its storage independently of the receiver.

        let State::Collection(view) = resp else {
            panic!("expected a Table view")
        };

        assert!(matches!(*view, Table::Slice(_)));
        assert!(view.persistent().is_err());
        assert_eq!(view.count(txn.id()).await.unwrap(), 3);
        assert!(
            view.read_row(txn.id(), &[1_u64.into()])
                .await
                .unwrap()
                .is_some()
        );

        assert_eq!(view.limit(1).unwrap().count(txn.id()).await.unwrap(), 1);

        let before = table.read_row(txn.id(), &[1_u64.into()]).await.unwrap();
        assert!(
            view.upsert_row(&txn, vec![1_u64.into()], vec!["changed".into()])
                .await
                .is_err()
        );

        assert!(
            view.update(&txn, b_table::Range::default(), Map::new())
                .await
                .is_err()
        );

        assert!(
            view.truncate(&txn, b_table::Range::default())
                .await
                .is_err()
        );

        assert_eq!(
            table.read_row(txn.id(), &[1_u64.into()]).await.unwrap(),
            before
        );

        let collection = crate::Collection::Table(view);
        let owner = crate::Collection::from(table);
        assert!(collection.sync().await.is_err());
        assert!(collection.sync_all().await.is_err());
        assert!(collection.restore_from(&txn, &owner).await.is_err());
        assert!(owner.restore_from(&txn, &collection).await.is_err());
        assert!(collection.commit(txn.id()).await.is_err());
        assert!(collection.rollback(&txn.id()).await.is_err());
        assert!(collection.finalize(&txn.id()).await.is_err());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn local_table_uses_native_routes_without_transaction_deltas() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(21);
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

        let count = tc_ir::Route::<State>::route(&table, &[segment("count")]).expect("count route");
        assert_eq!(
            count.get().expect("GET handler")(&txn, Scalar::Value(Value::None))
                .await
                .expect("count local rows"),
            State::Count(2)
        );

        let root = tc_ir::Route::<State>::route(&table, &[]).expect("root route");
        root.put().expect("PUT handler")(
            &txn,
            Scalar::Value(Value::Tuple(vec![Value::from(3_u64)])),
            State::from(Scalar::Value(Value::from("gamma"))),
        )
        .await
        .expect("upsert local row");

        let root = tc_ir::Route::<State>::route(&table, &[]).expect("root route");
        let row = root.get().expect("GET handler")(
            &txn,
            Scalar::Value(Value::Tuple(vec![Value::from(3_u64)])),
        )
        .await
        .expect("read local row");

        assert_eq!(
            row,
            State::Value(Value::Tuple(vec![Value::from(3_u64), Value::from("gamma")]))
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn decoded_table_is_local_not_persistent() {
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
        let txn = MockTxn::fresh(22);
        let decoded: crate::table::DecodedTablePayload<MockTxn> =
            destream_json::try_decode(txn.clone(), stream)
                .await
                .expect("decode Table payload");

        assert!(matches!(decoded.table, Table::Local(_)));
        assert_eq!(decoded.table.count(tx(22)).await.expect("count rows"), 2);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_table_key() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");
        let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        match resp {
            State::Value(value) => {
                let Value::Tuple(row) = value else {
                    panic!("expected tuple value")
                };

                assert_eq!(row, vec![Value::from(2_u64), Value::from("beta")]);
            }
            other => panic!("expected row value, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_columns() {
    Box::pin(async {
        use safecast::CastFrom;

        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("columns")]).expect("columns");
        let resp = handler.get().expect("GET handler")(&txn, Scalar::Value(Value::None))
            .await
            .expect("response");

        match resp {
            State::Value(value) => {
                let Value::Tuple(cols) = value else {
                    panic!("expected tuple value")
                };

                let expected = simple_schema()
                    .column_schema()
                    .map(Value::cast_from)
                    .collect::<Vec<_>>();

                assert_eq!(cols, expected);
            }
            other => panic!("expected value, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_count_all() {
    Box::pin(async {
        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("count")]).expect("count");
        let resp = handler.get().expect("GET handler")(&txn, Scalar::Value(tc_value::Value::None))
            .await
            .expect("response");

        assert_eq!(resp, State::Count(3));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_count_key() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("count")]).expect("count");
        let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        assert_eq!(resp, State::Count(1));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_count_missing_key() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("count")]).expect("count");
        let req = Scalar::Value(Value::Tuple(vec![Value::from(999_u64)]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        assert_eq!(resp, State::Count(0));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_contains_key() {
    Box::pin(async {
        use safecast::CastFrom;

        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler =
            tc_ir::Route::<State>::route(&table, &[segment("contains")]).expect("contains");

        let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        match resp {
            State::Value(value) => {
                let Value::Number(n) = value else {
                    panic!("expected boolean value")
                };

                assert!(bool::cast_from(n));
            }
            other => panic!("expected bool, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_contains_missing() {
    Box::pin(async {
        use safecast::CastFrom;

        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler =
            tc_ir::Route::<State>::route(&table, &[segment("contains")]).expect("contains");

        let req = Scalar::Value(Value::Tuple(vec![Value::from(999_u64)]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        match resp {
            State::Value(value) => {
                let Value::Number(n) = value else {
                    panic!("expected boolean value")
                };

                assert!(!bool::cast_from(n));
            }
            other => panic!("expected bool, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_key_columns() {
    Box::pin(async {
        use safecast::CastFrom;

        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler =
            tc_ir::Route::<State>::route(&table, &[segment("key_columns")]).expect("key_columns");

        let resp = handler.get().expect("GET handler")(&txn, Scalar::Value(Value::None))
            .await
            .expect("response");

        match resp {
            State::Value(value) => {
                let Value::Tuple(cols) = value else {
                    panic!("expected tuple value")
                };

                let expected = simple_schema()
                    .key_schema()
                    .map(Value::cast_from)
                    .collect::<Vec<_>>();

                assert_eq!(cols, expected);
            }
            other => panic!("expected value, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_key_names() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler =
            tc_ir::Route::<State>::route(&table, &[segment("key_names")]).expect("key_names");

        let resp = handler.get().expect("GET handler")(&txn, Scalar::Value(Value::None))
            .await
            .expect("response");

        assert_eq!(resp, State::Value(Value::Tuple(vec![Value::from("id")])));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_limit() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("limit")]).expect("limit");
        let req = Scalar::Value(Value::from(2_u64));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        assert!(matches!(resp, State::Collection(_)));
        use futures::FutureExt;

        for (table, source_count) in [(table, 3), (make_local_table_with_data(&txn).await, 2)] {
            for (limit, expected) in [
                (Value::from(0_u64), 0),
                (Value::from(1_i8), 1),
                (Value::from(2_f64), 2),
                (Value::from(u64::MAX), source_count),
            ] {
                let handler = tc_ir::Route::<State>::route(&table, &[segment("limit")])
                    .unwrap()
                    .get()
                    .unwrap();

                let State::Collection(limited) = handler(&txn, Scalar::Value(limit)).await.unwrap()
                else {
                    panic!("limited Table")
                };

                assert_eq!(limited.count(txn.id()).await.unwrap(), expected);
                assert_eq!(limited.is_empty(txn.id()).await.unwrap(), expected == 0);
            }

            let context = txn.context().await.unwrap();
            let held = context.write().await;
            for limit in [
                Value::from(-1_i64),
                Value::from(1.5_f64),
                Value::from(f64::NAN),
                Value::from(f64::INFINITY),
                Value::from(f64::NEG_INFINITY),
                Value::from((u64::MAX as u128 + 1) as f64),
                Value::from(true),
                Value::Number("1+1i".parse().unwrap()),
                Value::from("2"),
                Value::None,
            ] {
                let handler = tc_ir::Route::<State>::route(&table, &[segment("limit")])
                    .unwrap()
                    .get()
                    .unwrap();

                let error = handler(&txn, Scalar::Value(limit))
                    .now_or_never()
                    .expect("invalid limits reject without storage access")
                    .unwrap_err();

                assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
                assert!(!error.requires_recovery());
            }

            drop(held);
            assert_eq!(table.count(txn.id()).await.unwrap(), source_count);
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_order() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("order")]).expect("order");
        let req = Scalar::Value(Value::Tuple(vec![Value::from("id")]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        assert!(matches!(resp, State::Collection(_)));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn get_select() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("select")]).expect("select");
        let req = Scalar::Value(Value::Tuple(vec![Value::from("id"), Value::from("label")]));
        let resp = handler.get().expect("GET handler")(&txn, req)
            .await
            .expect("response");

        assert!(matches!(resp, State::Collection(_)));
        use futures::TryStreamExt;

        let columns = vec!["id".parse().unwrap(), "label".parse().unwrap()];
        for (table, source_count) in [(table, 3), (make_local_table_with_data(&txn).await, 2)] {
            for limit in [0, 2, 10] {
                let limited = table.limit(limit).unwrap();
                let direct = limited.select(columns.clone()).unwrap();
                let intermediate = Table::from(limited);
                let delegated = intermediate.select(&columns).unwrap();
                let collection = crate::Collection::from(intermediate);
                let handler = tc_ir::Route::<State>::route(&collection, &[segment("select")])
                    .unwrap()
                    .get()
                    .unwrap();

                let State::Collection(routed) = handler(
                    &txn,
                    Scalar::Value(Value::Tuple(vec!["id".into(), "label".into()])),
                )
                .await
                .unwrap() else {
                    panic!("selected Table")
                };

                assert_eq!(routed.schema(), direct.schema());
                assert_eq!(delegated.schema(), direct.schema());
                let expected: Vec<_> = direct
                    .rows(txn.id())
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();

                let delegated: Vec<_> = delegated
                    .rows(txn.id())
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();

                let actual: Vec<_> = routed
                    .row_stream(txn.id())
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();

                assert_eq!(expected.len(), limit.min(source_count) as usize);
                assert_eq!(delegated, expected);
                assert_eq!(actual, expected);
                assert_eq!(
                    routed.count(txn.id()).await.unwrap(),
                    limit.min(source_count)
                );

                assert_eq!(routed.is_empty(txn.id()).await.unwrap(), limit == 0);
                assert!(!routed.is_persistent());
                assert!(
                    routed
                        .upsert_row(&txn, vec![1_u64.into()], vec!["changed".into()])
                        .await
                        .is_err()
                );

                for selected in [
                    vec![],
                    vec!["label".parse().unwrap()],
                    vec!["id".parse().unwrap(), "id".parse().unwrap()],
                ] {
                    let limited = table.limit(limit).unwrap();
                    assert!(limited.select(selected.clone()).is_err());
                    assert!(Table::from(limited).select(&selected).is_err());
                }
            }

            assert_eq!(table.count(txn.id()).await.unwrap(), source_count);
            assert_eq!(
                table
                    .read_row(txn.id(), &[1_u64.into()])
                    .await
                    .unwrap()
                    .unwrap()[1],
                Value::from("alpha")
            );
        }
    })
    .await;
}

// ── PUT handler: upsert ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn put_upsert_via_key() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");

        let key = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
        let value = State::from(Scalar::Value(Value::Tuple(vec![Value::from("updated")])));

        handler.put().expect("PUT handler")(&txn, key, value)
            .await
            .expect("upsert ok");

        let row = table.read_row(tx(30), &[Value::from(2_u64)]).await.unwrap();
        assert!(row.is_some());
        assert_eq!(
            row.unwrap().as_ref(),
            &[Value::from(2_u64), Value::from("updated")]
        );
    })
    .await;
}

// ── PUT handler: update ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn put_update_all() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");

        let mut value_map = Map::new();
        value_map.insert(
            "label".parse().expect("Id"),
            Scalar::Value(Value::from("updated")),
        );

        handler.put().expect("PUT handler")(
            &txn,
            Scalar::Value(Value::None),
            State::from(Scalar::Map(value_map)),
        )
        .await
        .expect("update ok");

        for id in [1_u64, 2_u64, 3_u64] {
            let row = table.read_row(tx(30), &[Value::from(id)]).await.unwrap();
            assert!(row.is_some(), "row {id} should exist");
            assert_eq!(
                row.unwrap().as_ref(),
                &[Value::from(id), Value::from("updated")]
            );
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn put_update_range() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");

        let key_selector = Value::Tuple(vec![Value::Tuple(vec![
            Value::from("id"),
            Value::Tuple(vec![
                Value::Tuple(vec![Value::from("in"), Value::from(1_u64)]),
                Value::Tuple(vec![Value::from("ex"), Value::from(3_u64)]),
            ]),
        ])]);

        let mut value_map = Map::new();
        value_map.insert(
            "label".parse().expect("Id"),
            Scalar::Value(Value::from("range_updated")),
        );

        handler.put().expect("PUT handler")(
            &txn,
            Scalar::Value(key_selector),
            State::from(Scalar::Map(value_map)),
        )
        .await
        .expect("update ok");

        let row1 = table.read_row(tx(30), &[Value::from(1_u64)]).await.unwrap();
        assert_eq!(
            row1.unwrap().as_ref(),
            &[Value::from(1_u64), Value::from("range_updated")]
        );

        let row2 = table.read_row(tx(30), &[Value::from(2_u64)]).await.unwrap();
        assert_eq!(
            row2.unwrap().as_ref(),
            &[Value::from(2_u64), Value::from("range_updated")]
        );

        let row3 = table.read_row(tx(30), &[Value::from(3_u64)]).await.unwrap();
        assert_eq!(
            row3.unwrap().as_ref(),
            &[Value::from(3_u64), Value::from("gamma")]
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn update_direct_method() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;

        let mut updates = tc_ir::Map::new();
        updates.insert("label".parse().expect("Id"), Value::from("method_updated"));

        table
            .update(&txn, b_table::Range::default(), updates)
            .await
            .expect("update");

        for id in [1_u64, 2_u64, 3_u64] {
            let row = table.read_row(tx(30), &[Value::from(id)]).await.unwrap();
            assert_eq!(
                row.unwrap().as_ref(),
                &[Value::from(id), Value::from("method_updated")]
            );
        }
    })
    .await;
}

// ── POST handler ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_slice() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(20);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");

        let mut req = Map::new();
        req.insert(
            "id".parse().expect("id"),
            State::from(Scalar::Value(Value::Tuple(vec![
                Value::Tuple(vec![Value::from("in"), Value::from(1_u64)]),
                Value::Tuple(vec![Value::from("ex"), Value::from(3_u64)]),
            ]))),
        );

        let resp = handler.post().expect("POST handler")(&txn, req)
            .await
            .expect("response");

        assert!(matches!(resp, State::Collection(_)));
    })
    .await;
}

// ── DELETE handler ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn delete_key() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");

        let req = Scalar::Value(Value::Tuple(vec![Value::from(2_u64)]));
        handler.delete().expect("DELETE handler")(&txn, req)
            .await
            .expect("delete ok");

        let row = table.read_row(tx(30), &[Value::from(2_u64)]).await.unwrap();
        assert!(row.is_none(), "row should be deleted");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn delete_all_truncates() {
    Box::pin(async {
        use tc_value::Value;

        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[]).expect("root");

        let req = Scalar::Value(Value::None);
        handler.delete().expect("DELETE handler")(&txn, req)
            .await
            .expect("truncate ok");

        assert!(
            table.is_empty(tx(30)).await.unwrap(),
            "table should be empty"
        );
    })
    .await;
}

// ── KeyOrRange parsing ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn key_or_range_all() {
    Box::pin(async {
        use tc_value::Value;

        let (_root, table) = make_table_with_data().await;
        let kor = KeyOrRange::try_from_value(table.schema(), Value::None).expect("parse None");
        assert!(matches!(kor, KeyOrRange::All));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn key_or_range_key() {
    Box::pin(async {
        use tc_value::Value;

        let (_root, table) = make_table_with_data().await;
        let kor =
            KeyOrRange::try_from_value(table.schema(), Value::Tuple(vec![Value::from(1_u64)]))
                .expect("parse key");

        match kor {
            KeyOrRange::Key(key) => assert_eq!(key, vec![Value::from(1_u64)]),
            other => panic!("expected Key, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn key_or_range_range() {
    Box::pin(async {
        use tc_value::Value;

        let (_root, table) = make_table_with_data().await;
        let selector = Value::Tuple(vec![Value::Tuple(vec![
            Value::from("id"),
            Value::Tuple(vec![
                Value::Tuple(vec![Value::from("in"), Value::from(1_u64)]),
                Value::Tuple(vec![Value::from("ex"), Value::from(3_u64)]),
            ]),
        ])]);

        let kor = KeyOrRange::try_from_value(table.schema(), selector).expect("parse range");
        assert!(matches!(kor, KeyOrRange::Range(_)));
    })
    .await;
}

// ── Method-not-supported ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn put_on_count_rejected() {
    Box::pin(async {
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("count")]).expect("count");
        assert!(handler.put().is_none());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn delete_on_columns_rejected() {
    Box::pin(async {
        let (_root, table) = make_table_with_data().await;
        let handler = tc_ir::Route::<State>::route(&table, &[segment("columns")]).expect("columns");
        assert!(handler.delete().is_none());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn insert_is_strict_and_is_empty_is_direct() {
    Box::pin(async {
        let txn = MockTxn::fresh(30);
        let (_root, table) = make_table_with_data().await;
        let insert =
            tc_ir::Route::<State>::route(&table, &[segment("insert")]).expect("insert route");

        let params: Map<State> = [
            (
                "key".parse().expect("key id"),
                State::from(Value::Tuple(vec![Value::from(4_u64)])),
            ),
            (
                "values".parse().expect("values id"),
                State::from(Value::Tuple(vec![Value::from("delta")])),
            ),
        ]
        .into_iter()
        .collect();

        let key = Scalar::Value(Value::Tuple(vec![Value::from(4_u64)]));
        let values = params
            .get(&"values".parse::<tc_ir::Id>().expect("values id"))
            .unwrap()
            .clone();

        insert.put().expect("PUT handler")(&txn, key.clone(), values.clone())
            .await
            .expect("insert row");

        assert!(
            table
                .contains_row(tx(30), &[Value::from(4_u64)])
                .await
                .unwrap()
        );

        let insert =
            tc_ir::Route::<State>::route(&table, &[segment("insert")]).expect("insert route");

        let duplicate = insert.put().expect("PUT handler")(&txn, key, values).await;
        assert!(
            duplicate.is_err(),
            "strict insert must reject a visible key"
        );

        let is_empty =
            tc_ir::Route::<State>::route(&table, &[segment("is_empty")]).expect("is_empty route");

        assert_eq!(
            is_empty.get().expect("GET handler")(&txn, Scalar::Value(Value::None))
                .await
                .expect("is_empty"),
            State::from(Value::from(false))
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn local_slice_update_and_truncate_stream_through_delegated_storage() {
    Box::pin(async {
        use std::ops::Bound;

        let txn = MockTxn::fresh(40);
        let table = make_local_table_with_data(&txn).await;
        let range: b_table::Range<tc_ir::Id, Value> = [(
            "id".parse().expect("id"),
            (
                Bound::Included(Value::from(1_u64)),
                Bound::Excluded(Value::from(2_u64)),
            ),
        )]
        .into_iter()
        .collect();

        let slice = Table::from(table.slice(range.clone(), &[], false).expect("slice"));
        let handler = tc_ir::Route::<State>::route(&slice, &[]).expect("slice route");

        let rejected = handler.put().expect("PUT handler")(
            &txn,
            Scalar::Value(Value::None),
            State::from(Scalar::Map(
                [(
                    "label".parse().expect("label"),
                    Scalar::Value(Value::from("updated")),
                )]
                .into_iter()
                .collect(),
            )),
        )
        .await
        .expect_err("returned slices are read-only");

        assert!(rejected.to_string().contains("cannot update"));
        assert_eq!(
            table
                .read_row(txn.id(), &[1_u64.into()])
                .await
                .unwrap()
                .unwrap()[1],
            Value::from("alpha")
        );

        // Selection mutations target the original owner with explicit bounds.
        table
            .update(
                &txn,
                range.clone(),
                [("label".parse().unwrap(), "updated".into())]
                    .into_iter()
                    .collect(),
            )
            .await
            .expect("update owner selection");

        assert_eq!(
            table
                .read_row(txn.id(), &[Value::from(1_u64)])
                .await
                .expect("read updated row")
                .expect("updated row")
                .as_ref(),
            &[Value::from(1_u64), Value::from("updated")]
        );

        assert_eq!(
            table
                .read_row(txn.id(), &[Value::from(2_u64)])
                .await
                .expect("read untouched row")
                .expect("untouched row")
                .as_ref(),
            &[Value::from(2_u64), Value::from("beta")]
        );

        let handler = tc_ir::Route::<State>::route(&slice, &[]).expect("slice route");
        handler.delete().expect("DELETE handler")(&txn, Scalar::Value(Value::None))
            .await
            .expect_err("returned slices are read-only");

        table
            .truncate(&txn, range)
            .await
            .expect("truncate owner selection");

        assert!(
            !table
                .contains_row(txn.id(), &[Value::from(1_u64)])
                .await
                .expect("test truncated row")
        );

        assert!(
            table
                .contains_row(txn.id(), &[Value::from(2_u64)])
                .await
                .expect("test untouched row")
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn invalid_local_replacement_preserves_the_row_and_indexes() {
    Box::pin(async {
        let txn = MockTxn::fresh(25);
        let base = simple_schema();
        let schema = TableSchema::new(
            base.key_schema().collect(),
            base.column_schema().skip(base.key().len()).collect(),
            vec![("by_label".into(), vec!["label".parse().unwrap()])],
            StorageConfig::default(),
        )
        .unwrap();

        let native = b_table::TableLock::create(
            schema,
            crate::persistence::value_order(),
            txn.subcontext_unique().context().await.unwrap(),
        )
        .await
        .unwrap();

        for (key, value) in [(1_u64, "alpha"), (2, "beta")] {
            native
                .write()
                .await
                .upsert(vec![key.into()], vec![value.into()])
                .await
                .unwrap();
        }

        let table = Table::Local(native);

        let before = table.read_row(txn.id(), &[1_u64.into()]).await.unwrap();
        for values in [
            Value::from(42_u64),
            Value::Tuple(Vec::new()),
            Value::Tuple(vec!["one".into(), "extra".into()]),
        ] {
            let handler = tc_ir::Route::<State>::route(&table, &[]).unwrap();
            assert!(
                handler.put().unwrap()(
                    &txn,
                    Scalar::Value(Value::Tuple(vec![1_u64.into()])),
                    State::from(values),
                )
                .await
                .is_err()
            );

            assert_eq!(
                table.read_row(txn.id(), &[1_u64.into()]).await.unwrap(),
                before
            );

            assert_eq!(table.count(txn.id()).await.unwrap(), 2);
        }

        let handler = tc_ir::Route::<State>::route(&table, &[]).unwrap();
        handler.put().unwrap()(
            &txn,
            Scalar::Value(Value::Tuple(vec![1_u64.into()])),
            State::from(Value::from("changed")),
        )
        .await
        .unwrap();

        assert_eq!(
            table
                .read_row(txn.id(), &[1_u64.into()])
                .await
                .unwrap()
                .unwrap()
                .as_slice(),
            &[1_u64.into(), "changed".into()]
        );

        assert_eq!(table.count(txn.id()).await.unwrap(), 2);
        let Table::Local(native) = table else {
            panic!("native fixture")
        };

        native.validate().await.unwrap();
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn ordered_view_predicates_reject_unsupported_queries() {
    Box::pin(async {
        let (_root, persistent) = make_table_with_data().await;
        let txn = MockTxn::fresh(22);
        let encoded =
            destream_json::encode((Value::cast_from(simple_schema()), Vec::<Value>::new()))
                .unwrap();

        let local: crate::table::DecodedTablePayload<MockTxn> =
            destream_json::try_decode(txn.clone(), encoded)
                .await
                .unwrap();

        for owner in [persistent, local.table] {
            let expected = owner.count(tx(22)).await.unwrap();
            for reverse in [false, true] {
                let supported = owner.order_by(&["id".parse().unwrap()], reverse).unwrap();
                assert_eq!(supported.count(tx(22)).await.unwrap(), expected);
                assert_eq!(supported.is_empty(tx(22)).await.unwrap(), expected == 0);
                for column in ["nonexistent", "label"] {
                    let slice = owner.order_by(&[column.parse().unwrap()], reverse).unwrap();
                    let error = slice.rows(tx(22)).await.err().expect("unsupported order");
                    assert!(!error.requires_recovery());
                    assert_eq!(slice.count(tx(22)).await.unwrap_err().code(), error.code());
                    assert_eq!(
                        slice.is_empty(tx(22)).await.unwrap_err().code(),
                        error.code()
                    );

                    let mut callbacks = 0;
                    assert!(
                        slice
                            .for_each_row_in_order(tx(22), |_| callbacks += 1)
                            .await
                            .is_err()
                    );

                    assert_eq!(callbacks, 0);
                    let columns = vec!["id".parse().unwrap()];
                    for view in [
                        Table::from(slice.clone()),
                        Table::from(slice.limit(0)),
                        Table::from(slice.limit(1)),
                        Table::from(slice.select(columns.clone()).unwrap()),
                        Table::from(slice.limit(0).select(columns).unwrap()),
                    ] {
                        for route in ["count", "is_empty"] {
                            let handler = tc_ir::Route::<State>::route(&view, &[segment(route)])
                                .unwrap()
                                .get()
                                .unwrap();

                            let failure =
                                handler(&txn, Scalar::Value(Value::None)).await.unwrap_err();

                            assert_eq!(failure.code(), error.code());
                            assert!(!failure.requires_recovery());
                        }
                    }
                }
            }
        }
    })
    .await;
}
