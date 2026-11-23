use std::ops::Bound;
use std::str::FromStr;
use std::sync::Arc;

use freqfs::Cache;
use pathlink::{PathLabel, PathSegment, path_label};
use safecast::TryCastFrom;
use tc_error::TCError;
use tc_ir::{Scalar, Transact, Transaction};
use tc_value::Value;

use super::*;
use crate::StorageContext;
use crate::btree::BTreeColumnSchema;
use crate::collection::BTreeView;
use crate::test::{TempRoot, TestTxn};

const SERVICE: PathLabel = path_label(&["service", "example", "data", "1.0.0"]);

#[allow(dead_code)]
#[derive(Debug)]
enum TestState {
    Value(Value),
    Tuple(Vec<TestState>),
    Map(Map<TestState>),
    Collection(Collection<TestTxn>),
}

tokio::task_local! {
    static STATE_CLONES: std::cell::Cell<usize>;
}

impl Clone for TestState {
    fn clone(&self) -> Self {
        let _ = STATE_CLONES.try_with(|count| count.set(count.get() + 1));
        match self {
            Self::Value(value) => Self::Value(value.clone()),
            Self::Tuple(values) => Self::Tuple(values.clone()),
            Self::Map(values) => Self::Map(values.clone()),
            Self::Collection(collection) => Self::Collection(collection.clone()),
        }
    }
}

impl From<crate::table::Table<TestTxn>> for TestState {
    fn from(table: crate::table::Table<TestTxn>) -> Self {
        Self::Collection(Collection::from(table))
    }
}

impl From<Value> for TestState {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<Scalar> for TestState {
    fn from(scalar: Scalar) -> Self {
        Self::Value(Value::try_cast_from(scalar, |_| "value").expect("test scalar"))
    }
}

impl From<Collection<TestTxn>> for TestState {
    fn from(collection: Collection<TestTxn>) -> Self {
        Self::Collection(collection)
    }
}

impl From<u64> for TestState {
    fn from(value: u64) -> Self {
        Self::Value(Value::from(value))
    }
}

impl tc_ir::StateInstance for TestState {
    type Transaction = TestTxn;
}

impl CollectionState for TestState {
    type Txn = TestTxn;

    fn into_scalar(self) -> TCResult<Scalar> {
        self.into_value().map(Scalar::Value)
    }

    fn into_value(self) -> TCResult<Value> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Tuple(values) => values
                .into_iter()
                .map(Self::into_value)
                .collect::<TCResult<_>>()
                .map(Value::Tuple),
            Self::Map(_) | Self::Collection(_) => {
                Err(TCError::bad_request("test state is not a value"))
            }
        }
    }

    fn into_tuple(self) -> TCResult<Vec<Self>> {
        match self {
            Self::Tuple(values) => Ok(values),
            Self::Value(Value::Tuple(values)) => Ok(values.into_iter().map(Self::from).collect()),
            other => Err(TCError::bad_request(format!(
                "expected tuple, found {}",
                std::any::type_name_of_val(&other)
            ))),
        }
    }

    fn into_map(self) -> TCResult<Map<Self>> {
        match self {
            Self::Map(values) => Ok(values),
            _ => Err(TCError::bad_request("expected map")),
        }
    }

    fn into_tensor(self) -> TCResult<Tensor<Self::Txn>> {
        match self {
            Self::Collection(Collection::Tensor(tensor)) => Ok(tensor),
            _ => Err(TCError::bad_request("expected tensor")),
        }
    }

    fn as_number(&self) -> Option<number_general::Number> {
        match self {
            Self::Value(Value::Number(number)) => Some(*number),
            _ => None,
        }
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::Value(value) if matches!(&value, Value::None))
    }
}

fn segment(path: &str) -> PathSegment {
    PathSegment::from_str(path).expect("path segment")
}

async fn btree_view() -> (Arc<TempRoot>, BTreeView<TestTxn>) {
    let root = TempRoot::new("route");
    std::fs::create_dir_all(root.join("persistent")).expect("persistent root");
    std::fs::create_dir_all(root.join("txn")).expect("transaction root");

    let cache = Cache::<crate::PersistentFile>::new(
        16 * 1024 * 1024,
        None,
        0,
        std::time::Duration::from_secs(3),
    );

    let persistent = Arc::clone(&cache)
        .load(root.join("persistent"))
        .expect("persistent directory");

    let columns = vec![BTreeColumnSchema {
        name: "id".to_string(),
        dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
        max_size: None,
    }];

    let schema = crate::btree::BTreeSchema::try_from(columns.as_slice()).unwrap();
    let btree = crate::btree::BTree::<TestTxn>::try_with_schema(persistent, schema)
        .await
        .unwrap();

    let view = BTreeView::new(columns, btree).unwrap();

    (root, view)
}

#[tokio::test]
async fn unmatched_literal_put_returns_owned_inputs() {
    let txn = TestTxn::fresh(1);
    let link = SERVICE.into();
    let text = "payload".repeat(4096);
    let original = text.as_ptr();
    let result = Collection::from_put(
        &txn,
        &link,
        TestState::from(7_u64),
        TestState::from(Value::String(text)),
    )
    .await
    .unwrap();

    let LiteralPut::Unmatched {
        key: TestState::Value(Value::Number(key)),
        value: TestState::Value(Value::String(value)),
    } = result
    else {
        panic!("expected original PUT inputs");
    };

    assert_eq!(key, 7_u64.into());
    assert_eq!(value.as_ptr(), original);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn native_materialization_and_cold_hash_preserve_schema_and_contents() {
    Box::pin(async {
        use crate::collection::CollectionSchema;
        use crate::table::{Column, TableSchema};
        use crate::tensor::TensorSchema;
        use tc_ir::Public;

        for kind in 0..3 {
            let root = TempRoot::new("native-hash");
            for name in ["source", "snapshot"] {
                std::fs::create_dir(root.join(name)).unwrap();
            }

            let cache = || {
                Cache::<crate::PersistentFile>::new(
                    1 << 24,
                    None,
                    0,
                    std::time::Duration::from_secs(3),
                )
            };

            let schema = match kind {
                0 => CollectionSchema::BTree(vec![BTreeColumnSchema {
                    name: "id".into(),
                    dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
                    max_size: None,
                }]),
                1 => CollectionSchema::Table(
                    TableSchema::new(
                        vec![Column {
                            name: "id".parse().unwrap(),
                            dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
                        }],
                        vec![Column {
                            name: "value".parse().unwrap(),
                            dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
                        }],
                        vec![],
                        crate::btree::StorageConfig::default(),
                    )
                    .unwrap(),
                ),
                _ => CollectionSchema::Tensor(
                    TensorSchema::new(
                        tc_value::NumberType::Float(number_general::FloatType::F64),
                        vec![4].into(),
                        None,
                    )
                    .unwrap(),
                ),
            };

            let source = Collection::<TestTxn>::create(
                cache().load(root.join("source")).unwrap(),
                schema.clone(),
            )
            .await
            .unwrap();

            let txn = TestTxn::fresh(1);
            let (path, key, value) = match kind {
                0 => (
                    vec![segment("insert")],
                    Value::None,
                    Value::Tuple(vec![7_u64.into()]),
                ),
                1 => (
                    vec![segment("insert")],
                    Value::Tuple(vec![1_u64.into()]),
                    Value::Tuple(vec![7_u64.into()]),
                ),
                _ => (vec![], Value::Tuple(vec![0_u64.into()]), Value::from(7_f64)),
            };

            source
                .put(&txn, &path, Scalar::from(key), TestState::from(value))
                .await
                .unwrap();

            source.commit(txn.id()).await.unwrap();
            let expected = source.hash(txn.id()).await.unwrap();
            source.finalize(&txn.id()).await.unwrap();
            let reader = txn.at(2);
            let snapshot = source
                .materialize(&reader, cache().load(root.join("snapshot")).unwrap())
                .await
                .unwrap();

            snapshot.sync_all().await.unwrap();
            drop(snapshot);
            let reopened =
                Collection::<TestTxn>::load(cache().load(root.join("snapshot")).unwrap(), schema)
                    .await
                    .unwrap();

            assert_eq!(reopened.hash(reader.id()).await.unwrap(), expected);
        }
    })
    .await;
}

#[tokio::test]
async fn tensor_route_rejects_unknown_path_and_invalid_shape() {
    let txn = TestTxn::fresh(1);
    let tensor = Tensor::dense_f64(&txn, vec![2], vec![1.0, 2.0])
        .await
        .expect("tensor");

    let scalar = tc_ir::Route::<TestState>::route(&tensor, &[])
        .unwrap()
        .get()
        .unwrap()(&txn, Scalar::Value(Value::Tuple(vec![0_u64.into()])))
    .await
    .unwrap();

    assert_eq!(scalar.as_number(), Some(1.0.into()));

    assert!(tc_ir::Route::<TestState>::route(&tensor, &[segment("unknown")]).is_none());

    for name in ["copy", "copy_values"] {
        let route = tc_ir::Route::<TestState>::route(&tensor, &[segment(name)]).unwrap();
        let output = route.get().unwrap()(&txn, Scalar::Value(Value::None))
            .await
            .unwrap();

        let TestState::Collection(Collection::Tensor(copy)) = output else {
            panic!("Tensor copy route returned a non-Tensor value")
        };

        assert_eq!(copy.read_value(txn.id(), &[0]).await.unwrap(), 1.0.into());
        assert_eq!(copy.read_value(txn.id(), &[1]).await.unwrap(), 2.0.into());
    }

    STATE_CLONES
        .scope(std::cell::Cell::new(0), async {
            for right in [
                TestState::from(Value::Number(2.0.into())),
                TestState::from(Collection::Tensor(tensor.clone())),
            ] {
                let params = [("r".parse().unwrap(), right)].into_iter().collect();
                let result = tc_ir::Route::<TestState>::route(&tensor, &[segment("add")])
                    .unwrap()
                    .post()
                    .unwrap()(&txn, params)
                .await
                .unwrap();

                let sum = result.into_tensor().unwrap();
                assert_eq!(sum.read_value(txn.id(), &[1]).await.unwrap(), 4.0.into());
            }

            let missing = tc_ir::Route::<TestState>::route(&tensor, &[segment("add")])
                .unwrap()
                .post()
                .unwrap()(&txn, Map::new())
            .await
            .unwrap_err();

            assert!(missing.to_string().contains("r"));
            let null = [("r".parse().unwrap(), TestState::from(Value::None))]
                .into_iter()
                .collect();

            let error = tc_ir::Route::<TestState>::route(&tensor, &[segment("add")])
                .unwrap()
                .post()
                .unwrap()(&txn, null)
            .await
            .unwrap_err();

            assert!(error.to_string().contains("expected tensor"));

            for params in [
                Map::new(),
                [
                    ("axes".parse().unwrap(), TestState::from(Value::None)),
                    ("keepdims".parse().unwrap(), TestState::from(Value::None)),
                ]
                .into_iter()
                .collect(),
            ] {
                let result = tc_ir::Route::<TestState>::route(&tensor, &[segment("sum")])
                    .unwrap()
                    .post()
                    .unwrap()(&txn, params)
                .await
                .unwrap();

                assert_eq!(result.as_number(), Some(3.0.into()));
            }

            let literal = || {
                TestState::Tuple(vec![
                    TestState::from(Value::Number(1.0.into())),
                    TestState::from(Value::Number(2.0.into())),
                ])
            };

            let shape = || TestState::Tuple(vec![2_u64.into()]);
            let key = TestState::Tuple(vec![TestState::from(Value::from("f64")), shape()]);
            let value = crate::tensor::route::tensor_literal(&txn, key, literal())
                .await
                .unwrap();

            assert_eq!(value.read_value(txn.id(), &[1]).await.unwrap(), 2.0.into());
            for (fields, message) in [
                (vec![], "Tensor literal key must be"),
                (
                    vec![
                        TestState::from(Value::None),
                        TestState::from(Value::None),
                        TestState::from(Value::None),
                    ],
                    "expected a numeric sparse axis",
                ),
                (
                    vec![
                        TestState::from(Value::from("invalid")),
                        TestState::from(Value::None),
                    ],
                    "unsupported Tensor dtype",
                ),
                (
                    vec![TestState::from(Value::from("f64")), shape()],
                    "Tensor literal value count differs from shape",
                ),
            ] {
                let error = crate::tensor::route::tensor_literal(
                    &txn,
                    TestState::Tuple(fields),
                    TestState::Tuple(vec![]),
                )
                .await
                .unwrap_err();

                assert!(error.to_string().contains(message), "{error}");
            }

            let key = TestState::Tuple(vec![
                TestState::from(Value::from("f64")),
                shape(),
                0_u64.into(),
            ]);

            let entries = TestState::Tuple(vec![TestState::Tuple(vec![
                TestState::Tuple(vec![1_u64.into()]),
                TestState::from(Value::Number(2.5.into())),
            ])]);

            let sparse = crate::tensor::route::tensor_literal(&txn, key, entries)
                .await
                .unwrap();

            assert_eq!(sparse.read_value(txn.id(), &[0]).await.unwrap(), 0.0.into());
            assert_eq!(sparse.read_value(txn.id(), &[1]).await.unwrap(), 2.5.into());
            for key in [
                Scalar::default(),
                Scalar::from(Value::Tuple(vec![0_u64.into()])),
            ] {
                let transposed = tc_ir::Route::<TestState>::route(&tensor, &[segment("transpose")])
                    .unwrap()
                    .get()
                    .unwrap()(&txn, key)
                .await
                .unwrap()
                .into_tensor()
                .unwrap();

                assert_eq!(
                    transposed.read_value(txn.id(), &[1]).await.unwrap(),
                    2.0.into()
                );
            }

            #[cfg(feature = "complex")]
            {
                let key = TestState::Tuple(vec![TestState::from(Value::from("c64")), shape()]);
                let complex = crate::tensor::route::tensor_literal(&txn, key, literal())
                    .await
                    .unwrap();

                let params = [(
                    "r".parse().unwrap(),
                    TestState::from(Value::Number(2.0.into())),
                )]
                .into_iter()
                .collect();

                let added = tc_ir::Route::<TestState>::route(&complex, &[segment("add")])
                    .unwrap()
                    .post()
                    .unwrap()(&txn, params)
                .await
                .unwrap()
                .into_tensor()
                .unwrap();

                assert_eq!(added.read_value(txn.id(), &[1]).await.unwrap(), 4.0.into());
            }

            assert_eq!(STATE_CLONES.with(std::cell::Cell::get), 0);
        })
        .await;

    let route =
        tc_ir::Route::<TestState>::route(&tensor, &[segment("reshape")]).expect("reshape route");

    let get = route.get().expect("GET handler");
    let txn = TestTxn::fresh(1);
    let err = get(&txn, Scalar::Value(Value::String("invalid".into())))
        .await
        .expect_err("invalid reshape request");

    assert!(!err.to_string().is_empty());
}

#[tokio::test]
async fn btree_route_rejects_unknown_path_and_invalid_row() {
    let (_root, view) = btree_view().await;
    assert!(tc_ir::Route::<TestState>::route(&view, &[segment("unknown")]).is_none());

    let route =
        tc_ir::Route::<TestState>::route(&view, &[segment("contains")]).expect("contains route");

    let get = route.get().expect("GET handler");
    let txn = TestTxn::fresh(1);
    let err = get(&txn, Scalar::Value(Value::String("not a BTree row".into())))
        .await
        .expect_err("invalid BTree row");

    assert!(!err.to_string().is_empty());
}

#[tokio::test]
async fn btree_returned_views_cannot_mutate_or_drive_owner_lifecycle() {
    let (_root, owner) = btree_view().await;
    let txn = TestTxn::fresh(1);
    for value in 1..=3_u64 {
        owner
            .persistent()
            .unwrap()
            .insert_row(&txn, vec![value.into()])
            .await
            .unwrap();
    }

    let full = tc_ir::Route::<TestState>::route(&owner, &[])
        .unwrap()
        .get()
        .unwrap()(&txn, Scalar::default())
    .await
    .unwrap();

    let TestState::Collection(Collection::BTree(full)) = full else {
        panic!("expected a BTree view");
    };

    let all = (Bound::Unbounded, Bound::Unbounded);
    let selected = owner.slice((Bound::Included(2_u64.into()), Bound::Unbounded), false);
    assert!(
        !selected
            .contains_row(txn.id(), &[1_u64.into()])
            .await
            .unwrap()
    );

    assert!(
        selected
            .contains_row(txn.id(), &[2_u64.into()])
            .await
            .unwrap()
    );

    let views = [
        *full.clone(),
        full.slice(all.clone(), false),
        owner.slice(all.clone(), false),
        owner.slice(all.clone(), true).slice(all, false),
        selected,
    ];

    let owner = Collection::BTree(Box::new(owner));
    for view in views {
        assert!(view.persistent().is_err());
        let insert = tc_ir::Route::<TestState>::route(&view, &[segment("insert")])
            .unwrap()
            .put()
            .unwrap();

        assert!(
            insert(
                &txn,
                Scalar::default(),
                TestState::from(Value::Tuple(vec![4_u64.into()]))
            )
            .await
            .is_err()
        );

        for path in [Vec::new(), vec![segment("delete")]] {
            let delete = tc_ir::Route::<TestState>::route(&view, &path)
                .unwrap()
                .delete()
                .unwrap();

            assert!(
                delete(&txn, Scalar::Value(Value::Tuple(vec![2_u64.into()])))
                    .await
                    .is_err()
            );
        }

        let view = Collection::BTree(Box::new(view));
        assert!(!view.is_persistent());
        assert!(view.as_btree().is_none());
        assert!(view.sync().await.is_err());
        assert!(view.sync_all().await.is_err());
        assert!(view.commit(txn.id()).await.is_err());
        assert!(view.rollback(&txn.id()).await.is_err());
        assert!(view.finalize(&txn.id()).await.is_err());
        assert!(view.restore_from(&txn, &owner).await.is_err());
        assert!(owner.restore_from(&txn, &view).await.is_err());
    }

    let copied = full
        .materialize(&txn, txn.subcontext_unique().context().await.unwrap())
        .await
        .unwrap();

    copied
        .persistent()
        .unwrap()
        .insert_row(&txn, vec![4_u64.into()])
        .await
        .unwrap();

    assert_eq!(copied.count(txn.id()).await.unwrap(), 4);
    assert_eq!(owner.as_btree().unwrap().count(txn.id()).await.unwrap(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn tensor_handlers_reject_unused_parameters() {
    use futures::FutureExt;

    use tc_ir::Route;

    Box::pin(async {
        let txn = TestTxn::fresh(1);
        let tensor = Tensor::dense_f64(&txn, vec![2, 2], vec![1., 2., 3., 4.])
            .await
            .unwrap();

        let extra = || ("unused".parse().unwrap(), TestState::from(Value::None));
        let scalar = || TestState::from(Value::from(2.));
        let operand = || TestState::from(Collection::Tensor(tensor.clone()));
        let mut cases: Vec<(&str, Map<TestState>)> = Vec::new();
        for name in [
            "add", "sub", "mul", "div", "pow", "rem", "log", "eq", "ne", "gt", "ge", "lt", "le",
            "and", "or", "xor",
        ] {
            for right in [scalar(), operand()] {
                cases.push((
                    name,
                    [("r".parse().unwrap(), right), extra()]
                        .into_iter()
                        .collect(),
                ));
            }
        }

        for name in [
            "not", "abs", "exp", "ln", "round", "sin", "asin", "sinh", "cos", "acos", "cosh",
            "tan", "atan", "tanh", "is_nan", "is_inf", "mt", "diag", "all", "any",
        ] {
            cases.push((name, [extra()].into_iter().collect()));
        }

        #[cfg(feature = "complex")]
        for name in [
            "conj", "re", "im", "angle", "mh", "fft", "ifft", "fft2", "ifft2",
        ] {
            cases.push((name, [extra()].into_iter().collect()));
        }

        for name in ["sum", "product", "min", "max", "mean", "norm", "std"] {
            // A misspelled optional argument must not silently select a full reduction.
            cases.push((
                name,
                [("axis".parse().unwrap(), TestState::from(0_u64))]
                    .into_iter()
                    .collect(),
            ));
        }

        cases.push((
            "matmul",
            [("r".parse().unwrap(), operand()), extra()]
                .into_iter()
                .collect(),
        ));

        cases.push((
            "cond",
            [
                ("then".parse().unwrap(), operand()),
                ("or_else".parse().unwrap(), operand()),
                extra(),
            ]
            .into_iter()
            .collect(),
        ));

        cases.push((
            "transpose",
            [
                ("perm".parse().unwrap(), TestState::from(Value::None)),
                extra(),
            ]
            .into_iter()
            .collect(),
        ));

        cases.push((
            "broadcast_reduce",
            [
                (
                    "target_shape".parse().unwrap(),
                    TestState::from(Value::Tuple(vec![2_u64.into(), 2_u64.into()])),
                ),
                extra(),
            ]
            .into_iter()
            .collect(),
        ));

        for (name, params) in cases {
            let handler = Route::<TestState>::route(&tensor, &[segment(name)])
                .unwrap()
                .post()
                .unwrap();

            let error = handler(&txn, params)
                .now_or_never()
                .expect("argument rejection precedes asynchronous execution")
                .unwrap_err();

            assert_eq!(error.code(), tc_error::ErrorKind::BadRequest, "{name}");
            assert!(
                error.message().starts_with("unexpected Tensor parameter"),
                "{name}: {error}"
            );

            assert!(!error.requires_recovery());
        }

        for params in [
            Map::new(),
            [
                ("axes".parse().unwrap(), TestState::from(Value::None)),
                ("keepdims".parse().unwrap(), TestState::from(Value::None)),
            ]
            .into_iter()
            .collect(),
        ] {
            let result = Route::<TestState>::route(&tensor, &[segment("sum")])
                .unwrap()
                .post()
                .unwrap()(&txn, params)
            .await
            .unwrap();

            assert_eq!(result.as_number(), Some(10.0.into()));
        }

        for keepdims in [false, true] {
            let params = [
                ("axes".parse().unwrap(), TestState::from(0_u64)),
                (
                    "keepdims".parse().unwrap(),
                    TestState::from(Value::from(keepdims)),
                ),
            ]
            .into_iter()
            .collect();

            let result = Route::<TestState>::route(&tensor, &[segment("sum")])
                .unwrap()
                .post()
                .unwrap()(&txn, params)
            .await
            .unwrap()
            .into_tensor()
            .unwrap();

            assert_eq!(
                result.shape(),
                if keepdims { &[1, 2][..] } else { &[2][..] }
            );

            let coord = if keepdims { vec![0, 0] } else { vec![0] };
            assert_eq!(
                result.read_value(txn.id(), &coord).await.unwrap(),
                4.0.into()
            );
        }

        for key in [
            Value::Tuple(vec![]),
            Value::Tuple(vec![0_u64.into(), 1_u64.into()]),
        ] {
            let error = Route::<TestState>::route(&tensor, &[segment("flip")])
                .unwrap()
                .get()
                .unwrap()(&txn, Scalar::Value(key))
            .await
            .unwrap_err();

            assert!(error.message().contains("exactly one axis"));
        }

        for key in [Value::from(0_u64), Value::Tuple(vec![0_u64.into()])] {
            let flipped = Route::<TestState>::route(&tensor, &[segment("flip")])
                .unwrap()
                .get()
                .unwrap()(&txn, Scalar::Value(key))
            .await
            .unwrap()
            .into_tensor()
            .unwrap();

            assert_eq!(
                flipped.read_value(txn.id(), &[0, 0]).await.unwrap(),
                3.0.into()
            );
        }

        let error = Route::<TestState>::route(&tensor, &[segment("add")])
            .unwrap()
            .post()
            .unwrap()(&txn, [extra()].into_iter().collect())
        .await
        .unwrap_err();

        assert!(error.message().contains("missing Tensor parameter r"));
        let params = [
            (
                "r".parse().unwrap(),
                TestState::from(Value::from("invalid operand")),
            ),
            extra(),
        ]
        .into_iter()
        .collect();

        let error = Route::<TestState>::route(&tensor, &[segment("add")])
            .unwrap()
            .post()
            .unwrap()(&txn, params)
        .await
        .unwrap_err();

        assert!(!error.message().starts_with("unexpected Tensor parameter"));
        tensor.write_value(&txn, &[0, 0], 7.0.into()).await.unwrap();
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn btree_views_validate_schema_pairing_and_share_metadata() {
    use tc_ir::IntoView;

    Box::pin(async {
        let txn = TestTxn::fresh(1);
        let columns = vec![BTreeColumnSchema {
            name: "id".into(),
            dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
            max_size: Some(32_u64.into()),
        }];

        let dir = txn.subcontext_unique().context().await.unwrap();
        let schema = crate::btree::BTreeSchema::try_from(columns.as_slice()).unwrap();

        let raw = crate::btree::BTree::<TestTxn>::try_with_schema(dir.clone(), schema)
            .await
            .unwrap();

        let held = dir.write().await;
        let wrong_type = vec![BTreeColumnSchema {
            dtype: tc_value::ValueType::String,
            ..columns[0].clone()
        }];

        for invalid in [
            Vec::new(),
            vec![columns[0].clone(), columns[0].clone()],
            wrong_type,
        ] {
            let error = BTreeView::new(invalid, raw.clone()).unwrap_err();
            assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
            assert!(!error.requires_recovery());
        }

        let view = BTreeView::new(columns.clone(), raw.clone()).unwrap();
        let alias = view.clone();
        let slice = view.slice((Bound::Unbounded, Bound::Unbounded), false);
        assert!(std::ptr::eq(view.schema(), alias.schema()));
        assert!(std::ptr::eq(view.schema(), slice.schema()));
        let nested = slice.slice(
            (Bound::Included(0_u64.into()), Bound::Excluded(2_u64.into())),
            false,
        );

        assert!(std::ptr::eq(view.schema(), nested.schema()));
        drop(held);
        raw.insert_row(&txn, vec![1_u64.into()]).await.unwrap();
        assert_eq!(nested.count(txn.id()).await.unwrap(), 1);
        let terminal = alias.into_view(txn.clone()).await.unwrap();
        assert_eq!(terminal.schema, columns);
        drop(terminal);
        drop(slice);
        drop(nested);
        let metadata = view.schema().as_ptr();
        let mut terminal = view.into_view(txn.clone()).await.unwrap();
        assert_eq!(
            terminal.schema.as_ptr(),
            metadata,
            "unique metadata moves into the wire header"
        );

        assert_eq!(
            futures::TryStreamExt::try_next(&mut terminal.keys)
                .await
                .unwrap()
                .unwrap(),
            vec![Value::from(1_u64)]
        );

        assert!(
            futures::TryStreamExt::try_next(&mut terminal.keys)
                .await
                .unwrap()
                .is_none()
        );

        drop(terminal);
        raw.insert_row(&txn, vec![2_u64.into()]).await.unwrap();
        let untyped =
            crate::btree::BTree::<TestTxn>::new(txn.subcontext_unique().context().await.unwrap())
                .await
                .unwrap();

        assert!(BTreeView::new(columns, untyped.clone()).is_err());
        untyped
            .insert_row(
                &txn,
                vec![Value::from("raw trees still accept untyped keys")],
            )
            .await
            .unwrap();
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restoration_uses_one_semantic_schema_rule() {
    use std::time::Duration;

    use futures::FutureExt;

    use crate::btree::{BTree, BTreeSchema, StorageConfig};
    use crate::table::{Column, PersistentTable, TableSchema};
    use crate::tensor::TensorSchema;
    use crate::test::tx;
    use tc_value::{NumberType, ValueType};

    Box::pin(async {
        let txn = TestTxn::fresh(10);
        let columns = vec![BTreeColumnSchema {
            name: "id".into(),
            dtype: ValueType::Number(NumberType::Number),
            max_size: None,
        }];

        let schema = BTreeSchema::try_from(columns.as_slice()).unwrap();
        let raw = BTree::<TestTxn>::try_with_schema(
            txn.subcontext_unique().context().await.unwrap(),
            schema,
        )
        .await
        .unwrap();

        let target = BTreeView::new(columns.clone(), raw.clone()).unwrap();
        let mismatches = [
            BTreeColumnSchema {
                name: "other".into(),
                ..columns[0].clone()
            },
            BTreeColumnSchema {
                dtype: ValueType::String,
                ..columns[0].clone()
            },
            BTreeColumnSchema {
                max_size: Some(64_u64.into()),
                ..columns[0].clone()
            },
        ];

        for column in mismatches {
            let columns = vec![column];
            let schema = BTreeSchema::try_from(columns.as_slice()).unwrap();
            let source = BTree::<TestTxn>::try_with_schema(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
            )
            .await
            .unwrap();

            let source = BTreeView::new(columns, source).unwrap();
            let context = txn.context().await.unwrap();
            let held = context.write().await;
            let direct = target
                .restore_from(&txn, &source)
                .now_or_never()
                .expect("schema rejection precedes staging")
                .unwrap_err();

            let through_enum = Collection::BTree(Box::new(target.clone()))
                .restore_from(&txn, &Collection::BTree(Box::new(source)))
                .now_or_never()
                .expect("enum delegates synchronous rejection")
                .unwrap_err();

            assert_eq!(direct.code(), tc_error::ErrorKind::BadRequest);
            assert_eq!(through_enum.code(), direct.code());
            assert!(!direct.requires_recovery());
            drop(held);
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(2), target.count(tx(11)))
                    .await
                    .unwrap()
                    .unwrap(),
                0
            );
        }

        let later = txn.with_id(tx(12));
        let source_schema = BTreeSchema::new(
            StorageConfig {
                order: 4,
                block_size: 8192,
            },
            1,
            Some(vec![columns[0].dtype.clone()]),
        )
        .unwrap();

        let source = BTree::<TestTxn>::try_with_schema(
            txn.subcontext_unique().context().await.unwrap(),
            source_schema,
        )
        .await
        .unwrap();

        source.insert_row(&later, vec![5_u64.into()]).await.unwrap();
        let source = BTreeView::new(columns, source).unwrap();
        target.restore_from(&later, &source).await.unwrap();
        assert!(
            target
                .contains_row(later.id(), &[5_u64.into()])
                .await
                .unwrap()
        );

        raw.rollback(later.id()).unwrap();

        let table_schema = |key: &str, dtype: ValueType, index: &str, storage| {
            TableSchema::new(
                vec![Column {
                    name: key.parse().unwrap(),
                    dtype,
                }],
                vec![Column {
                    name: "value".parse().unwrap(),
                    dtype: ValueType::Number(NumberType::Number),
                }],
                vec![(index.into(), vec!["value".parse().unwrap()])],
                storage,
            )
            .unwrap()
        };

        let schema = table_schema(
            "id",
            ValueType::Number(NumberType::Number),
            "by_value",
            StorageConfig::default(),
        );

        let target = PersistentTable::<TestTxn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            schema.clone(),
        )
        .await
        .unwrap();

        for schema in [
            table_schema(
                "other",
                ValueType::Number(NumberType::Number),
                "by_value",
                StorageConfig::default(),
            ),
            table_schema(
                "id",
                ValueType::String,
                "by_value",
                StorageConfig::default(),
            ),
            table_schema(
                "id",
                ValueType::Number(NumberType::Number),
                "other_index",
                StorageConfig::default(),
            ),
        ] {
            let source = PersistentTable::<TestTxn>::try_new(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
            )
            .await
            .unwrap();

            let context = txn.context().await.unwrap();
            let held = context.write().await;
            let direct = target
                .restore_from(&txn, &source)
                .now_or_never()
                .expect("schema rejection precedes staging")
                .unwrap_err();

            let through_enum = Collection::from(target.clone())
                .restore_from(&txn, &Collection::from(source))
                .now_or_never()
                .expect("enum delegates synchronous rejection")
                .unwrap_err();

            assert_eq!(direct.code(), tc_error::ErrorKind::BadRequest);
            assert_eq!(through_enum.code(), direct.code());
            drop(held);
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(2), target.count(tx(11)))
                    .await
                    .unwrap()
                    .unwrap(),
                0
            );
        }

        let source_schema = table_schema(
            "id",
            ValueType::Number(NumberType::Number),
            "by_value",
            StorageConfig {
                order: 4,
                block_size: 8192,
            },
        );

        let source = PersistentTable::<TestTxn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            source_schema,
        )
        .await
        .unwrap();

        source
            .upsert_row(&later, vec![1_u64.into()], vec![8_u64.into()])
            .await
            .unwrap();

        Collection::from(target.clone())
            .restore_from(&later, &Collection::from(source))
            .await
            .unwrap();

        assert_eq!(
            target
                .read_row(later.id(), &[1_u64.into()])
                .await
                .unwrap()
                .unwrap()
                .as_slice(),
            &[1_u64.into(), 8_u64.into()]
        );

        target.rollback(later.id()).unwrap();

        for target_axis in [None, Some(0), Some(1)] {
            let dtype = NumberType::Float(number_general::FloatType::F64);
            let schema = TensorSchema::new(dtype, vec![2, 3].into(), target_axis).unwrap();
            let target =
                Tensor::create(txn.subcontext_unique().context().await.unwrap(), schema, 2)
                    .await
                    .unwrap();

            for source_axis in [None, Some(0), Some(1)] {
                if source_axis == target_axis {
                    continue;
                }

                let schema = TensorSchema::new(dtype, vec![2, 3].into(), source_axis).unwrap();
                let source =
                    Tensor::create(txn.subcontext_unique().context().await.unwrap(), schema, 3)
                        .await
                        .unwrap();

                let context = txn.context().await.unwrap();
                let held = context.write().await;
                let direct = target
                    .restore_from(&txn, &source)
                    .now_or_never()
                    .expect("schema rejection precedes staging")
                    .unwrap_err();

                let through_enum = Collection::from(target.clone())
                    .restore_from(&txn, &Collection::from(source))
                    .now_or_never()
                    .expect("enum delegates synchronous rejection")
                    .unwrap_err();

                assert_eq!(direct.code(), tc_error::ErrorKind::BadRequest);
                assert_eq!(through_enum.code(), direct.code());
                drop(held);
                assert_eq!(
                    tokio::time::timeout(
                        Duration::from_secs(2),
                        target.read_value(tx(11), &[0, 0])
                    )
                    .await
                    .unwrap()
                    .unwrap(),
                    0.0.into()
                );
            }

            let schema = TensorSchema::new(dtype, vec![2, 3].into(), target_axis).unwrap();
            let source =
                Tensor::create(txn.subcontext_unique().context().await.unwrap(), schema, 3)
                    .await
                    .unwrap();

            source
                .write_value(&later, &[1, 2], 9.0.into())
                .await
                .unwrap();

            target.restore_from(&later, &source).await.unwrap();
            assert_eq!(
                target.read_value(later.id(), &[1, 2]).await.unwrap(),
                9.0.into()
            );

            target.rollback(&later.id()).await.unwrap();
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concrete_views_preserve_error_policy_and_release_sources() {
    use std::cell::RefCell;

    use futures::TryStreamExt;

    use crate::CollectionView;
    use crate::btree::StorageConfig;
    use crate::collection::CollectionSchema;
    use crate::table::{Column, TableSchema};
    use crate::test::storage::STORAGE_ERRORS;
    use tc_ir::IntoView;
    use tc_value::{NumberType, ValueType};

    Box::pin(async {
        for table in [false, true] {
            let txn = TestTxn::fresh(1);
            let dir = txn.subcontext_unique().context().await.unwrap();
            let dtype = ValueType::Number(NumberType::Number);
            let schema = if table {
                CollectionSchema::Table(
                    TableSchema::new(
                        vec![Column {
                            name: "id".parse().unwrap(),
                            dtype: dtype.clone(),
                        }],
                        vec![Column {
                            name: "value".parse().unwrap(),
                            dtype,
                        }],
                        vec![],
                        StorageConfig::default(),
                    )
                    .unwrap(),
                )
            } else {
                CollectionSchema::BTree(vec![BTreeColumnSchema {
                    name: "id".into(),
                    dtype,
                    max_size: None,
                }])
            };

            let owner = Collection::<TestTxn>::create(dir.clone(), schema)
                .await
                .unwrap();

            let native = if table {
                dir.read().await.get_dir("primary").unwrap().clone()
            } else {
                dir.clone()
            };

            let root = native
                .read()
                .await
                .get_file("00000000-0000-0000-0000-000000000000")
                .unwrap()
                .clone();

            for direct in [false, true] {
                *root.write::<b_table::Node<Value>>().await.unwrap() =
                    b_tree::Node::Leaf(vec![Vec::new()]);

                STORAGE_ERRORS
                    .scope(RefCell::new(Vec::new()), async {
                        let mut projected = if direct {
                            match owner.clone() {
                                Collection::BTree(view) => (*view)
                                    .into_view(txn.clone())
                                    .await
                                    .map(CollectionView::BTree),
                                Collection::Table(view) => (*view)
                                    .into_view(txn.clone())
                                    .await
                                    .map(CollectionView::Table),
                                Collection::Tensor(_) => unreachable!(),
                            }
                        } else {
                            owner.clone().into_view(txn.clone()).await
                        };

                        let error = match &mut projected {
                            Err(error) => error.clone(),
                            Ok(CollectionView::BTree(view)) => {
                                view.keys.try_next().await.unwrap_err()
                            }
                            Ok(CollectionView::Table(view)) => {
                                view.rows.try_next().await.unwrap_err()
                            }
                            Ok(CollectionView::Tensor(_)) => unreachable!(),
                        };

                        assert!(error.requires_recovery());
                        assert_eq!(
                            STORAGE_ERRORS.with(|errors| errors.borrow().clone()),
                            vec![true]
                        );

                        *root.try_write::<b_table::Node<Value>>().unwrap() =
                            b_tree::Node::Leaf(Vec::new());

                        match &owner {
                            Collection::BTree(view) => view
                                .persistent()
                                .unwrap()
                                .insert_row(&txn, vec![7_u64.into()])
                                .await
                                .unwrap(),
                            Collection::Table(view) => view
                                .persistent()
                                .unwrap()
                                .upsert_row(&txn, vec![7_u64.into()], vec![8_u64.into()])
                                .await
                                .unwrap(),
                            Collection::Tensor(_) => unreachable!(),
                        }

                        // Keep an exhausted/erroring projection alive during the write.
                        drop(projected);
                    })
                    .await;
            }
        }
    })
    .await;
}

#[test]
fn semantic_schema_construction_moves_fields_and_reports_errors() {
    use crate::btree::BTreeSchema;
    use crate::collection::CollectionSchema;
    use crate::tensor::TensorSchema;
    use tc_value::class::NativeClass;

    let name = String::from("identifier");
    let pointer = name.as_ptr();
    let schema = Value::Tuple(vec![Value::Tuple(vec![
        Value::String(name),
        Value::from(tc_value::ValueType::String.path().to_string()),
    ])]);

    let CollectionSchema::BTree(columns) =
        CollectionSchema::try_from((crate::BTreeType.path(), schema)).unwrap()
    else {
        panic!("BTree schema")
    };

    assert_eq!(columns[0].name.as_ptr(), pointer);
    let native = BTreeSchema::try_from(columns.as_slice()).unwrap();
    assert_eq!(b_tree::Schema::len(&native), 1);
    assert_eq!(columns[0].name.as_ptr(), pointer);
    assert!(BTreeSchema::try_from([].as_slice()).is_err());

    for value in [
        Value::None,
        Value::Tuple(vec![]),
        Value::Tuple(vec![Value::None]),
    ] {
        let error = CollectionSchema::try_from((crate::BTreeType.path(), value)).unwrap_err();
        assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
        assert!(!error.requires_recovery());
    }

    let unknown = crate::BTreeType.path().append(pathlink::label("unknown"));
    assert!(CollectionSchema::try_from((unknown, Value::None)).is_err());

    let dtype =
        tc_value::number_type_path(&tc_value::NumberType::Float(number_general::FloatType::F64));

    for fields in [
        vec![],
        vec![Value::from(dtype.to_string())],
        vec![Value::None, Value::Tuple(vec![2_u64.into()])],
        vec![Value::from(dtype.to_string()), Value::None],
        vec![Value::from(dtype.to_string()), Value::Tuple(vec![])],
        vec![
            Value::from(dtype.to_string()),
            Value::Tuple(vec![0_u64.into()]),
        ],
        vec![
            Value::from(dtype.to_string()),
            Value::Tuple(vec![Value::from(-1_i64)]),
        ],
        vec![
            Value::from(dtype.to_string()),
            Value::Tuple(vec![2_u64.into()]),
            Value::None,
        ],
        vec![
            Value::from(dtype.to_string()),
            Value::Tuple(vec![2_u64.into()]),
            1_u64.into(),
        ],
        vec![
            Value::from(dtype.to_string()),
            Value::Tuple(vec![2_u64.into()]),
            0_u64.into(),
            Value::None,
        ],
    ] {
        let value = Value::Tuple(fields);
        let direct = TensorSchema::try_from(value.clone()).unwrap_err();
        let delegated = CollectionSchema::try_from((crate::TensorType.path(), value)).unwrap_err();
        assert_eq!(direct.code(), delegated.code());
        assert_eq!(direct.message(), delegated.message());
        assert!(!delegated.requires_recovery());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concrete_and_enum_lifecycle_decisions_agree() {
    use safecast::CastFrom;

    use crate::btree::StorageConfig;
    use crate::collection::CollectionSchema;
    use crate::table::{Column, Table, TableSchema};
    use crate::test::tx;

    Box::pin(async {
        let txn = TestTxn::fresh(10);
        let table_schema = TableSchema::new(
            vec![Column {
                name: "id".parse().unwrap(),
                dtype: tc_value::ValueType::String,
            }],
            Vec::new(),
            Vec::new(),
            StorageConfig::default(),
        )
        .unwrap();

        for schema in [
            CollectionSchema::BTree(vec![BTreeColumnSchema {
                name: "id".into(),
                dtype: tc_value::ValueType::String,
                max_size: None,
            }]),
            CollectionSchema::Table(table_schema.clone()),
        ] {
            let owner = Collection::<TestTxn>::create(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
            )
            .await
            .unwrap();

            let view = match &owner {
                Collection::BTree(tree) => {
                    tree.commit(tx(10)).await.unwrap();
                    owner.commit(tx(10)).await.unwrap();
                    assert!(tree.rollback(&tx(10)).await.is_err());
                    tree.rollback(&tx(12)).await.unwrap();
                    tree.finalize(&tx(11)).await.unwrap();
                    Collection::BTree(Box::new(
                        tree.slice((Bound::Unbounded, Bound::Unbounded), false),
                    ))
                }
                Collection::Table(table) => {
                    table.commit(tx(10)).await.unwrap();
                    owner.commit(tx(10)).await.unwrap();
                    assert!(table.rollback(&tx(10)).await.is_err());
                    table.rollback(&tx(12)).await.unwrap();
                    table.finalize(&tx(11)).await.unwrap();
                    Collection::from(table.read_view())
                }
                Collection::Tensor(_) => unreachable!(),
            };

            assert!(owner.commit(tx(10)).await.is_err());
            owner.rollback(&tx(12)).await.unwrap();
            owner.finalize(&tx(11)).await.unwrap();
            owner.sync().await.unwrap();
            owner.sync_all().await.unwrap();
            match &view {
                Collection::BTree(tree) => {
                    assert!(tree.commit(tx(13)).await.is_err());
                    assert!(tree.rollback(&tx(13)).await.is_err());
                    assert!(tree.finalize(&tx(13)).await.is_err());
                    assert!(tree.sync().await.is_err());
                    assert!(tree.sync_all().await.is_err());
                }
                Collection::Table(table) => {
                    assert!(table.commit(tx(13)).await.is_err());
                    assert!(table.rollback(&tx(13)).await.is_err());
                    assert!(table.finalize(&tx(13)).await.is_err());
                }
                Collection::Tensor(_) => unreachable!(),
            }

            assert!(view.commit(tx(13)).await.is_err());
            assert!(view.rollback(&tx(13)).await.is_err());
            assert!(view.finalize(&tx(13)).await.is_err());
            owner.commit(tx(14)).await.unwrap();
            owner.commit(tx(14)).await.unwrap();
        }

        let encoded =
            destream_json::encode((Value::cast_from(table_schema), Vec::<Value>::new())).unwrap();

        let local = destream_json::try_decode::<_, _, crate::table::DecodedTablePayload<TestTxn>>(
            txn.clone(),
            encoded,
        )
        .await
        .unwrap()
        .table;

        assert!(matches!(local, Table::Local(_)));
        for _ in 0..2 {
            local.commit(tx(10)).await.unwrap();
            local.rollback(&tx(10)).await.unwrap();
            local.finalize(&tx(10)).await.unwrap();
            let delegated = Collection::from(local.clone());
            delegated.commit(tx(10)).await.unwrap();
            delegated.rollback(&tx(10)).await.unwrap();
            delegated.finalize(&tx(10)).await.unwrap();
            assert_eq!(local.count(tx(10)).await.unwrap(), 0);
        }
    })
    .await;
}
