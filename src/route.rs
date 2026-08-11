use pathlink::Link;
use tc_error::{TCError, TCResult};
use tc_ir::{Handler, Map, NativeClass, Scalar};
use tc_value::Value;

use crate::Collection;
use crate::btree::{BTreeRoute, BTreeRoutes};
use crate::class::CollectionType;
use crate::table::public::{TableRoute, TableRoutes};
use crate::tensor::{Tensor, TensorRoute, TensorRoutes};

/// The minimal conversion boundary required by collection route handlers.
///
/// Implementations live in the caller's state crate so collection code never
/// depends on a particular universal state representation.
pub trait CollectionState:
    tc_ir::StateInstance<Transaction = Self::Txn>
    + Clone
    + Send
    + 'static
    + From<Scalar>
    + From<Collection<Self::Txn>>
    + From<crate::table::Table<Self::Txn>>
    + From<Value>
    + From<u64>
{
    type Txn: crate::StorageContext;

    fn into_scalar(self) -> TCResult<Scalar>;
    fn into_value(self) -> TCResult<Value>;
    fn into_tuple(self) -> TCResult<Vec<Self>>;
    fn into_map(self) -> TCResult<Map<Self>>;
    fn into_tensor(self) -> TCResult<Tensor>;
    fn is_none(&self) -> bool;
}

/// Routes exposed by a collection value.
#[derive(Clone)]
pub enum CollectionRoutes<S: CollectionState> {
    BTree(Box<BTreeRoutes<S>>),
    Table(Box<TableRoutes<S>>),
    Tensor(TensorRoutes<S>),
}

/// A resolved native collection handler.
pub enum CollectionRoute<S: CollectionState> {
    BTree(Box<BTreeRoute<S>>),
    Table(Box<TableRoute<S>>),
    Tensor(TensorRoute<S>),
}

impl<S: CollectionState> tc_ir::Route<S> for CollectionRoutes<S> {
    type Handler = CollectionRoute<S>;

    fn route(&self, path: &[pathlink::PathSegment]) -> Option<Self::Handler> {
        match self {
            Self::BTree(routes) => tc_ir::Route::route(routes.as_ref(), path)
                .map(Box::new)
                .map(CollectionRoute::BTree),
            Self::Table(routes) => tc_ir::Route::route(routes.as_ref(), path)
                .map(Box::new)
                .map(CollectionRoute::Table),
            Self::Tensor(routes) => tc_ir::Route::route(routes, path).map(CollectionRoute::Tensor),
        }
    }
}

impl<Txn: crate::StorageContext> Collection<Txn> {
    pub fn routes<S: CollectionState<Txn = Txn>>(&self) -> Option<CollectionRoutes<S>> {
        match self {
            Self::BTree(view) => Some(CollectionRoutes::BTree(Box::new(BTreeRoutes::new(
                (**view).clone(),
            )))),
            Self::Table(table) => Some(CollectionRoutes::Table(Box::new(TableRoutes::new(
                (**table).clone(),
            )))),
            Self::Tensor(tensor) => {
                Some(CollectionRoutes::Tensor(TensorRoutes::new(tensor.clone())))
            }
        }
    }

    pub fn from_put<S: CollectionState<Txn = Txn>>(
        link: &Link,
        key: S,
        value: S,
    ) -> TCResult<Option<S>> {
        match CollectionType::from_path(link.path()) {
            Some(CollectionType::Tensor(_)) => crate::tensor::route::tensor_literal(key, value)
                .map(|tensor| Some(S::from(Self::Tensor(tensor)))),
            Some(CollectionType::BTree(_) | CollectionType::Table(_)) | None => Ok(None),
        }
    }
}

impl<S: CollectionState> Handler<S> for CollectionRoute<S> {
    async fn get(&self, txn: &S::Txn, key: Scalar) -> TCResult<S> {
        match self {
            Self::BTree(route) => Handler::get(route.as_ref(), txn, key).await,
            Self::Table(route) => Handler::get(route.as_ref(), txn, key).await,
            Self::Tensor(route) => Handler::get(route, txn, key).await,
        }
    }

    async fn put(&self, txn: &S::Txn, key: Scalar, value: S) -> TCResult<()> {
        match self {
            Self::Table(route) => Handler::put(route.as_ref(), txn, key, value).await,
            Self::BTree(_) | Self::Tensor(_) => Err(TCError::method_not_allowed(
                tc_ir::Method::Put,
                "collection",
            )),
        }
    }

    async fn post(&self, txn: &S::Txn, params: Map<S>) -> TCResult<S> {
        match self {
            Self::BTree(route) => Handler::post(route.as_ref(), txn, params).await,
            Self::Table(route) => Handler::post(route.as_ref(), txn, params).await,
            Self::Tensor(route) => Handler::post(route, txn, params).await,
        }
    }

    async fn delete(&self, txn: &S::Txn, key: Scalar) -> TCResult<()> {
        match self {
            Self::BTree(route) => Handler::delete(route.as_ref(), txn, key).await,
            Self::Table(route) => Handler::delete(route.as_ref(), txn, key).await,
            Self::Tensor(_) => Err(TCError::method_not_allowed(tc_ir::Method::Delete, "tensor")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::sync::Arc;

    use freqfs::Cache;
    use pathlink::{Link, PathSegment};
    use safecast::TryCastFrom;
    use tc_ir::{Claim, NetworkTime, Route, Transaction, TxnId};
    use tc_value::Value;

    use super::*;
    use crate::btree::BTreeColumnSchema;
    use crate::collection::BTreeView;

    #[allow(dead_code)]
    #[derive(Clone, Debug)]
    enum TestState {
        Value(Value),
        Tuple(Vec<TestState>),
        Map(Map<TestState>),
        Collection(Collection<TestTxn>),
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
        fn into_tensor(self) -> TCResult<Tensor> {
            match self {
                Self::Collection(Collection::Tensor(tensor)) => Ok(tensor),
                _ => Err(TCError::bad_request("expected tensor")),
            }
        }
        fn is_none(&self) -> bool {
            matches!(self, Self::Value(Value::None))
        }
    }

    #[derive(Clone, Debug)]
    struct TestTxn {
        id: TxnId,
        claim: Claim,
        root: freqfs::DirLock<crate::PersistentFile>,
        path: Vec<String>,
    }

    impl TestTxn {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "tc-collection-route-txn-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
            std::fs::create_dir_all(&root).expect("transaction root");
            let cache = Cache::<crate::PersistentFile>::new(
                16 * 1024 * 1024,
                None,
                0,
                std::time::Duration::from_secs(3),
            );
            let root = cache.load(root).expect("load transaction root");
            Self {
                id: TxnId::from_parts(NetworkTime::from_nanos(1), 1),
                claim: Claim::new(Link::from_str("/test").expect("claim"), umask::Mode::all()),
                root,
                path: Vec::new(),
            }
        }
    }
    impl Transaction for TestTxn {
        fn id(&self) -> TxnId {
            self.id
        }
        fn timestamp(&self) -> NetworkTime {
            self.id().timestamp()
        }
        fn claim(&self) -> &Claim {
            &self.claim
        }
    }

    impl crate::StorageContext for TestTxn {
        fn context(
            &self,
        ) -> impl std::future::Future<Output = TCResult<freqfs::DirLock<crate::PersistentFile>>> + Send
        {
            let root = self.root.clone();
            let mut path = vec![self.id.to_string()];
            path.extend(self.path.clone());
            async move {
                let mut current = root;
                for name in path {
                    let next = {
                        let mut dir = current.write().await;
                        dir.get_or_create_dir(name).map_err(TCError::internal)?
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
            self.subcontext("literal")
        }

        fn materialized_tensor_bytes(&self) -> usize {
            256 * 1024 * 1024
        }
    }

    fn segment(path: &str) -> PathSegment {
        PathSegment::from_str(path).expect("path segment")
    }

    fn btree_routes() -> BTreeRoutes<TestState> {
        let root = std::env::temp_dir().join(format!("tc-collection-route-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
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
        let btree = crate::btree::BTree::<TestTxn>::new(persistent);
        BTreeRoutes::new(BTreeView::new(
            vec![BTreeColumnSchema {
                name: "id".to_string(),
                dtype: tc_value::ValueType::Number,
                max_size: None,
            }],
            btree,
        ))
    }

    #[tokio::test]
    async fn tensor_route_rejects_unknown_path_and_invalid_shape() {
        let routes = TensorRoutes::<TestState>::new(
            Tensor::dense_f64(vec![2], vec![1.0, 2.0]).expect("tensor"),
        );
        assert!(routes.route(&[segment("unknown")]).is_none());

        let route = routes.route(&[segment("reshape")]).expect("reshape route");
        let err = route
            .get(
                &TestTxn::new(),
                Scalar::Value(Value::String("invalid".into())),
            )
            .await
            .expect_err("invalid reshape request");
        assert!(!err.to_string().is_empty());
    }

    #[tokio::test]
    async fn btree_route_rejects_unknown_path_and_invalid_row() {
        let routes = btree_routes();
        assert!(routes.route(&[segment("unknown")]).is_none());

        let route = routes
            .route(&[segment("contains")])
            .expect("contains route");
        let err = route
            .get(
                &TestTxn::new(),
                Scalar::Value(Value::String("not a BTree row".into())),
            )
            .await
            .expect_err("invalid BTree row");
        assert!(!err.to_string().is_empty());
    }
}
