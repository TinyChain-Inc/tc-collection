use std::fmt;
use std::future::Future;
use std::pin::Pin;

use pathlink::PathSegment;
use tc_error::{bad_request, TCError, TCResult};
use tc_ir::{
    HandleDelete, HandleGet, HandlePost, HandlePut, Handler, Map, Method, Route, Scalar,
    Transaction, Transact, TxnId,
};
use tc_value::Value;

use crate::btree::BTree;
use crate::table::{
    PersistentTable, TableResponse, TableRoute, TableRouter,
};
use crate::table::public::scalar_to_value;

#[derive(Debug, Clone)]
pub enum Collection {
    BTree(BTree),
    Table(PersistentTable),
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Self::BTree(btree)
    }
}

impl From<PersistentTable> for Collection {
    fn from(table: PersistentTable) -> Self {
        Self::Table(table)
    }
}

impl Collection {
    pub fn as_btree(&self) -> Option<&BTree> {
        match self {
            Self::BTree(btree) => Some(btree),
            _ => None,
        }
    }

    pub fn into_btree(self) -> BTree {
        match self {
            Self::BTree(btree) => btree,
            _ => panic!("Collection is not a BTree"),
        }
    }

    pub fn as_table(&self) -> Option<&PersistentTable> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }

    pub fn into_table(self) -> PersistentTable {
        match self {
            Self::Table(table) => table,
            _ => panic!("Collection is not a Table"),
        }
    }

    /// Build a [`CollectionRouter`] from this collection.
    ///
    /// This is the primary entry point for routing: the host calls
    /// `collection.router()` to obtain a [`CollectionRouter`] that implements
    /// [`tc_ir::Route`], then calls `router.route(&path)` to resolve handlers.
    ///
    /// Without this call, the routing layer is unreachable — `Collection`
    /// itself does not implement `Route` because the v2 `Route` trait returns
    /// borrowed handler references, requiring pre-constructed handlers.
    pub fn router(self) -> CollectionRouter {
        CollectionRouter::new(self)
    }
}

impl Transact for Collection {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => Transact::commit(btree, txn_id).await,
            Self::Table(table) => Transact::commit(table, txn_id).await,
        }
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::rollback(btree, &txn_id).await,
                Self::Table(table) => Transact::rollback(table, &txn_id).await,
            }
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::finalize(btree, &txn_id).await,
                Self::Table(table) => Transact::finalize(table, &txn_id).await,
            }
        }
    }
}

// ─── Collection-level routing ──────────────────────────────────────────

/// A single route handler at the `Collection` level.
///
/// Either delegates to a [`TableRoute`] sub-handler or handles the
/// collection-level `schema` route (v1 `public/table.rs` `SchemaHandler`).
#[derive(Clone)]
pub enum CollectionRoute {
    /// Delegate to a table sub-route handler.
    Table(TableRoute),
    /// `GET <collection>/schema` — return the collection's schema as a `Value`.
    Schema(PersistentTable),
}

impl fmt::Debug for CollectionRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Table(route) => f.debug_tuple("Table").field(route).finish(),
            Self::Schema(_) => f.write_str("Schema"),
        }
    }
}

type GetFut<'a> = Pin<Box<dyn Future<Output = TCResult<TableResponse>> + Send + 'a>>;
type PutFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
type PostFut<'a> = Pin<Box<dyn Future<Output = TCResult<TableResponse>> + Send + 'a>>;
type DeleteFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;

impl<T: Transaction + ?Sized> HandleGet<T> for CollectionRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = TableResponse;
    type Error = TCError;
    type Fut<'a>
        = GetFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn get<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::Table(route) => route.get(txn, request),
            Self::Schema(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                if value != Value::None {
                    return Err(bad_request!("schema route takes no parameters"));
                }
                Ok(Box::pin(async move {
                    Ok(TableResponse::Value(table.schema().to_value()))
                }))
            }
        }
    }
}

impl<T: Transaction + ?Sized> HandlePut<T> for CollectionRoute {
    type Request = Map<Scalar>;
    type RequestContext = ();
    type Response = ();
    type Error = TCError;
    type Fut<'a>
        = PutFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn put<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::Table(route) => route.put(txn, request),
            Self::Schema(_) => Err(<Self as Handler<T>>::method_not_supported(Method::Put)),
        }
    }
}

impl<T: Transaction + ?Sized> HandlePost<T> for CollectionRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = TableResponse;
    type Error = TCError;
    type Fut<'a>
        = PostFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn post<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::Table(route) => route.post(txn, request),
            Self::Schema(_) => Err(<Self as Handler<T>>::method_not_supported(Method::Post)),
        }
    }
}

impl<T: Transaction + ?Sized> HandleDelete<T> for CollectionRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = ();
    type Error = TCError;
    type Fut<'a>
        = DeleteFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn delete<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::Table(route) => route.delete(txn, request),
            Self::Schema(_) => Err(<Self as Handler<T>>::method_not_supported(Method::Delete)),
        }
    }
}

/// Router for a [`Collection`] instance.
///
/// Implements [`Route`] by dispatching to [`TableRouter`] sub-routes for the
/// `Table` variant, and exposes the collection-level `schema` route.
///
/// `CollectionRouter` pre-constructs all route handlers because the `Route`
/// trait returns borrowed references (`&'a Handler`) — handlers cannot be
/// built on the fly. The 9 table sub-routes are `CollectionRoute::Table`
/// wrappers around `TableRouter`'s handlers; the `schema` route is
/// collection-level.
///
/// Route table:
/// | Path | Handler |
/// |------|---------|
/// | `[]` | table root (read/slice/mutate) |
/// | `["columns"]` | column names |
/// | `["contains"]` | row presence |
/// | `["count"]` | row count |
/// | `["key_columns"]` | key column names |
/// | `["key_names"]` | key column names |
/// | `["limit"]` | row cap |
/// | `["order"]` | ordering |
/// | `["select"]` | column projection |
/// | `["schema"]` | collection schema |
pub struct CollectionRouter {
    table: CollectionRoute,
    columns: CollectionRoute,
    contains: CollectionRoute,
    count: CollectionRoute,
    key_columns: CollectionRoute,
    key_names: CollectionRoute,
    limit: CollectionRoute,
    order: CollectionRoute,
    select: CollectionRoute,
    schema: CollectionRoute,
}

impl CollectionRouter {
    /// Build a router from a [`Collection`].
    ///
    /// Currently only the `Table` variant is routed (BTree routing is a
    /// separate issue).
    pub fn new(collection: Collection) -> Self {
        match collection {
            Collection::Table(table) => {
                let router = TableRouter::new(table.clone());
                Self {
                    table: CollectionRoute::Table(router.table.clone()),
                    columns: CollectionRoute::Table(router.columns.clone()),
                    contains: CollectionRoute::Table(router.contains.clone()),
                    count: CollectionRoute::Table(router.count.clone()),
                    key_columns: CollectionRoute::Table(router.key_columns.clone()),
                    key_names: CollectionRoute::Table(router.key_names.clone()),
                    limit: CollectionRoute::Table(router.limit.clone()),
                    order: CollectionRoute::Table(router.order.clone()),
                    select: CollectionRoute::Table(router.select.clone()),
                    schema: CollectionRoute::Schema(table),
                }
            }
            Collection::BTree(_) => {
                panic!("CollectionRouter does not yet support BTree routing")
            }
        }
    }
}

impl fmt::Debug for CollectionRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CollectionRouter")
            .field("table", &self.table)
            .field("schema", &self.schema)
            .finish()
    }
}

impl Route for CollectionRouter {
    type Handler = CollectionRoute;

    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<&'a Self::Handler> {
        if path.is_empty() {
            Some(&self.table)
        } else if path.len() == 1 {
            match path[0].as_str() {
                "columns" => Some(&self.columns),
                "contains" => Some(&self.contains),
                "count" => Some(&self.count),
                "key_columns" => Some(&self.key_columns),
                "key_names" => Some(&self.key_names),
                "limit" => Some(&self.limit),
                "order" => Some(&self.order),
                "select" => Some(&self.select),
                "schema" => Some(&self.schema),
                _ => None,
            }
        } else {
            None
        }
    }
}
