//! Public API route handlers for a transactional [`PersistentTable`].
//!
//! Ports the v1 `table/public.rs` routing logic.  Each route is a separate
//! handler **struct** (not an enum variant) with a `From` impl, following
//! the v1 pattern.  Handlers are generic over the response type `Resp`,
//! which must support `From<Collection>` and `From<Value>` (and `From<u64>`
//! for `CountHandler`) — this mirrors v1's `State: From<Collection> + From<Value>
//! + From<u64>` bounds.
//!
//! The host calls [`route`] to construct the appropriate handler for a given
//! path, then invokes the verb trait method.  No `Route` trait impl or router
//! struct is needed — the host owns routing, consistent with `AGENTS.md`
//! ("keep routing logic shard-local and lean").
//!
//! ## Module layout
//!
//! - [`handler`] — individual handler structs + verb trait impls
//! - [`selector`] — `KeyOrRange` and `cast_into_range` selector parsing

pub mod handler;
pub mod selector;

use freqfs::DirLock;

use crate::btree::PersistentFile;
use crate::table::PersistentTable;

pub use handler::{
    ContainsHandler, CopyHandler, CountHandler, CreateHandler, LimitHandler, OrderHandler,
    SelectHandler, TableHandler,
};

/// Construct the appropriate handler for the given table and path.
///
/// Returns `None` if the path does not match any known route.
///
/// This is the v2 analogue of v1's `route` function in `table/public.rs`.
/// The host calls this to obtain a handler, then invokes the verb trait
/// method (`get`, `put`, `post`, or `delete`).
///
/// Route table (parity port §4):
///
/// | Path | Handler |
/// |------|---------|
/// | `[]` | [`TableHandler`] (read/slice/upsert/update/truncate/delete) |
/// | `["columns"]` | [`handler::SchemaHandler`] (column names) |
/// | `["contains"]` | [`ContainsHandler`] |
/// | `["count"]` | [`CountHandler`] |
/// | `["key_columns"]` | [`handler::SchemaHandler`] (key column ids) |
/// | `["key_names"]` | [`handler::SchemaHandler`] (key column ids) |
/// | `["limit"]` | [`LimitHandler`] |
/// | `["order"]` | [`OrderHandler`] |
/// | `["select"]` | [`SelectHandler`] |
pub fn route<'a, Resp>(
    table: &'a PersistentTable,
    path: &[pathlink::PathSegment],
) -> Option<Box<dyn RouteHandler<Resp> + 'a>>
where
    Resp: From<crate::Collection> + From<tc_value::Value> + From<u64> + Clone + Send + 'static,
{
    if path.is_empty() {
        Some(Box::new(TableHandler::from(table.clone())))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "columns" => Some(Box::new(handler::SchemaHandler::new(
                table.clone(),
                handler::column_schema,
            ))),
            "contains" => Some(Box::new(ContainsHandler::from(table.clone()))),
            "count" => Some(Box::new(CountHandler::from(table.clone()))),
            "key_columns" => Some(Box::new(handler::SchemaHandler::new(
                table.clone(),
                handler::key_columns,
            ))),
            "key_names" => Some(Box::new(handler::SchemaHandler::new(
                table.clone(),
                handler::key_names,
            ))),
            "limit" => Some(Box::new(LimitHandler::from(table.clone()))),
            "order" => Some(Box::new(OrderHandler::from(table.clone()))),
            "select" => Some(Box::new(SelectHandler::from(table.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

/// Trait object returned by [`route`].  Each handler struct implements this
/// via its `HandleGet`/`HandlePut`/`HandlePost`/`HandleDelete` impls.
///
/// This is the v2 analogue of v1's `Box<dyn Handler<'a, State> + 'a>`.
/// Unlike v1, the verb traits have associated types that make them
/// non-object-safe, so we use a custom dispatch trait that erases the
/// future types while preserving the request/response types.
pub trait RouteHandler<Resp>: Send + Sync {
    fn get(
        &self,
        txn: &dyn tc_ir::Transaction,
        request: tc_ir::Scalar,
    ) -> tc_error::TCResult<
        std::pin::Pin<
            Box<dyn std::future::Future<Output = tc_error::TCResult<Resp>> + Send + '_>,
        >,
    >;

    fn put(
        &self,
        txn: &dyn tc_ir::Transaction,
        request: tc_ir::Map<tc_ir::Scalar>,
    ) -> tc_error::TCResult<
        std::pin::Pin<
            Box<dyn std::future::Future<Output = tc_error::TCResult<()>> + Send + '_>,
        >,
    >;

    fn post(
        &self,
        txn: &dyn tc_ir::Transaction,
        request: tc_ir::Scalar,
    ) -> tc_error::TCResult<
        std::pin::Pin<
            Box<dyn std::future::Future<Output = tc_error::TCResult<Resp>> + Send + '_>,
        >,
    >;

    fn delete(
        &self,
        txn: &dyn tc_ir::Transaction,
        request: tc_ir::Scalar,
    ) -> tc_error::TCResult<
        std::pin::Pin<
            Box<dyn std::future::Future<Output = tc_error::TCResult<()>> + Send + '_>,
        >,
    >;
}

/// Static route handler for table construction.
///
/// Ported from v1 `Static` in `table/public.rs`:
/// - `GET /state/collection/table` → [`CreateHandler`] (create a new table)
/// - `POST /state/collection/table/copy_from` → [`CopyHandler`]
pub struct Static {
    root: DirLock<PersistentFile>,
}

impl Static {
    pub fn new(root: DirLock<PersistentFile>) -> Self {
        Self { root }
    }

    /// Construct the appropriate static handler for the given path.
    ///
    /// | Path | Handler |
    /// |------|---------|
    /// | `[]` | [`CreateHandler`] |
    /// | `["copy_from"]` | [`CopyHandler`] |
    pub fn route<Resp>(
        &self,
        path: &[pathlink::PathSegment],
    ) -> Option<Box<dyn StaticRouteHandler<Resp> + '_>>
    where
        Resp: From<crate::Collection> + Clone + Send + 'static,
    {
        if path.is_empty() {
            Some(Box::new(CreateHandler::new(self.root.clone())))
        } else if path.len() == 1 && path[0].as_str() == "copy_from" {
            Some(Box::new(CopyHandler::new(self.root.clone())))
        } else {
            None
        }
    }
}

pub trait StaticRouteHandler<Resp>: Send + Sync {
    fn get(
        &self,
        txn: &dyn tc_ir::Transaction,
        request: tc_ir::Scalar,
    ) -> tc_error::TCResult<
        std::pin::Pin<
            Box<dyn std::future::Future<Output = tc_error::TCResult<Resp>> + Send + '_>,
        >,
    >;

    fn post(
        &self,
        txn: &dyn tc_ir::Transaction,
        request: tc_ir::Map<tc_ir::Scalar>,
    ) -> tc_error::TCResult<
        std::pin::Pin<
            Box<dyn std::future::Future<Output = tc_error::TCResult<Resp>> + Send + '_>,
        >,
    >;
}

#[cfg(test)]
mod tests;
