//! Public API route handlers for a transactional [`PersistentTable`].
//!
//! Ports the v1 `table/public.rs` routing logic.  Each route is a separate
//! handler **struct** (not an enum variant) with a `From` impl, following
//! the v1 pattern.  Handlers are generic over the response type `State`,
//! which must support `From<Table>` and `From<Value>` (and `From<u64>` for
//! `CountHandler`). Table handlers only produce their owned `Table` type.
//!
//! [`TableRoutes`] resolves a path to a [`TableRoute`], which implements the
//! same native [`tc_ir::Handler`] contract as every other collection. Routing
//! and execution remain table-owned and serialization-free.
//!
//! ## Module layout
//!
//! - [`handler`] — individual handler structs + verb trait impls
//! - [`selector`] — `KeyOrRange` and `cast_into_range` selector parsing

pub mod handler;
pub mod selector;

use crate::table::Table;

pub use handler::{
    ContainsHandler, CountHandler, LimitHandler, OrderHandler, SelectHandler, TableHandler,
};
/// Owned routes for one persistent table.
#[derive(Clone)]
pub struct TableRoutes<State: crate::CollectionState> {
    table: Table<State::Txn>,
    state: std::marker::PhantomData<fn() -> State>,
}

impl<State: crate::CollectionState> TableRoutes<State> {
    pub fn new(table: Table<State::Txn>) -> Self {
        Self {
            table,
            state: std::marker::PhantomData,
        }
    }
}

/// A concrete table route handler.
pub enum TableRoute<State: crate::CollectionState> {
    Table(TableHandler<State::Txn>),
    Columns(handler::SchemaHandler<State::Txn>),
    Contains(ContainsHandler<State::Txn>),
    Count(CountHandler<State::Txn>),
    KeyColumns(handler::SchemaHandler<State::Txn>),
    Limit(LimitHandler<State::Txn>),
    Order(OrderHandler<State::Txn>),
    Select(SelectHandler<State::Txn>),
    State(std::marker::PhantomData<fn() -> State>),
}

impl<State: crate::CollectionState> tc_ir::Route<State> for TableRoutes<State> {
    type Handler = TableRoute<State>;

    fn route(&self, path: &[pathlink::PathSegment]) -> Option<Self::Handler> {
        let route = if path.is_empty() {
            TableRoute::Table(TableHandler::from(self.table.clone()))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "columns" => TableRoute::Columns(handler::SchemaHandler::new(
                    self.table.clone(),
                    handler::column_schema,
                )),
                "contains" => TableRoute::Contains(ContainsHandler::from(self.table.clone())),
                "count" => TableRoute::Count(CountHandler::from(self.table.clone())),
                "key_columns" => TableRoute::KeyColumns(handler::SchemaHandler::new(
                    self.table.clone(),
                    handler::key_columns,
                )),
                "limit" => TableRoute::Limit(LimitHandler::from(self.table.clone())),
                "order" => TableRoute::Order(OrderHandler::from(self.table.clone())),
                "select" => TableRoute::Select(SelectHandler::from(self.table.clone())),
                _ => return None,
            }
        } else {
            return None;
        };

        Some(route)
    }
}

/// Resolve a persistent table route for the caller's universal state type.
pub fn route<State: crate::CollectionState>(
    table: &(impl Clone + Into<Table<State::Txn>>),
    path: &[pathlink::PathSegment],
) -> Option<TableRoute<State>> {
    tc_ir::Route::route(&TableRoutes::new(table.clone().into()), path)
}

impl<State> tc_ir::Handler<State> for TableRoute<State>
where
    State: crate::CollectionState,
{
    async fn get(&self, txn: &State::Txn, request: tc_ir::Scalar) -> tc_error::TCResult<State> {
        match self {
            Self::Table(handler) => handler.get(txn, request)?.await,
            Self::Columns(handler) | Self::KeyColumns(handler) => handler.get(txn, request)?.await,
            Self::Contains(handler) => handler.get(txn, request)?.await,
            Self::Count(handler) => handler.get(txn, request)?.await,
            Self::Limit(handler) => handler.get(txn, request)?.await,
            Self::Order(handler) => handler.get(txn, request)?.await,
            Self::Select(handler) => handler.get(txn, request)?.await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }

    async fn put(
        &self,
        txn: &State::Txn,
        key: tc_ir::Scalar,
        value: State,
    ) -> tc_error::TCResult<()> {
        let request = [
            ("key".parse().expect("static key id"), key),
            (
                "value".parse().expect("static value id"),
                value.into_scalar()?,
            ),
        ]
        .into_iter()
        .collect();
        match self {
            Self::Table(handler) => handler.put(txn, request)?.await,
            Self::Columns(handler) | Self::KeyColumns(handler) => handler.put(txn, request)?.await,
            Self::Contains(handler) => handler.put(txn, request)?.await,
            Self::Count(handler) => handler.put(txn, request)?.await,
            Self::Limit(handler) => handler.put(txn, request)?.await,
            Self::Order(handler) => handler.put(txn, request)?.await,
            Self::Select(handler) => handler.put(txn, request)?.await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }

    async fn post(
        &self,
        txn: &State::Txn,
        request: tc_ir::Map<State>,
    ) -> tc_error::TCResult<State> {
        let request = request
            .into_iter()
            .map(|(id, value)| value.into_scalar().map(|value| (id, value)))
            .collect::<tc_error::TCResult<tc_ir::Map<_>>>()?;
        match self {
            Self::Table(handler) => handler.post(txn, request)?.await,
            Self::Columns(handler) | Self::KeyColumns(handler) => handler.post(txn, request)?.await,
            Self::Contains(handler) => handler.post(txn, request)?.await,
            Self::Count(handler) => handler.post(txn, request)?.await,
            Self::Limit(handler) => handler.post(txn, request)?.await,
            Self::Order(handler) => handler.post(txn, request)?.await,
            Self::Select(handler) => handler.post(txn, request)?.await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }

    async fn delete(&self, txn: &State::Txn, request: tc_ir::Scalar) -> tc_error::TCResult<()> {
        match self {
            Self::Table(handler) => handler.delete(txn, request)?.await,
            Self::Columns(handler) | Self::KeyColumns(handler) => {
                handler.delete(txn, request)?.await
            }
            Self::Contains(handler) => handler.delete(txn, request)?.await,
            Self::Count(handler) => handler.delete(txn, request)?.await,
            Self::Limit(handler) => handler.delete(txn, request)?.await,
            Self::Order(handler) => handler.delete(txn, request)?.await,
            Self::Select(handler) => handler.delete(txn, request)?.await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }
}

#[cfg(test)]
mod tests;
