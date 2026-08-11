//! Public API route handlers for a transactional [`PersistentTable`].
//!
//! [`TableRoutes`] resolves a path to a [`TableRoute`], which implements the
//! same native [`tc_ir::Handler`] contract as every other collection. Routing
//! and execution remain table-owned and serialization-free.
//!
//! ## Module layout
//!
//! - [`handler`] — individual route handlers
//! - [`selector`] — `KeyOrRange` and `cast_into_range` selector parsing

pub mod handler;
pub mod selector;

use crate::table::Table;

pub use handler::{
    ContainsHandler, CountHandler, InsertHandler, IsEmptyHandler, LimitHandler, OrderHandler,
    SelectHandler, TableHandler,
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
    Insert(InsertHandler<State::Txn>),
    IsEmpty(IsEmptyHandler<State::Txn>),
    KeyColumns(handler::SchemaHandler<State::Txn>),
    KeyNames(handler::SchemaHandler<State::Txn>),
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
                "insert" => TableRoute::Insert(InsertHandler::from(self.table.clone())),
                "is_empty" => TableRoute::IsEmpty(IsEmptyHandler::from(self.table.clone())),
                "key_columns" => TableRoute::KeyColumns(handler::SchemaHandler::new(
                    self.table.clone(),
                    handler::key_columns,
                )),
                "key_names" => TableRoute::KeyNames(handler::SchemaHandler::new(
                    self.table.clone(),
                    handler::key_names,
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

async fn dispatch_delete<State, H>(
    handler: &H,
    txn: &State::Txn,
    request: tc_ir::Scalar,
) -> tc_error::TCResult<()>
where
    State: crate::CollectionState,
    H: tc_ir::Handler<State>,
{
    tc_ir::Handler::delete(handler, txn, request).await
}

impl<State> tc_ir::Handler<State> for TableRoute<State>
where
    State: crate::CollectionState,
{
    async fn get(&self, txn: &State::Txn, request: tc_ir::Scalar) -> tc_error::TCResult<State> {
        match self {
            Self::Table(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::Columns(handler) | Self::KeyColumns(handler) | Self::KeyNames(handler) => {
                tc_ir::Handler::get(handler, txn, request).await
            }
            Self::Contains(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::Count(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::Insert(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::IsEmpty(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::Limit(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::Order(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::Select(handler) => tc_ir::Handler::get(handler, txn, request).await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }

    async fn put(
        &self,
        txn: &State::Txn,
        key: tc_ir::Scalar,
        value: State,
    ) -> tc_error::TCResult<()> {
        match self {
            Self::Table(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::Columns(handler) | Self::KeyColumns(handler) | Self::KeyNames(handler) => {
                tc_ir::Handler::put(handler, txn, key, value).await
            }
            Self::Contains(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::Count(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::Insert(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::IsEmpty(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::Limit(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::Order(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::Select(handler) => tc_ir::Handler::put(handler, txn, key, value).await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }

    async fn post(
        &self,
        txn: &State::Txn,
        request: tc_ir::Map<State>,
    ) -> tc_error::TCResult<State> {
        match self {
            Self::Table(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::Columns(handler) | Self::KeyColumns(handler) | Self::KeyNames(handler) => {
                tc_ir::Handler::post(handler, txn, request).await
            }
            Self::Contains(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::Count(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::Insert(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::IsEmpty(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::Limit(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::Order(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::Select(handler) => tc_ir::Handler::post(handler, txn, request).await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }

    async fn delete(&self, txn: &State::Txn, request: tc_ir::Scalar) -> tc_error::TCResult<()> {
        match self {
            Self::Table(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::Columns(handler) | Self::KeyColumns(handler) | Self::KeyNames(handler) => {
                dispatch_delete::<State, _>(handler, txn, request).await
            }
            Self::Contains(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::Count(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::Insert(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::IsEmpty(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::Limit(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::Order(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::Select(handler) => dispatch_delete::<State, _>(handler, txn, request).await,
            Self::State(_) => unreachable!("route state marker is never constructed"),
        }
    }
}

#[cfg(test)]
mod tests;
