//! Public API route handlers for a transactional [`PersistentTable`].
//!
//! [`Table`] resolves each path directly to its table-owned native handler.
//! Routing and execution remain serialization-free.
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
impl<State, Txn> tc_ir::Route<State> for Table<Txn>
where
    State: crate::CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    fn route(&self, path: &[pathlink::PathSegment]) -> Option<Box<dyn tc_ir::Handler<State> + '_>> {
        route_owned(self.clone(), path)
    }
}

fn route_owned<'a, State, Txn>(
    table: Table<Txn>,
    path: &[pathlink::PathSegment],
) -> Option<Box<dyn tc_ir::Handler<State> + 'a>>
where
    State: crate::CollectionState<Txn = Txn>,
    Txn: crate::StorageContext + 'a,
{
    let handler: Box<dyn tc_ir::Handler<State> + 'a> = if path.is_empty() {
        Box::new(TableHandler::from(table.clone()))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "columns" => Box::new(handler::SchemaHandler::new(
                table.clone(),
                handler::column_schema,
            )),
            "contains" => Box::new(ContainsHandler::from(table.clone())),
            "count" => Box::new(CountHandler::from(table.clone())),
            "insert" => Box::new(InsertHandler::from(table.clone())),
            "is_empty" => Box::new(IsEmptyHandler::from(table.clone())),
            "key_columns" => Box::new(handler::SchemaHandler::new(
                table.clone(),
                handler::key_columns,
            )),
            "key_names" => Box::new(handler::SchemaHandler::new(
                table.clone(),
                handler::key_names,
            )),
            "limit" => Box::new(LimitHandler::from(table.clone())),
            "order" => Box::new(OrderHandler::from(table.clone())),
            "select" => Box::new(SelectHandler::from(table.clone())),
            _ => return None,
        }
    } else {
        return None;
    };

    Some(handler)
}

/// Resolve a Table-compatible value to its owned native handler.
pub fn route<'a, State: crate::CollectionState>(
    table: &'a (impl Clone + Into<Table<State::Txn>>),
    path: &[pathlink::PathSegment],
) -> Option<Box<dyn tc_ir::Handler<State> + 'a>> {
    route_owned(table.clone().into(), path)
}

#[cfg(test)]
mod tests;
