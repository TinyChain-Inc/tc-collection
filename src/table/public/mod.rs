//! Public API route handlers for [`Table`] owners and views.
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
    fn route<'a>(
        &'a self,
        path: &[pathlink::PathSegment],
    ) -> Option<Box<dyn tc_ir::Handler<'a, State> + 'a>> {
        let handler: Box<dyn tc_ir::Handler<'a, State> + 'a> = if path.is_empty() {
            Box::new(TableHandler::from(self))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "columns" => Box::new(handler::SchemaHandler::new(self, handler::column_schema)),
                "contains" => Box::new(ContainsHandler::from(self)),
                "count" => Box::new(CountHandler::from(self)),
                "insert" => Box::new(InsertHandler::from(self)),
                "is_empty" => Box::new(IsEmptyHandler::from(self)),
                "key_columns" => Box::new(handler::SchemaHandler::new(self, handler::key_columns)),
                "key_names" => Box::new(handler::SchemaHandler::new(self, handler::key_names)),
                "limit" => Box::new(LimitHandler::from(self)),
                "order" => Box::new(OrderHandler::from(self)),
                "select" => Box::new(SelectHandler::from(self)),
                _ => return None,
            }
        } else {
            return None;
        };

        Some(handler)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/table/public/tests.rs"]
mod tests;
