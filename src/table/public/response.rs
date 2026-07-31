//! Response types from table route handlers.
//!
//! Scalar responses (count, bool, schema, row) use the `Value` variant.
//! Table/view responses use the appropriate view variant.  The host/server
//! layer is responsible for serializing the response (streaming for views).

use tc_value::Value;

use super::super::file::PersistentTable;
use super::super::view::{Limited, Selection, TableSlice};

/// Response from a table route handler.
#[derive(Clone, Debug)]
pub enum TableResponse {
    /// A scalar value (count, bool, schema, row, etc.).
    Value(Value),
    /// A full table reference.
    Table(PersistentTable),
    /// A range/order/reverse slice view.
    Slice(TableSlice),
    /// A row-cap limited view.
    Limited(Limited),
    /// A column-projection view.
    Selection(Selection),
}

impl From<Value> for TableResponse {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<PersistentTable> for TableResponse {
    fn from(table: PersistentTable) -> Self {
        Self::Table(table)
    }
}

impl From<TableSlice> for TableResponse {
    fn from(slice: TableSlice) -> Self {
        Self::Slice(slice)
    }
}
