//! Transactional Table implementation split by concern:
//! - `schema`: table schema validation, key/column encoding, and index definitions.
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `stream`: permit-bound row stream with lazy `limit`/`select` transforms.
//! - `view`: lazy view structs (`TableSlice`, `Limited`, `Selection`).
//! - `public`: public API route handlers (ports v1 `public.rs`).
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod file;
pub mod public;
mod schema;
mod stream;
mod view;

pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};
pub use stream::Rows;
pub use view::{Limited, Selection, TableSlice};

pub use b_table::{ColumnRange, Range, Row};

/// A relational database table, or a view of one.
///
/// Ported from v1 `Table<Txn, FE>` enum.  All view types convert into this
/// via `From`, and this converts into [`crate::Collection`] via `From`.
#[derive(Clone, Debug)]
pub enum Table {
    File(PersistentTable),
    Slice(TableSlice),
    Limited(Limited),
    Selection(Selection),
}

impl From<PersistentTable> for Table {
    fn from(table: PersistentTable) -> Self {
        Self::File(table)
    }
}

impl From<TableSlice> for Table {
    fn from(slice: TableSlice) -> Self {
        Self::Slice(slice)
    }
}

impl From<Limited> for Table {
    fn from(limited: Limited) -> Self {
        Self::Limited(limited)
    }
}

impl From<Selection> for Table {
    fn from(selection: Selection) -> Self {
        Self::Selection(selection)
    }
}

impl Table {
    pub fn schema(&self) -> &TableSchema {
        match self {
            Self::File(t) => t.schema(),
            Self::Slice(t) => t.schema(),
            Self::Limited(t) => t.schema(),
            Self::Selection(t) => t.schema(),
        }
    }
}

#[cfg(test)]
mod tests;
