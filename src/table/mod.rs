//! Transactional Table implementation split by concern:
//! - `schema`: table schema validation, key/column encoding, and index definitions.
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `stream`: permit-bound row stream with lazy `limit`/`select` transforms.
//! - `view`: lazy view structs (`TableSlice`, `Limited`, `Selection`).
//! - `public`: public API route handlers (ports v1 `public.rs`).
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod codec;
mod file;
pub mod public;
mod schema;
mod stream;
mod view;

pub use codec::DecodedTablePayload;
pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};
pub use stream::Rows;
pub use view::{Limited, Selection, TableSlice};

pub use b_table::{ColumnRange, Range, Row};
use futures::{StreamExt, stream::BoxStream};
use tc_error::TCResult;
use tc_ir::TxnId;
use tc_value::Value;

/// A relational database table, or a view of one.
///
/// Ported from v1 `Table<Txn, FE>` enum.  All view types convert into this
/// via `From`, and this converts into [`crate::Collection`] via `From`.
#[derive(Clone, Debug)]
pub enum Table<Txn = ()> {
    File(PersistentTable<Txn>),
    Slice(TableSlice<Txn>),
    Limited(Limited<Txn>),
    Selection(Selection<Txn>),
}

impl<Txn> From<PersistentTable<Txn>> for Table<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self::File(table)
    }
}

impl<Txn> From<TableSlice<Txn>> for Table<Txn> {
    fn from(slice: TableSlice<Txn>) -> Self {
        Self::Slice(slice)
    }
}

impl<Txn> From<Limited<Txn>> for Table<Txn> {
    fn from(limited: Limited<Txn>) -> Self {
        Self::Limited(limited)
    }
}

impl<Txn> From<Selection<Txn>> for Table<Txn> {
    fn from(selection: Selection<Txn>) -> Self {
        Self::Selection(selection)
    }
}

impl<Txn> Table<Txn> {
    pub fn schema(&self) -> &TableSchema {
        match self {
            Self::File(t) => t.schema(),
            Self::Slice(t) => t.schema(),
            Self::Limited(t) => t.schema(),
            Self::Selection(t) => t.schema(),
        }
    }

    /// Return a lazy row stream for this table or view at `txn_id`.
    pub async fn row_stream(
        &self,
        txn_id: TxnId,
    ) -> TCResult<BoxStream<'static, Result<Row<Value>, std::io::Error>>> {
        let rows = match self {
            Self::File(table) => table
                .rows(txn_id, Range::default(), Vec::new(), false)
                .await?
                .boxed(),
            Self::Slice(table) => table.rows(txn_id).await?.boxed(),
            Self::Limited(table) => table.rows(txn_id).await?.boxed(),
            Self::Selection(table) => table.rows(txn_id).await?.boxed(),
        };

        Ok(rows)
    }
}

#[cfg(test)]
mod tests;
