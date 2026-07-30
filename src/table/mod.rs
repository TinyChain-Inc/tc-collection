//! Transactional Table implementation split by concern:
//! - `schema`: table schema validation, key/column encoding, and index definitions.
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `stream`: permit-bound row stream with lazy `limit`/`select` transforms.
//! - `view`: lazy view structs (`TableSlice`, `Limited`, `Selection`).
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod file;
mod schema;
mod stream;
mod view;

pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};
pub use stream::Rows;
pub use view::{Limited, Selection, TableSlice};

pub use b_table::{ColumnRange, Range, Row};

#[cfg(test)]
mod tests;
