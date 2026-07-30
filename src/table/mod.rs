//! Transactional Table implementation split by concern:
//! - `schema`: table schema validation, key/column encoding, and index definitions.
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod file;
mod schema;

pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};

pub use b_table::{ColumnRange, Range, Row};

#[cfg(test)]
mod tests;
