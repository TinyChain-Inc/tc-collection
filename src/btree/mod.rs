//! Transactional BTree implementation split by concern:
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `codec`: destream codec bindings for BTree.
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod codec;
mod file;
mod route;
mod stream;

pub use codec::{BTreeColumnSchema, DecodedBTreePayload};
pub use file::{BTree, BTreeSchema, BTreeSlice, StorageConfig};
pub use stream::Keys;

#[cfg(test)]
mod tests;
