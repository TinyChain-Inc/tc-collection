//! Transactional BTree implementation split by concern:
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `codec`: destream codec bindings for BTree.
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod codec;
mod file;

pub use codec::{
    BTreeColumnSchema, BTreeDecodeContext, DecodedBTreePayload,
};
pub use file::{BTree, BTreeSchema, BTreeSlice, StorageConfig};
pub use crate::PersistentFile;

#[cfg(test)]
mod tests;
