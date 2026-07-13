//! Transactional BTree implementation split by concern:
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `stream`: persistent on-disk file codec bindings for freqfs/b-tree nodes.
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod codec;
mod file;
mod stream;

pub use codec::{
    BTreeColumnSchema, BTreeDecodeContext, DecodedBTreePayload,
};
pub use file::{BTree, BTreeSlice, BTreeStorageConfig};
pub use stream::PersistentFile;

#[cfg(test)]
mod tests;
