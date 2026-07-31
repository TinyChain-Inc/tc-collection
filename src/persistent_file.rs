//! Persistent file adapter for on-disk collection node loading/saving.
//!
//! This type wraps `b_tree::Node` for use with `freqfs` and is shared by
//! BTree, Table, and other collection types — it is not specific to any
//! one collection variant.

use std::fs::Metadata;

use freqfs::{FileLoad, FileSave};
use safecast::AsType;
use tc_value::Value;

type NodeFile = b_tree::Node<Vec<Vec<Value>>>;

#[derive(Clone)]
pub enum PersistentFile {
    Node(NodeFile),
}

impl From<NodeFile> for PersistentFile {
    fn from(node: NodeFile) -> Self {
        Self::Node(node)
    }
}

impl AsType<NodeFile> for PersistentFile {
    fn as_type(&self) -> Option<&NodeFile> {
        match self {
            Self::Node(node) => Some(node),
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut NodeFile> {
        match self {
            Self::Node(node) => Some(node),
        }
    }

    fn into_type(self) -> Option<NodeFile> {
        match self {
            Self::Node(node) => Some(node),
        }
    }
}

impl FileLoad for PersistentFile {
    async fn load(
        path: &std::path::Path,
        file: tokio::fs::File,
        metadata: Metadata,
    ) -> std::io::Result<Self> {
        NodeFile::load(path, file, metadata).await.map(Self::Node)
    }
}

impl FileSave for PersistentFile {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        match self {
            Self::Node(node) => node.save(file).await,
        }
    }
}
