//! The collection-owned file contract used by persistent BTree and Table nodes.

use freqfs::{FileLoad, FileSave};
use get_size::GetSize;
use safecast::AsType;
use tc_value::Value;

/// The only file value owned by `tc-collection`.
pub type CollectionNode = b_tree::Node<Vec<Vec<Value>>>;

/// A caller-owned file composition capable of containing collection nodes.
pub trait CollectionFile:
    Clone
    + FileLoad
    + FileSave
    + GetSize
    + AsType<CollectionNode>
    + From<CollectionNode>
    + Send
    + Sync
    + 'static
{
}

impl<T> CollectionFile for T where
    T: Clone
        + FileLoad
        + FileSave
        + GetSize
        + AsType<CollectionNode>
        + From<CollectionNode>
        + Send
        + Sync
        + 'static
{
}

/// Collection-only file composition used by standalone callers and tests.
#[derive(Clone)]
pub enum PersistentFile {
    Node(CollectionNode),
}

impl From<CollectionNode> for PersistentFile {
    fn from(node: CollectionNode) -> Self {
        Self::Node(node)
    }
}

impl AsType<CollectionNode> for PersistentFile {
    fn as_type(&self) -> Option<&CollectionNode> {
        let Self::Node(node) = self;
        Some(node)
    }

    fn as_type_mut(&mut self) -> Option<&mut CollectionNode> {
        let Self::Node(node) = self;
        Some(node)
    }

    fn into_type(self) -> Option<CollectionNode> {
        let Self::Node(node) = self;
        Some(node)
    }
}

impl FileLoad for PersistentFile {
    async fn load(
        path: &std::path::Path,
        file: tokio::fs::File,
        metadata: std::fs::Metadata,
    ) -> std::io::Result<Self> {
        CollectionNode::load(path, file, metadata)
            .await
            .map(Self::Node)
    }
}

impl FileSave for PersistentFile {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        let Self::Node(node) = self;
        node.save(file).await
    }
}

impl GetSize for PersistentFile {
    fn get_size(&self) -> usize {
        let Self::Node(node) = self;
        node.get_size()
    }
}
