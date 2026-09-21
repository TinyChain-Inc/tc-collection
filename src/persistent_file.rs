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

/// Collection-only file type used by standalone callers and tests.
///
/// This adapter explicitly uses TBON, preserving its existing stored representation.
///
/// `CollectionNode` and `AsType` are both defined by dependency crates, so
/// Rust's orphan rules prohibit implementing `AsType<CollectionNode>` directly
/// for `CollectionNode`. This local newtype exists only to provide that required
/// `freqfs` projection; it is not an extensible file-variant registry.
#[derive(Clone)]
pub struct PersistentFile(CollectionNode);

impl From<CollectionNode> for PersistentFile {
    fn from(node: CollectionNode) -> Self {
        Self(node)
    }
}

impl AsType<CollectionNode> for PersistentFile {
    fn as_type(&self) -> Option<&CollectionNode> {
        Some(&self.0)
    }

    fn as_type_mut(&mut self) -> Option<&mut CollectionNode> {
        Some(&mut self.0)
    }

    fn into_type(self) -> Option<CollectionNode> {
        Some(self.0)
    }
}

impl freqfs::FileLoad for PersistentFile {
    async fn load(
        _: &std::path::Path,
        file: tokio::fs::File,
        _: std::fs::Metadata,
    ) -> std::io::Result<Self> {
        tbon::de::read_from((), file)
            .await
            .map(Self)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }
}
impl freqfs::FileSave for PersistentFile {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        use futures::TryStreamExt;
        use tokio::io::AsyncWriteExt;
        let mut stream = tbon::en::encode(&self.0).map_err(std::io::Error::other)?;
        let mut size = 0;
        while let Some(chunk) = stream.try_next().await.map_err(std::io::Error::other)? {
            file.write_all(&chunk).await?;
            size += chunk.len() as u64;
        }
        Ok(size)
    }
}

impl GetSize for PersistentFile {
    fn get_size(&self) -> usize {
        self.0.get_size()
    }
}
