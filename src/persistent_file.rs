//! The collection-owned file contract used by persistent BTree and Table nodes.

use freqfs::{FileLoad, FileSave};
use get_size::GetSize;
use safecast::AsType;
use tc_value::Value;

/// A native BTree node shared by BTree and Table storage.
pub type CollectionNode = b_tree::Node<Vec<Vec<Value>>>;

/// A caller-owned file composition containing native nodes.
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
/// This adapter selects TBON for native nodes.
/// Domain codecs remain format-neutral.
#[derive(Clone)]
pub enum PersistentFile {
    Node(CollectionNode),
}

safecast::as_type!(PersistentFile, Node, CollectionNode);

impl<'en> destream::en::ToStream<'en> for PersistentFile {
    fn to_stream<E: destream::en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Node(node) => node.to_stream(encoder),
        }
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
            .map(Self::Node)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))
    }
}

impl freqfs::FileSave for PersistentFile {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        use futures::TryStreamExt;
        use tokio::io::AsyncWriteExt;
        let mut stream = tbon::en::encode(self).map_err(std::io::Error::other)?;
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
        match self {
            Self::Node(node) => node.get_size(),
        }
    }
}
