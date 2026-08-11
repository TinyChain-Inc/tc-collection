use freqfs::DirLock;
use tc_error::{TCError, TCResult};

use crate::{PersistentFile, StorageContext};

/// The delegated directory identity of a persistent collection.
///
/// A literal owns an already-unique transaction child. A named collection
/// stores canonical URI components and derives its child from the invocation.
#[derive(Clone)]
pub enum CollectionDir {
    Transaction,
    Literal(DirLock<PersistentFile>),
    Named(Vec<String>),
}

impl CollectionDir {
    pub fn literal(dir: DirLock<PersistentFile>) -> Self {
        Self::Literal(dir)
    }

    pub fn named(uri: &pathlink::Link, class: &str) -> TCResult<Self> {
        let path = uri
            .path()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let prefix = ["state", "collection", class];
        if path.len() <= prefix.len()
            || path
                .iter()
                .zip(prefix)
                .any(|(segment, prefix)| segment != prefix)
        {
            return Err(TCError::bad_request(format!(
                "expected a canonical /state/collection/{class}/... URI, got {uri}"
            )));
        }

        Ok(Self::Named(path))
    }

    pub async fn resolve<Txn: StorageContext>(
        &self,
        txn: &Txn,
    ) -> TCResult<DirLock<PersistentFile>> {
        match self {
            Self::Transaction => txn.context().await,
            Self::Literal(dir) => Ok(dir.clone()),
            Self::Named(path) => {
                let mut txn = txn.clone();
                for segment in path {
                    txn = txn.subcontext(segment.clone());
                }
                txn.context().await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn persistent_mutations_use_collection_directory_identity() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for path in ["src/btree/file.rs", "src/table/file.rs"] {
            let source = std::fs::read_to_string(root.join(path)).expect("collection source");
            assert!(
                !source.contains("txn.context().await"),
                "{path} must resolve its delta directory through CollectionDir"
            );
        }

        for path in ["src/btree/codec.rs", "src/table/codec.rs"] {
            let source = std::fs::read_to_string(root.join(path)).expect("codec source");
            assert!(
                source.contains("subcontext_unique"),
                "{path} must allocate literal collection storage in a unique transaction child"
            );
        }
    }
}
