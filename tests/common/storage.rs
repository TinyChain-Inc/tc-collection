//! Private caller-owned storage shared by unit tests and the opt-in benchmark.
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use freqfs::DirLock;

use super::{PersistentFile, StorageContext};
use tc_error::TCResult;
use tc_ir::{NetworkTime, Transaction, TxnId};

#[derive(Debug)]
pub struct TempRoot(PathBuf);

impl TempRoot {
    pub fn new(name: &str) -> Arc<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        for attempt in 0u64.. {
            let path = std::env::temp_dir().join(format!(
                "tc-collection-{name}-{}-{stamp}-{attempt}",
                std::process::id()
            ));

            match std::fs::create_dir(&path) {
                Ok(()) => return Arc::new(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create fixture directory: {error}"),
            }
        }

        unreachable!("directory-name space exhausted")
    }
}

impl Deref for TempRoot {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempRoot {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Debug)]
pub struct TestTxn {
    id: TxnId,
    root: DirLock<PersistentFile>,
    path: Vec<String>,
    next: Arc<AtomicU64>,
    materialized_bytes: usize,
    // Last field: release the directory handle before the cleanup owner.
    temp: Option<Arc<TempRoot>>,
}

impl TestTxn {
    pub fn new(id: TxnId, root: DirLock<PersistentFile>) -> Self {
        Self {
            id,
            root,
            path: Vec::new(),
            next: Arc::new(AtomicU64::new(0)),
            materialized_bytes: 256 * 1024 * 1024,
            temp: None,
        }
    }

    pub fn owned(id: TxnId, capacity: usize, materialized_bytes: usize) -> (Arc<TempRoot>, Self) {
        let temp = TempRoot::new("txn");
        let cache = freqfs::Cache::<PersistentFile>::new(capacity, None, 0, Duration::from_secs(3));
        let root = cache
            .load(temp.to_path_buf())
            .expect("load transaction root");

        let mut txn = Self::new(id, root);
        txn.materialized_bytes = materialized_bytes;
        txn.temp = Some(temp.clone());
        (temp, txn)
    }

    pub fn with_id(&self, id: TxnId) -> Self {
        Self { id, ..self.clone() }
    }

    pub fn at(&self, nonce: u16) -> Self {
        self.with_id(TxnId::from_parts(NetworkTime::from_nanos(1), nonce))
    }
}

impl Transaction for TestTxn {
    fn id(&self) -> TxnId {
        self.id
    }
}

#[cfg(test)]
tokio::task_local! {
    pub(crate) static STORAGE_ERRORS: std::cell::RefCell<Vec<bool>>;
}

impl StorageContext for TestTxn {
    type File = PersistentFile;

    fn storage_error(&self, error: tc_error::TCError) -> tc_error::TCError {
        #[cfg(test)]
        let _ = STORAGE_ERRORS.try_with(|errors| {
            errors.borrow_mut().push(error.requires_recovery());
        });

        error
    }

    async fn context(&self) -> TCResult<DirLock<PersistentFile>> {
        let mut dir = self.root.clone();
        for name in std::iter::once(self.id.to_string()).chain(self.path.clone()) {
            let next = dir.write().await.get_or_create_dir(name)?;
            dir = next;
        }

        Ok(dir)
    }

    fn subcontext(&self, name: impl Into<String>) -> Self {
        let mut child = self.clone();
        child.path.push(name.into());
        child
    }

    fn subcontext_unique(&self) -> Self {
        self.subcontext(format!(
            "value-{}",
            self.next.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn materialized_tensor_bytes(&self) -> usize {
        self.materialized_bytes
    }
}
