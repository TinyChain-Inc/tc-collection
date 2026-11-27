use std::path::Path;
use std::sync::Arc;

use freqfs::Cache;
use tc_ir::{NetworkTime, TxnId};

use crate::tensor::{Tensor, TensorSchema};
use crate::{PersistentFile, StorageContext};

pub(crate) mod depth;
pub(crate) mod storage;
pub(crate) mod tensor;

pub use storage::{TempRoot, TestTxn};

// General storage tests use a generous resident target so cache pressure is not
// an incidental part of their scenario. Pressure tests pass smaller capacities.
const DEFAULT_CACHE_BYTES: usize = 16 * 1024 * 1024;

pub fn tx(nonce: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), nonce)
}

pub fn init_root(name: &str) -> Arc<TempRoot> {
    let root = TempRoot::new(name);
    std::fs::create_dir(root.join("persistent")).expect("create persistent root");
    std::fs::create_dir(root.join("txn")).expect("create txn root");
    root
}

pub fn load_roots(root: &Path) -> (freqfs::DirLock<PersistentFile>, TestTxn) {
    let cache = Cache::<PersistentFile>::new(
        DEFAULT_CACHE_BYTES,
        None,
        0,
        std::time::Duration::from_secs(3),
    );

    let persistent = Arc::clone(&cache)
        .load(root.join("persistent"))
        .expect("load persistent root");

    let txn = Arc::clone(&cache)
        .load(root.join("txn"))
        .expect("load txn root");

    (persistent, TestTxn::new(tx(0), txn))
}

#[tokio::test]
async fn transaction_fixture_isolates_equal_ids_and_shared_subcontexts() {
    use tc_ir::Transaction;

    use crate::StorageContext;

    let id = tensor::id(10);
    let (left_root, left) =
        TestTxn::owned(id, DEFAULT_CACHE_BYTES, storage::DEFAULT_MATERIALIZED_BYTES);

    let (right_root, right) =
        TestTxn::owned(id, DEFAULT_CACHE_BYTES, storage::DEFAULT_MATERIALIZED_BYTES);

    assert_ne!(left_root.as_ref().as_ref(), right_root.as_ref().as_ref());
    assert_eq!(left.id(), right.id());
    let first = left.subcontext_unique().context().await.unwrap();
    let second = left.clone().subcontext_unique().context().await.unwrap();
    first.write().await.create_dir("first".into()).unwrap();
    assert!(second.read().await.get_dir("first").is_none());
    assert!(
        right
            .context()
            .await
            .unwrap()
            .read()
            .await
            .get_dir("first")
            .is_none()
    );
}

#[test]
fn fixture_cleanup_on_drop_and_unwind() {
    for unwind in [false, true] {
        let root = TempRoot::new("cleanup");
        let path = root.to_path_buf();
        let result = std::panic::catch_unwind(move || {
            let _root = root;
            if unwind {
                panic!("injected fixture unwind");
            }
        });

        assert_eq!(result.is_err(), unwind);
        assert!(!path.exists());
    }
}

impl TestTxn {
    pub fn fresh(nonce: u16) -> Self {
        Self::owned(
            tc_ir::TxnId::from_parts(tc_ir::NetworkTime::from_nanos(1), nonce),
            DEFAULT_CACHE_BYTES,
            storage::DEFAULT_MATERIALIZED_BYTES,
        )
        .1
    }
}
