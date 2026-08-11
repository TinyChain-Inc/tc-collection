//! Host-delegated transaction storage capability.

use freqfs::DirLock;
use tc_error::TCResult;
use tc_ir::Transaction as IrTransaction;

use crate::PersistentFile;

/// Host-local storage delegated by a protocol transaction.
///
/// The protocol identity is inherited unchanged by child contexts. Only the
/// host-issued implementation knows how to resolve the current directory.
pub trait StorageContext: IrTransaction + Clone + Send + Sync {
    fn context(
        &self,
    ) -> impl std::future::Future<Output = TCResult<DirLock<PersistentFile>>> + Send;

    fn subcontext(&self, name: impl Into<String>) -> Self;
    fn subcontext_unique(&self) -> Self;
    fn materialized_tensor_bytes(&self) -> usize;
}
