//! Host-delegated transaction storage capability.

use freqfs::DirLock;

use tc_error::TCResult;
use tc_ir::Transaction as IrTransaction;

/// Host-local storage delegated by a protocol transaction.
///
/// The protocol identity is inherited unchanged by child contexts. Only the
/// caller-provided implementation knows how to resolve the current directory.
pub trait StorageContext: IrTransaction + Clone + Send + Sync + 'static {
    type File: crate::CollectionFile;

    /// Apply caller recovery policy before storage errors lose their native provenance.
    /// Standalone callers retain the error and own recovery; host contexts may terminate.
    fn storage_error(&self, error: tc_error::TCError) -> tc_error::TCError {
        error
    }

    fn context(&self) -> impl std::future::Future<Output = TCResult<DirLock<Self::File>>> + Send;

    fn subcontext(&self, name: impl Into<String>) -> Self;
    fn subcontext_unique(&self) -> Self;
    /// Maximum supplied element bytes for native Tensor literals, checked after
    /// allocation. This does not size storage blocks or bound streamed tensors.
    fn materialized_tensor_bytes(&self) -> usize;
}
