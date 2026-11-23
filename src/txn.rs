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

/// Construct an independent transactional value from a transaction-visible source.
///
/// The destination must be empty, unpublished storage exclusively delegated by
/// the caller. Copying preserves source visibility and observable ordering and
/// synchronizes the constructed value; it does not decide the source transaction.
/// Failed or cancelled copying may leave partial destination contents, whose
/// cleanup belongs to the caller. Sources may be borrowed for scoped copying.
pub trait CopyFrom<Source>: Sized {
    type Txn: StorageContext;

    fn copy_from(
        txn: &Self::Txn,
        dir: DirLock<<Self::Txn as StorageContext>::File>,
        source: Source,
    ) -> impl std::future::Future<Output = TCResult<Self>> + Send;
}

/// Stage a replacement in an existing mutable collection's transaction.
///
/// Source and destination must be compatible owners, not read-only views.
/// The source is read at the supplied transaction; successful staging replaces
/// pending destination state without deciding or synchronizing the transaction.
/// Self-restoration is a no-op only for a writable transaction. Conflicting
/// reservations reject without waiting. Failed or cancelled staging preserves
/// the previous pending state; the caller owns unpublished workspace cleanup.
pub trait Restore {
    type Txn: StorageContext;

    fn restore(
        &self,
        txn: &Self::Txn,
        source: &Self,
    ) -> impl std::future::Future<Output = TCResult<()>> + Send;
}
