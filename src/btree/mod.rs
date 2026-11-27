//! Transactional BTree implementation split by concern:
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `codec`: destream codec bindings for BTree.
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod codec;
mod file;
mod route;
mod schema;
mod stream;

use std::ops::Bound;

use file::{BTreeBase, BTreeSlice};
use tc_error::TCResult;
use tc_ir::{Sha256Hash, Transact, TxnId};
use tc_value::Value;
pub use schema::{BTreeColumnSchema, BTreeSchema, StorageConfig};
pub use stream::Keys;

#[cfg(test)]
#[path = "../../tests/unit/btree/tests.rs"]
mod tests;

#[derive(Clone, Debug)]
pub enum BTree<Txn: crate::StorageContext> {
    File(BTreeBase<Txn>),
    Slice(Box<BTreeSlice<Txn>>),
}

impl<Txn: crate::StorageContext> BTree<Txn> {
    pub async fn insert_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        self.persistent()?.insert_row(txn, key).await
    }

    pub async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        self.persistent()?.delete_row(txn, key).await
    }

    pub async fn create(dir: freqfs::DirLock<Txn::File>, schema: BTreeSchema) -> TCResult<Self> {
        Ok(Self::File(BTreeBase::try_with_schema(dir, schema).await?))
    }

    pub async fn load(dir: freqfs::DirLock<Txn::File>, schema: BTreeSchema) -> TCResult<Self> {
        Ok(Self::File(Box::pin(BTreeBase::load(dir, schema)).await?))
    }

    /// Complete accepted recovery work on an exclusively owned, unpublished value.
    /// A durable decision skips the callback; otherwise it runs once before commit.
    pub(crate) async fn recover<Replay, Replayed>(
        &self,
        id: TxnId,
        replay: Replay,
    ) -> tc_error::TCResult<()>
    where
        Replay: FnOnce() -> Replayed + Send,
        Replayed: std::future::Future<Output = tc_error::TCResult<()>> + Send,
    {
        self.persistent()?.recover(id, replay).await
    }

    pub fn schema(&self) -> &BTreeSchema {
        match self {
            Self::File(base) => base.schema(),
            Self::Slice(view) => view.schema(),
        }
    }

    pub(crate) fn is_persistent(&self) -> bool {
        matches!(self, Self::File(_))
    }

    /// Return the mutable owner only when this value owns persistent storage.
    fn persistent(&self) -> tc_error::TCResult<&BTreeBase<Txn>> {
        match self {
            Self::File(btree) => Ok(btree),
            Self::Slice(_) => Err(tc_error::TCError::bad_request(
                "expected a persistent BTree owner",
            )),
        }
    }

    /// Select a read-only view, including when the selected range is unbounded.
    pub fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        let slice = match self {
            Self::File(btree) => btree.slice(bounds, reverse),
            Self::Slice(slice) => slice.slice(bounds, reverse),
        };

        Self::Slice(Box::new(slice))
    }

    pub async fn keys(&self, txn_id: TxnId) -> tc_error::TCResult<crate::btree::Keys> {
        match self {
            Self::File(btree) => {
                btree
                    .keys(txn_id, (Bound::Unbounded, Bound::Unbounded), false)
                    .await
            }
            Self::Slice(slice) => slice.keys(txn_id).await,
        }
    }

    pub(crate) async fn hash_contents(&self, id: TxnId) -> tc_error::TCResult<Sha256Hash> {
        async_hash::hash_try_stream::<async_hash::Sha256, _, _, _>(Box::pin(self.keys(id)).await?)
            .await
    }

    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> tc_error::TCResult<bool> {
        match self {
            Self::File(btree) => btree.contains_row(txn_id, key).await,
            Self::Slice(slice) => slice.contains_row(txn_id, key).await,
        }
    }

    pub async fn count(&self, txn_id: TxnId) -> tc_error::TCResult<u64> {
        match self {
            Self::File(btree) => btree.count(txn_id).await,
            Self::Slice(slice) => slice.count(txn_id).await,
        }
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> tc_error::TCResult<bool> {
        match self {
            Self::File(btree) => btree.is_empty(txn_id).await,
            Self::Slice(slice) => slice.is_empty(txn_id).await,
        }
    }
}

impl<Txn: crate::StorageContext> crate::Restore for BTree<Txn> {
    type Txn = Txn;

    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let target = self.persistent()?;
        let original = source.persistent()?;
        target.restore(txn, original).await
    }
}

impl<Txn: crate::StorageContext> Transact for BTree<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        Transact::commit(self.persistent()?, txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) -> TCResult<()> {
        Transact::rollback(self.persistent()?, txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) -> TCResult<()> {
        Transact::finalize(self.persistent()?, txn_id).await
    }
}
