use crate::btree::BTree;
use tc_ir::{Transact, TxnId};

#[derive(Debug, Clone)]
pub enum Collection {
    BTree(BTree),
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Self::BTree(btree)
    }
}

impl Collection {
    pub fn as_btree(&self) -> Option<&BTree> {
        match self {
            Self::BTree(btree) => Some(btree),
        }
    }

    pub fn into_btree(self) -> BTree {
        match self {
            Self::BTree(btree) => btree,
        }
    }
}

impl Transact for Collection {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).expect("Collection commit failed"),
        }
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match self {
                Self::BTree(btree) => btree.rollback(txn_id).expect("Collection rollback failed"),
            }
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match self {
                Self::BTree(btree) => btree
                    .finalize(txn_id)
                    .await
                    .expect("Collection finalize failed"),
            }
        }
    }
}
