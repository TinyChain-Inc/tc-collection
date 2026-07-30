use crate::btree::BTree;
use crate::table::PersistentTable;
use tc_ir::{Transact, TxnId};

#[derive(Debug, Clone)]
pub enum Collection {
    BTree(BTree),
    Table(PersistentTable),
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Self::BTree(btree)
    }
}

impl From<PersistentTable> for Collection {
    fn from(table: PersistentTable) -> Self {
        Self::Table(table)
    }
}

impl Collection {
    pub fn as_btree(&self) -> Option<&BTree> {
        match self {
            Self::BTree(btree) => Some(btree),
            _ => None,
        }
    }

    pub fn into_btree(self) -> BTree {
        match self {
            Self::BTree(btree) => btree,
            _ => panic!("Collection is not a BTree"),
        }
    }

    pub fn as_table(&self) -> Option<&PersistentTable> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }

    pub fn into_table(self) -> PersistentTable {
        match self {
            Self::Table(table) => table,
            _ => panic!("Collection is not a Table"),
        }
    }
}

impl Transact for Collection {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => Transact::commit(btree, txn_id).await,
            Self::Table(table) => Transact::commit(table, txn_id).await,
        }
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::rollback(btree, &txn_id).await,
                Self::Table(table) => Transact::rollback(table, &txn_id).await,
            }
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::finalize(btree, &txn_id).await,
                Self::Table(table) => Transact::finalize(table, &txn_id).await,
            }
        }
    }
}
