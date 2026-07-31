use tc_ir::{Transact, TxnId};

use crate::btree::BTree;
use crate::table::{PersistentTable, Table};

#[derive(Debug, Clone)]
pub enum Collection {
    BTree(BTree),
    Table(Table),
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Self::BTree(btree)
    }
}

impl From<PersistentTable> for Collection {
    fn from(table: PersistentTable) -> Self {
        Self::Table(table.into())
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Self {
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

    pub fn as_table(&self) -> Option<&Table> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }

    pub fn into_table(self) -> Table {
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
            Self::Table(table) => match table {
                Table::File(t) => Transact::commit(t, txn_id).await,
                _ => {}
            },
        }
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::rollback(btree, &txn_id).await,
                Self::Table(Table::File(t)) => Transact::rollback(t, &txn_id).await,
                _ => {}
            }
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = ()> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::finalize(btree, &txn_id).await,
                Self::Table(Table::File(t)) => Transact::finalize(t, &txn_id).await,
                _ => {}
            }
        }
    }
}
