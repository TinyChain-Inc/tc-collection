use std::cmp::Ordering;
use std::ops::Bound;

use collate::Collate;
use safecast::TryCastFrom;
use tc_ir::{Transact, TxnId};
use tc_value::{Value, ValueCollator};

use crate::btree::{BTree, BTreeColumnSchema};
use crate::table::{PersistentTable, Table};
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct BTreeView<Txn> {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree<Txn>,
    pub bounds: (Bound<Value>, Bound<Value>),
    pub reverse: bool,
}

impl<Txn> Clone for BTreeView<Txn> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            btree: self.btree.clone(),
            bounds: self.bounds.clone(),
            reverse: self.reverse,
        }
    }
}

impl<Txn> BTreeView<Txn> {
    pub fn new(schema: Vec<BTreeColumnSchema>, btree: BTree<Txn>) -> Self {
        Self {
            schema,
            btree,
            bounds: (Bound::Unbounded, Bound::Unbounded),
            reverse: false,
        }
    }

    pub fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        Self {
            schema: self.schema.clone(),
            btree: self.btree.clone(),
            bounds: (
                max_lower_bound(self.bounds.0.clone(), bounds.0),
                min_upper_bound(self.bounds.1.clone(), bounds.1),
            ),
            reverse,
        }
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        self.btree
            .finalized_key_stream_in(self.bounds.clone(), self.reverse)
            .await
    }
}

#[derive(Debug)]
pub enum Collection<Txn> {
    BTree(Box<BTreeView<Txn>>),
    Table(Box<Table<Txn>>),
    Tensor(Tensor),
}

impl<Txn: Clone> Clone for Collection<Txn> {
    fn clone(&self) -> Self {
        match self {
            Self::BTree(btree) => Self::BTree(Box::new((**btree).clone())),
            Self::Table(table) => Self::Table(Box::new((**table).clone())),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
        }
    }
}

impl<Txn> From<PersistentTable<Txn>> for Collection<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self::Table(Box::new(table.into()))
    }
}

fn max_lower_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => Bound::Included(max_value(left, right)),
        (Bound::Included(left), Bound::Excluded(right)) => {
            if compare_values(&left, &right) == Ordering::Greater {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if compare_values(&left, &right) == Ordering::Less {
                Bound::Included(right)
            } else {
                Bound::Excluded(left)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => Bound::Excluded(max_value(left, right)),
    }
}

fn min_upper_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => Bound::Included(min_value(left, right)),
        (Bound::Included(left), Bound::Excluded(right)) => {
            if compare_values(&left, &right) == Ordering::Less {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if compare_values(&left, &right) != Ordering::Greater {
                Bound::Excluded(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => Bound::Excluded(min_value(left, right)),
    }
}

fn compare_values(left: &Value, right: &Value) -> Ordering {
    ValueCollator::default().cmp(left, right)
}

fn max_value(left: Value, right: Value) -> Value {
    if compare_values(&left, &right) == Ordering::Less {
        right
    } else {
        left
    }
}

fn min_value(left: Value, right: Value) -> Value {
    if compare_values(&left, &right) == Ordering::Greater {
        right
    } else {
        left
    }
}

impl<Txn> TryCastFrom<Collection<Txn>> for Tensor {
    fn can_cast_from(collection: &Collection<Txn>) -> bool {
        matches!(collection, Collection::Tensor(_))
    }

    fn opt_cast_from(collection: Collection<Txn>) -> Option<Self> {
        match collection {
            Collection::Tensor(tensor) => Some(tensor),
            Collection::BTree(_) | Collection::Table(_) => None,
        }
    }
}

impl<Txn> From<Table<Txn>> for Collection<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self::Table(Box::new(table))
    }
}

impl<Txn> Collection<Txn> {
    pub fn as_btree(&self) -> Option<&BTree<Txn>> {
        match self {
            Self::BTree(btree) => Some(&btree.btree),
            _ => None,
        }
    }

    pub fn into_btree(self) -> BTree<Txn> {
        match self {
            Self::BTree(btree) => btree.btree,
            _ => panic!("Collection is not a BTree"),
        }
    }

    pub fn as_table(&self) -> Option<&Table<Txn>> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }

    pub fn into_table(self) -> Table<Txn> {
        match self {
            Self::Table(table) => *table,
            _ => panic!("Collection is not a Table"),
        }
    }
}

impl<Txn> Transact for Collection<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(btree) => Transact::commit(&btree.btree, txn_id).await?,
            Self::Table(table) => {
                if let Table::File(t) = table.as_ref() {
                    Transact::commit(t, txn_id).await?;
                }
            }
            Self::Tensor(_) => {}
        }

        Ok(())
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::rollback(&btree.btree, &txn_id).await?,
                Self::Table(table) => {
                    if let Table::File(t) = table.as_ref() {
                        Transact::rollback(t, &txn_id).await?;
                    }
                }
                Self::Tensor(_) => {}
            }

            Ok(())
        }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::finalize(&btree.btree, &txn_id).await?,
                Self::Table(table) => {
                    if let Table::File(t) = table.as_ref() {
                        Transact::finalize(t, &txn_id).await?;
                    }
                }
                Self::Tensor(_) => {}
            }

            Ok(())
        }
    }
}
