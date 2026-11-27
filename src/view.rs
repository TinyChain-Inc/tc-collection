use futures::StreamExt;
use futures::stream::BoxStream;
use tc_error::{TCError, TCResult};
use tc_ir::IntoView;
use tc_value::Value;

use crate::Collection;
use crate::btree::{BTree, BTreeColumnSchema, Keys};
use crate::table::{Table, TableSchema};
use crate::tensor::TensorView as NativeTensorView;

mod encode;

/// A transaction-consistent BTree representation prepared for terminal projection.
pub struct BTreeView {
    schema: Vec<BTreeColumnSchema>,
    keys: Keys,
}

/// A transaction-consistent Table representation prepared for terminal projection.
pub struct TableView {
    schema: TableSchema,
    rows: BoxStream<'static, TCResult<Value>>,
}

/// The terminal view of a collection. Route execution returns `Collection`, not this type.
pub enum CollectionView {
    BTree(BTreeView),
    Table(TableView),
    Tensor(NativeTensorView),
}

impl<Txn> IntoView for Collection<Txn>
where
    Txn: crate::StorageContext + 'static,
{
    type Txn = Txn;
    type View = CollectionView;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        match self {
            Collection::BTree(view) => Ok(CollectionView::BTree((*view).into_view(txn).await?)),
            Collection::Table(table) => Ok(CollectionView::Table((*table).into_view(txn).await?)),
            Collection::Tensor(tensor) => Ok(CollectionView::Tensor(tensor.into_view(txn).await?)),
        }
    }
}

impl<Txn: crate::StorageContext> IntoView for BTree<Txn> {
    type Txn = Txn;

    type View = BTreeView;

    async fn into_view(self, txn: Txn) -> tc_error::TCResult<Self::View> {
        let keys = self
            .keys(txn.id())
            .await
            .map_err(|error| txn.storage_error(error))?;

        let keys = keys.map_stream(|stream| {
            stream
                .map(move |row| row.map_err(|error| txn.storage_error(error)))
                .boxed()
        });

        Ok(BTreeView {
            schema: self.schema().columns().to_vec(),
            keys,
        })
    }
}

impl<Txn: crate::StorageContext> IntoView for Table<Txn> {
    type Txn = Txn;

    type View = TableView;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        let schema = self.schema().clone();
        let rows = self
            .rows(txn.id())
            .await
            .map_err(|error| txn.storage_error(error))?
            .map(move |row| {
                row.map(|row| Value::Tuple(row.into_vec()))
                    .map_err(TCError::from)
                    .map_err(|error| txn.storage_error(error))
            })
            .boxed();

        Ok(TableView { schema, rows })
    }
}

#[cfg(test)]
#[path = "../tests/unit/view.rs"]
mod tests;
