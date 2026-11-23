use futures::stream::BoxStream;
use tc_error::TCResult;
use tc_ir::IntoView;
use tc_value::Value;

use crate::Collection;
use crate::btree::{BTreeColumnSchema, Keys};
use crate::table::TableSchema;
use crate::tensor::TensorView as NativeTensorView;

/// A transaction-consistent BTree representation prepared for terminal projection.
pub struct BTreeView {
    pub(crate) schema: Vec<BTreeColumnSchema>,
    pub(crate) keys: Keys,
}

/// A transaction-consistent Table representation prepared for terminal projection.
pub struct TableView {
    pub(crate) schema: TableSchema,
    pub(crate) rows: BoxStream<'static, TCResult<Value>>,
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
