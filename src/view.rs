use futures::{StreamExt, stream::BoxStream};
use tc_error::{TCError, TCResult};
use tc_ir::IntoView;
use tc_value::Value;

use crate::{
    Collection,
    btree::{BTreeColumnSchema, Keys},
    table::TableSchema,
    tensor::Tensor,
};

/// A transaction-consistent BTree representation prepared for terminal projection.
pub struct BTreeView {
    pub(crate) schema: Vec<BTreeColumnSchema>,
    pub(crate) keys: Keys,
    pub(crate) arity: usize,
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
    Tensor(Tensor),
}

impl<Txn> IntoView for Collection<Txn>
where
    Txn: crate::StorageContext + 'static,
{
    type Txn = Txn;
    type View = CollectionView;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        match self {
            Collection::BTree(view) => {
                let arity = view.schema.len();
                let keys = view
                    .btree
                    .keys(txn.id(), view.bounds, view.reverse)
                    .await
                    .map_err(TCError::from)?;
                Ok(CollectionView::BTree(BTreeView {
                    schema: view.schema,
                    keys,
                    arity,
                }))
            }
            Collection::Table(table) => {
                let schema = table.schema().clone();
                let rows: BoxStream<'static, TCResult<Value>> = table
                    .row_stream(txn.id())
                    .await?
                    .map(|row| {
                        row.map(|row| Value::Tuple(row.into_vec()))
                            .map_err(TCError::from)
                    })
                    .boxed();
                Ok(CollectionView::Table(TableView { schema, rows }))
            }
            Collection::Tensor(tensor) => {
                let bytes = tensor.retained_bytes().ok_or_else(|| {
                    TCError::payload_too_large(
                        "tensor allocation size overflow",
                        tc_error::Pressure::new(
                            "/host/resource/tensor/materialized",
                            tc_error::PressureReason::AllocationFailed,
                        ),
                    )
                })?;
                if bytes > txn.materialized_tensor_bytes() {
                    return Err(TCError::payload_too_large(
                        "materialized tensor exceeds the configured host limit",
                        tc_error::Pressure::new(
                            "/host/resource/tensor/materialized",
                            tc_error::PressureReason::QuotaExceeded,
                        ),
                    ));
                }
                Ok(CollectionView::Tensor(tensor))
            }
        }
    }
}
