//! Transaction-consistent Tensor streams prepared for encoding.

use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use number_general::Number;
use tc_error::TCResult;
use tc_ir::IntoView;

use super::{Tensor, TensorSchema};
use crate::StorageContext;

pub struct TensorView {
    pub(crate) schema: TensorSchema,
    pub(crate) contents: Contents,
}

pub(crate) enum Contents {
    Dense(BoxStream<'static, TCResult<Number>>),
    Sparse(super::SparseElements),
}

impl<Txn: StorageContext> IntoView for Tensor<Txn> {
    type Txn = Txn;

    type View = TensorView;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        let schema = self.schema()?;
        let contents = if schema.is_sparse() {
            Contents::Sparse(
                self.sparse_elements(txn.id())
                    .await
                    .map_err(|error| txn.storage_error(error))?
                    .map_err(move |error| txn.storage_error(error))
                    .boxed(),
            )
        } else {
            Contents::Dense(
                self.elements(txn.id())
                    .await
                    .map_err(|error| txn.storage_error(error))?
                    .map_err(move |error| txn.storage_error(error))
                    .boxed(),
            )
        };

        Ok(TensorView { schema, contents })
    }
}
