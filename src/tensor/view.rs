//! Transaction-consistent Tensor streams prepared for encoding.

use destream::en;
use futures::{StreamExt, TryStreamExt};
use futures::stream::BoxStream;
use number_general::Number;
use tc_error::TCResult;
use tc_ir::IntoView;

use super::{Tensor, TensorSchema};
use crate::StorageContext;

pub struct TensorView {
    schema: TensorSchema,
    contents: Contents,
}

enum Contents {
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

impl<'en> en::IntoStream<'en> for TensorView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let schema = self.schema.logical_value();
        match self.contents {
            Contents::Sparse(entries) => {
                (schema, en::SeqStream::from(entries)).into_stream(encoder)
            }
            Contents::Dense(values) => (schema, en::SeqStream::from(values)).into_stream(encoder),
        }
    }
}
