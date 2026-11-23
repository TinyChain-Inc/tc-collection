use number_general::Number;
use pathlink::Link;
use tc_error::TCResult;
use tc_ir::{Map, Scalar};
use tc_value::Value;
use tc_value::class::NativeClass;

use crate::Collection;
use crate::class::CollectionType;
use crate::tensor::Tensor;

/// The minimal conversion boundary required by collection route handlers.
///
/// Implementations live in the caller's state crate so collection code never
/// depends on a particular universal state representation.
pub trait CollectionState:
    tc_ir::StateInstance<Transaction = Self::Txn>
    + Clone
    + Send
    + 'static
    + From<Scalar>
    + From<Collection<Self::Txn>>
    + From<crate::table::Table<Self::Txn>>
    + From<Value>
    + From<u64>
{
    type Txn: crate::StorageContext;

    fn into_scalar(self) -> TCResult<Scalar>;
    fn into_value(self) -> TCResult<Value>;
    fn into_tuple(self) -> TCResult<Vec<Self>>;
    fn into_map(self) -> TCResult<Map<Self>>;
    fn into_tensor(self) -> TCResult<Tensor<Self::Txn>>;

    /// Inspect an embedded number without consuming or coercing this value.
    fn as_number(&self) -> Option<Number>;

    fn is_none(&self) -> bool;
}

impl<S, Txn> tc_ir::Route<S> for Collection<Txn>
where
    S: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    fn route<'a>(
        &'a self,
        path: &[pathlink::PathSegment],
    ) -> Option<Box<dyn tc_ir::Handler<'a, S> + 'a>> {
        match self {
            Self::BTree(view) => tc_ir::Route::route(view.as_ref(), path),
            Self::Table(table) => tc_ir::Route::route(table.as_ref(), path),
            Self::Tensor(tensor) => tc_ir::Route::route(tensor, path),
        }
    }
}

/// The result of attempting native literal construction with owned inputs.
pub enum LiteralPut<S> {
    Created(S),
    Unmatched { key: S, value: S },
}

impl<Txn: crate::StorageContext> Collection<Txn> {
    pub async fn from_put<S: CollectionState<Txn = Txn>>(
        txn: &Txn,
        link: &Link,
        key: S,
        value: S,
    ) -> TCResult<LiteralPut<S>> {
        match CollectionType::from_path(link.path()) {
            Some(CollectionType::Tensor(_)) => {
                crate::tensor::route::tensor_literal(txn, key, value)
                    .await
                    .map(|tensor| LiteralPut::Created(S::from(Self::Tensor(tensor))))
            }
            Some(CollectionType::BTree(_) | CollectionType::Table(_)) | None => {
                Ok(LiteralPut::Unmatched { key, value })
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/route/tests.rs"]
mod tests;
