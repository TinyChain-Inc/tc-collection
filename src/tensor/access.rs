use fensor::{
    Layout, Shape, TensorElement, TensorExpression, TensorFileEntry, TensorGeometry, TensorRead,
    TensorTransform,
};
use tc_error::{TCError, TCResult};
use tc_ir::TxnId;

use super::storage::{Snapshot, TensorBase};
use crate::StorageContext;

type Reservation<'a, Txn, T> = (&'a TensorBase<Txn, T>, std::ops::Range<u64>);

/// A typed Tensor payload with private storage, geometry, and expression ownership.
/// Only a base can mutate storage; transformations always produce read-only values.
#[derive(Clone)]
pub struct TensorAccess<Txn: StorageContext, T: TensorElement>
where
    Txn::File: TensorFileEntry<T>,
{
    source: Source<Txn, T>,
}

#[derive(Clone)]
enum Source<Txn: StorageContext, T: TensorElement>
where
    Txn::File: TensorFileEntry<T>,
{
    Base(Box<TensorBase<Txn, T>>),
    Geometry(Box<fensor::TensorView<TensorBase<Txn, T>>>),
    Expression(Box<TensorExpression<T>>, TxnId),
}

impl<Txn: StorageContext, T: TensorElement> TensorGeometry for TensorAccess<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    type DType = T;

    fn dtype(&self) -> fensor::NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        match &self.source {
            Source::Base(owner) => owner.shape(),
            Source::Geometry(view) => view.shape(),
            Source::Expression(expr, _) => expr.shape(),
        }
    }

    fn layout(&self) -> Layout {
        match &self.source {
            Source::Base(owner) => owner.layout(),
            Source::Geometry(view) => view.layout(),
            Source::Expression(expr, _) => expr.layout(),
        }
    }
}

macro_rules! transform {
    ($method:ident, $arg:ident, $ty:ty) => {
        pub(super) fn $method(self, $arg: $ty) -> TCResult<Self> {
            let result = match self.source {
                Source::Base(owner) => {
                    Self::from_view(fensor::TensorView::new(*owner).$method($arg)?)
                }
                Source::Geometry(view) => Self::from_view(view.$method($arg)?),
                Source::Expression(expr, id) => Self::computed((*expr).$method($arg)?, id),
            };

            super::TensorSchema::validate_geometry(result.shape(), result.layout())?;
            Ok(result)
        }
    };
}

impl<Txn: StorageContext, T: TensorElement> TensorAccess<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    pub(super) fn computed(expression: TensorExpression<T>, id: TxnId) -> Self {
        Self {
            source: Source::Expression(Box::new(expression), id),
        }
    }

    pub(super) fn from_base(base: TensorBase<Txn, T>) -> Self {
        Self {
            source: Source::Base(Box::new(base)),
        }
    }

    fn from_view(view: fensor::TensorView<TensorBase<Txn, T>>) -> Self {
        Self {
            source: Source::Geometry(Box::new(view)),
        }
    }

    pub(super) fn into_read_only(self) -> Self {
        match self.source {
            Source::Base(owner) => Self::from_view(fensor::TensorView::new(*owner)),
            _ => self,
        }
    }

    pub(super) fn base(&self) -> TCResult<&TensorBase<Txn, T>> {
        self.as_base()
            .ok_or_else(|| TCError::unsupported("Tensor views are read-only"))
    }

    pub(super) fn as_base(&self) -> Option<&TensorBase<Txn, T>> {
        match &self.source {
            Source::Base(owner) => Some(owner),
            _ => None,
        }
    }

    pub(super) fn is_persistent(&self) -> bool {
        self.as_base().is_some()
    }

    pub(super) async fn expression(&self, id: TxnId) -> TCResult<TensorExpression<T>> {
        match &self.source {
            Source::Base(owner) => Ok(TensorExpression::new(fensor::TensorView::new(
                owner
                    .read_snapshot(id, 0..owner.geometry().block_count())
                    .await?,
            ))?),
            Source::Geometry(view) => {
                let owner = view.source();
                let snapshot = owner
                    .read_snapshot(id, view.logical_block_range(owner.geometry())?)
                    .await?;

                Self::snapshot_expression(view, snapshot)
            }
            // A returned expression independently retains its sources and read leases.
            Source::Expression(expr, bound) => {
                Ok(Self::bound_expression(expr, *bound, id)?.clone())
            }
        }
    }

    pub(super) fn try_expression(&self, id: TxnId) -> TCResult<TensorExpression<T>> {
        self.try_expression_with(id, std::iter::empty())
    }

    /// Coalesce the bounded operand list before acquiring any owner reservation.
    pub(super) fn expressions<const N: usize>(
        inputs: [&Self; N],
        id: TxnId,
    ) -> TCResult<[TensorExpression<T>; N]> {
        let mut requests: Vec<Reservation<'_, Txn, T>> = Vec::with_capacity(N);
        for input in &inputs {
            let Some((owner, range)) = input.reservation()? else {
                continue;
            };

            if let Some((_, existing)) = requests
                .iter_mut()
                .find(|(previous, _)| previous.same(owner))
            {
                existing.start = existing.start.min(range.start);
                existing.end = existing.end.max(range.end);
            } else {
                requests.push((owner, range));
            }
        }

        // Failure drops every acquired snapshot. Never wait while holding another owner:
        // reciprocal expressions could otherwise deadlock across collection owners.
        let snapshots = requests
            .iter()
            .map(|(owner, range)| owner.try_snapshot(id, range.clone()))
            .collect::<TCResult<Vec<_>>>()?;

        inputs
            .into_iter()
            .map(|input| {
                input.try_expression_with(
                    id,
                    requests
                        .iter()
                        .zip(&snapshots)
                        .map(|((owner, _), snapshot)| (*owner, snapshot)),
                )
            })
            .collect::<TCResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| unreachable!("operand count changed"))
    }

    pub(super) fn reservation(&self) -> TCResult<Option<Reservation<'_, Txn, T>>> {
        match &self.source {
            Source::Base(base) => Ok(Some((base, 0..base.geometry().block_count()))),
            Source::Geometry(view) => Ok(Some((
                view.source(),
                view.logical_block_range(view.source().geometry())?,
            ))),
            Source::Expression(..) => Ok(None),
        }
    }

    pub(super) fn try_expression_with<'a>(
        &self,
        id: TxnId,
        snapshots: impl IntoIterator<Item = (&'a TensorBase<Txn, T>, &'a Snapshot<Txn::File, T>)>,
    ) -> TCResult<TensorExpression<T>> {
        let snapshot = |owner: &TensorBase<Txn, T>, range| {
            if let Some((_, snapshot)) = snapshots.into_iter().find(|(base, _)| base.same(owner)) {
                // Coalesced operands retain the same snapshot independently.
                Ok(snapshot.clone())
            } else {
                owner.try_snapshot(id, range)
            }
        };

        match &self.source {
            Source::Base(owner) => Ok(TensorExpression::new(fensor::TensorView::new(snapshot(
                owner,
                0..owner.geometry().block_count(),
            )?))?),
            Source::Geometry(view) => Self::snapshot_expression(
                view,
                snapshot(
                    view.source(),
                    view.logical_block_range(view.source().geometry())?,
                )?,
            ),
            Source::Expression(expr, bound) => {
                Ok(Self::bound_expression(expr, *bound, id)?.clone())
            }
        }
    }

    fn snapshot_expression(
        view: &fensor::TensorView<TensorBase<Txn, T>>,
        snapshot: Snapshot<Txn::File, T>,
    ) -> TCResult<TensorExpression<T>> {
        Ok(TensorExpression::new(view.with_source(snapshot)?)?)
    }

    fn bound_expression(
        expression: &TensorExpression<T>,
        bound: TxnId,
        id: TxnId,
    ) -> TCResult<&TensorExpression<T>> {
        if id != bound {
            return Err(TCError::bad_request(
                "computed Tensor belongs to a different transaction",
            ));
        }

        Ok(expression)
    }

    pub(super) async fn read_value(&self, id: TxnId, coord: &[u64]) -> TCResult<T> {
        match &self.source {
            Source::Base(owner) => owner.read_value(id, coord).await,
            Source::Geometry(view) => {
                view.source()
                    .read_value(id, &view.resolve_base_coord(coord)?)
                    .await
            }
            Source::Expression(expr, bound) => Ok(Self::bound_expression(expr, *bound, id)?
                .read_value(coord)
                .await?),
        }
    }

    transform!(reshape, shape, Shape);
    transform!(broadcast, shape, Shape);
    transform!(slice, range, fensor::Range);
    transform!(transpose, permutation, Option<fensor::Axes>);
    transform!(flip, axis, usize);
    transform!(squeeze, axes, fensor::Axes);
    transform!(unsqueeze, axes, fensor::Axes);
}
