use fensor::{
    Layout, Shape, TensorArray, TensorElement, TensorExpression, TensorFileEntry, TensorGeometry,
    TensorRead, TensorTransform, TensorViewSemantics,
};
use freqfs::DirLock;
use futures::{StreamExt, TryStreamExt};
use tc_error::{Pressure, PressureReason, TCError, TCResult};
use tc_ir::TxnId;

use super::storage::{Owner, Snapshot};
use crate::StorageContext;

/// Only owners mutate storage; geometric views and computed expressions are read-only.
#[derive(Clone)]
pub(super) enum Typed<Txn: StorageContext, T: TensorElement>
where
    Txn::File: TensorFileEntry<T>,
{
    Owner(Box<Owner<Txn, T>>),
    View(Box<fensor::TensorView<Owner<Txn, T>>>),
    Computed(Box<TensorExpression<T>>, TxnId),
}

impl<Txn: StorageContext, T: TensorElement> TensorGeometry for Typed<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    type DType = T;

    fn dtype(&self) -> fensor::NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        match self {
            Self::Owner(owner) => owner.shape(),
            Self::View(view) => view.shape(),
            Self::Computed(expr, _) => expr.shape(),
        }
    }

    fn layout(&self) -> Layout {
        match self {
            Self::Owner(owner) => owner.layout(),
            Self::View(view) => view.layout(),
            Self::Computed(expr, _) => expr.layout(),
        }
    }
}

macro_rules! transform {
    ($method:ident, $arg:ident, $ty:ty) => {
        pub fn $method(self, $arg: $ty) -> TCResult<Self> {
            let result = match self {
                Self::Owner(owner) => {
                    Self::View(Box::new(fensor::TensorView::new(*owner).$method($arg)?))
                }
                Self::View(view) => Self::View(Box::new(view.$method($arg)?)),
                Self::Computed(expr, id) => Self::computed((*expr).$method($arg)?, id),
            };

            super::TensorSchema::validate_geometry(result.shape(), result.layout())?;
            Ok(result)
        }
    };
}

impl<Txn: StorageContext, T: TensorElement> Typed<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    pub fn computed(expression: TensorExpression<T>, id: TxnId) -> Self {
        Self::Computed(Box::new(expression), id)
    }

    pub async fn dense(txn: &Txn, shape: Vec<u64>, values: Vec<T>) -> TCResult<Self> {
        // This checks supplied element bytes after allocation, not spare capacity
        // or aggregate working memory. Storage blocks have an independent bound.
        if std::mem::size_of_val(values.as_slice()) > txn.materialized_tensor_bytes() {
            return Err(TCError::payload_too_large(
                "Tensor buffer exceeds the delegated allocation limit",
                Pressure::new(
                    "/host/resource/tensor/materialized",
                    PressureReason::QuotaExceeded,
                ),
            ));
        }

        let schema = super::TensorSchema::new(T::dtype(), shape.into(), None)?;
        if values.len() as u64 != schema.shape().iter().product::<u64>() {
            return Err(TCError::bad_request(
                "Tensor literal value count differs from shape",
            ));
        }

        let capacity = fensor::MAX_BLOCK_CAPACITY;
        let tensor = fensor::Tensor::from_values(
            txn.subcontext_unique().context().await?,
            schema.into_native(),
            capacity,
            futures::stream::iter(values.into_iter().map(Ok::<_, fensor::Error>)),
        )
        .await?;

        Ok(Self::stored(tensor))
    }

    pub fn stored(native: fensor::Tensor<Txn::File, T>) -> Self {
        Self::Owner(Box::new(Owner::new(native)))
    }

    pub fn into_read_only(self) -> Self {
        match self {
            Self::Owner(owner) => Self::View(Box::new(fensor::TensorView::new(*owner))),
            view => view,
        }
    }

    pub fn owner(&self) -> TCResult<&Owner<Txn, T>> {
        match self {
            Self::Owner(owner) => Ok(owner),
            _ => Err(TCError::unsupported("Tensor views are read-only")),
        }
    }

    pub fn is_persistent(&self) -> bool {
        matches!(self, Self::Owner(_))
    }

    pub async fn expression(&self, id: TxnId) -> TCResult<TensorExpression<T>> {
        match self {
            Self::Owner(owner) => Ok(TensorExpression::new(fensor::TensorView::new(
                owner
                    .read_snapshot(id, 0..owner.geometry().block_count())
                    .await?,
            ))?),
            Self::View(view) => {
                let source = view
                    .source()
                    .read_snapshot(id, view.logical_block_range(view.source().geometry())?)
                    .await?;

                Self::snapshot_expression(view, source)
            }
            Self::Computed(expr, bound) if id == *bound => Ok((**expr).clone()),
            Self::Computed(_, _) => Err(TCError::bad_request(
                "computed Tensor belongs to a different transaction",
            )),
        }
    }

    pub async fn as_dense(&self, id: TxnId) -> TCResult<Self> {
        Ok(Self::computed(self.expression(id).await?.into_dense(), id))
    }

    pub fn try_expression(&self, id: TxnId) -> TCResult<TensorExpression<T>> {
        match self {
            Self::Owner(owner) => Ok(TensorExpression::new(fensor::TensorView::new(
                owner.try_snapshot(id, 0..owner.geometry().block_count())?,
            ))?),
            Self::View(view) => {
                let source = view
                    .source()
                    .try_snapshot(id, view.logical_block_range(view.source().geometry())?)?;

                Self::snapshot_expression(view, source)
            }
            Self::Computed(expr, bound) if id == *bound => Ok((**expr).clone()),
            Self::Computed(_, _) => Err(TCError::bad_request(
                "computed Tensor belongs to a different transaction",
            )),
        }
    }

    /// Coalesce the bounded operand list before acquiring any owner reservation.
    pub fn expressions<const N: usize>(
        inputs: [&Self; N],
        id: TxnId,
    ) -> TCResult<[TensorExpression<T>; N]> {
        let mut requests: Vec<(&Owner<Txn, T>, std::ops::Range<u64>)> = Vec::with_capacity(N);
        for input in &inputs {
            let (owner, range) = match input {
                Self::Owner(owner) => (owner.as_ref(), 0..owner.geometry().block_count()),
                Self::View(view) => (
                    view.source(),
                    view.logical_block_range(view.source().geometry())?,
                ),
                Self::Computed(..) => continue,
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
            .map(|input| match input {
                Self::Owner(owner) => {
                    let position = requests
                        .iter()
                        .position(|(reserved, _)| reserved.same(owner))
                        .expect("reserved source");

                    Ok(TensorExpression::new(fensor::TensorView::new(
                        snapshots[position].clone(),
                    ))?)
                }
                Self::View(view) => {
                    let position = requests
                        .iter()
                        .position(|(owner, _)| owner.same(view.source()))
                        .expect("reserved source");

                    Self::snapshot_expression(view, snapshots[position].clone())
                }
                Self::Computed(_, _) => input.try_expression(id),
            })
            .collect::<TCResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| unreachable!("operand count changed"))
    }

    fn snapshot_expression(
        view: &fensor::TensorView<Owner<Txn, T>>,
        snapshot: Snapshot<Txn::File, T>,
    ) -> TCResult<TensorExpression<T>> {
        Ok(TensorExpression::new(view.clone().with_source(snapshot)?)?)
    }

    pub async fn read_value(&self, id: TxnId, coord: &[u64]) -> TCResult<T> {
        match self {
            Self::Owner(owner) => owner.read_value(id, coord).await,
            Self::View(view) => {
                view.source()
                    .read_value(id, &view.resolve_base_coord(coord)?)
                    .await
            }
            Self::Computed(_, _) => {
                let value = self.expression(id).await?.read_value(coord).await?;
                Ok(value)
            }
        }
    }

    pub async fn write_value(&self, txn: &Txn, coord: &[u64], value: T) -> TCResult<()> {
        self.owner()?.write_value(txn, coord, value).await
    }

    pub async fn copy_into(
        &self,
        id: TxnId,
        dir: DirLock<Txn::File>,
        capacity: usize,
    ) -> TCResult<Self> {
        let expression = self.expression(id).await?;
        let layout = Self::copy_layout(&expression)?;
        let native = fensor::Tensor::copy_from(dir, &expression, layout, capacity).await?;
        native.sync().await?;
        Ok(Self::stored(native))
    }

    pub async fn copy_values(
        &self,
        id: TxnId,
        dir: DirLock<Txn::File>,
        capacity: usize,
    ) -> TCResult<Self> {
        let expression = self.expression(id).await?;
        let layout = Self::copy_layout(&expression)?;
        let native = fensor::Tensor::copy_from_values(dir, &expression, layout, capacity).await?;
        native.sync().await?;
        Ok(Self::stored(native))
    }

    fn copy_layout(expression: &TensorExpression<T>) -> TCResult<Layout> {
        super::TensorSchema::validate_geometry(expression.shape(), expression.layout())?;
        Ok(expression.layout())
    }

    pub async fn assign(
        &self,
        txn: &Txn,
        selection: Option<fensor::Range>,
        source: Self,
    ) -> TCResult<()> {
        let owner = self.owner()?;
        let view = fensor::TensorView::new(owner.clone());
        let view = if let Some(selection) = selection {
            view.slice(selection)?
        } else {
            view
        };

        if !view.supports_write_through() || view.shape() != source.shape() {
            return Err(TCError::bad_request(
                "Tensor assignment requires writable geometry and matching shapes",
            ));
        }

        // Read a consistent target while building an unpublished replacement.
        // Consuming the source lets an aliased computed value release its leases
        // before publication. Another retained alias still produces a conflict.
        let range = view.logical_block_range(owner.geometry())?;
        let read_range = match &source {
            Self::Owner(source) if owner.same(source) => 0..owner.geometry().block_count(),
            Self::View(source) if owner.same(source.source()) => {
                let other = source.logical_block_range(owner.geometry())?;
                range.start.min(other.start)..range.end.max(other.end)
            }
            _ => range.clone(),
        };

        let base = owner.try_snapshot(txn.id(), read_range)?;
        let stamp = owner.pending_stamp(txn.id())?;
        let expression = match &source {
            Self::Owner(source) if owner.same(source) => {
                TensorExpression::new(fensor::TensorView::new(base.clone()))?
            }
            Self::View(source) if owner.same(source.source()) => {
                Self::snapshot_expression(source, base.clone())?
            }
            source => source.try_expression(txn.id())?,
        };

        drop(source);
        let staged = owner.stage(txn).await?;
        let mut blocks = view.updates_from(owner.geometry(), expression)?;
        while let Some(updates) = blocks.try_next().await? {
            staged.apply_updates(&base, updates).await?;
        }

        drop(blocks);
        drop(base);
        let _write = owner
            .core
            .try_acquire_write_permit(txn.id(), view.logical_block_range(owner.geometry())?)?;

        // The semaphore has no atomic upgrade. Detect any same-transaction
        // mutation during the read-to-write gap before publishing the buffer.
        owner.publish_if_unchanged(txn, staged, stamp).await
    }

    pub async fn fill(&self, txn: &Txn, value: T) -> TCResult<()> {
        let owner = self.owner()?;
        let _permit = owner
            .core
            .try_acquire_write_permit(txn.id(), 0..owner.geometry().block_count())?;

        owner.assert_writable(txn.id())?;
        let staged = owner.stage(txn).await?;
        for block in 0..owner.geometry().block_count() {
            let values = vec![value; owner.geometry().block_len(block)?];
            staged.write(block, values).await?;
        }

        owner.publish(txn, staged).await
    }

    pub async fn fill_range(&self, txn: &Txn, range: fensor::Range, value: T) -> TCResult<()> {
        let owner = self.owner()?;
        let view = fensor::TensorView::new(owner.clone()).slice(range)?;
        if view.is_base_tensor() {
            return self.fill(txn, value).await;
        }

        let size = view.size()?;
        let values = futures::stream::iter(0..size).map(move |_| Ok(value));
        Self::write_elements(&view, txn, values).await
    }

    async fn write_elements(
        view: &fensor::TensorView<Owner<Txn, T>>,
        txn: &Txn,
        values: impl futures::Stream<Item = TCResult<T>> + Send,
    ) -> TCResult<()> {
        if !view.supports_write_through() {
            return Err(TCError::unsupported("Tensor geometry is read-only"));
        }

        let owner = view.source();
        let _permit = owner
            .core
            .try_acquire_write_permit(txn.id(), view.logical_block_range(owner.geometry())?)?;

        owner.assert_writable(txn.id())?;
        let snapshot = owner.snapshot(txn.id(), None)?;
        let staged = owner.stage(txn).await?;
        let size = view.size()?;
        let mut start = 0;
        futures::pin_mut!(values);
        while start < size {
            let len = (size - start).min(fensor::MAX_BATCH_ELEMENTS as u64) as usize;
            let mut batch = Vec::with_capacity(len);
            for _ in 0..len {
                batch.push(
                    values
                        .try_next()
                        .await?
                        .ok_or_else(|| TCError::bad_request("too few Tensor assignment values"))?,
                );
            }

            let updates = view.plan_updates(owner.geometry(), start, batch)?;
            staged.apply_updates(&snapshot, updates).await?;
            start += len as u64;
        }

        if values.try_next().await?.is_some() {
            return Err(TCError::bad_request("too many Tensor assignment values"));
        }

        owner.publish(txn, staged).await
    }

    pub fn commit(&self, id: TxnId) -> TCResult<()> {
        // owner() rejects views and expressions. Only the mutable owner can
        // publish its pending blocks; commit does not flush canonical storage.
        self.owner()?.core.commit(id)
    }

    pub fn rollback(&self, id: TxnId) -> TCResult<()> {
        // Discard pending blocks through the same lifecycle owner as BTree/Table.
        self.owner()?.core.rollback(id)
    }

    pub async fn finalize(&self, id: TxnId) -> TCResult<()> {
        // The shared owner applies covered block replacements before advancing
        // the frontier; fensor owns each native logical-block replacement.
        self.owner()?.core.finalize(id).await
    }

    pub async fn sync(&self) -> TCResult<()> {
        let owner = self.owner()?;
        owner.canonical()?.sync().await?;
        Ok(())
    }

    pub async fn sync_all(&self) -> TCResult<()> {
        let owner = self.owner()?;
        owner.canonical()?.sync_all().await?;
        Ok(())
    }

    transform!(reshape, shape, Shape);
    transform!(broadcast, shape, Shape);
    transform!(slice, range, fensor::Range);
    transform!(transpose, permutation, Option<fensor::Axes>);
    transform!(flip, axis, usize);
    transform!(squeeze, axes, fensor::Axes);
    transform!(unsqueeze, axes, fensor::Axes);
}

impl<Txn: StorageContext, T: TensorElement> crate::Restore for Typed<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    type Txn = Txn;

    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let target = self.owner()?;
        let source = source.owner()?;
        if target.schema() != source.schema()
            || target.geometry().sparse_axis() != source.geometry().sparse_axis()
        {
            return Err(TCError::bad_request("Tensor restoration schema mismatch"));
        }

        target.assert_writable(txn.id())?;
        if target.same(source) {
            return Ok(());
        }

        let _write = target
            .core
            .try_acquire_write_permit(txn.id(), 0..target.geometry().block_count())?;

        target.assert_writable(txn.id())?;
        let snapshot = source.try_snapshot(txn.id(), 0..source.geometry().block_count())?;
        let view = fensor::TensorView::new(snapshot);
        let expression = TensorExpression::new(view)?;
        let staged = target.stage(txn).await?;
        // Consume all values, including zeros, in canonical block order.
        let mut batches = expression.into_blocks()?;
        let mut remaining = Vec::new().into_iter();
        for block in 0..target.geometry().block_count() {
            let len = target.geometry().block_len(block)?;
            if remaining.len() == 0 {
                let batch = batches
                    .try_next()
                    .await?
                    .ok_or_else(|| TCError::internal("restoration length mismatch"))?;

                if batch.len() == len {
                    staged.write(block, batch).await?;
                    continue;
                }

                remaining = batch.into_iter();
            }

            let mut values = Vec::with_capacity(len);
            while values.len() < len {
                if let Some(value) = remaining.next() {
                    values.push(value);
                } else {
                    remaining = batches
                        .try_next()
                        .await?
                        .ok_or_else(|| TCError::internal("restoration length mismatch"))?
                        .into_iter();
                }
            }

            staged.write(block, values).await?;
        }

        if remaining.len() != 0 || batches.try_next().await?.is_some() {
            return Err(TCError::internal("restoration length mismatch"));
        }

        target.install(txn.id(), staged)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/typed.rs"]
mod tests;
