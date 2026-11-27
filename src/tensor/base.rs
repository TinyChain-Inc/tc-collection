//! Writable behavior belongs to the transactional Tensor base.

use fensor::{TensorElement, TensorFileEntry, TensorGeometry};
use freqfs::DirLock;
use futures::{StreamExt, TryStreamExt};
use tc_error::{Pressure, PressureReason, TCError, TCResult};

use super::access::TensorAccess;
use super::storage::TensorBase;
use crate::StorageContext;
use crate::persistence::storage::Store;

impl<Txn: StorageContext, T: TensorElement> TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    pub(super) async fn dense(txn: &Txn, shape: Vec<u64>, values: Vec<T>) -> TCResult<Self> {
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

        Self::from_values(
            txn,
            schema,
            futures::stream::iter(values.into_iter().map(Ok)),
        )
        .await
    }

    pub(super) async fn from_values<E>(
        txn: &Txn,
        schema: super::TensorSchema,
        values: impl futures::Stream<Item = Result<T, E>> + Send,
    ) -> Result<Self, E>
    where
        E: From<TCError> + From<fensor::Error>,
    {
        let store = Store::create(txn.subcontext_unique().context().await?).await?;
        let native = fensor::Tensor::from_values(
            store.base().clone(),
            schema.into_native(),
            fensor::MAX_BLOCK_CAPACITY,
            values,
        )
        .await?;

        Ok(Self::from_native(native, store).await?)
    }

    pub(super) async fn from_sparse_entries<E>(
        txn: &Txn,
        schema: super::TensorSchema,
        entries: impl futures::Stream<Item = Result<(Vec<u64>, T), E>> + Send,
    ) -> Result<Self, E>
    where
        E: From<TCError> + From<fensor::Error>,
    {
        let store = Store::create(txn.subcontext_unique().context().await?).await?;
        let layout = schema.layout();
        let entries = entries.map(|entry| -> Result<_, E> {
            let (coord, value) = entry?;
            if value == T::ZERO {
                return Err(TCError::bad_request("sparse Tensor entries must be nonzero").into());
            }

            Ok((coord, value))
        });
        let native = fensor::Tensor::from_sparse_elements(
            store.base().clone(),
            schema.into_native(),
            layout,
            fensor::MAX_BLOCK_CAPACITY,
            entries,
        )
        .await?;

        Ok(Self::from_native(native, store).await?)
    }

    pub(super) async fn assign(
        &self,
        txn: &Txn,
        selection: Option<fensor::Range>,
        source: TensorAccess<Txn, T>,
    ) -> TCResult<()> {
        let selection = self.geometry().write_selection(selection)?;
        if selection.size() == 0 || selection.shape() != source.shape() {
            return Err(TCError::bad_request(
                "Tensor assignment requires a nonempty selection and matching shapes",
            ));
        }

        let complete = selection.is_full();
        // Read a consistent target while building an unpublished replacement.
        // Consuming the source lets an aliased computed value release its leases
        // before publication. Another retained alias still produces a conflict.
        let range = selection.block_range();
        let read_range = match source.reservation()? {
            Some((base, other)) if self.same(base) => {
                range.start.min(other.start)..range.end.max(other.end)
            }
            _ => range.clone(),
        };

        let base = self.try_snapshot(txn.id(), read_range)?;
        let stamp = self.pending_stamp(txn.id())?;
        let geometry = self.geometry();
        let sparse = geometry.sparse_axis().is_some();
        let staged = if complete
            && let Some(owner) = source.as_base()
            && sparse == owner.geometry().sparse_axis().is_some()
        {
            let other;
            let snapshot = if self.same(owner) {
                &base
            } else {
                other = owner.try_snapshot(txn.id(), 0..owner.geometry().block_count())?;
                &other
            };

            self.stage_full_blocks(txn, geometry.read_blocks(snapshot)?)
                .await?
        } else {
            let expression = source.try_expression_with(txn.id(), [(self, &base)])?;
            drop(source);

            if complete && sparse && matches!(expression.layout(), fensor::Layout::Sparse { .. }) {
                let blocks = geometry.pack_sparse_blocks(expression.into_sparse_elements()?)?;
                self.stage_full_blocks(txn, blocks).await?
            } else {
                let staged = self.stage(txn, complete).await?;
                let mut blocks = selection.updates_from(expression)?;
                while let Some(updates) = blocks.try_next().await? {
                    staged
                        .apply_updates((!complete).then_some(&base), updates)
                        .await?;
                }

                staged
            }
        };

        drop(base);
        let _write = self.try_acquire_write_permit(txn.id(), range)?;

        // The semaphore has no atomic upgrade. Detect any same-transaction
        // mutation during the read-to-write gap before publishing the buffer.
        self.publish_if_unchanged(txn, staged, stamp).await
    }

    pub(super) async fn fill_range(
        &self,
        txn: &Txn,
        range: fensor::Range,
        value: T,
    ) -> TCResult<()> {
        let selection = self.geometry().write_selection(Some(range))?;
        if selection.size() == 0 {
            return Err(TCError::bad_request(
                "Tensor fill requires a nonempty selection",
            ));
        }

        if selection.is_full() {
            return self.fill(txn, value).await;
        }

        let _permit = self.try_acquire_write_permit(txn.id(), selection.block_range())?;

        let snapshot = self.snapshot(txn.id(), None)?;
        let staged = self.stage(txn, false).await?;
        let size = selection.size();
        let mut start = 0;
        while start < size {
            let len = (size - start).min(fensor::MAX_BATCH_ELEMENTS as u64) as usize;
            let updates = selection.plan_updates(start, vec![value; len])?;
            staged.apply_updates(Some(&snapshot), updates).await?;
            start += len as u64;
        }

        self.publish(txn, staged, |_| Ok(())).await
    }
}

impl<Txn: StorageContext, T: TensorElement> crate::CopyFrom<&TensorAccess<Txn, T>>
    for TensorBase<Txn, T>
where
    Txn::File: TensorFileEntry<T>,
{
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: DirLock<Txn::File>,
        source: &TensorAccess<Txn, T>,
    ) -> TCResult<Self> {
        let expression = source.expression(txn.id()).await?;
        super::TensorSchema::validate_geometry(expression.shape(), expression.layout())?;
        let store = Store::create(dir).await?;
        let native = fensor::Tensor::copy_from(
            store.base().clone(),
            &expression,
            expression.layout(),
            fensor::MAX_BLOCK_CAPACITY,
        )
        .await?;

        Self::from_native(native, store).await
    }
}
