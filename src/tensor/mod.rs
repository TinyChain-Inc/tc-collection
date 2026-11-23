//! Transactional Tensor values backed by fensor's native storage and expressions.

use std::fmt;

use fensor::TensorGeometry;
pub use fensor::{Axes, AxisRange, Layout, Range, Shape};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use number_general::Number;
use safecast::CastInto;

use crate::StorageContext;
use crate::tensor::typed::Typed;
pub(crate) use schema::MAX_TENSOR_METADATA_RANK;
pub use schema::TensorSchema;
use tc_error::{TCError, TCResult};
use tc_ir::{Transact, TxnId};
use tc_value::NumberType;
pub use view::TensorView;

pub type SparseElements = BoxStream<'static, TCResult<(Vec<u64>, Number)>>;

#[macro_use]
mod macros;

#[derive(Clone)]
pub struct Tensor<Txn: StorageContext> {
    data: Data<Txn>,
}

tensor_dtypes!($;
    integers [
        U8: u8 => NumberType::UInt(number_general::UIntType::U8), dense_u8;
        U16: u16 => NumberType::UInt(number_general::UIntType::U16), dense_u16;
        U32: u32 => NumberType::UInt(number_general::UIntType::U32), dense_u32;
        U64: u64 => NumberType::UInt(number_general::UIntType::U64), dense_u64;
        I8: i8 => NumberType::Int(number_general::IntType::I8), dense_i8;
        I16: i16 => NumberType::Int(number_general::IntType::I16), dense_i16;
        I32: i32 => NumberType::Int(number_general::IntType::I32), dense_i32;
        I64: i64 => NumberType::Int(number_general::IntType::I64), dense_i64;
    ]
    floats [
        F32: f32 => NumberType::Float(number_general::FloatType::F32), dense_f32;
        F64: f64 => NumberType::Float(number_general::FloatType::F64), dense_f64;
    ]
    complex [
        #[cfg(feature = "complex")]
        C32: fensor::complex::Complex32 =>
            NumberType::Complex(number_general::ComplexType::C32), dense_c32 => F32;
        #[cfg(feature = "complex")]
        C64: fensor::complex::Complex64 =>
            NumberType::Complex(number_general::ComplexType::C64), dense_c64 => F64;
    ]
);

#[derive(Clone, Debug)]
pub enum TensorReduceResult<Txn: StorageContext> {
    Scalar(Number),
    Tensor(Tensor<Txn>),
}

impl<Txn: StorageContext> fmt::Debug for Tensor<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.number_type())
            .field("shape", &self.shape())
            .field("layout", &self.layout())
            .finish()
    }
}

impl<Txn: StorageContext> Tensor<Txn> {
    pub(crate) async fn hash_contents(&self, id: TxnId) -> TCResult<tc_ir::Sha256Hash> {
        dispatch!(&self.data, value => hash::blocks(value.expression(id).await?.into_blocks()?).await)
    }

    pub fn shape(&self) -> &[u64] {
        dispatch!(&self.data, value => value.shape())
    }

    pub fn layout(&self) -> Layout {
        dispatch!(&self.data, value => value.layout())
    }

    pub fn size(&self) -> TCResult<u64> {
        dispatch!(&self.data, value => value.size())
            .map_err(|_| TCError::bad_request("Tensor size overflows u64"))
    }

    pub fn number_type(&self) -> NumberType {
        dispatch!(&self.data, value => value.dtype())
    }

    pub fn schema(&self) -> TCResult<TensorSchema> {
        TensorSchema::from_geometry(self.number_type(), self.shape().into(), self.layout())
    }

    pub fn is_persistent(&self) -> bool {
        dispatch!(&self.data, value => value.is_persistent())
    }

    /// Return a read-only alias. Transforming or cloning it never grants ownership.
    pub(crate) fn into_read_only(self) -> Self {
        Self {
            data: dispatch!(@wrap self.data, value => value.into_read_only()),
        }
    }

    /// Evaluate once into independent unpublished native storage.
    pub async fn materialize(&self, txn: &Txn, dir: DirLock<Txn::File>) -> TCResult<Self> {
        let capacity = fensor::MAX_BLOCK_CAPACITY;
        Ok(Self {
            data: dispatch!(@wrap &self.data, value => value.copy_into(txn.id(), dir, capacity).await?),
        })
    }

    /// Interpret implicit zeros as dense values without copying or persisting data.
    /// Explicit copying materializes the result in independent storage.
    pub async fn as_dense(&self, id: TxnId) -> TCResult<Self> {
        Ok(Self {
            data: dispatch!(@wrap &self.data, value => value.as_dense(id).await?),
        })
    }

    pub async fn create(
        dir: DirLock<Txn::File>,
        schema: TensorSchema,
        max_capacity: usize,
    ) -> TCResult<Self> {
        let data = dtype_factory!(schema.dtype(), T => fensor::Tensor::<Txn::File, T>::create(
            dir, schema.native().clone(), schema.layout(), max_capacity
        ).await?, dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {dtype}"))));

        Ok(Self { data })
    }

    pub async fn load(dir: DirLock<Txn::File>, schema: TensorSchema) -> TCResult<Self> {
        let data = dtype_factory!(schema.dtype(), T => fensor::Tensor::<Txn::File, T>::load(dir).await?,
            dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {dtype}"))));

        let tensor = Self { data };
        if tensor.schema()? != schema {
            return Err(TCError::bad_request("Tensor storage schema mismatch"));
        }

        Ok(tensor)
    }

    dense_literals!();

    pub async fn fill(&self, txn: &Txn, number: Number) -> TCResult<()> {
        dispatch!(&self.data, value => value.fill(txn, number.cast_into()).await)
    }

    /// Fill a selection through its mutable owner, without exposing a writable view.
    /// Explicit gather selections remain read-only.
    pub async fn fill_range(&self, txn: &Txn, range: Range, number: Number) -> TCResult<()> {
        dispatch!(&self.data, value => value.fill_range(txn, range, number.cast_into()).await)
    }

    pub async fn read_value(&self, id: TxnId, coord: &[u64]) -> TCResult<Number> {
        dispatch!(&self.data, value => Ok(Number::from(value.read_value(id, coord).await?)))
    }

    pub async fn write_value(&self, txn: &Txn, coord: &[u64], number: Number) -> TCResult<()> {
        dispatch!(&self.data, value => value.write_value(txn, coord, number.cast_into()).await)
    }

    pub async fn elements(&self, id: TxnId) -> TCResult<BoxStream<'static, TCResult<Number>>> {
        let stream = dispatch!(&self.data, value => value.expression(id).await?.into_blocks()?
            .map(|batch| batch.map(|values| futures::stream::iter(
                values.into_iter().map(|value| Ok::<_, TCError>(Number::from(value)))
            )).map_err(TCError::from)).try_flatten().boxed());

        // The source owns its leases; termination must release it before outer drop.
        Ok(crate::stream::GuardedStream::new(stream, ()).boxed())
    }

    /// Ordered nonzero values from native occupied-block traversal.
    pub async fn sparse_elements(&self, id: TxnId) -> TCResult<SparseElements> {
        if !matches!(self.layout(), Layout::Sparse { .. }) {
            return Err(TCError::bad_request(
                "sparse iteration requires sparse Tensor",
            ));
        }

        let stream = dispatch!(&self.data, value => value.expression(id).await?.into_sparse_elements()?
            .map_ok(|(coord, value)| (coord, Number::from(value)))
            .map_err(TCError::from).boxed());

        // The source owns its leases; termination must release it before outer drop.
        Ok(crate::stream::GuardedStream::new(stream, ()).boxed())
    }

    /// Explicitly copy all logical values, including implicit sparse zeros.
    /// Use this when indexed sparse copying cannot preserve the source's order.
    /// The destination is independent; transaction visibility and source leases
    /// are acquired in the same way as [`Self::materialize`].
    pub async fn copy_values(&self, txn: &Txn, dir: DirLock<Txn::File>) -> TCResult<Self> {
        let capacity = fensor::MAX_BLOCK_CAPACITY;
        Ok(Self {
            data: dispatch!(@wrap &self.data, value => value.copy_values(txn.id(), dir, capacity).await?),
        })
    }

    pub async fn write_tensor(&self, txn: &Txn, source: Self) -> TCResult<()> {
        self.assign(txn, None, source).await
    }

    /// Assign a selection through its mutable owner. Persistent application callers
    /// must deliver this mutation through the owner's Chain with these bounds.
    /// Explicit gather selections remain read-only.
    pub async fn write_tensor_range(&self, txn: &Txn, range: Range, source: Self) -> TCResult<()> {
        self.assign(txn, Some(range), source).await
    }

    async fn assign(&self, txn: &Txn, range: Option<Range>, source: Self) -> TCResult<()> {
        dispatch_pair!(&self.data, source.data, target, source => target.assign(txn, range, source).await, Err(TCError::bad_request("Tensor dtype mismatch")))
    }

    pub async fn restore_from(&self, txn: &Txn, source: &Self) -> TCResult<()> {
        dispatch_pair!(&self.data, &source.data, target, source => target.restore_from(txn, source).await, Err(TCError::bad_request("Tensor dtype mismatch")))
    }

    pub async fn sync(&self) -> TCResult<()> {
        dispatch!(&self.data, value => value.sync().await)
    }

    pub async fn sync_all(&self) -> TCResult<()> {
        dispatch!(&self.data, value => value.sync_all().await)
    }

    tensor_transforms! {
        reshape(shape: Shape);
        broadcast(shape: Shape);
        slice(range: Range);
        transpose(permutation: Option<Axes>);
        flip(axis: usize);
        squeeze(axes: Axes);
        unsqueeze(axes: Axes);
    }
}

impl<Txn: StorageContext> Transact for Tensor<Txn> {
    async fn commit(&self, id: TxnId) -> TCResult<()> {
        dispatch!(&self.data, value => value.commit(id))
    }

    async fn rollback(&self, id: &TxnId) -> TCResult<()> {
        dispatch!(&self.data, value => value.rollback(*id))
    }

    async fn finalize(&self, id: &TxnId) -> TCResult<()> {
        dispatch!(&self.data, value => value.finalize(*id).await)
    }
}

mod codec;
mod hash;
mod ops;
pub(crate) mod route;
mod schema;
mod storage;
#[cfg(test)]
#[path = "../../tests/unit/tensor/tests.rs"]
mod tests;
mod typed;
mod view;
