//! Transactional Tensor values backed by fensor's native storage and expressions.

use std::fmt;

use fensor::TensorGeometry;
pub use fensor::{Axes, AxisRange, Layout, Range, Shape};
use freqfs::DirLock;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use number_general::{DType, Number, NumberClass};
use safecast::{CastFrom, CastInto};
use tc_error::{TCError, TCResult};
use tc_ir::{Transact, TxnId};
use tc_value::NumberType;

use crate::StorageContext;
pub use access::TensorAccess;
pub(crate) use schema::MAX_TENSOR_METADATA_RANK;
pub use schema::TensorSchema;
use storage::TensorBase;
pub use view::TensorView;

pub type SparseElements = BoxStream<'static, TCResult<(Vec<u64>, Number)>>;

fn checked_value<T: DType + CastFrom<Number>>(value: Number) -> TCResult<T> {
    T::dtype()
        .try_cast(value)
        .map(CastInto::cast_into)
        .map_err(TCError::bad_request)
}

fn complex_value(re: Number, im: Number) -> TCResult<Number> {
    let component = |number| {
        if !matches!(number, Number::Float(_) | Number::Int(_) | Number::UInt(_)) {
            return Err(TCError::bad_request("expected a real Tensor component"));
        }

        number_general::FloatType::F64
            .try_cast(number)
            .map(f64::from)
            .map_err(TCError::bad_request)
    };

    Ok(Number::from([component(re)?, component(im)?]))
}

#[macro_use]
mod macros;

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
            NumberType::Complex(number_general::ComplexType::C32), dense_c32;
        #[cfg(feature = "complex")]
        C64: fensor::complex::Complex64 =>
            NumberType::Complex(number_general::ComplexType::C64), dense_c64;
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
        dispatch!(self, value => hash::blocks(value.expression(id).await?.into_blocks()?).await)
    }

    pub fn shape(&self) -> &[u64] {
        dispatch!(self, value => value.shape())
    }

    pub fn layout(&self) -> Layout {
        dispatch!(self, value => value.layout())
    }

    pub fn size(&self) -> TCResult<u64> {
        dispatch!(self, value => value.size())
            .map_err(|_| TCError::bad_request("Tensor size overflows u64"))
    }

    pub fn number_type(&self) -> NumberType {
        dispatch!(self, value => value.dtype())
    }

    pub fn schema(&self) -> TCResult<TensorSchema> {
        TensorSchema::from_geometry(self.number_type(), self.shape().into(), self.layout())
    }

    pub fn is_persistent(&self) -> bool {
        dispatch!(self, value => value.is_persistent())
    }

    /// Return a read-only alias. Transforming or cloning it never grants ownership.
    pub(crate) fn into_read_only(self) -> Self {
        dispatch!(@wrap self, value => value.into_read_only())
    }

    /// Interpret implicit zeros as dense values without copying or persisting data.
    /// Explicit copying materializes the result in independent storage.
    pub async fn as_dense(&self, id: TxnId) -> TCResult<Self> {
        Ok(
            dispatch!(@wrap self, value => TensorAccess::computed(value.expression(id).await?.into_dense(), id)),
        )
    }

    pub async fn create(
        dir: DirLock<Txn::File>,
        schema: TensorSchema,
        max_capacity: usize,
    ) -> TCResult<Self> {
        let data = dtype_factory!(schema.dtype(), T => TensorBase::<Txn, T>::create(dir, &schema, max_capacity).await?, dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {dtype}"))));

        Ok(data)
    }

    pub async fn load(dir: DirLock<Txn::File>, schema: TensorSchema) -> TCResult<Self> {
        let tensor = dtype_factory!(schema.dtype(), T => TensorBase::<Txn, T>::load(dir).await?,
            dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {dtype}"))));

        if tensor.schema()? != schema {
            return Err(TCError::bad_request("Tensor storage schema mismatch"));
        }

        Ok(tensor)
    }

    dense_literals!();

    pub async fn fill(&self, txn: &Txn, number: Number) -> TCResult<()> {
        dispatch!(self, value => value.base()?.fill(txn, checked_value(number)?).await)
    }

    /// Fill a selection through its mutable owner, without exposing a writable view.
    /// Explicit gather selections remain read-only.
    pub async fn fill_range(&self, txn: &Txn, range: Range, number: Number) -> TCResult<()> {
        dispatch!(self, value => value.base()?.fill_range(txn, range, checked_value(number)?).await)
    }

    pub async fn read_value(&self, id: TxnId, coord: &[u64]) -> TCResult<Number> {
        dispatch!(self, value => Ok(Number::from(value.read_value(id, coord).await?)))
    }

    pub async fn write_value(&self, txn: &Txn, coord: &[u64], number: Number) -> TCResult<()> {
        dispatch!(self, value => value.base()?.write_value(txn, coord, checked_value(number)?).await)
    }

    pub async fn elements(&self, id: TxnId) -> TCResult<BoxStream<'static, TCResult<Number>>> {
        let stream = dispatch!(self, value => value.expression(id).await?.into_blocks()?
            .map(|batch| batch.map(|values| futures::stream::iter(
                values.into_iter().map(|value| Ok::<_, TCError>(Number::from(value)))
            )).map_err(TCError::from)).try_flatten().boxed());

        // The source owns its leases; termination must release it before outer drop.
        Ok(crate::stream::GuardedStream::new(stream, ()).boxed())
    }

    /// Ordered nonzero values, using occupied blocks when the geometry permits.
    pub async fn sparse_elements(&self, id: TxnId) -> TCResult<SparseElements> {
        let stream = dispatch!(self, value => value.expression(id).await?.into_sparse_elements()?
            .map_ok(|(coord, value)| (coord, Number::from(value)))
            .map_err(TCError::from).boxed());

        // The source owns its leases; termination must release it before outer drop.
        Ok(crate::stream::GuardedStream::new(stream, ()).boxed())
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
        dispatch_pair!(self, source, target, source => target.base()?.assign(txn, range, source).await, Err(TCError::bad_request("Tensor dtype mismatch")))
    }

    /// Complete accepted recovery work on an exclusively owned, unpublished value.
    /// A durable decision skips the callback; otherwise it runs once before commit.
    pub(crate) async fn recover<Replay, Replayed>(
        &self,
        id: TxnId,
        replay: Replay,
    ) -> tc_error::TCResult<()>
    where
        Replay: FnOnce() -> Replayed + Send,
        Replayed: std::future::Future<Output = tc_error::TCResult<()>> + Send,
    {
        dispatch!(self, value => value.base()?.recover(id, replay).await)
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
        dispatch!(self, value => value.base()?.commit(id).await)
    }

    async fn rollback(&self, id: &TxnId) -> TCResult<()> {
        dispatch!(self, value => value.base()?.rollback(id).await)
    }

    async fn finalize(&self, id: &TxnId) -> TCResult<()> {
        dispatch!(self, value => value.base()?.finalize(id).await)
    }
}

mod access;
mod base;
mod codec;
mod hash;
mod ops;
pub(crate) mod route;
mod schema;
mod storage;
mod view;

impl<Txn: crate::StorageContext> crate::CopyFrom<&Tensor<Txn>> for Tensor<Txn> {
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
        source: &Tensor<Txn>,
    ) -> tc_error::TCResult<Self> {
        Ok(
            dispatch!(@wrap source, value => TensorAccess::from_base(TensorBase::copy_from(txn, dir, value).await?)),
        )
    }
}

impl<Txn: crate::StorageContext> crate::Restore for Tensor<Txn> {
    type Txn = Txn;

    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        dispatch_pair!(self, source, target, source => target.base()?.restore(txn, source.base()?).await, Err(TCError::bad_request("Tensor dtype mismatch")))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/tests.rs"]
mod tests;
