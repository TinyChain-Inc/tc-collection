//! In-memory Tensor collection primitives and format-neutral tensor encoding.

use destream::{
    IntoStream, de,
    en::{self, EncodeSeq},
};
use ha_ndarray::{ArrayBuf, Buffer, NDArray, NDArrayRead};
use number_general::{FloatType, Number, UIntType};
use tc_value::{NumberType, number_type_path};

mod add;
mod broadcast_reduce;
mod core;
mod dtype;
mod matmul;
mod transpose;
mod wire;

pub use ha_ndarray::{AxisRange, Range};
/// In-memory dense tensor data.
#[derive(Clone, Debug)]
pub enum Tensor {
    F32(Box<ArrayBuf<f32, Buffer<f32>>>),
    F64(Box<ArrayBuf<f64, Buffer<f64>>>),
    U64(Box<ArrayBuf<u64, Buffer<u64>>>),
}

#[derive(Clone, Debug)]
pub enum TensorReduceResult {
    Scalar(Number),
    Tensor(Tensor),
}

impl<'en> en::IntoStream<'en> for Tensor {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut seq = encoder.encode_seq(Some(2))?;
        match self {
            Tensor::F32(array) => {
                let schema = (
                    number_type_path(&NumberType::Float(FloatType::F32)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(en::Error::custom)?
                    .to_slice()
                    .map_err(en::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
            Tensor::F64(array) => {
                let schema = (
                    number_type_path(&NumberType::Float(FloatType::F64)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(en::Error::custom)?
                    .to_slice()
                    .map_err(en::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
            Tensor::U64(array) => {
                let schema = (
                    number_type_path(&NumberType::UInt(UIntType::U64)).to_string(),
                    array
                        .shape()
                        .iter()
                        .map(|dim| *dim as u64)
                        .collect::<Vec<_>>(),
                );
                seq.encode_element(schema)?;
                let values = array
                    .buffer()
                    .map_err(en::Error::custom)?
                    .to_slice()
                    .map_err(en::Error::custom)?
                    .into_vec();
                seq.encode_element(values)?;
            }
        }
        seq.end()
    }
}

impl<'en> en::ToStream<'en> for Tensor {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.clone().into_stream(encoder)
    }
}

impl de::FromStream for Tensor {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct TensorVisitor;

        impl de::Visitor for TensorVisitor {
            type Value = Tensor;

            fn expecting() -> &'static str {
                "a TinyChain tensor payload"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let (dtype_path, shape): (String, Vec<u64>) = seq
                    .next_element(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing tensor schema"))?;
                let dtype = wire::tensor_dtype_from_wire(&dtype_path).ok_or_else(|| {
                    de::Error::invalid_value(
                        dtype_path,
                        "a TinyChain numeric type path for tensor dtype",
                    )
                })?;
                let shape = wire::coerce_shape(shape).map_err(de::Error::custom)?;
                let values = seq
                    .next_element::<Vec<Number>>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing tensor values"))?;

                wire::tensor_from_parts(dtype, shape, values).map_err(de::Error::custom)
            }
        }

        decoder.decode_seq(TensorVisitor).await
    }
}

pub use add::{broadcast_add, exact_shape_add};
pub use broadcast_reduce::broadcast_reduce_sum;
pub use dtype::{TensorDtypeGuard, TensorOpError, tensor_op_result};
pub use matmul::batched_matmul;
pub use transpose::{tensor_transpose, transpose_output_shape};
