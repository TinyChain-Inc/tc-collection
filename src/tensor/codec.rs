//! Format-neutral Tensor payloads.

use destream::de;
use number_general::Number;
use tc_error::TCError;
use tc_value::Value;

use super::{NumberType, Tensor, TensorAccess, TensorBase, TensorSchema, checked_value};
use crate::StorageContext;

impl<Txn: StorageContext> de::FromStream for Tensor<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let mut payload = decoder.open_container(de::Kind::Seq, None).await?;
        require_field::<D::Error>(&payload)?;
        let mut schema = TensorSchema::from_stream((), decoder).await?;
        decoder.finish_child(&mut payload).await?;
        require_field::<D::Error>(&payload)?;
        let mut rows = decoder.open_container(de::Kind::Seq, None).await?;
        let mut first_pair =
            if rows.slot() != de::Slot::End && decoder.peek_kind().await? == de::Kind::Seq {
                Some(decoder.open_container(de::Kind::Seq, Some(2)).await?)
            } else {
                None
            };

        // A sparse pair starts with a coordinate sequence. A complex number
        // starts with a real component. Only sparse contents may be empty.
        let sparse = rows.slot() == de::Slot::End
            || (first_pair
                .as_ref()
                .is_some_and(|pair| pair.slot() != de::Slot::End)
                && decoder.peek_kind().await? == de::Kind::Seq);
        let first_value = if sparse {
            schema = schema.into_default_sparse();
            None
        } else if let Some(pair) = first_pair.take() {
            let value = decode_complex(pair, decoder).await?;
            decoder.finish_child(&mut rows).await?;
            Some(value)
        } else {
            None
        };

        Ok(dtype_factory!(schema.dtype(), T => {
            let native = if schema.is_sparse() {
                let rank = schema.shape().len();
                let entries = futures::stream::try_unfold(
                    (payload, rows, &mut *decoder, first_pair),
                    move |(payload, mut rows, decoder, first_pair)| async move {
                        if rows.slot() == de::Slot::End {
                            finish_payload(payload, decoder).await.map_err(InputError::Decode)?;
                            return Ok(None);
                        }

                        let pair = match first_pair {
                            Some(pair) => pair,
                            None => decoder.open_container(de::Kind::Seq, Some(2)).await.map_err(InputError::Decode)?,
                        };
                        let (coord, number) = decode_entry(rank, pair, decoder)
                            .await
                            .map_err(InputError::Decode)?;

                        decoder.finish_child(&mut rows).await.map_err(InputError::Decode)?;
                        Ok(Some(((coord, checked_value::<T>(number)?), (payload, rows, decoder, None))))
                    }
                );

                TensorBase::<Txn, T>::from_sparse_entries(&txn, schema, entries).await
            } else {
                let values = futures::stream::try_unfold(
                    (payload, rows, &mut *decoder, first_value),
                    |(payload, mut rows, decoder, first_value)| async move {
                        if let Some(number) = first_value {
                            return Ok(Some((checked_value::<T>(number)?, (payload, rows, decoder, None))));
                        }

                        if rows.slot() == de::Slot::End {
                            finish_payload(payload, decoder).await.map_err(InputError::Decode)?;
                            return Ok(None);
                        }

                        let number = decode_number(decoder)
                            .await
                            .map_err(InputError::Decode)?;

                        decoder.finish_child(&mut rows).await.map_err(InputError::Decode)?;
                        Ok(Some((checked_value::<T>(number)?, (payload, rows, decoder, None))))
                    }
                );

                TensorBase::<Txn, T>::from_values(&txn, schema, values).await
            };

            native.map_err(InputError::decode)?
        }, dtype => return Err(de::Error::custom(format!("unsupported Tensor dtype {dtype}")))))
    }
}

// Decode failures retain the decoder's original error; native failures use its
// normal conversion. Neither constructor needs to erase or spawn the input stream.
enum InputError<E> {
    Decode(E),
    Native(fensor::Error),
    Collection(TCError),
}

impl<E> From<TCError> for InputError<E> {
    fn from(error: TCError) -> Self {
        Self::Collection(error)
    }
}

impl<E> From<fensor::Error> for InputError<E> {
    fn from(error: fensor::Error) -> Self {
        Self::Native(error)
    }
}

impl<E: de::Error> InputError<E> {
    fn decode(self) -> E {
        match self {
            Self::Decode(error) => error,
            Self::Native(error) => E::custom(error),
            Self::Collection(error) => E::custom(error),
        }
    }
}

async fn finish_payload<D: de::Decoder>(
    mut payload: de::Container,
    decoder: &mut D,
) -> Result<(), D::Error> {
    decoder.finish_child(&mut payload).await?;
    if payload.slot() == de::Slot::End {
        Ok(())
    } else {
        Err(de::Error::custom("unexpected Tensor payload field"))
    }
}

// Shapes and sparse coordinates are bounded protocol metadata. Never reserve
// from a decoder's unchecked sequence-size hint.
async fn decode_dimensions<D: de::Decoder>(
    limit: usize,
    decoder: &mut D,
) -> Result<Vec<u64>, D::Error> {
    let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
    let mut dims = Vec::new();
    while sequence.slot() != de::Slot::End {
        let dim = super::schema::dimension(decoder.decode_any(NumericLeaf).await?)
            .map_err(de::Error::custom)?;
        decoder.finish_child(&mut sequence).await?;
        if dims.len() == limit {
            return Err(de::Error::custom("too many Tensor dimensions"));
        }

        dims.push(dim);
    }

    Ok(dims)
}

fn require_field<E: de::Error>(sequence: &de::Container) -> Result<(), E> {
    if sequence.slot() == de::Slot::End {
        Err(E::custom("expected sequence element is missing"))
    } else {
        Ok(())
    }
}

/// Read the shared logical wire schema before inspecting the contents.
impl de::FromStream for TensorSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
        require_field::<D::Error>(&sequence)?;
        let dtype = String::from_stream((), decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        let dtype = super::schema::dtype(Value::from(dtype)).map_err(de::Error::custom)?;

        require_field::<D::Error>(&sequence)?;
        let shape = decode_dimensions(super::MAX_TENSOR_METADATA_RANK, decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        if sequence.slot() != de::Slot::End {
            de::IgnoredAny::from_stream((), decoder).await?;
            decoder.finish_child(&mut sequence).await?;
            return Err(de::Error::custom("unexpected Tensor schema field"));
        }

        TensorSchema::logical(dtype, shape.into(), false).map_err(de::Error::custom)
    }
}

async fn decode_entry<D: de::Decoder>(
    rank: usize,
    mut sequence: de::Container,
    decoder: &mut D,
) -> Result<(Vec<u64>, Number), D::Error> {
    require_field::<D::Error>(&sequence)?;
    let coord = decode_dimensions(rank, decoder).await?;
    decoder.finish_child(&mut sequence).await?;
    if coord.len() != rank {
        return Err(de::Error::custom("sparse coordinate rank mismatch"));
    }

    require_field::<D::Error>(&sequence)?;
    let value = decode_number(decoder).await?;
    decoder.finish_child(&mut sequence).await?;
    require_pair_end::<D::Error>(&sequence)?;

    Ok((coord, value))
}

async fn decode_number<D: de::Decoder>(decoder: &mut D) -> Result<Number, D::Error> {
    match decoder.peek_kind().await? {
        de::Kind::Seq => {
            let pair = decoder.open_container(de::Kind::Seq, Some(2)).await?;
            decode_complex(pair, decoder).await
        }
        de::Kind::Leaf => decoder.decode_any(NumericLeaf).await,
        de::Kind::Map => Err(de::Error::custom("expected numeric Tensor value")),
    }
}

async fn decode_complex<D: de::Decoder>(
    mut pair: de::Container,
    decoder: &mut D,
) -> Result<Number, D::Error> {
    require_field::<D::Error>(&pair)?;
    let re = decoder.decode_any(NumericLeaf).await?;
    decoder.finish_child(&mut pair).await?;
    require_field::<D::Error>(&pair)?;
    let im = decoder.decode_any(NumericLeaf).await?;
    decoder.finish_child(&mut pair).await?;
    require_pair_end::<D::Error>(&pair)?;
    super::complex_value(re, im).map_err(de::Error::custom)
}

// The decoder widens smaller real primitives losslessly through the visitor
// defaults. Keep typed conversion separate from recognition of a numeric token.
struct NumericLeaf;

impl de::Visitor for NumericLeaf {
    type Value = Number;

    fn expecting() -> &'static str {
        "a numeric Tensor value"
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Number, E> {
        Ok(value.into())
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Number, E> {
        Ok(value.into())
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Number, E> {
        Ok(value.into())
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Number, E> {
        Ok(value.into())
    }
}

fn require_pair_end<E: de::Error>(pair: &de::Container) -> Result<(), E> {
    if pair.slot() == de::Slot::End {
        Ok(())
    } else {
        Err(E::custom("unexpected Tensor pair field"))
    }
}
