//! Format-neutral Tensor payloads.

use destream::{de, en};
use futures::{StreamExt, TryStreamExt};
use number_general::Number;
use safecast::CastInto;

use super::{Data, NumberType, Tensor, TensorSchema, TensorView, Typed};
use crate::StorageContext;
use tc_value::Value;

impl<'en> en::IntoStream<'en> for TensorView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self.contents {
            super::view::Contents::Sparse(entries) => {
                let entries = entries.map_ok(|(coord, number)| {
                    Value::Tuple(vec![
                        Value::Tuple(coord.into_iter().map(Value::from).collect()),
                        Value::Number(number),
                    ])
                });

                (
                    Value::from(self.schema),
                    en::SeqStream::from(entries.boxed()),
                )
                    .into_stream(encoder)
            }
            super::view::Contents::Dense(values) => {
                (Value::from(self.schema), en::SeqStream::from(values)).into_stream(encoder)
            }
        }
    }
}

impl<Txn: StorageContext> de::FromStream for Tensor<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor<Txn>(Txn);

        impl<Txn: StorageContext> de::Visitor for Visitor<Txn> {
            type Value = Tensor<Txn>;

            fn expecting() -> &'static str {
                "a Tensor [schema, values] payload"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let schema = seq.expect_next::<TensorSchema>(()).await?;
                let dir = self
                    .0
                    .subcontext_unique()
                    .context()
                    .await
                    .map_err(de::Error::custom)?;

                let capacity = fensor::MAX_BLOCK_CAPACITY;
                let Values(tensor) = seq
                    .expect_next::<Values<Txn>>((dir, schema, capacity))
                    .await?;

                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected Tensor payload field"));
                }

                Ok(tensor)
            }
        }

        decoder.decode_seq(Visitor(txn)).await
    }
}

// Decode failures retain the decoder's original error; native failures use its
// normal conversion. Neither constructor needs to erase or spawn the input stream.
enum InputError<E> {
    Decode(E),
    Native(fensor::Error),
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
        }
    }
}

struct Values<Txn: StorageContext>(Tensor<Txn>);

impl<Txn: StorageContext> de::FromStream for Values<Txn> {
    type Context = (freqfs::DirLock<Txn::File>, TensorSchema, usize);

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor<Txn: StorageContext>(<Values<Txn> as de::FromStream>::Context);

        impl<Txn: StorageContext> de::Visitor for Visitor<Txn> {
            type Value = Values<Txn>;

            fn expecting() -> &'static str {
                "streamed Tensor values"
            }

            async fn visit_seq<A: de::SeqAccess>(self, seq: A) -> Result<Self::Value, A::Error> {
                let (dir, schema, capacity) = self.0;
                let data = dtype_factory!(schema.dtype(), T => {
                    let native = if schema.is_sparse() {
                        let rank = schema.native().shape().len();
                        let entries = futures::stream::try_unfold(seq, move |mut seq| async move {
                            let entry = seq.next_element::<Entry>(rank).await.map_err(InputError::Decode)?;
                            match entry {
                                Some(Entry(coord, number)) => {
                                    if number == Number::from(0u8) {
                                        return Err(InputError::Decode(de::Error::custom("sparse Tensor entries must be nonzero")));
                                    }
                                    Ok(Some(((coord, number.cast_into()), seq)))
                                }
                                None => Ok(None),
                            }
                        });
                        fensor::Tensor::<Txn::File, T>::from_sparse_elements(
                            dir, schema.native().clone(), schema.layout(), capacity, entries
                        ).await
                    } else {
                        let values = futures::stream::try_unfold(seq, |mut seq| async move {
                            let value = seq.next_element::<Number>(()).await.map_err(InputError::Decode)?;
                            Ok(value.map(|value| (value.cast_into(), seq)))
                        });
                        fensor::Tensor::<Txn::File, T>::from_values(
                            dir, schema.native().clone(), capacity, values
                        ).await
                    };
                    native.map_err(InputError::decode)?
                }, dtype => return Err(de::Error::custom(format!("unsupported Tensor dtype {dtype}"))));

                Ok(Values(Tensor { data }))
            }
        }

        decoder.decode_seq(Visitor::<Txn>(context)).await
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
        let dim = <u64 as de::FromStream>::from_stream((), decoder).await?;
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

impl de::FromStream for TensorSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
        require_field::<D::Error>(&sequence)?;
        let dtype = String::from_stream((), decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        let path = dtype
            .parse::<pathlink::PathBuf>()
            .map_err(de::Error::custom)?;

        let dtype = tc_value::number_type_from_path(&path)
            .ok_or_else(|| de::Error::custom("invalid Tensor dtype"))?;

        require_field::<D::Error>(&sequence)?;
        let shape = decode_dimensions(super::MAX_TENSOR_METADATA_RANK, decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        let sparse_axis = if sequence.slot() == de::Slot::End {
            None
        } else {
            let axis = Number::from_stream((), decoder).await?;
            decoder.finish_child(&mut sequence).await?;
            let axis = super::schema::dimension(axis).map_err(de::Error::custom)?;
            Some(usize::try_from(axis).map_err(de::Error::custom)?)
        };

        if sequence.slot() != de::Slot::End {
            de::IgnoredAny::from_stream((), decoder).await?;
            decoder.finish_child(&mut sequence).await?;
            return Err(de::Error::custom("unexpected Tensor schema field"));
        }

        TensorSchema::new(dtype, shape.into(), sparse_axis).map_err(de::Error::custom)
    }
}

struct Entry(Vec<u64>, Number);

impl de::FromStream for Entry {
    type Context = usize;

    async fn from_stream<D: de::Decoder>(rank: usize, decoder: &mut D) -> Result<Self, D::Error> {
        let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
        require_field::<D::Error>(&sequence)?;
        let coord = decode_dimensions(rank, decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        if coord.len() != rank {
            return Err(de::Error::custom("sparse coordinate rank mismatch"));
        }

        require_field::<D::Error>(&sequence)?;
        let value = Number::from_stream((), decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        if sequence.slot() != de::Slot::End {
            de::IgnoredAny::from_stream((), decoder).await?;
            decoder.finish_child(&mut sequence).await?;
            return Err(de::Error::custom("unexpected sparse entry field"));
        }

        Ok(Entry(coord, value))
    }
}
