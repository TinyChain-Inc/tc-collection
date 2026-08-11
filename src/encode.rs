use destream::en::{self, EncodeMap};
use futures::StreamExt;
use safecast::CastFrom;
use tc_error::TCError;
use tc_ir::NativeClass;
use tc_value::Value;

use crate::view::{BTreeView, CollectionView, TableView};
use crate::{BTreeType, TableType, TensorType};

impl<'en> en::IntoStream<'en> for BTreeView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let arity = self.arity;
        let keys = self.keys.map(move |row| {
            row.and_then(|row| {
                if row.len() != arity {
                    return Err(TCError::internal(format!(
                        "BTree row arity {} does not match schema arity {arity}",
                        row.len()
                    )));
                }

                Ok(if arity == 1 {
                    row.into_iter().next().expect("unary BTree row")
                } else {
                    Value::Tuple(row)
                })
            })
        });

        (self.schema, en::SeqStream::from(keys)).into_stream(encoder)
    }
}

impl<'en> en::IntoStream<'en> for TableView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (
            Value::cast_from(self.schema),
            en::SeqStream::from(self.rows),
        )
            .into_stream(encoder)
    }
}

impl<'en> en::IntoStream<'en> for CollectionView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        match self {
            Self::BTree(view) => map.encode_entry(BTreeType.path().to_string(), view)?,
            Self::Table(view) => map.encode_entry(TableType.path().to_string(), view)?,
            Self::Tensor(tensor) => map.encode_entry(TensorType.path().to_string(), tensor)?,
        }
        map.end()
    }
}
