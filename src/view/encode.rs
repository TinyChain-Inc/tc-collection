use destream::en::{self, EncodeMap};
use futures::StreamExt;
use safecast::CastFrom;
use tc_value::Value;
use tc_value::class::NativeClass;

use crate::{BTreeType, TableType, TensorType};
use super::{BTreeView, CollectionView, TableView};

impl<'en> en::IntoStream<'en> for BTreeView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let arity = self.schema.len();
        let keys = self.keys.map(move |row| {
            row.map(|row| {
                if arity == 1 {
                    row.into_iter().next().expect("validated unary BTree row")
                } else {
                    Value::Tuple(row)
                }
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
