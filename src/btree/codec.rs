use destream::{de, en};
use number_general::Number;
use safecast::{CastFrom, TryCastFrom};
use tc_value::{Value, ValueType};

use super::file::BTreeBase;
use super::{BTree, BTreeColumnSchema, BTreeSchema};

impl From<BTreeColumnSchema> for Value {
    fn from(column: BTreeColumnSchema) -> Self {
        let mut fields = vec![Value::from(column.name), Value::cast_from(column.dtype)];

        if let Some(size) = column.max_size {
            fields.push(Value::Number(size));
        }

        Value::Tuple(fields)
    }
}

impl TryCastFrom<Value> for BTreeColumnSchema {
    fn can_cast_from(value: &Value) -> bool {
        matches!(value, Value::Tuple(fields) if Self::can_cast_from(fields))
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(fields) => Self::opt_cast_from(fields),
            _ => None,
        }
    }
}

impl TryCastFrom<Vec<Value>> for BTreeColumnSchema {
    fn can_cast_from(fields: &Vec<Value>) -> bool {
        (2..=3).contains(&fields.len())
            && matches!(&fields[0], Value::String(_))
            && ValueType::can_cast_from(&fields[1])
            && fields.get(2).is_none_or(Number::can_cast_from)
    }

    fn opt_cast_from(fields: Vec<Value>) -> Option<Self> {
        if !(2..=3).contains(&fields.len()) {
            return None;
        }

        let mut fields = fields.into_iter();
        let Value::String(name) = fields.next()? else {
            return None;
        };

        let dtype = ValueType::opt_cast_from(fields.next()?)?;
        let max_size = match fields.next() {
            Some(Value::Number(size)) => Some(size),
            None => None,
            _ => return None,
        };

        Some(Self {
            name,
            dtype,
            max_size,
        })
    }
}

impl<'en> en::IntoStream<'en> for BTreeColumnSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        Value::from(self).into_stream(encoder)
    }
}

impl de::FromStream for BTreeColumnSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        // Tagged Value tuples belong to the Value codec, including its bounds.
        if decoder.peek_kind().await? == de::Kind::Map {
            return Self::try_cast_from(Value::from_stream((), decoder).await?, |_| {
                de::Error::custom("invalid BTree schema column [name, dtype, optional max_size]")
            });
        }

        struct Visitor;

        impl de::Visitor for Visitor {
            type Value = BTreeColumnSchema;

            fn expecting() -> &'static str {
                "BTree schema column [name, dtype, optional max_size]"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut fields = Vec::with_capacity(3);
                fields.push(seq.expect_next::<Value>(()).await?);
                fields.push(seq.expect_next::<Value>(()).await?);
                if let Some(size) = seq.next_element::<Value>(()).await? {
                    fields.push(size);
                    if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                        return Err(de::Error::custom("unexpected BTree schema column field"));
                    }
                }

                BTreeColumnSchema::try_cast_from(fields, |_| {
                    de::Error::custom(
                        "invalid BTree schema column [name, dtype, optional max_size]",
                    )
                })
            }
        }

        decoder.decode_seq(Visitor).await
    }
}

impl<Txn: crate::StorageContext> de::FromStream for BTree<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let mut payload = decoder.open_container(de::Kind::Seq, None).await?;
        if payload.slot() == de::Slot::End {
            return Err(de::Error::custom("BTree payload must be [schema, rows]"));
        }

        let schema = Vec::<BTreeColumnSchema>::from_stream((), decoder).await?;
        decoder.finish_child(&mut payload).await?;
        if payload.slot() == de::Slot::End {
            return Err(de::Error::custom("BTree payload must be [schema, rows]"));
        }

        let native_schema = BTreeSchema::try_from(schema).map_err(de::Error::custom)?;
        let tree = BTreeBase::unpublished(
            txn.subcontext_unique()
                .context()
                .await
                .map_err(de::Error::custom)?,
            native_schema,
        )
        .await
        .map_err(de::Error::custom)?;

        let mut rows = decoder.open_container(de::Kind::Seq, None).await?;
        while rows.slot() != de::Slot::End {
            let value = Value::from_stream((), decoder).await?;
            tree.load_literal_row(value)
                .await
                .map_err(de::Error::custom)?;

            decoder.finish_child(&mut rows).await?;
        }

        decoder.finish_child(&mut payload).await?;
        if payload.slot() != de::Slot::End {
            <de::IgnoredAny as de::FromStream>::from_stream((), decoder).await?;
            return Err(de::Error::custom("BTree payload must be [schema, rows]"));
        }

        tree.finish_creation().await.map_err(de::Error::custom)?;
        Ok(Self::File(tree))
    }
}
