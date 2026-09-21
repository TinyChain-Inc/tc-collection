use destream::{de, en};
use number_general::Number;
use safecast::TryCastFrom;
use tc_value::class::NativeClass;
use tc_value::{Value, ValueType};

use super::{BTree, BTreeSchema};

#[derive(Clone, Debug)]
pub struct BTreeColumnSchema {
    pub name: String,
    pub dtype: ValueType,
    pub max_size: Option<Number>,
}

impl From<BTreeColumnSchema> for Value {
    fn from(column: BTreeColumnSchema) -> Self {
        let mut fields = vec![
            Value::from(column.name),
            Value::from(column.dtype.path().to_string()),
        ];
        if let Some(size) = column.max_size {
            fields.push(Value::Number(size));
        }

        Value::Tuple(fields)
    }
}

impl TryCastFrom<Value> for BTreeColumnSchema {
    fn can_cast_from(value: &Value) -> bool {
        Self::opt_cast_from(value.clone()).is_some()
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        let Value::Tuple(fields) = value else {
            return None;
        };
        if !(2..=3).contains(&fields.len()) {
            return None;
        }

        let mut fields = fields.into_iter();
        let Value::String(name) = fields.next()? else {
            return None;
        };
        let Value::String(dtype) = fields.next()? else {
            return None;
        };
        let path = dtype.as_str().parse::<pathlink::PathBuf>().ok()?;
        let max_size = match fields.next() {
            Some(Value::Number(size)) => Some(size),
            None => None,
            _ => return None,
        };

        Some(Self {
            name: name.to_string(),
            dtype: ValueType::from_path(&path)?,
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
        Self::try_cast_from(Value::from_stream((), decoder).await?, |_| {
            de::Error::custom("invalid BTree schema column [name, dtype, optional max_size]")
        })
    }
}

impl TryCastFrom<Vec<BTreeColumnSchema>> for BTreeSchema {
    fn can_cast_from(columns: &Vec<BTreeColumnSchema>) -> bool {
        !columns.is_empty()
    }

    fn opt_cast_from(columns: Vec<BTreeColumnSchema>) -> Option<Self> {
        if columns.is_empty() {
            return None;
        }

        Some(Self::from_key_types(
            columns.into_iter().map(|column| column.dtype).collect(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct DecodedBTreePayload<Txn: crate::StorageContext> {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree<Txn>,
}

struct BTreeRows<Txn: crate::StorageContext>(std::marker::PhantomData<fn() -> Txn>);

struct BTreeRowsContext<Txn: crate::StorageContext> {
    btree: BTree<Txn>,
}

fn decode_err(action: &str, err: impl std::fmt::Display) -> String {
    format!("{action}: {err}")
}

impl<Txn: crate::StorageContext> de::FromStream for BTreeRows<Txn> {
    type Context = BTreeRowsContext<Txn>;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct RowsVisitor<Txn: crate::StorageContext> {
            btree: BTree<Txn>,
        }

        impl<Txn: crate::StorageContext> de::Visitor for RowsVisitor<Txn> {
            type Value = BTreeRows<Txn>;

            fn expecting() -> &'static str {
                "a list of BTree rows"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                while let Some(row_value) = seq.next_element::<Value>(()).await? {
                    self.btree
                        .load_literal_row(row_value)
                        .await
                        .map_err(|err| {
                            de::Error::custom(decode_err("failed to load BTree literal row", err))
                        })?;
                }

                Ok(BTreeRows(std::marker::PhantomData))
            }
        }

        decoder
            .decode_seq(RowsVisitor {
                btree: context.btree,
            })
            .await
    }
}

impl<Txn: crate::StorageContext> de::FromStream for DecodedBTreePayload<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let txn = context.subcontext_unique();
        let persistent_dir = txn.context().await.map_err(|err| {
            de::Error::custom(decode_err("failed to allocate BTree literal", err))
        })?;

        struct PayloadVisitor<Txn: crate::StorageContext> {
            persistent_dir: freqfs::DirLock<Txn::File>,
            txn: std::marker::PhantomData<fn() -> Txn>,
        }

        impl<Txn: crate::StorageContext> de::Visitor for PayloadVisitor<Txn> {
            type Value = DecodedBTreePayload<Txn>;

            fn expecting() -> &'static str {
                "a BTree payload [schema, rows]"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let schema = seq
                    .next_element::<Vec<BTreeColumnSchema>>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing BTree schema"))?;

                let btree_schema = BTreeSchema::try_cast_from(schema.clone(), |schema| {
                    de::Error::custom(format!("invalid BTree schema: {schema:?}"))
                })?;
                let btree = BTree::with_schema(self.persistent_dir, btree_schema);
                seq.next_element::<BTreeRows<Txn>>(BTreeRowsContext {
                    btree: btree.clone(),
                })
                .await?
                .ok_or_else(|| de::Error::custom("missing BTree rows"))?;

                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("BTree payload must be [schema, rows]"));
                }

                Ok(DecodedBTreePayload { schema, btree })
            }
        }

        decoder
            .decode_seq(PayloadVisitor {
                persistent_dir,
                txn: std::marker::PhantomData,
            })
            .await
    }
}
