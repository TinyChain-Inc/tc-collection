use destream::{
    de,
    en::{self, EncodeSeq},
};
use number_general::Number;
use safecast::TryCastFrom;
use tc_ir::NativeClass;
use tc_value::{Value, ValueType};

use super::{BTree, BTreeSchema, PersistentFile};

#[derive(Clone, Debug)]
pub struct BTreeColumnSchema {
    pub name: String,
    pub dtype: ValueType,
    pub max_size: Option<Number>,
}

impl<'en> en::IntoStream<'en> for BTreeColumnSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut seq = encoder.encode_seq(Some(if self.max_size.is_some() { 3 } else { 2 }))?;
        seq.encode_element(self.name)?;
        seq.encode_element(self.dtype.path().to_string())?;
        if let Some(max_size) = self.max_size {
            seq.encode_element(max_size)?;
        }
        seq.end()
    }
}

impl de::FromStream for BTreeColumnSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct ColumnVisitor;

        impl de::Visitor for ColumnVisitor {
            type Value = BTreeColumnSchema;

            fn expecting() -> &'static str {
                "a BTree schema column [name, dtype, optional max_size]"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let name = seq
                    .next_element::<String>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing BTree schema column name"))?;

                let dtype = seq
                    .next_element::<String>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing BTree schema column dtype"))?;

                let path = dtype.parse::<pathlink::PathBuf>().map_err(|err| {
                    de::Error::custom(format!("invalid BTree schema dtype {dtype:?}: {err}"))
                })?;

                let dtype = ValueType::from_path(path.as_ref()).ok_or_else(|| {
                    de::Error::custom(format!(
                        "unsupported BTree schema dtype path {dtype}; expected a /state/scalar/value/... URI"
                    ))
                })?;

                let max_size = seq.next_element::<Number>(()).await?;

                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom(
                        "BTree schema column entries must have length 2 or 3",
                    ));
                }

                Ok(BTreeColumnSchema {
                    name,
                    dtype,
                    max_size,
                })
            }
        }

        decoder.decode_seq(ColumnVisitor).await
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
pub struct DecodedBTreePayload<Txn> {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree<Txn>,
}

struct BTreeRows<Txn>(std::marker::PhantomData<fn() -> Txn>);

struct BTreeRowsContext<Txn> {
    btree: BTree<Txn>,
}

fn decode_err(action: &str, err: impl std::fmt::Display) -> String {
    format!("{action}: {err}")
}

impl<Txn> de::FromStream for BTreeRows<Txn> {
    type Context = BTreeRowsContext<Txn>;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct RowsVisitor<Txn> {
            btree: BTree<Txn>,
        }

        impl<Txn> de::Visitor for RowsVisitor<Txn> {
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

        struct PayloadVisitor<Txn> {
            persistent_dir: freqfs::DirLock<PersistentFile>,
            txn: std::marker::PhantomData<fn() -> Txn>,
        }

        impl<Txn> de::Visitor for PayloadVisitor<Txn> {
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
                let btree = BTree::literal(self.persistent_dir, btree_schema);
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
