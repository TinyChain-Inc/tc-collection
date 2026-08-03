use destream::{
    de,
    en::{self, EncodeSeq},
};
use number_general::Number;
use tc_error::TCError;
use tc_ir::NativeClass;
use tc_value::{Value, ValueType};

use super::{BTree, PersistentFile};

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

#[derive(Clone)]
pub struct BTreeDecodeContext {
    persistent_dir: freqfs::DirLock<PersistentFile>,
    txn_root: freqfs::DirLock<PersistentFile>,
}

impl BTreeDecodeContext {
    pub fn new(
        persistent_dir: freqfs::DirLock<PersistentFile>,
        txn_root: freqfs::DirLock<PersistentFile>,
    ) -> Self {
        Self {
            persistent_dir,
            txn_root,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodedBTreePayload {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree,
}

struct BTreeRows {
    rows: Vec<Vec<Value>>,
}

fn decode_err(action: &str, err: impl std::fmt::Display) -> String {
    format!("{action}: {err}")
}

fn btree_schema_value_types(schema: &[BTreeColumnSchema]) -> Result<Vec<ValueType>, TCError> {
    if schema.is_empty() {
        return Err(tc_error::bad_request!(
            "BTree schema must have at least one column"
        ));
    }

    Ok(schema.iter().map(|column| column.dtype.clone()).collect())
}

fn normalize_btree_row(row: Value, key_types: &[ValueType]) -> Result<Vec<Value>, TCError> {
    let key_arity = key_types.len();

    let values = if key_arity == 1 {
        vec![row]
    } else {
        match row {
            Value::Tuple(items) => {
                if items.len() == key_arity {
                    items
                } else {
                    Err(tc_error::bad_request!(
                        "BTree row arity {} does not match schema arity {}",
                        items.len(),
                        key_arity
                    ))?
                }
            }
            other => Err(tc_error::bad_request!(
                "BTree row must be a tuple of length {} but got {:?}",
                key_arity,
                other
            ))?,
        }
    };

    for (i, (value, expected)) in values.iter().zip(key_types.iter()).enumerate() {
        let actual = value.class();
        if &actual != expected {
            return Err(tc_error::bad_request!(
                "BTree row column {i} expected {:?} but got {:?}",
                expected,
                actual
            ));
        }
    }

    Ok(values)
}

impl de::FromStream for BTreeRows {
    type Context = Vec<ValueType>;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct RowsVisitor {
            key_types: Vec<ValueType>,
        }

        impl de::Visitor for RowsVisitor {
            type Value = BTreeRows;

            fn expecting() -> &'static str {
                "a list of BTree rows"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let key_types = self.key_types;
                let mut rows = Vec::new();

                while let Some(row_value) = seq.next_element::<Value>(()).await? {
                    let row =
                        normalize_btree_row(row_value, &key_types).map_err(de::Error::custom)?;
                    rows.push(row);
                }

                Ok(BTreeRows { rows })
            }
        }

        decoder.decode_seq(RowsVisitor { key_types: context }).await
    }
}

impl de::FromStream for DecodedBTreePayload {
    type Context = BTreeDecodeContext;

    async fn from_stream<D: de::Decoder>(
        context: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct PayloadVisitor {
            persistent_dir: freqfs::DirLock<PersistentFile>,
            txn_root: freqfs::DirLock<PersistentFile>,
        }

        impl de::Visitor for PayloadVisitor {
            type Value = DecodedBTreePayload;

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

                let key_types = btree_schema_value_types(&schema).map_err(de::Error::custom)?;

                let btree =
                    BTree::with_key_types(self.persistent_dir, self.txn_root, key_types.clone());
                let rows = seq
                    .next_element::<BTreeRows>(key_types)
                    .await?
                    .ok_or_else(|| de::Error::custom("missing BTree rows"))?;

                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("BTree payload must be [schema, rows]"));
                }

                btree.load_literal_rows(rows.rows).await.map_err(|err| {
                    de::Error::custom(decode_err("failed to load BTree literal rows", err))
                })?;

                Ok(DecodedBTreePayload {
                    schema,
                    btree,
                })
            }
        }

        decoder
            .decode_seq(PayloadVisitor {
                persistent_dir: context.persistent_dir,
                txn_root: context.txn_root,
            })
            .await
    }
}
