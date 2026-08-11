use destream::de;
use safecast::TryCastFrom;
use tc_value::{Value, ValueCollator};

use super::{LocalTable, Table, TableSchema};

#[derive(Clone, Debug)]
pub struct DecodedTablePayload<Txn> {
    pub table: Table<Txn>,
}

struct Rows;

impl de::FromStream for Rows {
    type Context = LocalTable;

    async fn from_stream<D: de::Decoder>(
        table: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor {
            table: LocalTable,
        }

        impl de::Visitor for Visitor {
            type Value = Rows;

            fn expecting() -> &'static str {
                "a sequence of Table rows"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let schema = self.table.schema().clone();
                let key_len = schema.key().len();
                let mut table = self.table.write().await;
                while let Some(row) = seq.next_element::<Value>(()).await? {
                    let Value::Tuple(row) = row else {
                        return Err(de::Error::custom("table row must be a tuple"));
                    };
                    if row.len() != schema.column_count() {
                        return Err(de::Error::custom(format!(
                            "table row has {} columns but schema has {}",
                            row.len(),
                            schema.column_count()
                        )));
                    }
                    table
                        .upsert(row[..key_len].to_vec(), row[key_len..].to_vec())
                        .await
                        .map_err(de::Error::custom)?;
                }
                Ok(Rows)
            }
        }

        decoder.decode_seq(Visitor { table }).await
    }
}

impl<Txn: crate::StorageContext> de::FromStream for DecodedTablePayload<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let txn = txn.subcontext_unique();
        let dir = txn.context().await.map_err(de::Error::custom)?;

        struct Visitor<Txn> {
            dir: freqfs::DirLock<crate::PersistentFile>,
            txn: std::marker::PhantomData<fn() -> Txn>,
        }

        impl<Txn> de::Visitor for Visitor<Txn> {
            type Value = DecodedTablePayload<Txn>;

            fn expecting() -> &'static str {
                "a Table payload [schema, rows]"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let schema = seq
                    .next_element::<Value>(())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing Table schema"))?;
                let schema = TableSchema::try_cast_from(schema, |schema| {
                    de::Error::custom(format!("invalid Table schema: {schema:?}"))
                })?;
                let table = LocalTable::create(schema, ValueCollator::default(), self.dir)
                    .map_err(de::Error::custom)?;

                seq.next_element::<Rows>(table.clone())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing Table rows"))?;
                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("Table payload must be [schema, rows]"));
                }

                Ok(DecodedTablePayload {
                    table: Table::Local(table),
                })
            }
        }

        decoder
            .decode_seq(Visitor {
                dir,
                txn: std::marker::PhantomData,
            })
            .await
    }
}
