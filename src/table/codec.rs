use destream::de;
use safecast::TryCastFrom;
use tc_value::{Value, ValueCollator};

use super::file::{TableFile, upsert};
use super::{Table, TableSchema};

#[derive(Clone, Debug)]
pub struct DecodedTablePayload<Txn: crate::StorageContext> {
    pub table: Table<Txn>,
}

struct Rows<Txn: crate::StorageContext>(std::marker::PhantomData<fn() -> Txn>);

impl<Txn: crate::StorageContext> de::FromStream for Rows<Txn> {
    type Context = TableFile<Txn::File>;

    async fn from_stream<D: de::Decoder>(
        table: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor<Txn: crate::StorageContext> {
            table: TableFile<Txn::File>,
        }

        impl<Txn: crate::StorageContext> de::Visitor for Visitor<Txn> {
            type Value = Rows<Txn>;

            fn expecting() -> &'static str {
                "a sequence of Table rows"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let schema = self.table.schema().clone();
                let key_len = schema.key().len();
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
                    upsert(
                        &self.table,
                        row[..key_len].to_vec(),
                        row[key_len..].to_vec(),
                    )
                    .await
                    .map_err(de::Error::custom)?;
                }
                Ok(Rows(std::marker::PhantomData))
            }
        }

        decoder.decode_seq(Visitor::<Txn> { table }).await
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

        struct Visitor<Txn: crate::StorageContext> {
            dir: freqfs::DirLock<Txn::File>,
            txn: std::marker::PhantomData<fn() -> Txn>,
        }

        impl<Txn: crate::StorageContext> de::Visitor for Visitor<Txn> {
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
                let table = b_table::TableLock::create(schema, ValueCollator::default(), self.dir)
                    .map_err(de::Error::custom)?;

                seq.next_element::<Rows<Txn>>(table.clone())
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
