use destream::de;
use safecast::TryCastFrom;
use tc_value::Value;

use super::{PersistentTable, TableSchema};

#[derive(Clone, Debug)]
pub struct DecodedTablePayload<Txn> {
    pub table: PersistentTable<Txn>,
}

struct Rows<Txn>(std::marker::PhantomData<fn() -> Txn>);

impl<Txn> de::FromStream for Rows<Txn> {
    type Context = PersistentTable<Txn>;

    async fn from_stream<D: de::Decoder>(
        table: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor<Txn> {
            table: PersistentTable<Txn>,
        }

        impl<Txn> de::Visitor for Visitor<Txn> {
            type Value = Rows<Txn>;

            fn expecting() -> &'static str {
                "a sequence of Table rows"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                while let Some(row) = seq.next_element::<Value>(()).await? {
                    self.table
                        .load_literal_row(row)
                        .await
                        .map_err(de::Error::custom)?;
                }
                Ok(Rows(std::marker::PhantomData))
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
                let table = PersistentTable::literal(self.dir, schema);

                seq.next_element::<Rows<Txn>>(table.clone())
                    .await?
                    .ok_or_else(|| de::Error::custom("missing Table rows"))?;
                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("Table payload must be [schema, rows]"));
                }

                Ok(DecodedTablePayload { table })
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
