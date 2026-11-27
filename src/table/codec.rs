use destream::de;
use tc_value::Value;

use super::{PersistentTable, Table, TableSchema};

impl<Txn: crate::StorageContext> de::FromStream for Table<Txn> {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let mut payload = decoder.open_container(de::Kind::Seq, None).await?;
        if payload.slot() == de::Slot::End {
            return Err(de::Error::custom("Table payload must be [schema, rows]"));
        }

        let value = Value::from_stream((), decoder).await?;
        decoder.finish_child(&mut payload).await?;
        if payload.slot() == de::Slot::End {
            return Err(de::Error::custom("Table payload must be [schema, rows]"));
        }

        let schema = TableSchema::try_from(value).map_err(de::Error::custom)?;
        let table = PersistentTable::unpublished(
            txn.subcontext_unique()
                .context()
                .await
                .map_err(de::Error::custom)?,
            schema,
        )
        .await
        .map_err(de::Error::custom)?;

        let mut rows = decoder.open_container(de::Kind::Seq, None).await?;
        while rows.slot() != de::Slot::End {
            let Value::Tuple(row) = Value::from_stream((), decoder).await? else {
                return Err(de::Error::custom("table row must be a tuple"));
            };

            table
                .load_literal_row(row)
                .await
                .map_err(de::Error::custom)?;

            decoder.finish_child(&mut rows).await?;
        }

        decoder.finish_child(&mut payload).await?;
        if payload.slot() != de::Slot::End {
            <de::IgnoredAny as de::FromStream>::from_stream((), decoder).await?;
            return Err(de::Error::custom("Table payload must be [schema, rows]"));
        }

        table.finish_creation().await.map_err(de::Error::custom)?;
        Ok(Table::File(table))
    }
}
