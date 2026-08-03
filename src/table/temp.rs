//! A temporary, non-transactional `Table` backed by `b-table::TableLock`.
//!
//! A `TempTable` exists only for the duration of a single transaction (or
//! request). It supports the same public read/write methods as a
//! transactional [`PersistentTable`] but has no commit/rollback/finalize
//! lifecycle — it is not a member of a hosted `Service`.
//!
//! Used by the `create` and `copy_from` static route handlers to construct
//! a table that the host can then route against within the same request.

use std::fmt;

use b_table::{Range, Row, TableLock};
use collate::Collator;
use freqfs::DirLock;
use futures::TryStreamExt;
use tc_value::Value;

use super::schema::{TableIndexSchema, TableSchema};
use crate::PersistentFile;

type TempVersion = TableLock<TableSchema, TableIndexSchema, Collator<Value>, PersistentFile>;

#[derive(Clone)]
pub struct TempTable {
    table: TempVersion,
}

impl fmt::Debug for TempTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TempTable")
            .field("schema", self.schema())
            .finish()
    }
}

impl TempTable {
    /// Create a new temporary table with the given schema.
    pub fn create(
        schema: TableSchema,
        dir: DirLock<PersistentFile>,
    ) -> std::io::Result<Self> {
        let collator = Collator::<Value>::default();
        let table = TableLock::create(schema, collator, dir)?;
        Ok(Self { table })
    }

    pub fn schema(&self) -> &TableSchema {
        self.table.schema()
    }

    /// Read a single row by key.
    pub async fn read_row(&self, key: &[Value]) -> Option<Row<Value>> {
        let schema = self.schema().clone();
        let range = schema.range_from_key(key).ok()?;
        let view = self.table.read().await;
        let mut rows = view.rows(range, &[], false, None).await.ok()?;
        rows.try_next().await.ok().flatten()
    }

    /// Check if a row with the given key exists.
    pub async fn contains_row(&self, key: &[Value]) -> bool {
        self.read_row(key).await.is_some()
    }

    /// Count all rows in the table.
    pub async fn count(&self) -> u64 {
        let view = self.table.read().await;
        view.count(Range::default()).await.unwrap_or(0)
    }

    /// Return `true` if the table has no rows.
    pub async fn is_empty(&self) -> bool {
        let view = self.table.read().await;
        view.is_empty(Range::default()).await.unwrap_or(true)
    }

    /// Insert or update a row.
    pub async fn upsert_row(&self, key: Vec<Value>, values: Vec<Value>) -> std::io::Result<()> {
        let mut view = self.table.write().await;
        view.upsert(key, values)
            .await
            .map(|_| ())
            .map_err(std::io::Error::other)
    }

    /// Delete a row by key.
    pub async fn delete_row(&self, key: &[Value]) -> std::io::Result<bool> {
        let mut view = self.table.write().await;
        view.delete_row(key).await
    }

    /// Copy rows from a source `PersistentTable` into this temp table.
    pub async fn copy_from(
        &self,
        txn_id: tc_ir::TxnId,
        source: &super::file::PersistentTable,
    ) -> std::io::Result<()> {
        let key_len = self.schema().key().len();
        let mut rows = source
            .rows(txn_id, Range::default(), Vec::new(), false)
            .await
            .map_err(std::io::Error::other)?;

        while let Some(row) = rows.try_next().await? {
            let row_vec = row.into_vec();
            let key: Vec<Value> = row_vec[..key_len].to_vec();
            let values: Vec<Value> = row_vec[key_len..].to_vec();
            self.upsert_row(key, values).await?;
        }

        Ok(())
    }
}

impl From<TempTable> for crate::Collection {
    fn from(table: TempTable) -> Self {
        Self::Table(Box::new(super::Table::from(table)))
    }
}
