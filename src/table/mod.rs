//! Transactional Table implementation split by concern:
//! - `schema`: table schema validation, key/column encoding, and index definitions.
//! - `file`: runtime behavior, transaction state, and query/mutation logic.
//! - `stream`: permit-bound row stream with lazy `limit`/`select` transforms.
//! - `view`: lazy view structs (`TableSlice`, `Limited`, `Selection`).
//! - `public`: public API route handlers (ports v1 `public.rs`).
//! - `tests`: behavioral regression coverage for transactional visibility semantics.
mod codec;
mod file;
pub mod public;
mod schema;
mod stream;
mod view;

pub use codec::DecodedTablePayload;
pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};
pub use stream::Rows;
pub use view::{Limited, Selection, TableSlice};

pub use b_table::{ColumnRange, Range, Row};
use futures::{StreamExt, stream::BoxStream};
use tc_error::{TCError, TCResult};
use tc_ir::TxnId;
use tc_ir::{Id, Map};
use tc_value::{Value, ValueCollator};

/// A transaction-local table backed directly by `b-table`.
///
/// Its directory is owned by the enclosing transaction workspace, but its
/// reads and writes do not participate in the persistent transaction protocol.
pub type LocalTable =
    b_table::TableLock<TableSchema, TableIndexSchema, ValueCollator, crate::PersistentFile>;

/// A relational database table, or a view of one.
///
/// Ported from v1 `Table<Txn, FE>` enum.  All view types convert into this
/// via `From`, and this converts into [`crate::Collection`] via `From`.
#[derive(Clone)]
pub enum Table<Txn = ()> {
    File(PersistentTable<Txn>),
    Local(LocalTable),
    Slice(TableSlice<Txn>),
    Limited(Limited<Txn>),
    Selection(Selection<Txn>),
}

impl<Txn> std::fmt::Debug for Table<Txn> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("schema", self.schema())
            .finish()
    }
}

impl<Txn> From<PersistentTable<Txn>> for Table<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self::File(table)
    }
}

impl<Txn> From<LocalTable> for Table<Txn> {
    fn from(table: LocalTable) -> Self {
        Self::Local(table)
    }
}

impl<Txn> From<TableSlice<Txn>> for Table<Txn> {
    fn from(slice: TableSlice<Txn>) -> Self {
        Self::Slice(slice)
    }
}

impl<Txn> From<Limited<Txn>> for Table<Txn> {
    fn from(limited: Limited<Txn>) -> Self {
        Self::Limited(limited)
    }
}

impl<Txn> From<Selection<Txn>> for Table<Txn> {
    fn from(selection: Selection<Txn>) -> Self {
        Self::Selection(selection)
    }
}

impl<Txn> Table<Txn> {
    pub fn schema(&self) -> &TableSchema {
        match self {
            Self::File(t) => t.schema(),
            Self::Local(t) => t.schema(),
            Self::Slice(t) => t.schema(),
            Self::Limited(t) => t.schema(),
            Self::Selection(t) => t.schema(),
        }
    }

    /// Return a lazy row stream for this table or view at `txn_id`.
    pub async fn row_stream(
        &self,
        txn_id: TxnId,
    ) -> TCResult<BoxStream<'static, Result<Row<Value>, std::io::Error>>> {
        let rows = match self {
            Self::File(table) => table
                .rows(txn_id, Range::default(), Vec::new(), false)
                .await?
                .boxed(),
            Self::Local(table) => table.clone().into_read().await.into_rows().await?,
            Self::Slice(table) => table.rows(txn_id).await?.boxed(),
            Self::Limited(table) => table.rows(txn_id).await?.boxed(),
            Self::Selection(table) => table.rows(txn_id).await?.boxed(),
        };

        Ok(rows)
    }

    pub(crate) fn slice(
        &self,
        range: Range<Id, Value>,
        order: &[Id],
        reverse: bool,
    ) -> TCResult<TableSlice<Txn>> {
        match self {
            Self::File(table) => Ok(table.slice(range, order, reverse)),
            Self::Local(table) => Ok(TableSlice::local(
                table.clone(),
                range,
                order.to_vec(),
                reverse,
            )),
            _ => Err(TCError::bad_request(
                "cannot slice an already transformed Table view",
            )),
        }
    }

    pub(crate) fn order_by(&self, columns: &[Id], reverse: bool) -> TCResult<TableSlice<Txn>> {
        self.slice(Range::default(), columns, reverse)
    }

    pub(crate) fn limit(&self, limit: u64) -> TCResult<Limited<Txn>> {
        self.slice(Range::default(), &[], false)
            .map(|slice| slice.limit(limit))
    }

    pub(crate) fn select(&self, columns: &[Id]) -> TCResult<Selection<Txn>> {
        self.slice(Range::default(), &[], false)
            .map(|slice| slice.select(columns.to_vec()))
    }

    pub(crate) async fn read_row(&self, txn_id: TxnId, key: &[Value]) -> Option<Row<Value>> {
        match self {
            Self::File(table) => table.read_row(txn_id, key).await,
            Self::Local(table) => table.read().await.get_row(key).await.ok().flatten(),
            _ => None,
        }
    }

    pub(crate) async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> bool {
        match self {
            Self::File(table) => table.contains_row(txn_id, key).await,
            Self::Local(table) => table.read().await.contains(key).await.unwrap_or(false),
            _ => false,
        }
    }

    pub(crate) async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::File(table) => Ok(table.count(txn_id).await),
            Self::Local(table) => table
                .read()
                .await
                .count(Range::default())
                .await
                .map_err(TCError::from),
            Self::Slice(table) => Ok(table.count(txn_id).await),
            Self::Limited(table) => Ok(table.count(txn_id).await),
            Self::Selection(table) => Ok(table.count(txn_id).await),
        }
    }

    pub(crate) async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::File(table) => Ok(table.is_empty(txn_id).await),
            Self::Local(table) => table
                .read()
                .await
                .is_empty(Range::default())
                .await
                .map_err(TCError::from),
            Self::Slice(table) => Ok(table.is_empty(txn_id).await),
            Self::Limited(table) => Ok(table.is_empty(txn_id).await),
            Self::Selection(table) => Ok(table.is_empty(txn_id).await),
        }
    }

    pub(crate) async fn upsert_row(
        &self,
        txn: &Txn,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> TCResult<()>
    where
        Txn: crate::StorageContext,
    {
        match self {
            Self::File(table) => table
                .upsert_row(txn, key, values)
                .await
                .map_err(TCError::from),
            Self::Local(table) => table.write().await.upsert(key, values).await.map(|_| ()),
            _ => Err(TCError::bad_request("cannot mutate a Table view")),
        }
    }

    pub(crate) async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()>
    where
        Txn: crate::StorageContext,
    {
        match self {
            Self::File(table) => table.delete_row(txn, key).await.map_err(TCError::from),
            Self::Local(table) => table
                .write()
                .await
                .delete_row(&key)
                .await
                .map(|_| ())
                .map_err(TCError::from),
            _ => Err(TCError::bad_request("cannot mutate a Table view")),
        }
    }

    pub(crate) async fn truncate(&self, txn: &Txn, range: Range<Id, Value>) -> TCResult<()>
    where
        Txn: crate::StorageContext,
    {
        match self {
            Self::File(table) => table.truncate(txn, range).await.map_err(TCError::from),
            Self::Local(table) => table
                .write()
                .await
                .delete_range(range)
                .await
                .map(|_| ())
                .map_err(TCError::from),
            _ => Err(TCError::bad_request("cannot mutate a Table view")),
        }
    }

    pub(crate) async fn update(
        &self,
        txn: &Txn,
        range: Range<Id, Value>,
        values: Map<Value>,
    ) -> TCResult<()>
    where
        Txn: crate::StorageContext,
    {
        match self {
            Self::File(table) => table
                .update(txn, range, values)
                .await
                .map_err(TCError::from),
            Self::Local(table) => {
                let value_columns = self.schema().values();
                for name in values.keys() {
                    if !value_columns.contains(name) {
                        return Err(TCError::bad_request(format!(
                            "cannot update key column {name}"
                        )));
                    }
                }

                let _ = (table, range);
                Err(TCError::method_not_allowed(
                    tc_ir::Method::Put,
                    "bulk update of a local Table",
                ))
            }
            _ => Err(TCError::bad_request("cannot mutate a Table view")),
        }
    }
}

#[cfg(test)]
mod tests;
