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

pub use b_table::{ColumnRange, Range, Row};
use futures::TryStreamExt;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Map, Transact, TxnId};
use tc_value::Value;

pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};
pub use stream::Rows;
use view::{Limited, Selection, TableSlice};

/// A relational database table, or a view of one.
///
/// Views recursively delegate to their source while retaining read-only access.
#[derive(Clone)]
pub enum Table<Txn: crate::StorageContext> {
    File(PersistentTable<Txn>),
    Slice(TableSlice<Txn>),
    Limited(Limited<Txn>),
    Selection(Selection<Txn>),
}

impl<Txn: crate::StorageContext> std::fmt::Debug for Table<Txn> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("schema", self.schema())
            .finish()
    }
}

impl<Txn: crate::StorageContext> From<PersistentTable<Txn>> for Table<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self::File(table)
    }
}

impl<Txn: crate::StorageContext> Table<Txn> {
    pub(crate) async fn hash_contents(&self, id: TxnId) -> TCResult<tc_ir::Sha256Hash> {
        Ok(async_hash::hash_try_stream::<async_hash::Sha256, _, _, _>(
            Box::pin(self.rows(id)).await?.map_ok(|row| row.into_vec()),
        )
        .await?)
    }

    pub fn is_persistent(&self) -> bool {
        matches!(self, Self::File(_))
    }

    /// Return an alias with read access only, including for a full-table read.
    pub(crate) fn read_view(&self) -> Self {
        match self {
            Self::File(table) => table.slice(Range::default(), &[], false),
            _ => self.clone(),
        }
    }

    fn persistent(&self) -> TCResult<&PersistentTable<Txn>> {
        match self {
            Self::File(table) => Ok(table),
            _ => Err(TCError::bad_request("expected a persistent Table owner")),
        }
    }

    /// Complete accepted recovery work on an exclusively owned, unpublished value.
    /// A durable decision skips the callback; otherwise it runs once before commit.
    pub(crate) async fn recover<Replay, Replayed>(
        &self,
        id: TxnId,
        replay: Replay,
    ) -> tc_error::TCResult<()>
    where
        Replay: FnOnce() -> Replayed + Send,
        Replayed: std::future::Future<Output = tc_error::TCResult<()>> + Send,
    {
        self.persistent()?.recover(id, replay).await
    }

    pub fn schema(&self) -> &TableSchema {
        match self {
            Self::File(t) => t.schema(),
            Self::Slice(t) => t.schema(),
            Self::Limited(t) => t.schema(),
            Self::Selection(t) => t.schema(),
        }
    }

    /// Acquire a transaction-consistent guarded stream for this Table value.
    pub async fn rows(&self, id: TxnId) -> TCResult<Rows> {
        match self {
            Self::File(table) => table.rows(id, Range::default(), Vec::new(), false).await,
            Self::Slice(view) => view.rows(id).await,
            Self::Limited(view) => view.rows(id).await,
            Self::Selection(view) => view.rows(id).await,
        }
    }

    pub fn order_by(&self, columns: &[Id], reverse: bool) -> TCResult<Self> {
        self.slice(Range::default(), columns, reverse)
    }

    pub(crate) async fn read_row(
        &self,
        txn_id: TxnId,
        key: &[Value],
    ) -> TCResult<Option<Row<Value>>> {
        match self {
            Self::File(table) => table.read_row(txn_id, key).await,
            Self::Slice(table) => table.read_row(txn_id, key).await,
            _ => Err(TCError::bad_request(
                "cannot read a row from this Table view",
            )),
        }
    }

    pub(crate) async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        match self {
            Self::File(table) => table.contains_row(txn_id, key).await,
            Self::Slice(table) => table.read_row(txn_id, key).await.map(|row| row.is_some()),
            _ => Err(TCError::bad_request(
                "cannot test membership of this Table view",
            )),
        }
    }

    pub async fn count(&self, id: TxnId) -> TCResult<u64> {
        match self {
            Self::File(table) => table.count(id).await,
            _ => Ok(self.rows(id).await?.try_count().await?),
        }
    }

    pub async fn is_empty(&self, id: TxnId) -> TCResult<bool> {
        match self {
            Self::File(table) => table.is_empty(id).await,
            _ => Ok(self.rows(id).await?.try_is_empty().await?),
        }
    }

    pub(crate) async fn upsert_row(
        &self,
        txn: &Txn,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> TCResult<()> {
        match self {
            Self::File(table) => table.upsert_row(txn, key, values).await,
            _ => Err(TCError::bad_request("cannot mutate a read-only Table view")),
        }
    }

    pub(crate) async fn insert_row(
        &self,
        txn: &Txn,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> TCResult<()> {
        match self {
            Self::File(table) => table.insert_row(txn, key, values).await,
            _ => Err(TCError::bad_request(
                "cannot insert into a read-only Table view",
            )),
        }
    }

    pub(crate) async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        match self {
            Self::File(table) => table.delete_row(txn, key).await,
            _ => Err(TCError::bad_request(
                "cannot delete a row from a read-only Table view",
            )),
        }
    }

    pub(crate) async fn truncate(&self, txn: &Txn, range: Range<Id, Value>) -> TCResult<()> {
        match self {
            Self::File(table) => table.truncate(txn, range).await,
            _ => Err(TCError::bad_request(
                "cannot truncate a read-only Table view",
            )),
        }
    }

    pub(crate) async fn update(
        &self,
        txn: &Txn,
        range: Range<Id, Value>,
        values: Map<Value>,
    ) -> TCResult<()> {
        match self {
            Self::File(table) => table.update(txn, range, values).await,
            _ => Err(TCError::bad_request("cannot update a read-only Table view")),
        }
    }
}

impl<Txn: crate::StorageContext> Transact for Table<Txn> {
    async fn commit(&self, txn_id: TxnId) -> TCResult<()> {
        Transact::commit(self.persistent()?, txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) -> TCResult<()> {
        self.persistent()?.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) -> TCResult<()> {
        self.persistent()?.finalize(txn_id).await
    }
}

impl<Txn: crate::StorageContext> crate::Restore for Table<Txn> {
    type Txn = Txn;

    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        self.persistent()?.restore(txn, source.persistent()?).await
    }
}

#[cfg(test)]
#[path = "../../tests/unit/table/tests.rs"]
mod tests;
