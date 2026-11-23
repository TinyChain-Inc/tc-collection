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
use collate::CollateRef;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Map, Transact, TxnId};
use tc_value::Value;

pub use codec::DecodedTablePayload;
pub use file::PersistentTable;
pub use schema::{Column, TableIndexSchema, TableSchema};
pub use stream::Rows;
pub use view::{Limited, Selection, TableSlice};

/// A relational database table, or a view of one.
///
/// Ported from v1 `Table<Txn, FE>` enum.  All view types convert into this
/// via `From`, and this converts into [`crate::Collection`] via `From`.
#[derive(Clone)]
pub enum Table<Txn: crate::StorageContext> {
    File(PersistentTable<Txn>),
    Local(file::TableFile<Txn::File>),
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

impl<Txn: crate::StorageContext> From<file::TableFile<Txn::File>> for Table<Txn> {
    fn from(table: file::TableFile<Txn::File>) -> Self {
        Self::Local(table)
    }
}

impl<Txn: crate::StorageContext> From<TableSlice<Txn>> for Table<Txn> {
    fn from(slice: TableSlice<Txn>) -> Self {
        Self::Slice(slice)
    }
}

impl<Txn: crate::StorageContext> From<Limited<Txn>> for Table<Txn> {
    fn from(limited: Limited<Txn>) -> Self {
        Self::Limited(limited)
    }
}

impl<Txn: crate::StorageContext> From<Selection<Txn>> for Table<Txn> {
    fn from(selection: Selection<Txn>) -> Self {
        Self::Selection(selection)
    }
}

impl<Txn: crate::StorageContext> Table<Txn> {
    pub(crate) async fn hash_contents(&self, id: TxnId) -> TCResult<tc_ir::Sha256Hash> {
        Ok(async_hash::hash_try_stream::<async_hash::Sha256, _, _, _>(
            Box::pin(self.row_stream(id))
                .await?
                .map_ok(|row| row.into_vec()),
        )
        .await?)
    }

    pub fn is_persistent(&self) -> bool {
        matches!(self, Self::File(_))
    }

    /// Return an alias with read access only, including for a full-table read.
    pub(crate) fn read_view(&self) -> Self {
        match self {
            Self::File(table) => Self::Slice(table.slice(Range::default(), &[], false)),
            Self::Local(table) => Self::Slice(TableSlice::local(
                table.clone(),
                Range::default(),
                Vec::new(),
                false,
            )),
            _ => self.clone(),
        }
    }

    pub(crate) fn persistent(&self) -> TCResult<&PersistentTable<Txn>> {
        match self {
            Self::File(table) => Ok(table),
            _ => Err(TCError::bad_request("expected a persistent Table owner")),
        }
    }

    /// Synchronize canonical storage of a persistent owner.
    pub async fn sync(&self) -> TCResult<()> {
        self.persistent()?.sync().await.map_err(Into::into)
    }

    /// Explicitly make canonical storage durable without publishing pending versions.
    pub async fn sync_all(&self) -> TCResult<()> {
        self.persistent()?.sync_all().await.map_err(Into::into)
    }

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
            Self::Local(table) => file::row_stream(table, Range::default(), &[], false).await?,
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
            Self::Slice(table) => {
                let slice = table.slice(range)?;
                Ok(if order.is_empty() && !reverse {
                    slice
                } else {
                    slice.order_by(order, reverse)
                })
            }
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
        match self {
            Self::Limited(table) => table.select(columns.to_vec()),
            _ => self
                .slice(Range::default(), &[], false)
                .and_then(|slice| slice.select(columns.to_vec())),
        }
    }

    pub(crate) async fn read_row(
        &self,
        txn_id: TxnId,
        key: &[Value],
    ) -> TCResult<Option<Row<Value>>> {
        match self {
            Self::File(table) => table.read_row(txn_id, key).await,
            Self::Local(table) => table.get_row(key).await.map_err(TCError::from),
            Self::Slice(table) => table.read_row(txn_id, key).await,
            _ => Err(TCError::bad_request(
                "cannot read a row from this Table view",
            )),
        }
    }

    pub(crate) async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> TCResult<bool> {
        match self {
            Self::File(table) => table.contains_row(txn_id, key).await,
            Self::Local(table) => table.contains(key).await.map_err(TCError::from),
            Self::Slice(table) => table.read_row(txn_id, key).await.map(|row| row.is_some()),
            _ => Err(TCError::bad_request(
                "cannot test membership of this Table view",
            )),
        }
    }

    pub(crate) async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::File(table) => table.count(txn_id).await,
            Self::Local(table) => table
                .read()
                .await
                .count(Range::default())
                .await
                .map_err(TCError::from),
            Self::Slice(table) => table.count(txn_id).await,
            Self::Limited(table) => table.count(txn_id).await,
            Self::Selection(table) => table.count(txn_id).await,
        }
    }

    pub(crate) async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::File(table) => table.is_empty(txn_id).await,
            Self::Local(table) => table
                .read()
                .await
                .is_empty(Range::default())
                .await
                .map_err(TCError::from),
            Self::Slice(table) => table.is_empty(txn_id).await,
            Self::Limited(table) => table.is_empty(txn_id).await,
            Self::Selection(table) => table.is_empty(txn_id).await,
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
            Self::Local(table) => file::upsert(table, key, values)
                .await
                .map_err(TCError::from),
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
            Self::Local(table) => {
                if table.contains(&key).await.map_err(TCError::from)? {
                    return Err(TCError::bad_request(format!(
                        "cannot insert Table row: key {key:?} already exists"
                    )));
                }

                file::upsert(table, key, values)
                    .await
                    .map_err(TCError::from)
            }
            _ => Err(TCError::bad_request(
                "cannot insert into a read-only Table view",
            )),
        }
    }

    pub(crate) async fn delete_row(&self, txn: &Txn, key: Vec<Value>) -> TCResult<()> {
        match self {
            Self::File(table) => table.delete_row(txn, key).await,
            Self::Local(table) => file::delete_row(table, &key)
                .await
                .map(|_| ())
                .map_err(TCError::from),
            _ => Err(TCError::bad_request(
                "cannot delete a row from a read-only Table view",
            )),
        }
    }

    pub(crate) async fn truncate(&self, txn: &Txn, range: Range<Id, Value>) -> TCResult<()> {
        match self {
            Self::File(table) => table.truncate(txn, range).await,
            Self::Local(table) => view::truncate_local(table, txn, range).await,
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
            Self::Local(table) => view::update_local(table, txn, range, values).await,
            _ => Err(TCError::bad_request("cannot update a read-only Table view")),
        }
    }
}

impl<Txn: crate::StorageContext> Transact for Table<Txn> {
    async fn commit(&self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Local(_) => Ok(()),
            _ => Transact::commit(self.persistent()?, txn_id).await,
        }
    }

    fn rollback(&self, txn_id: &TxnId) -> impl std::future::Future<Output = TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match self {
                Self::Local(_) => Ok(()),
                _ => Transact::rollback(self.persistent()?, &txn_id).await,
            }
        }
    }

    fn finalize(&self, txn_id: &TxnId) -> impl std::future::Future<Output = TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match self {
                Self::Local(_) => Ok(()),
                _ => Transact::finalize(self.persistent()?, &txn_id).await,
            }
        }
    }
}

impl<Txn: crate::StorageContext> crate::CopyFrom<&Table<Txn>> for PersistentTable<Txn> {
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
        source: &Table<Txn>,
    ) -> tc_error::TCResult<Self> {
        match source {
            Table::File(_) | Table::Local(_) => {}
            Table::Slice(view) => view.validate_copy()?,
            Table::Limited(view) => view.validate_copy()?,
            Table::Selection(view) => view.validate_copy()?,
        }

        let target = PersistentTable::try_new(dir, source.schema().clone()).await?;

        let mut rows = source.row_stream(txn.id()).await?;
        let collator = crate::persistence::value_order();
        let key_len = source.schema().key().len();
        if let Some(mut previous) = rows.try_next().await? {
            while let Some(row) = rows.try_next().await? {
                let previous_key = &previous[..key_len];
                let key = &row[..key_len];
                let tied = collator.range_collator().cmp_ref(previous_key, key).is_eq();
                if previous_key == key || (tied && target.contains_literal_key(key).await?) {
                    return Err(freqfs::IntegrityError::invalid_data(
                        "ambiguous Table primary key during copying",
                    )
                    .into());
                }

                if collator.key_collator().cmp(&previous, &row).is_gt() {
                    return Err(TCError::bad_request(
                        "cannot materialize Table rows outside canonical key order",
                    ));
                }

                let completed = std::mem::replace(&mut previous, row);
                target.load_literal_row(completed.into_vec()).await?;
            }

            target.load_literal_row(previous.into_vec()).await?;
        }

        target.sync().await?;
        Ok(target)
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
