//! Lazy view structs: `TableSlice`, `Limited`, and `Selection`.
//!
//! Each view holds a reference to its source and applies its transform
//! (range/order/reverse, row cap, column projection) lazily during row
//! streaming — no rows are eagerly copied (v1 no-materialization invariant).
use std::fmt;

use b_table::{Range, Row};
use futures::TryStreamExt;
use tc_ir::{Id, TxnId};
use tc_value::Value;

use super::LocalTable;
use super::file::PersistentTable;
use super::schema::TableSchema;
use super::stream::Rows;

enum TableSource<Txn> {
    File(PersistentTable<Txn>),
    Local(LocalTable),
}

impl<Txn> Clone for TableSource<Txn> {
    fn clone(&self) -> Self {
        match self {
            Self::File(table) => Self::File(table.clone()),
            Self::Local(table) => Self::Local(table.clone()),
        }
    }
}

impl<Txn> TableSource<Txn> {
    fn schema(&self) -> &TableSchema {
        match self {
            Self::File(table) => table.schema(),
            Self::Local(table) => table.schema(),
        }
    }

    async fn rows(
        &self,
        txn_id: TxnId,
        range: Range<Id, Value>,
        order: Vec<Id>,
        reverse: bool,
    ) -> tc_error::TCResult<Rows> {
        match self {
            Self::File(table) => table
                .rows(txn_id, range, order, reverse)
                .await
                .map_err(tc_error::TCError::from),
            Self::Local(table) => {
                let rows = table
                    .clone()
                    .into_read()
                    .await
                    .rows(range, &order, reverse, None)
                    .await?;
                Ok(Rows::local(rows))
            }
        }
    }

    async fn count(&self, txn_id: TxnId, range: Range<Id, Value>) -> u64 {
        match self {
            Self::File(table) => table.count_in(txn_id, range).await,
            Self::Local(table) => table.read().await.count(range).await.unwrap_or(0),
        }
    }

    async fn is_empty(&self, txn_id: TxnId, range: Range<Id, Value>) -> bool {
        match self {
            Self::File(table) => table.is_empty_in(txn_id, range).await,
            Self::Local(table) => table.read().await.is_empty(range).await.unwrap_or(true),
        }
    }
}

/// A range + order + reverse view over a [`PersistentTable`].
///
/// Constructed via [`PersistentTable::slice`] or [`PersistentTable::order_by`].
/// All operations delegate to the source table with the stored range, order,
/// and direction applied. The view is structural — it holds no row data.
pub struct TableSlice<Txn> {
    table: TableSource<Txn>,
    range: Range<Id, Value>,
    order: Vec<Id>,
    reverse: bool,
}

impl<Txn> Clone for TableSlice<Txn> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            range: self.range.clone(),
            order: self.order.clone(),
            reverse: self.reverse,
        }
    }
}

impl<Txn> fmt::Debug for TableSlice<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableSlice")
            .field("range", &self.range)
            .field("order", &self.order)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl<Txn> TableSlice<Txn> {
    fn from_source(
        table: TableSource<Txn>,
        range: Range<Id, Value>,
        order: Vec<Id>,
        reverse: bool,
    ) -> Self {
        Self {
            table,
            range,
            order,
            reverse,
        }
    }

    pub(crate) fn new(
        table: PersistentTable<Txn>,
        range: Range<Id, Value>,
        order: Vec<Id>,
        reverse: bool,
    ) -> Self {
        Self::from_source(TableSource::File(table), range, order, reverse)
    }

    pub(crate) fn local(
        table: LocalTable,
        range: Range<Id, Value>,
        order: Vec<Id>,
        reverse: bool,
    ) -> Self {
        Self::from_source(TableSource::Local(table), range, order, reverse)
    }

    pub fn schema(&self) -> &TableSchema {
        self.table.schema()
    }

    pub fn range(&self) -> &Range<Id, Value> {
        &self.range
    }

    pub fn order(&self) -> &[Id] {
        &self.order
    }

    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// Return a row stream over this slice's range, order, and direction.
    ///
    /// The stream holds a transactional read permit for its entire lifetime.
    pub async fn rows(&self, txn_id: TxnId) -> tc_error::TCResult<Rows> {
        self.table
            .rows(txn_id, self.range.clone(), self.order.clone(), self.reverse)
            .await
    }

    /// Count the visible rows in this slice at `txn_id`.
    pub async fn count(&self, txn_id: TxnId) -> u64 {
        self.table.count(txn_id, self.range.clone()).await
    }

    /// Return `true` if this slice has no visible rows at `txn_id`.
    pub async fn is_empty(&self, txn_id: TxnId) -> bool {
        self.table.is_empty(txn_id, self.range.clone()).await
    }

    /// Iterate visible rows in this slice, calling `on_row` for each.
    pub async fn for_each_row_in_order<F>(&self, txn_id: TxnId, on_row: F)
    where
        F: FnMut(Row<Value>),
    {
        let Ok(mut rows) = self.rows(txn_id).await else {
            return;
        };
        let mut on_row = on_row;
        while let Some(row) = rows.try_next().await.expect("read table row") {
            on_row(row);
        }
    }

    /// Cap this slice to at most `n` rows.
    pub fn limit(&self, n: u64) -> Limited<Txn> {
        Limited {
            source: self.clone(),
            limit: n,
        }
    }

    /// Project only the given `columns` from each row in this slice.
    pub fn select(&self, columns: Vec<Id>) -> Selection<Txn> {
        Selection {
            source: self.clone(),
            columns,
            limit: None,
        }
    }

    /// Further narrow this slice to a sub-range.
    ///
    /// The sub-range columns are merged with this slice's range (the sub-range
    /// takes precedence for shared columns).
    pub fn slice(&self, sub_range: Range<Id, Value>) -> TableSlice<Txn> {
        let mut combined = self.range.inner().clone();
        for (name, bound) in sub_range.into_inner() {
            combined.insert(name, bound);
        }
        TableSlice::from_source(
            self.table.clone(),
            combined.into(),
            self.order.clone(),
            self.reverse,
        )
    }
}

/// A row-cap view that yields at most `limit` rows from its source.
///
/// Constructed via [`TableSlice::limit`] or [`PersistentTable::limit`].
/// `count` streams rows and stops at the cap — no full materialization.
#[derive(Clone)]
pub struct Limited<Txn> {
    source: TableSlice<Txn>,
    limit: u64,
}

impl<Txn> fmt::Debug for Limited<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Limited")
            .field("limit", &self.limit)
            .finish()
    }
}

impl<Txn> Limited<Txn> {
    pub fn schema(&self) -> &TableSchema {
        self.source.schema()
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// Return a row stream capped to at most `limit` rows.
    pub async fn rows(&self, txn_id: TxnId) -> tc_error::TCResult<Rows> {
        let rows = self.source.rows(txn_id).await?;
        Ok(rows.limit(self.limit))
    }

    /// Count visible rows, capped at `limit`.
    ///
    /// This streams rows (with the limit applied via a lazy `take`) and counts
    /// them — no full source range is materialized.
    pub async fn count(&self, txn_id: TxnId) -> u64 {
        if self.limit == 0 {
            return 0;
        }
        let Ok(mut rows) = self.rows(txn_id).await else {
            return 0;
        };
        let mut count = 0_u64;
        while rows.try_next().await.expect("read limited row").is_some() {
            count += 1;
        }
        count
    }

    /// Return `true` if there are no visible rows or `limit` is zero.
    pub async fn is_empty(&self, txn_id: TxnId) -> bool {
        if self.limit == 0 {
            return true;
        }
        self.source.is_empty(txn_id).await
    }

    /// Iterate at most `limit` visible rows, calling `on_row` for each.
    pub async fn for_each_row_in_order<F>(&self, txn_id: TxnId, mut on_row: F)
    where
        F: FnMut(Row<Value>),
    {
        if self.limit == 0 {
            return;
        }
        let Ok(mut rows) = self.rows(txn_id).await else {
            return;
        };
        while let Some(row) = rows.try_next().await.expect("read limited row") {
            on_row(row);
        }
    }

    /// Project only the given `columns` from each row, preserving this
    /// view's row cap.
    pub fn select(&self, columns: Vec<Id>) -> Selection<Txn> {
        Selection {
            source: self.source.clone(),
            columns,
            limit: Some(self.limit),
        }
    }
}

/// A column-projection view that yields only the selected columns from each row.
///
/// Constructed via [`TableSlice::select`] or [`PersistentTable::select`].
/// The projection is applied lazily during streaming — no rows are copied
/// until the stream is polled.
#[derive(Clone)]
pub struct Selection<Txn> {
    source: TableSlice<Txn>,
    columns: Vec<Id>,
    limit: Option<u64>,
}

impl<Txn> fmt::Debug for Selection<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Selection")
            .field("columns", &self.columns)
            .finish()
    }
}

impl<Txn> Selection<Txn> {
    pub fn schema(&self) -> &TableSchema {
        self.source.schema()
    }

    pub fn columns(&self) -> &[Id] {
        &self.columns
    }

    /// Return a row stream with only the selected columns.
    ///
    /// If this selection was composed from a [`Limited`] view, the row cap
    /// is applied as a lazy `take` before the column projection.
    pub async fn rows(&self, txn_id: TxnId) -> tc_error::TCResult<Rows> {
        let rows = self.source.rows(txn_id).await?;
        let rows = if let Some(limit) = self.limit {
            rows.limit(limit)
        } else {
            rows
        };
        Ok(rows.select(self.schema(), &self.columns))
    }

    /// Count visible rows (column projection does not change row count).
    pub async fn count(&self, txn_id: TxnId) -> u64 {
        self.source.count(txn_id).await
    }

    /// Return `true` if there are no visible rows.
    pub async fn is_empty(&self, txn_id: TxnId) -> bool {
        self.source.is_empty(txn_id).await
    }

    /// Iterate visible rows with only the selected columns, calling `on_row`.
    pub async fn for_each_row_in_order<F>(&self, txn_id: TxnId, mut on_row: F)
    where
        F: FnMut(Row<Value>),
    {
        let indices = Self::column_indices(self.schema(), &self.columns);
        self.source
            .for_each_row_in_order(txn_id, |row| {
                let projected: Row<Value> = indices
                    .iter()
                    .filter_map(|&i| row.get(i).cloned())
                    .collect();
                on_row(projected);
            })
            .await;
    }

    fn column_indices(schema: &TableSchema, columns: &[Id]) -> Vec<usize> {
        let key = schema.key();
        let values = schema.values();
        let all: Vec<&Id> = key.iter().chain(values.iter()).collect();

        let mut indices = Vec::with_capacity(columns.len());
        for col in columns {
            if let Some(i) = all.iter().position(|name| *name == col) {
                indices.push(i);
            }
        }
        indices
    }
}
