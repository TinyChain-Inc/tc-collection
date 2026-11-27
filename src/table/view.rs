//! Read-only Table views delegate guarded traversal to their concrete sources.
use std::fmt;

use b_table::{Range, Row};
use collate::CollateRef;
use futures::TryStreamExt;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, TxnId};
use tc_value::{Value, ValueCollator};

use super::stream::{Projection, Rows};
use super::{PersistentTable, Table, TableSchema};

#[derive(Clone)]
pub struct TableSlice<Txn: crate::StorageContext> {
    table: PersistentTable<Txn>,
    range: Range<Id, Value>,
    order: Vec<Id>,
    reverse: bool,
}

impl<Txn: crate::StorageContext> fmt::Debug for TableSlice<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableSlice")
            .field("range", &self.range)
            .field("order", &self.order)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl<Txn: crate::StorageContext> TableSlice<Txn> {
    pub(super) fn new(
        table: PersistentTable<Txn>,
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

    pub(super) fn schema(&self) -> &TableSchema {
        self.table.schema()
    }

    fn slice(&self, range: Range<Id, Value>, order: &[Id], reverse: bool) -> TCResult<Self> {
        let range = self
            .range
            .intersection(range, &ValueCollator::default())
            .ok_or_else(|| TCError::bad_request("Table slice constraints are disjoint"))?;

        let (order, reverse) = if order.is_empty() && !reverse {
            (self.order.clone(), self.reverse)
        } else {
            (order.to_vec(), reverse)
        };

        Ok(Self::new(self.table.clone(), range, order, reverse))
    }

    pub(super) async fn read_row(&self, id: TxnId, key: &[Value]) -> TCResult<Option<Row<Value>>> {
        if !self.range.is_default() {
            return Err(TCError::bad_request(
                "cannot read a row from this Table view",
            ));
        }

        self.table.read_row(id, key).await
    }

    pub(super) async fn rows(&self, id: TxnId) -> TCResult<Rows> {
        self.table
            .rows(id, self.range.clone(), self.order.clone(), self.reverse)
            .await
    }
}

#[derive(Clone, Debug)]
pub struct Limited<Txn: crate::StorageContext> {
    source: Box<Table<Txn>>,
    limit: u64,
}

impl<Txn: crate::StorageContext> Limited<Txn> {
    pub(super) fn schema(&self) -> &TableSchema {
        self.source.schema()
    }

    pub(super) async fn rows(&self, id: TxnId) -> TCResult<Rows> {
        // Descriptions normalize to at most one limit and one projection.
        // Box this recursive future boundary; stream guards remain in Rows.
        Ok(Box::pin(self.source.rows(id)).await?.limit(self.limit))
    }
}

#[derive(Clone, Debug)]
pub struct Selection<Txn: crate::StorageContext> {
    source: Box<Table<Txn>>,
    schema: TableSchema,
    projection: Projection,
}

impl<Txn: crate::StorageContext> Selection<Txn> {
    fn new(source: Table<Txn>, columns: Vec<Id>) -> TCResult<Self> {
        let (schema, projection) = source.schema().project(columns)?;

        Ok(Self {
            source: Box::new(source),
            schema,
            projection,
        })
    }

    pub(super) fn schema(&self) -> &TableSchema {
        &self.schema
    }

    pub(super) async fn rows(&self, id: TxnId) -> TCResult<Rows> {
        Ok(Box::pin(self.source.rows(id))
            .await?
            .project(self.projection.clone()))
    }
}

impl<Txn: crate::StorageContext> crate::CopyFrom<&Table<Txn>> for PersistentTable<Txn> {
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
        source: &Table<Txn>,
    ) -> tc_error::TCResult<Self> {
        let mut current = source;
        loop {
            match current {
                Table::File(_) => break,
                Table::Slice(view) => {
                    if view.reverse || (!view.order.is_empty() && view.order != view.schema().key())
                    {
                        return Err(TCError::bad_request(
                            "cannot materialize a reordered Table view in canonical key order",
                        ));
                    }

                    break;
                }
                Table::Limited(view) => current = &view.source,
                Table::Selection(view) => current = &view.source,
            }
        }

        let target = PersistentTable::unpublished(dir, source.schema().clone()).await?;

        let mut rows = source.rows(txn.id()).await?;
        let collator = b_tree::RangeCollator::new(tc_value::ValueCollator::default());
        let key_len = source.schema().key().len();
        if let Some(mut previous) = rows.try_next().await? {
            while let Some(row) = rows.try_next().await? {
                let previous_key = &previous[..key_len];
                let key = &row[..key_len];
                let tied = collator.cmp_ref(previous_key, key).is_eq();
                if previous_key == key || (tied && target.contains_literal_key(key).await?) {
                    return Err(freqfs::IntegrityError::invalid_data(
                        "ambiguous Table primary key during copying",
                    )
                    .into());
                }

                if collator.cmp_keys(&previous, &row).is_gt() {
                    return Err(TCError::bad_request(
                        "cannot materialize Table rows outside canonical key order",
                    ));
                }

                let completed = std::mem::replace(&mut previous, row);
                target.load_literal_row(completed.into_vec()).await?;
            }

            target.load_literal_row(previous.into_vec()).await?;
        }

        target.finish_creation().await?;
        Ok(target)
    }
}

impl<Txn: crate::StorageContext> Table<Txn> {
    pub fn slice(&self, range: Range<Id, Value>, order: &[Id], reverse: bool) -> TCResult<Self> {
        match self {
            Self::File(table) => Ok(table.slice(range, order, reverse)),
            Self::Slice(view) => view.slice(range, order, reverse).map(Self::Slice),
            Self::Selection(view) => {
                for column in range.inner().keys().chain(order) {
                    if !view.schema().columns().any(|visible| visible == column) {
                        return Err(tc_error::not_found!("column not found: {column}"));
                    }
                }

                let source = view.source.slice(range, order, reverse)?;

                Ok(Self::Selection(Selection::new(
                    source,
                    view.schema().columns().cloned().collect(),
                )?))
            }
            Self::Limited(_) => Err(TCError::bad_request(
                "cannot slice or reorder a limited Table view",
            )),
        }
    }

    pub fn limit(&self, limit: u64) -> Self {
        match self {
            Self::Limited(view) => Self::Limited(Limited {
                source: view.source.clone(),
                limit: limit.min(view.limit),
            }),
            _ => Self::Limited(Limited {
                source: Box::new(self.clone()),
                limit,
            }),
        }
    }

    pub fn select(&self, columns: &[Id]) -> TCResult<Self> {
        match self {
            Self::Limited(view) => Ok(Self::Limited(Limited {
                source: Box::new(view.source.select(columns)?),
                limit: view.limit,
            })),
            Self::Selection(view) => {
                for column in columns {
                    if !view.schema().columns().any(|visible| visible == column) {
                        return Err(tc_error::not_found!("column not found: {column}"));
                    }
                }

                Ok(Self::Selection(Selection::new(
                    (*view.source).clone(),
                    columns.to_vec(),
                )?))
            }
            _ => Ok(Self::Selection(Selection::new(
                self.clone(),
                columns.to_vec(),
            )?)),
        }
    }
}
