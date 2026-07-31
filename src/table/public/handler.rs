//! Individual route handler structs.
//!
//! Each handler is a separate struct (not an enum variant), following the v1
//! pattern in `table/public.rs`.  Each has a `From` impl from `PersistentTable`
//! and implements [`super::RouteHandler`] via the verb trait methods.
//!
//! Handlers are generic over `State`, which must support the appropriate
//! `From` impls — this mirrors v1's `State: From<Collection> + From<Value>
//! + From<u64>` bounds.

use std::future::Future;
use std::pin::Pin;

use b_table::Range;
use safecast::{CastFrom, Match, TryCastInto};
use tc_error::{bad_request, TCError, TCResult};
use tc_ir::{Id, Map, Scalar, Transaction, TxnId};
use tc_value::Value;

use super::selector::{cast_into_range, KeyOrRange};
use super::RouteHandler;
use crate::table::{PersistentTable, TableSchema, TempTable};

// ─── SchemaHandler ─────────────────────────────────────────────────────

/// Handler for `columns`, `key_columns`, and `key_names` routes.
///
/// Holds a function pointer that extracts the schema `Value` from the table.
/// Ported from v1 `SchemaHandler<'a, T>`.
pub struct SchemaHandler {
    table: PersistentTable,
    schema_fn: fn(&PersistentTable) -> Value,
}

impl SchemaHandler {
    pub fn new(table: PersistentTable, schema_fn: fn(&PersistentTable) -> Value) -> Self {
        Self { table, schema_fn }
    }
}

/// Return the primary column names as a `Value::Tuple` of strings.
pub fn column_schema(table: &PersistentTable) -> Value {
    let columns = table
        .schema()
        .columns()
        .map(|c| Value::String(c.to_string()))
        .collect();
    Value::Tuple(columns)
}

/// Return the key column names as a `Value::Tuple` of strings.
pub fn key_columns(table: &PersistentTable) -> Value {
    let key = table
        .schema()
        .key()
        .iter()
        .map(|c| Value::String(c.to_string()))
        .collect();
    Value::Tuple(key)
}

/// Return the key column names (alias of [`key_columns`]).
pub fn key_names(table: &PersistentTable) -> Value {
    key_columns(table)
}

// ─── ContainsHandler ───────────────────────────────────────────────────

/// Handler for `<table>/contains` — check row presence (All / Key / Range).
///
/// Ported from v1 `ContainsHandler<Txn, FE>`.
#[derive(Clone)]
pub struct ContainsHandler {
    table: PersistentTable,
}

impl From<PersistentTable> for ContainsHandler {
    fn from(table: PersistentTable) -> Self {
        Self { table }
    }
}

// ─── CountHandler ──────────────────────────────────────────────────────

/// Handler for `<table>/count` — count rows (All / Key / Range).
///
/// Ported from v1 `CountHandler<T>`.
#[derive(Clone)]
pub struct CountHandler {
    table: PersistentTable,
}

impl From<PersistentTable> for CountHandler {
    fn from(table: PersistentTable) -> Self {
        Self { table }
    }
}

// ─── LimitHandler ──────────────────────────────────────────────────────

/// Handler for `<table>/limit` — cap the row stream.
///
/// Ported from v1 `LimitHandler<T>`.
#[derive(Clone)]
pub struct LimitHandler {
    table: PersistentTable,
}

impl From<PersistentTable> for LimitHandler {
    fn from(table: PersistentTable) -> Self {
        Self { table }
    }
}

// ─── OrderHandler ──────────────────────────────────────────────────────

/// Handler for `<table>/order` — order the row stream.
///
/// Ported from v1 `OrderHandler<T>`.
#[derive(Clone)]
pub struct OrderHandler {
    table: PersistentTable,
}

impl From<PersistentTable> for OrderHandler {
    fn from(table: PersistentTable) -> Self {
        Self { table }
    }
}

// ─── SelectHandler ─────────────────────────────────────────────────────

/// Handler for `<table>/select` — project columns.
///
/// Ported from v1 `SelectHandler<T>`.
#[derive(Clone)]
pub struct SelectHandler {
    table: PersistentTable,
}

impl From<PersistentTable> for SelectHandler {
    fn from(table: PersistentTable) -> Self {
        Self { table }
    }
}

// ─── TableHandler ──────────────────────────────────────────────────────

/// Handler for `<table>` — read / slice / upsert / update / truncate / delete.
///
/// Ported from v1 `TableHandler<Txn, FE>`.
#[derive(Clone)]
pub struct TableHandler {
    table: PersistentTable,
}

impl From<PersistentTable> for TableHandler {
    fn from(table: PersistentTable) -> Self {
        Self { table }
    }
}

// ─── CreateHandler ─────────────────────────────────────────────────────

/// Handler for `GET /state/collection/table` — create a new table from a schema.
///
/// Ported from v1 `CreateHandler`.
pub struct CreateHandler {
    root: freqfs::DirLock<crate::PersistentFile>,
}

impl CreateHandler {
    pub fn new(root: freqfs::DirLock<crate::PersistentFile>) -> Self {
        Self { root }
    }
}

// ─── CopyHandler ───────────────────────────────────────────────────────

/// Handler for `POST /state/collection/table/copy_from` — create + copy rows.
///
/// Ported from v1 `CopyHandler`.
pub struct CopyHandler {
    root: freqfs::DirLock<crate::PersistentFile>,
}

impl CopyHandler {
    pub fn new(root: freqfs::DirLock<crate::PersistentFile>) -> Self {
        Self { root }
    }
}

// ─── RouteHandler impls ────────────────────────────────────────────────

type GetFut<'a, State> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
type PutFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;

impl<State> RouteHandler<State> for TableHandler
where
    State: From<crate::Collection> + From<tc_value::Value> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            match kor {
                KeyOrRange::All => Ok(State::from(crate::Collection::from(table))),
                KeyOrRange::Range(range) => {
                    let slice = table.slice(range, &[], false);
                    Ok(State::from(crate::Collection::from(
                        crate::table::Table::from(slice),
                    )))
                }
                KeyOrRange::Key(key) => {
                    let row = table.read_row(txn_id, &key).await;
                    match row {
                        Some(row) => Ok(State::from(Value::Tuple(row.into_vec()))),
                        None => Ok(State::from(Value::None)),
                    }
                }
            }
        }))
    }

    fn put(
        &self,
        txn: &dyn Transaction,
        request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let mut params = request;
        let key_value: Value = params.require("key")?.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let value_scalar = params.require("value")?;

        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, key_value)?;
            match kor {
                KeyOrRange::All => {
                    let values: tc_ir::Map<Value> = value_scalar.try_cast_into(|v| bad_request!("invalid update values: {v:?}"))?;
                    table
                        .update(txn_id, Range::default(), values)
                        .await
                        .map_err(TCError::from)
                }
                KeyOrRange::Range(range) => {
                    let values: tc_ir::Map<Value> = value_scalar.try_cast_into(|v| bad_request!("invalid update values: {v:?}"))?;
                    table.update(txn_id, range, values).await.map_err(TCError::from)
                }
                KeyOrRange::Key(key) => {
                    let value: Value = value_scalar.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
                    let values = if let Value::Tuple(tuple) = value {
                        tuple
                    } else {
                        vec![value]
                    };
                    table.upsert_row(txn_id, key, values).await.map_err(TCError::from)
                }
            }
        }))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let range = cast_into_range(&table, value)?;
            let slice = table.slice(range, &[], false);
            Ok(State::from(crate::Collection::from(
                crate::table::Table::from(slice),
            )))
        }))
    }

    fn delete(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            match kor {
                KeyOrRange::All => {
                    table
                        .truncate(txn_id, Range::default())
                        .await
                        .map_err(TCError::from)
                }
                KeyOrRange::Key(key) => {
                    table.delete_row(txn_id, key).await.map_err(TCError::from)
                }
                KeyOrRange::Range(range) => {
                    table.truncate(txn_id, range).await.map_err(TCError::from)
                }
            }
        }))
    }
}

impl<State> RouteHandler<State> for ContainsHandler
where
    State: From<tc_value::Value> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            let filled = match kor {
                KeyOrRange::All => !table.is_empty(txn_id).await,
                KeyOrRange::Key(key) => table.contains_row(txn_id, &key).await,
                KeyOrRange::Range(range) => {
                    let slice = table.slice(range, &[], false);
                    !slice.is_empty(txn_id).await
                }
            };
            Ok(State::from(Value::from(filled)))
        }))
    }

    fn put(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(
            tc_ir::Method::Put,
            "contains",
        ))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(
            tc_ir::Method::Post,
            "contains",
        ))
    }

    fn delete(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(
            tc_ir::Method::Delete,
            "contains",
        ))
    }
}

impl<State> RouteHandler<State> for CountHandler
where
    State: From<u64> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            let count: u64 = match kor {
                KeyOrRange::All => table.count(txn_id).await,
                KeyOrRange::Key(key) => {
                    if table.contains_row(txn_id, &key).await {
                        1
                    } else {
                        0
                    }
                }
                KeyOrRange::Range(range) => {
                    let slice = table.slice(range, &[], false);
                    slice.count(txn_id).await
                }
            };
            Ok(State::from(count))
        }))
    }

    fn put(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "count"))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "count"))
    }

    fn delete(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "count"))
    }
}

impl<State> RouteHandler<State> for LimitHandler
where
    State: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let limit = match value {
                Value::Number(n) => u64::cast_from(n),
                other => {
                    return Err(bad_request!(
                        "limit must be a positive integer, not {other:?}"
                    ))
                }
            };
            let limited = table.limit(limit);
            Ok(State::from(crate::Collection::from(
                crate::table::Table::from(limited),
            )))
        }))
    }

    fn put(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "limit"))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "limit"))
    }

    fn delete(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "limit"))
    }
}

impl<State> RouteHandler<State> for OrderHandler
where
    State: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let (columns, reverse): (Vec<Id>, bool) = if value.matches::<(Vec<Id>, bool)>() {
                value.try_cast_into(|v| bad_request!("invalid order request: {v:?}"))?
            } else {
                let columns: Vec<Id> = value.try_cast_into(|v| bad_request!("invalid column list: {v:?}"))?;
                (columns, false)
            };
            let slice = table.order_by(&columns, reverse);
            Ok(State::from(crate::Collection::from(
                crate::table::Table::from(slice),
            )))
        }))
    }

    fn put(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "order"))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "order"))
    }

    fn delete(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "order"))
    }
}

impl<State> RouteHandler<State> for SelectHandler
where
    State: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let columns: Vec<Id> = value.try_cast_into(|v| bad_request!("invalid column list: {v:?}"))?;
            let selection = table.select(&columns);
            Ok(State::from(crate::Collection::from(
                crate::table::Table::from(selection),
            )))
        }))
    }

    fn put(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "select"))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "select"))
    }

    fn delete(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "select"))
    }
}

impl<State> RouteHandler<State> for SchemaHandler
where
    State: From<tc_value::Value> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        if value != Value::None {
            return Err(bad_request!("this route takes no parameters"));
        }
        let table = self.table.clone();
        let schema_fn = self.schema_fn;
        Ok(Box::pin(async move {
            Ok(State::from(schema_fn(&table)))
        }))
    }

    fn put(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "schema"))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "schema"))
    }

    fn delete(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "schema"))
    }
}

// ─── Static handlers ───────────────────────────────────────────────────

impl<State> super::StaticRouteHandler<State> for CreateHandler
where
    State: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let root = self.root.clone();
        let value: Value = request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let schema: TableSchema = value.try_cast_into(|v| bad_request!("invalid table schema: {v:?}"))?;
            let table = create_table(&root, txn_id, schema).await?;
            Ok(State::from(crate::Collection::from(table)))
        }))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(
            tc_ir::Method::Post,
            "create",
        ))
    }
}

impl<State> super::StaticRouteHandler<State> for CopyHandler
where
    State: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Get, "copy_from"))
    }

    fn post(
        &self,
        txn: &dyn Transaction,
        mut request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let root = self.root.clone();
        let schema_value: Value = request.require("schema")?.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let source_value = request.optional("source")?;
        request.expect_empty()?;

        Ok(Box::pin(async move {
            let schema: TableSchema = schema_value.try_cast_into(|v| bad_request!("invalid table schema: {v:?}"))?;
            let table = create_table(&root, txn_id, schema).await?;

            if let Some(source_scalar) = source_value {
                let source_value = source_scalar.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
                copy_inline_rows(&table, source_value).await?;
            }

            Ok(State::from(crate::Collection::from(table)))
        }))
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Create a new temporary [`TempTable`] under the given root directory.
///
/// A `TempTable` is backed by `b-table::TableLock` and is not transactional.
/// It supports the same read/write methods as a `PersistentTable` but has no
/// commit/rollback/finalize lifecycle.
async fn create_table(
    root: &freqfs::DirLock<crate::PersistentFile>,
    txn_id: TxnId,
    schema: TableSchema,
) -> TCResult<TempTable> {
    let dir_name = format!("table-{txn_id}");

    let table_dir = {
        let mut root = root.write().await;
        root.get_or_create_dir(dir_name)
            .map_err(TCError::internal)?
    };

    TempTable::create(schema, table_dir).map_err(TCError::internal)
}

/// Copy inline row data into a new table.
async fn copy_inline_rows(
    table: &TempTable,
    source: Value,
) -> TCResult<()> {
    let rows = match source {
        Value::Tuple(rows) => rows,
        Value::None => return Ok(()),
        other => {
            return Err(bad_request!(
                "copy_from source must be a tuple of rows, got {other:?}"
            ))
        }
    };

    let key_len = table.schema().key().len();
    let value_len = table.schema().values().len();

    for row_value in rows {
        let row = match row_value {
            Value::Tuple(row) => row,
            other => {
                return Err(bad_request!(
                    "each source row must be a tuple, got {other:?}"
                ))
            }
        };

        if row.len() != key_len + value_len {
            return Err(bad_request!(
                "source row has {} columns but schema expects {} ({} key + {} value)",
                row.len(),
                key_len + value_len,
                key_len,
                value_len
            ));
        }

        let key: Vec<Value> = row[..key_len].to_vec();
        let values: Vec<Value> = row[key_len..].to_vec();
        table
            .upsert_row(key, values)
            .await
            .map_err(TCError::internal)?;
    }

    Ok(())
}
