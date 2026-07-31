//! Individual route handler structs.
//!
//! Each handler is a separate struct (not an enum variant), following the v1
//! pattern in `table/public.rs`.  Each has a `From` impl from `PersistentTable`
//! and implements [`super::RouteHandler`] via the verb trait methods.
//!
//! Handlers are generic over `Resp`, which must support the appropriate
//! `From` impls — this mirrors v1's `State: From<Collection> + From<Value>
//! + From<u64>` bounds.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use b_table::Range;
use safecast::CastFrom;
use tc_error::{bad_request, TCError, TCResult};
use tc_ir::{Id, Map, Scalar, Transaction, TxnId};
use tc_value::Value;

use super::selector::{cast_into_range, KeyOrRange};
use super::RouteHandler;
use crate::table::{PersistentTable, TableSchema};

/// Convert a [`txn_lock::Error`] into a [`TCError`].
///
/// TODO: this should be `impl From<txn_lock::Error> for TCError` in the
/// `tc-error` crate (or `txn_lock`), but neither crate currently depends
/// on the other.
fn txn_err(err: txn_lock::Error) -> TCError {
    match err {
        txn_lock::Error::Committed => TCError::conflict("transaction already committed"),
        txn_lock::Error::Conflict => TCError::conflict(err),
        txn_lock::Error::Outdated => TCError::not_found("transaction has been finalized"),
        txn_lock::Error::WouldBlock => TCError::conflict("transactional lock would block"),
        txn_lock::Error::Background(cause) => TCError::internal(cause),
    }
}

/// Extract a [`Value`] from a [`Scalar`] request envelope.
///
/// TODO: this should be `impl TryCastFrom<Scalar> for Value` in the
/// `tc-ir` crate (where `Scalar` is defined).
pub(crate) fn scalar_to_value(scalar: Scalar) -> TCResult<Value> {
    match scalar {
        Scalar::Value(value) => Ok(value),
        Scalar::Tuple(items) => {
            let values = items
                .into_iter()
                .map(scalar_to_value)
                .collect::<TCResult<Vec<_>>>()?;
            Ok(Value::Tuple(values))
        }
        Scalar::Map(_) => Err(bad_request!("expected a value, not a map")),
        Scalar::Ref(_) => Err(bad_request!("expected a value, not a reference")),
        Scalar::Op(_) => Err(bad_request!("expected a value, not an op")),
    }
}

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
    root: freqfs::DirLock<crate::btree::PersistentFile>,
}

impl CreateHandler {
    pub fn new(root: freqfs::DirLock<crate::btree::PersistentFile>) -> Self {
        Self { root }
    }
}

// ─── CopyHandler ───────────────────────────────────────────────────────

/// Handler for `POST /state/collection/table/copy_from` — create + copy rows.
///
/// Ported from v1 `CopyHandler`.
pub struct CopyHandler {
    root: freqfs::DirLock<crate::btree::PersistentFile>,
}

impl CopyHandler {
    pub fn new(root: freqfs::DirLock<crate::btree::PersistentFile>) -> Self {
        Self { root }
    }
}

// ─── RouteHandler impls ────────────────────────────────────────────────

type GetFut<'a, Resp> = Pin<Box<dyn Future<Output = TCResult<Resp>> + Send + 'a>>;
type PutFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;

impl<Resp> RouteHandler<Resp> for TableHandler
where
    Resp: From<crate::Collection> + From<tc_value::Value> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            match kor {
                KeyOrRange::All => Ok(Resp::from(crate::Collection::from(table))),
                KeyOrRange::Range(range) => {
                    let slice = table.slice(range, &[], false);
                    Ok(Resp::from(crate::Collection::from(
                        crate::table::Table::from(slice),
                    )))
                }
                KeyOrRange::Key(key) => {
                    let row = table.read_row(txn_id, &key).await;
                    match row {
                        Some(row) => Ok(Resp::from(Value::Tuple(row.into_vec()))),
                        None => Ok(Resp::from(Value::None)),
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
        let key_value = scalar_to_value(params.require("key")?)?;
        let value_value = scalar_to_value(params.require("value")?)?;

        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, key_value)?;
            match kor {
                KeyOrRange::All => {
                    let values = value_map_from_value(value_value)?;
                    table
                        .update(txn_id, Range::default(), values)
                        .await
                        .map_err(txn_err)
                }
                KeyOrRange::Range(range) => {
                    let values = value_map_from_value(value_value)?;
                    table.update(txn_id, range, values).await.map_err(txn_err)
                }
                KeyOrRange::Key(key) => {
                    let values = if let Value::Tuple(tuple) = value_value {
                        tuple
                    } else {
                        vec![value_value]
                    };
                    table.upsert_row(txn_id, key, values).await.map_err(txn_err)
                }
            }
        }))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
        Ok(Box::pin(async move {
            let range = cast_into_range(&table, value)?;
            let slice = table.slice(range, &[], false);
            Ok(Resp::from(crate::Collection::from(
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
        let value = scalar_to_value(request)?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            match kor {
                KeyOrRange::All => {
                    table
                        .truncate(txn_id, Range::default())
                        .await
                        .map_err(txn_err)
                }
                KeyOrRange::Key(key) => {
                    table.delete_row(txn_id, key).await.map_err(txn_err)
                }
                KeyOrRange::Range(range) => {
                    table.truncate(txn_id, range).await.map_err(txn_err)
                }
            }
        }))
    }
}

impl<Resp> RouteHandler<Resp> for ContainsHandler
where
    Resp: From<tc_value::Value> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
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
            Ok(Resp::from(Value::from(filled)))
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
    ) -> TCResult<GetFut<'_, Resp>> {
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

impl<Resp> RouteHandler<Resp> for CountHandler
where
    Resp: From<u64> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
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
            Ok(Resp::from(count))
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
    ) -> TCResult<GetFut<'_, Resp>> {
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

impl<Resp> RouteHandler<Resp> for LimitHandler
where
    Resp: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
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
            Ok(Resp::from(crate::Collection::from(
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
    ) -> TCResult<GetFut<'_, Resp>> {
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

impl<Resp> RouteHandler<Resp> for OrderHandler
where
    Resp: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
        Ok(Box::pin(async move {
            let (columns, reverse) = parse_order_request(value)?;
            let slice = table.order_by(&columns, reverse);
            Ok(Resp::from(crate::Collection::from(
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
    ) -> TCResult<GetFut<'_, Resp>> {
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

impl<Resp> RouteHandler<Resp> for SelectHandler
where
    Resp: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let table = self.table.clone();
        let value = scalar_to_value(request)?;
        Ok(Box::pin(async move {
            let columns = parse_column_list(value)?;
            let selection = table.select(&columns);
            Ok(Resp::from(crate::Collection::from(
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
    ) -> TCResult<GetFut<'_, Resp>> {
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

impl<Resp> RouteHandler<Resp> for SchemaHandler
where
    Resp: From<tc_value::Value> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let value = scalar_to_value(request)?;
        if value != Value::None {
            return Err(bad_request!("this route takes no parameters"));
        }
        let table = self.table.clone();
        let schema_fn = self.schema_fn;
        Ok(Box::pin(async move {
            Ok(Resp::from(schema_fn(&table)))
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
    ) -> TCResult<GetFut<'_, Resp>> {
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

impl<Resp> super::StaticRouteHandler<Resp> for CreateHandler
where
    Resp: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        txn: &dyn Transaction,
        request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        let txn_id = txn.id();
        let root = self.root.clone();
        let value = scalar_to_value(request)?;
        Ok(Box::pin(async move {
            let schema = TableSchema::try_from_value(value)?;
            let table = create_table(&root, txn_id, schema).await?;
            Ok(Resp::from(crate::Collection::from(table)))
        }))
    }

    fn post(
        &self,
        _txn: &dyn Transaction,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, Resp>> {
        Err(TCError::method_not_allowed(
            tc_ir::Method::Post,
            "create",
        ))
    }
}

impl<Resp> super::StaticRouteHandler<Resp> for CopyHandler
where
    Resp: From<crate::Collection> + Clone + Send + 'static,
{
    fn get(
        &self,
        _txn: &dyn Transaction,
        _request: Scalar,
    ) -> TCResult<GetFut<'_, Resp>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Get, "copy_from"))
    }

    fn post(
        &self,
        txn: &dyn Transaction,
        mut request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, Resp>> {
        let txn_id = txn.id();
        let root = self.root.clone();
        let schema_value = scalar_to_value(request.require("schema")?)?;
        let source_value = request.optional("source")?;
        request.expect_empty()?;

        Ok(Box::pin(async move {
            let schema = TableSchema::try_from_value(schema_value)?;
            let table = create_table(&root, txn_id, schema).await?;

            if let Some(source_scalar) = source_value {
                let source_value = scalar_to_value(source_scalar)?;
                copy_inline_rows(&table, txn_id, source_value).await?;
            }

            Ok(Resp::from(crate::Collection::from(table)))
        }))
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Create a new [`PersistentTable`] under the given root directory.
async fn create_table(
    root: &freqfs::DirLock<crate::btree::PersistentFile>,
    txn_id: TxnId,
    schema: TableSchema,
) -> TCResult<PersistentTable> {
    let dir_name = format!("table-{txn_id}");

    let table_dir = {
        let mut root = root.write().await;
        root.get_or_create_dir(dir_name)
            .map_err(TCError::internal)?
    };

    let persistent_dir = {
        let mut table_dir = table_dir.write().await;
        table_dir
            .get_or_create_dir("persistent".to_string())
            .map_err(TCError::internal)?
    };

    let txn_dir = {
        let mut table_dir = table_dir.write().await;
        table_dir
            .get_or_create_dir("txn".to_string())
            .map_err(TCError::internal)?
    };

    Ok(PersistentTable::new(persistent_dir, txn_dir, schema))
}

/// Copy inline row data into a new table.
async fn copy_inline_rows(
    table: &PersistentTable,
    txn_id: TxnId,
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
            .upsert_row(txn_id, key, values)
            .await
            .map_err(TCError::internal)?;
    }

    Ok(())
}

/// Parse an order request: either `(columns, reverse)` or just `columns`.
fn parse_order_request(value: Value) -> TCResult<(Vec<Id>, bool)> {
    match value {
        Value::Tuple(ref tuple) if tuple.len() == 2 => {
            let columns = parse_column_list(tuple[0].clone())?;
            let reverse = match &tuple[1] {
                Value::Number(n) => {
                    let n: u64 = u64::cast_from(*n);
                    n != 0
                }
                Value::None => false,
                other => return Err(bad_request!("invalid reverse flag: {other:?}")),
            };
            Ok((columns, reverse))
        }
        other => {
            let columns = parse_column_list(other)?;
            Ok((columns, false))
        }
    }
}

/// Parse a column list from a `Value`.
fn parse_column_list(value: Value) -> TCResult<Vec<Id>> {
    match value {
        Value::Tuple(tuple) => tuple
            .into_iter()
            .map(|v| match v {
                Value::String(s) => s
                    .parse::<Id>()
                    .map_err(|e| bad_request!("invalid column name {s:?}: {e}")),
                other => Err(bad_request!("column name must be a string, got {other:?}")),
            })
            .collect(),
        Value::String(s) => {
            let id = s
                .parse::<Id>()
                .map_err(|e| bad_request!("invalid column name {s:?}: {e}"))?;
            Ok(vec![id])
        }
        other => Err(bad_request!("invalid column list: {other:?}")),
    }
}

/// Parse a `Map<Value>` (column → new value) from a `Value`.
fn value_map_from_value(value: Value) -> TCResult<HashMap<Id, Value>> {
    let tuple = match value {
        Value::Tuple(tuple) => tuple,
        Value::None => return Ok(HashMap::new()),
        other => return Err(bad_request!("invalid update values: {other:?}")),
    };

    let mut map = HashMap::new();
    for entry in tuple {
        let pair = match entry {
            Value::Tuple(pair) if pair.len() == 2 => pair,
            other => return Err(bad_request!("invalid update entry: {other:?}")),
        };
        let name = match &pair[0] {
            Value::String(s) => s
                .parse::<Id>()
                .map_err(|e| bad_request!("invalid column name {s:?}: {e}"))?,
            other => {
                return Err(bad_request!("column name must be a string, got {other:?}"))
            }
        };
        map.insert(name, pair[1].clone());
    }
    Ok(map)
}
