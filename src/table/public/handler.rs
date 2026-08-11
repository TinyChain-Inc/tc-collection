//! Individual route handler structs.
//!
//! Each handler is a separate struct (not an enum variant), following the v1
//! pattern in `table/public.rs`.  Each has a `From` impl from `PersistentTable`
//! and supplies direct implementations for the concrete route enum.
//!
//! Handlers are generic over `State`, which must support the appropriate
//! `From` impls — handlers return their owned `Table` type plus scalar values.

use std::future::Future;
use std::pin::Pin;

use b_table::Range;
use safecast::{CastFrom, Match, TryCastFrom, TryCastInto};
use tc_error::{TCError, TCResult, bad_request};
use tc_ir::{Id, Map, Scalar};
use tc_value::Value;

use super::selector::{KeyOrRange, cast_into_range};
use crate::CollectionState;
use crate::table::{PersistentTable, Table};

// ─── SchemaHandler ─────────────────────────────────────────────────────

/// Handler for `columns` and `key_columns` routes.
///
/// Holds a function pointer that extracts the schema `Value` from the table.
/// Ported from v1 `SchemaHandler<'a, T>`.
pub struct SchemaHandler<Txn> {
    table: PersistentTable<Txn>,
    schema_fn: fn(&PersistentTable<Txn>) -> Value,
}

impl<Txn> SchemaHandler<Txn> {
    pub fn new(table: PersistentTable<Txn>, schema_fn: fn(&PersistentTable<Txn>) -> Value) -> Self {
        Self { table, schema_fn }
    }
}

/// Return the primary column names as a `Value::Tuple` of strings.
pub fn column_schema<Txn>(table: &PersistentTable<Txn>) -> Value {
    let columns = table
        .schema()
        .columns()
        .map(|c| Value::String(c.to_string()))
        .collect();
    Value::Tuple(columns)
}

/// Return the key column names as a `Value::Tuple` of strings.
pub fn key_columns<Txn>(table: &PersistentTable<Txn>) -> Value {
    let key = table
        .schema()
        .key()
        .iter()
        .map(|c| Value::String(c.to_string()))
        .collect();
    Value::Tuple(key)
}

// ─── ContainsHandler ───────────────────────────────────────────────────

/// Handler for `<table>/contains` — check row presence (All / Key / Range).
///
/// Ported from v1 `ContainsHandler<Txn, FE>`.
#[derive(Clone)]
pub struct ContainsHandler<Txn> {
    table: PersistentTable<Txn>,
}

impl<Txn> From<PersistentTable<Txn>> for ContainsHandler<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self { table }
    }
}

// ─── CountHandler ──────────────────────────────────────────────────────

/// Handler for `<table>/count` — count rows (All / Key / Range).
///
/// Ported from v1 `CountHandler<T>`.
#[derive(Clone)]
pub struct CountHandler<Txn> {
    table: PersistentTable<Txn>,
}

impl<Txn> From<PersistentTable<Txn>> for CountHandler<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self { table }
    }
}

// ─── LimitHandler ──────────────────────────────────────────────────────

/// Handler for `<table>/limit` — cap the row stream.
///
/// Ported from v1 `LimitHandler<T>`.
#[derive(Clone)]
pub struct LimitHandler<Txn> {
    table: PersistentTable<Txn>,
}

impl<Txn> From<PersistentTable<Txn>> for LimitHandler<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self { table }
    }
}

// ─── OrderHandler ──────────────────────────────────────────────────────

/// Handler for `<table>/order` — order the row stream.
///
/// Ported from v1 `OrderHandler<T>`.
#[derive(Clone)]
pub struct OrderHandler<Txn> {
    table: PersistentTable<Txn>,
}

impl<Txn> From<PersistentTable<Txn>> for OrderHandler<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self { table }
    }
}

// ─── SelectHandler ─────────────────────────────────────────────────────

/// Handler for `<table>/select` — project columns.
///
/// Ported from v1 `SelectHandler<T>`.
#[derive(Clone)]
pub struct SelectHandler<Txn> {
    table: PersistentTable<Txn>,
}

impl<Txn> From<PersistentTable<Txn>> for SelectHandler<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self { table }
    }
}

// ─── TableHandler ──────────────────────────────────────────────────────

/// Handler for `<table>` — read / slice / upsert / update / truncate / delete.
///
/// Ported from v1 `TableHandler<Txn, FE>`.
#[derive(Clone)]
pub struct TableHandler<Txn> {
    table: PersistentTable<Txn>,
}

impl<Txn> From<PersistentTable<Txn>> for TableHandler<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self { table }
    }
}

// ─── Concrete route operations ──────────────────────────────────────────

type GetFut<'a, State> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
type PutFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;

fn update_values(value: Scalar) -> TCResult<Map<Value>> {
    let Scalar::Map(values) = value else {
        return Err(bad_request!("invalid update values: expected a map"));
    };

    values
        .into_iter()
        .map(|(name, value)| {
            value
                .try_cast_into(|value| bad_request!("invalid update value for {name}: {value:?}"))
                .map(|value| (name, value))
        })
        .collect()
}

impl<Txn: crate::StorageContext> TableHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            match kor {
                KeyOrRange::All => Ok(State::from(Table::from(table))),
                KeyOrRange::Range(range) => {
                    let slice = table.slice(range, &[], false);
                    Ok(State::from(Table::from(slice)))
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

    pub(super) fn put<'a>(&self, txn: &'a Txn, request: Map<Scalar>) -> TCResult<PutFut<'a>> {
        let table = self.table.clone();
        let mut params = request;
        let key_value: Value = params
            .require("key")?
            .try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let value_scalar = params.require("value")?;

        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, key_value)?;
            match kor {
                KeyOrRange::All => {
                    let values = update_values(value_scalar)?;
                    table
                        .update(txn, Range::default(), values)
                        .await
                        .map_err(TCError::from)
                }
                KeyOrRange::Range(range) => {
                    let values = update_values(value_scalar)?;
                    table
                        .update(txn, range, values)
                        .await
                        .map_err(TCError::from)
                }
                KeyOrRange::Key(key) => {
                    let value: Value = value_scalar
                        .try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
                    let values = if let Value::Tuple(tuple) = value {
                        tuple
                    } else {
                        vec![value]
                    };
                    table
                        .upsert_row(txn, key, values)
                        .await
                        .map_err(TCError::from)
                }
            }
        }))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value = Value::Tuple(
            request
                .into_iter()
                .map(|(id, scalar)| {
                    Value::try_cast_from(scalar, |s| bad_request!("expected a value, not {s:?}"))
                        .map(|value| Value::Tuple(vec![Value::String(id.to_string()), value]))
                })
                .collect::<TCResult<_>>()?,
        );
        Ok(Box::pin(async move {
            let range = cast_into_range(&table, value)?;
            let slice = table.slice(range, &[], false);
            Ok(State::from(Table::from(slice)))
        }))
    }

    pub(super) fn delete<'a>(&self, txn: &'a Txn, request: Scalar) -> TCResult<PutFut<'a>> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let kor = KeyOrRange::try_from_value(&table, value)?;
            match kor {
                KeyOrRange::All => table
                    .truncate(txn, Range::default())
                    .await
                    .map_err(TCError::from),
                KeyOrRange::Key(key) => table.delete_row(txn, key).await.map_err(TCError::from),
                KeyOrRange::Range(range) => table.truncate(txn, range).await.map_err(TCError::from),
            }
        }))
    }
}

impl<Txn: crate::StorageContext> ContainsHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
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

    pub(super) fn put(&self, _txn: &Txn, _request: Map<Scalar>) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "contains"))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "contains"))
    }

    pub(super) fn delete(&self, _txn: &Txn, _request: Scalar) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(
            tc_ir::Method::Delete,
            "contains",
        ))
    }
}

impl<Txn: crate::StorageContext> CountHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
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

    pub(super) fn put(&self, _txn: &Txn, _request: Map<Scalar>) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "count"))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "count"))
    }

    pub(super) fn delete(&self, _txn: &Txn, _request: Scalar) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "count"))
    }
}

impl<Txn: crate::StorageContext> LimitHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let limit = match value {
                Value::Number(n) => u64::cast_from(n),
                other => {
                    return Err(bad_request!(
                        "limit must be a positive integer, not {other:?}"
                    ));
                }
            };
            let limited = table.limit(limit);
            Ok(State::from(Table::from(limited)))
        }))
    }

    pub(super) fn put(&self, _txn: &Txn, _request: Map<Scalar>) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "limit"))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "limit"))
    }

    pub(super) fn delete(&self, _txn: &Txn, _request: Scalar) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "limit"))
    }
}

impl<Txn: crate::StorageContext> OrderHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let (columns, reverse): (Vec<Id>, bool) = if value.matches::<(Vec<Id>, bool)>() {
                value.try_cast_into(|v| bad_request!("invalid order request: {v:?}"))?
            } else {
                let columns: Vec<Id> =
                    value.try_cast_into(|v| bad_request!("invalid column list: {v:?}"))?;
                (columns, false)
            };
            let slice = table.order_by(&columns, reverse);
            Ok(State::from(Table::from(slice)))
        }))
    }

    pub(super) fn put(&self, _txn: &Txn, _request: Map<Scalar>) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "order"))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "order"))
    }

    pub(super) fn delete(&self, _txn: &Txn, _request: Scalar) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "order"))
    }
}

impl<Txn: crate::StorageContext> SelectHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        Ok(Box::pin(async move {
            let columns: Vec<Id> =
                value.try_cast_into(|v| bad_request!("invalid column list: {v:?}"))?;
            let selection = table.select(&columns);
            Ok(State::from(Table::from(selection)))
        }))
    }

    pub(super) fn put(&self, _txn: &Txn, _request: Map<Scalar>) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "select"))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "select"))
    }

    pub(super) fn delete(&self, _txn: &Txn, _request: Scalar) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "select"))
    }
}

impl<Txn: crate::StorageContext> SchemaHandler<Txn> {
    pub(super) fn get<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        request: Scalar,
    ) -> TCResult<GetFut<'_, State>> {
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        if value != Value::None {
            return Err(bad_request!("this route takes no parameters"));
        }
        let table = self.table.clone();
        let schema_fn = self.schema_fn;
        Ok(Box::pin(async move { Ok(State::from(schema_fn(&table))) }))
    }

    pub(super) fn put(&self, _txn: &Txn, _request: Map<Scalar>) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Put, "schema"))
    }

    pub(super) fn post<State: CollectionState<Txn = Txn>>(
        &self,
        _txn: &Txn,
        _request: Map<Scalar>,
    ) -> TCResult<GetFut<'_, State>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Post, "schema"))
    }

    pub(super) fn delete(&self, _txn: &Txn, _request: Scalar) -> TCResult<PutFut<'_>> {
        Err(TCError::method_not_allowed(tc_ir::Method::Delete, "schema"))
    }
}
