//! Route handler enum and verb trait implementations.
//!
//! [`TableRoute`] is the per-instance route handler enum.  Each variant
//! corresponds to one sub-route from the v1 route/API matrix (parity port §4).
//! The enum implements `HandleGet`, `HandlePut`, `HandlePost`, and
//! `HandleDelete`, dispatching to the appropriate [`PersistentTable`] method.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;

use b_table::Range;
use safecast::CastFrom;
use tc_error::{bad_request, TCError, TCResult};
use tc_ir::{
    HandleDelete, HandleGet, HandlePost, HandlePut, Handler, Id, Map, Method, Scalar,
    Transaction,
};
use tc_value::Value;

use super::super::file::PersistentTable;
use super::response::TableResponse;
use super::selector::{cast_into_range, column_schema, key_columns_value, key_names_value, KeyOrRange};

/// Convert a [`txn_lock::Error`] into a structured [`TCError`].
pub(crate) fn txn_err(err: txn_lock::Error) -> TCError {
    match err {
        txn_lock::Error::Committed => TCError::conflict("transaction already committed"),
        txn_lock::Error::Conflict => TCError::conflict(err),
        txn_lock::Error::Outdated => TCError::not_found("transaction has been finalized"),
        txn_lock::Error::WouldBlock => TCError::conflict("transactional lock would block"),
        txn_lock::Error::Background(cause) => TCError::internal(cause),
    }
}

/// Extract a [`Value`] from a [`Scalar`] request envelope.
pub fn scalar_to_value(scalar: Scalar) -> TCResult<Value> {
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

/// A single route handler for a [`PersistentTable`] instance.
///
/// Each variant corresponds to one sub-route identified in the v1
/// route/API matrix (parity port §4).  The enum dispatches verb trait
/// implementations to the appropriate [`PersistentTable`] method.
#[derive(Clone)]
pub enum TableRoute {
    /// `<table>` — read / slice / upsert / update / truncate / delete
    Table(PersistentTable),
    /// `<table>/columns` — return primary column names
    Columns(PersistentTable),
    /// `<table>/contains` — check row presence (All / Key / Range)
    Contains(PersistentTable),
    /// `<table>/count` — count rows (All / Key / Range)
    Count(PersistentTable),
    /// `<table>/key_columns` — return key column names
    KeyColumns(PersistentTable),
    /// `<table>/key_names` — return key column names (alias)
    KeyNames(PersistentTable),
    /// `<table>/limit` — cap the row stream
    Limit(PersistentTable),
    /// `<table>/order` — order the row stream
    Order(PersistentTable),
    /// `<table>/select` — project columns
    Select(PersistentTable),
}

impl fmt::Debug for TableRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Table(_) => "Table",
            Self::Columns(_) => "Columns",
            Self::Contains(_) => "Contains",
            Self::Count(_) => "Count",
            Self::KeyColumns(_) => "KeyColumns",
            Self::KeyNames(_) => "KeyNames",
            Self::Limit(_) => "Limit",
            Self::Order(_) => "Order",
            Self::Select(_) => "Select",
        };
        f.write_str(name)
    }
}

type GetFut<'a> = Pin<Box<dyn Future<Output = TCResult<TableResponse>> + Send + 'a>>;
type PutFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
type PostFut<'a> = Pin<Box<dyn Future<Output = TCResult<TableResponse>> + Send + 'a>>;
type DeleteFut<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;

impl<T: Transaction + ?Sized> HandleGet<T> for TableRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = TableResponse;
    type Error = TCError;
    type Fut<'a>
        = GetFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn get<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        let txn_id = txn.id();
        match self {
            Self::Table(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                Ok(Box::pin(async move {
                    let kor = KeyOrRange::try_from_value(&table, value)?;
                    match kor {
                        KeyOrRange::All => Ok(TableResponse::Table(table)),
                        KeyOrRange::Range(range) => {
                            let slice = table.slice(range, &[], false);
                            Ok(TableResponse::Slice(slice))
                        }
                        KeyOrRange::Key(key) => {
                            let row = table.read_row(txn_id, &key).await;
                            match row {
                                Some(row) => {
                                    Ok(TableResponse::Value(Value::Tuple(row.into_vec())))
                                }
                                None => Ok(TableResponse::Value(Value::None)),
                            }
                        }
                    }
                }))
            }

            Self::Columns(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                if value != Value::None {
                    return Err(bad_request!("columns route takes no parameters"));
                }
                Ok(Box::pin(async move {
                    Ok(TableResponse::Value(column_schema(&table)))
                }))
            }

            Self::Contains(table) => {
                let table = table.clone();
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
                    Ok(TableResponse::Value(Value::from(filled)))
                }))
            }

            Self::Count(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                Ok(Box::pin(async move {
                    let kor = KeyOrRange::try_from_value(&table, value)?;
                    let count = match kor {
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
                    Ok(TableResponse::Value(Value::from(count)))
                }))
            }

            Self::KeyColumns(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                if value != Value::None {
                    return Err(bad_request!("key_columns route takes no parameters"));
                }
                Ok(Box::pin(async move {
                    Ok(TableResponse::Value(key_columns_value(&table)))
                }))
            }

            Self::KeyNames(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                if value != Value::None {
                    return Err(bad_request!("key_names route takes no parameters"));
                }
                Ok(Box::pin(async move {
                    Ok(TableResponse::Value(key_names_value(&table)))
                }))
            }

            Self::Limit(table) => {
                let table = table.clone();
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
                    Ok(TableResponse::Limited(limited))
                }))
            }

            Self::Order(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                Ok(Box::pin(async move {
                    let (columns, reverse) = parse_order_request(value)?;
                    let slice = table.order_by(&columns, reverse);
                    Ok(TableResponse::Slice(slice))
                }))
            }

            Self::Select(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                Ok(Box::pin(async move {
                    let columns = parse_column_list(value)?;
                    let selection = table.select(&columns);
                    Ok(TableResponse::Selection(selection))
                }))
            }
        }
    }
}

impl<T: Transaction + ?Sized> HandlePut<T> for TableRoute {
    type Request = Map<Scalar>;
    type RequestContext = ();
    type Response = ();
    type Error = TCError;
    type Fut<'a>
        = PutFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn put<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        let txn_id = txn.id();
        match self {
            Self::Table(table) => {
                let table = table.clone();
                let mut params = request;
                let key_scalar = params.require("key")?;
                let value_scalar = params.require("value")?;
                let key_value = scalar_to_value(key_scalar)?;
                let value_value = scalar_to_value(value_scalar)?;

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
                            table
                                .upsert_row(txn_id, key, values)
                                .await
                                .map_err(txn_err)
                        }
                    }
                }))
            }
            _ => Err(<Self as Handler<T>>::method_not_supported(Method::Put)),
        }
    }
}

impl<T: Transaction + ?Sized> HandlePost<T> for TableRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = TableResponse;
    type Error = TCError;
    type Fut<'a>
        = PostFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn post<'a>(&'a self, _txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::Table(table) => {
                let table = table.clone();
                let value = scalar_to_value(request)?;
                Ok(Box::pin(async move {
                    let range = cast_into_range(&table, value)?;
                    let slice = table.slice(range, &[], false);
                    Ok(TableResponse::Slice(slice))
                }))
            }
            _ => Err(<Self as Handler<T>>::method_not_supported(Method::Post)),
        }
    }
}

impl<T: Transaction + ?Sized> HandleDelete<T> for TableRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = ();
    type Error = TCError;
    type Fut<'a>
        = DeleteFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn delete<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        let txn_id = txn.id();
        match self {
            Self::Table(table) => {
                let table = table.clone();
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
            _ => Err(<Self as Handler<T>>::method_not_supported(Method::Delete)),
        }
    }
}

// ─── parsing helpers ───────────────────────────────────────────────────

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
                other => {
                    return Err(bad_request!(
                        "invalid reverse flag: {other:?}"
                    ))
                }
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
                Value::String(s) => s.parse::<Id>().map_err(|e| {
                    bad_request!("invalid column name {s:?}: {e}")
                }),
                other => Err(bad_request!("column name must be a string, got {other:?}")),
            })
            .collect(),
        Value::String(s) => {
            let id = s.parse::<Id>().map_err(|e| {
                bad_request!("invalid column name {s:?}: {e}")
            })?;
            Ok(vec![id])
        }
        other => Err(bad_request!("invalid column list: {other:?}")),
    }
}

/// Parse a `Map<Value>` (column → new value) from a `Value`.
///
/// The value is a tuple of `(column_name, new_value)` pairs, matching the
/// v1 `Map<Value>::try_from(value)` semantics.
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
            Value::String(s) => s.parse::<Id>().map_err(|e| {
                bad_request!("invalid column name {s:?}: {e}")
            })?,
            other => return Err(bad_request!("column name must be a string, got {other:?}")),
        };
        map.insert(name, pair[1].clone());
    }
    Ok(map)
}
