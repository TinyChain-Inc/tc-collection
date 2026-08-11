//! Individual route handler structs.
//!
//! The route enum delegates each verb to these table-owned operations. This
//! module has no transport or serialization responsibilities.

use b_table::Range;
use safecast::{CastFrom, Match, TryCastInto};
use tc_error::{TCResult, bad_request};
use tc_ir::{Id, Map, Scalar};
use tc_value::Value;

use super::selector::{KeyOrRange, cast_into_range};
use crate::CollectionState;
use crate::table::Table;

/// Handler for schema and column-name routes.
///
/// Holds a function pointer that extracts the schema `Value` from the table.
/// Ported from v1 `SchemaHandler<'a, T>`.
pub struct SchemaHandler<Txn> {
    table: Table<Txn>,
    schema_fn: fn(&Table<Txn>) -> Value,
}

impl<Txn> SchemaHandler<Txn> {
    pub fn new(table: Table<Txn>, schema_fn: fn(&Table<Txn>) -> Value) -> Self {
        Self { table, schema_fn }
    }
}

/// Return all column definitions.
pub fn column_schema<Txn>(table: &Table<Txn>) -> Value {
    let columns = table
        .schema()
        .column_schema()
        .map(Value::cast_from)
        .collect();
    Value::Tuple(columns)
}

/// Return the primary-key column definitions.
pub fn key_columns<Txn>(table: &Table<Txn>) -> Value {
    let key = table.schema().key_schema().map(Value::cast_from).collect();
    Value::Tuple(key)
}

/// Return the primary-key column names.
pub fn key_names<Txn>(table: &Table<Txn>) -> Value {
    Value::Tuple(
        table
            .schema()
            .key()
            .iter()
            .map(|name| Value::String(name.to_string()))
            .collect(),
    )
}

/// Handler for `<table>/contains` — check row presence (All / Key / Range).
///
/// Ported from v1 `ContainsHandler<Txn, FE>`.
#[derive(Clone)]
pub struct ContainsHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for ContainsHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

/// Handler for `<table>/count` — count rows (All / Key / Range).
///
/// Ported from v1 `CountHandler<T>`.
#[derive(Clone)]
pub struct CountHandler<Txn> {
    table: Table<Txn>,
}

#[derive(Clone)]
pub struct IsEmptyHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for IsEmptyHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

#[derive(Clone)]
pub struct InsertHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for InsertHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

impl<Txn> From<Table<Txn>> for CountHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

/// Handler for `<table>/limit` — cap the row stream.
///
/// Ported from v1 `LimitHandler<T>`.
#[derive(Clone)]
pub struct LimitHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for LimitHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

/// Handler for `<table>/order` — order the row stream.
///
/// Ported from v1 `OrderHandler<T>`.
#[derive(Clone)]
pub struct OrderHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for OrderHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

/// Handler for `<table>/select` — project columns.
///
/// Ported from v1 `SelectHandler<T>`.
#[derive(Clone)]
pub struct SelectHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for SelectHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

/// Handler for `<table>` — read / slice / upsert / update / truncate / delete.
///
/// Ported from v1 `TableHandler<Txn, FE>`.
#[derive(Clone)]
pub struct TableHandler<Txn> {
    table: Table<Txn>,
}

impl<Txn> From<Table<Txn>> for TableHandler<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self { table }
    }
}

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

fn value_from_state<State: CollectionState>(state: State, expected: &str) -> TCResult<Value> {
    state
        .into_scalar()?
        .try_cast_into(|state| bad_request!("expected {expected}, not {state:?}"))
}

impl<State, Txn> tc_ir::Handler<State> for TableHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, txn: &Txn, request: Scalar) -> TCResult<State> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        match KeyOrRange::try_from_value(table.schema(), value)? {
            KeyOrRange::All => Ok(State::from(table)),
            KeyOrRange::Range(range) => {
                let slice = table.slice(range, &[], false)?;
                Ok(State::from(Table::from(slice)))
            }
            KeyOrRange::Key(key) => match table.read_row(txn_id, &key).await? {
                Some(row) => Ok(State::from(Value::Tuple(row.into_vec()))),
                None => Ok(State::from(Value::None)),
            },
        }
    }

    async fn put(&self, txn: &Txn, key: Scalar, value: State) -> TCResult<()> {
        let table = self.table.clone();
        let key: Value = key.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let value = value.into_scalar()?;
        match KeyOrRange::try_from_value(table.schema(), key)? {
            KeyOrRange::All => {
                table
                    .update(txn, Range::default(), update_values(value)?)
                    .await
            }
            KeyOrRange::Range(range) => table.update(txn, range, update_values(value)?).await,
            KeyOrRange::Key(key) => {
                let value: Value =
                    value.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
                let values = match value {
                    Value::Tuple(values) => values,
                    value => vec![value],
                };
                table.upsert_row(txn, key, values).await
            }
        }
    }

    async fn post(&self, _txn: &Txn, request: Map<State>) -> TCResult<State> {
        let table = self.table.clone();
        let value = Value::Tuple(
            request
                .into_iter()
                .map(|(id, state)| {
                    value_from_state(state, "a Table bound")
                        .map(|value| Value::Tuple(vec![Value::String(id.to_string()), value]))
                })
                .collect::<TCResult<_>>()?,
        );
        let range = cast_into_range(table.schema(), value)?;
        let slice = table.slice(range, &[], false)?;
        Ok(State::from(Table::from(slice)))
    }

    async fn delete(&self, txn: &Txn, request: Scalar) -> TCResult<()> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        match KeyOrRange::try_from_value(table.schema(), value)? {
            KeyOrRange::All => table.truncate(txn, Range::default()).await,
            KeyOrRange::Key(key) => table.delete_row(txn, key).await,
            KeyOrRange::Range(range) => table.truncate(txn, range).await,
        }
    }
}

impl<State, Txn> tc_ir::Handler<State> for ContainsHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, txn: &Txn, request: Scalar) -> TCResult<State> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let filled = match KeyOrRange::try_from_value(table.schema(), value)? {
            KeyOrRange::All => !table.is_empty(txn_id).await?,
            KeyOrRange::Key(key) => table.contains_row(txn_id, &key).await?,
            KeyOrRange::Range(range) => {
                let slice = table.slice(range, &[], false)?;
                !slice.is_empty(txn_id).await?
            }
        };
        Ok(State::from(Value::from(filled)))
    }
}

impl<State, Txn> tc_ir::Handler<State> for CountHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, txn: &Txn, request: Scalar) -> TCResult<State> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let count: u64 = match KeyOrRange::try_from_value(table.schema(), value)? {
            KeyOrRange::All => table.count(txn_id).await?,
            KeyOrRange::Key(key) => u64::from(table.contains_row(txn_id, &key).await?),
            KeyOrRange::Range(range) => {
                let slice = table.slice(range, &[], false)?;
                slice.count(txn_id).await?
            }
        };
        Ok(State::from(count))
    }
}

impl<State, Txn> tc_ir::Handler<State> for IsEmptyHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, txn: &Txn, request: Scalar) -> TCResult<State> {
        let txn_id = txn.id();
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let empty = match KeyOrRange::try_from_value(table.schema(), value)? {
            KeyOrRange::All => table.is_empty(txn_id).await?,
            KeyOrRange::Key(key) => !table.contains_row(txn_id, &key).await?,
            KeyOrRange::Range(range) => table.slice(range, &[], false)?.is_empty(txn_id).await?,
        };
        Ok(State::from(Value::from(empty)))
    }
}

impl<State, Txn> tc_ir::Handler<State> for InsertHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn post(&self, txn: &Txn, mut request: Map<State>) -> TCResult<State> {
        let table = self.table.clone();
        let key = value_from_state(request.require("key")?, "a Table key")?;
        let key: Vec<Value> =
            key.try_cast_into(|value| bad_request!("expected a Table key, not {value:?}"))?;
        let values = value_from_state(request.require("values")?, "Table values")?;
        let values: Vec<Value> =
            values.try_cast_into(|value| bad_request!("expected Table values, not {value:?}"))?;
        table.insert_row(txn, key, values).await?;
        Ok(State::from(Value::None))
    }
}

impl<State, Txn> tc_ir::Handler<State> for LimitHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, _txn: &Txn, request: Scalar) -> TCResult<State> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let Value::Number(limit) = value else {
            return Err(bad_request!(
                "limit must be a positive integer, not {value:?}"
            ));
        };
        Ok(State::from(Table::from(
            table.limit(u64::cast_from(limit))?,
        )))
    }
}

impl<State, Txn> tc_ir::Handler<State> for OrderHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, _txn: &Txn, request: Scalar) -> TCResult<State> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let (columns, reverse): (Vec<Id>, bool) = if value.matches::<(Vec<Id>, bool)>() {
            value.try_cast_into(|v| bad_request!("invalid order request: {v:?}"))?
        } else {
            let columns = value.try_cast_into(|v| bad_request!("invalid column list: {v:?}"))?;
            (columns, false)
        };
        Ok(State::from(Table::from(table.order_by(&columns, reverse)?)))
    }
}

impl<State, Txn> tc_ir::Handler<State> for SelectHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, _txn: &Txn, request: Scalar) -> TCResult<State> {
        let table = self.table.clone();
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        let columns: Vec<Id> =
            value.try_cast_into(|v| bad_request!("invalid column list: {v:?}"))?;
        Ok(State::from(Table::from(table.select(&columns)?)))
    }
}

impl<State, Txn> tc_ir::Handler<State> for SchemaHandler<Txn>
where
    State: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    async fn get(&self, _txn: &Txn, request: Scalar) -> TCResult<State> {
        let value: Value =
            request.try_cast_into(|s| bad_request!("expected a value, not {s:?}"))?;
        if value != Value::None {
            return Err(bad_request!("this route takes no parameters"));
        }
        Ok(State::from((self.schema_fn)(&self.table)))
    }
}
