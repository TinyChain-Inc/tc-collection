use std::collections::{HashMap, HashSet};
use std::fmt;

use b_table::{ColumnRange, IndexSchema, Range, Schema};
use b_tree::Schema as BTreeSchema;
use safecast::{CastFrom, TryCastFrom};

use super::stream::Projection;
use crate::btree::StorageConfig;
use tc_error::TCError;
use tc_ir::Id;
use tc_value::{Value, ValueType};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Column {
    pub name: Id,
    pub dtype: ValueType,
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", self.name, self.dtype)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableIndexSchema {
    column_names: Vec<Id>,
    column_types: Vec<ValueType>,
    storage: StorageConfig,
}

impl TableIndexSchema {
    pub fn new(columns: Vec<Column>, storage: StorageConfig) -> Result<Self, TCError> {
        if columns.is_empty() {
            return Err(tc_error::bad_request!(
                "table index must have at least one column"
            ));
        }

        let column_names = columns.iter().map(|c| c.name.clone()).collect();
        let column_types = columns.iter().map(|c| c.dtype.clone()).collect();

        Ok(Self {
            column_names,
            column_types,
            storage,
        })
    }

    pub fn column_types(&self) -> &[ValueType] {
        &self.column_types
    }

    pub fn storage(&self) -> &StorageConfig {
        &self.storage
    }

    pub fn into_columns(self) -> impl Iterator<Item = Column> {
        self.column_names
            .into_iter()
            .zip(self.column_types)
            .map(|(name, dtype)| Column { name, dtype })
    }
}

impl b_tree::Schema for TableIndexSchema {
    type Error = TCError;
    type Value = Value;

    fn block_size(&self) -> usize {
        self.storage.block_size
    }

    fn len(&self) -> usize {
        self.column_names.len()
    }

    fn order(&self) -> usize {
        self.storage.order
    }

    fn validate_prefix(&self, prefix: &[Value]) -> Result<(), Self::Error> {
        if prefix.is_empty() || prefix.len() > self.column_names.len() {
            return Err(tc_error::bad_request!(
                "index separator must contain 1..={} columns",
                self.column_names.len()
            ));
        }

        for (i, (value, expected)) in prefix.iter().zip(self.column_types.iter()).enumerate() {
            let actual = value.class();
            if !expected.accepts(&actual) {
                return Err(tc_error::bad_request!(
                    "column {i} expected {:?} but got {:?}",
                    expected,
                    actual
                ));
            }
        }

        Ok(())
    }

    fn validate_key(&self, key: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        if key.len() != self.column_names.len() {
            return Err(tc_error::bad_request!(
                "key arity {} does not match index column count {}",
                key.len(),
                self.column_names.len()
            ));
        }

        self.validate_prefix(&key)?;

        Ok(key)
    }
}

impl IndexSchema for TableIndexSchema {
    type Id = Id;

    fn columns(&self) -> &[Id] {
        &self.column_names
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableSchema {
    key: Vec<Id>,
    values: Vec<Id>,
    primary: TableIndexSchema,
    indices: Vec<(String, TableIndexSchema)>,
}

impl TableSchema {
    pub fn new(
        key: Vec<Column>,
        values: Vec<Column>,
        indices: Vec<(String, Vec<Id>)>,
        storage: StorageConfig,
    ) -> Result<Self, TCError> {
        if key.is_empty() {
            return Err(tc_error::bad_request!(
                "table schema must have at least one key column"
            ));
        }

        let key_names: Vec<Id> = key.iter().map(|c| c.name.clone()).collect();
        let value_names: Vec<Id> = values.iter().map(|c| c.name.clone()).collect();

        let mut column_map: HashMap<&Id, &Column> =
            HashMap::with_capacity(key.len() + values.len());

        for col in key.iter().chain(values.iter()) {
            column_map.insert(&col.name, col);
        }

        let mut primary_columns = Vec::with_capacity(key.len() + values.len());
        primary_columns.extend(key.iter().cloned());
        primary_columns.extend(values.iter().cloned());
        let primary = TableIndexSchema::new(primary_columns, storage)?;

        let mut aux_indices = Vec::with_capacity(indices.len());
        for (index_name, column_names) in indices {
            if column_names.is_empty() || column_names.iter().any(|name| key_names.contains(name)) {
                return Err(tc_error::bad_request!(
                    "index {index_name} must declare nonempty value columns; primary keys are appended automatically"
                ));
            }

            let mut index_columns = Vec::with_capacity(column_names.len() + key.len());
            for name in &column_names {
                let col = column_map.get(name).copied().ok_or_else(|| {
                    tc_error::bad_request!(
                        "index {index_name} references nonexistent column {name}"
                    )
                })?;

                index_columns.push(col.clone());
            }

            for col in &key {
                index_columns.push(col.clone());
            }

            let index_schema = TableIndexSchema::new(index_columns, storage)?;
            aux_indices.push((index_name, index_schema));
        }

        let schema = Self {
            key: key_names,
            values: value_names,
            primary,
            indices: aux_indices,
        };

        Schema::validate(&schema).map_err(TCError::bad_request)?;
        Ok(schema)
    }

    pub fn key(&self) -> &[Id] {
        &self.key
    }

    pub fn values(&self) -> &[Id] {
        &self.values
    }

    pub fn primary(&self) -> &TableIndexSchema {
        &self.primary
    }

    pub fn indices(&self) -> &[(String, TableIndexSchema)] {
        &self.indices
    }

    pub fn column_count(&self) -> usize {
        self.key.len() + self.values.len()
    }

    pub fn range_from_key(&self, key: &[Value]) -> Result<Range<Id, Value>, TCError> {
        if key.len() != self.key.len() {
            return Err(tc_error::bad_request!(
                "key arity {} does not match schema key arity {}",
                key.len(),
                self.key.len()
            ));
        }

        let mut range = HashMap::with_capacity(key.len());
        for (name, val) in self.key.iter().zip(key) {
            range.insert(name.clone(), ColumnRange::Eq(val.clone()));
        }

        Ok(range.into())
    }

    pub fn storage(&self) -> &StorageConfig {
        self.primary.storage()
    }

    pub(super) fn same_semantics(&self, other: &Self) -> bool {
        let same_columns = |left: &TableIndexSchema, right: &TableIndexSchema| {
            left.column_names == right.column_names && left.column_types == right.column_types
        };

        self.key == other.key
            && self.values == other.values
            && same_columns(&self.primary, &other.primary)
            && self.indices.len() == other.indices.len()
            && self.indices.iter().zip(&other.indices).all(
                |((name, index), (other_name, other_index))| {
                    name == other_name && same_columns(index, other_index)
                },
            )
    }

    pub(super) fn project(&self, columns: Vec<Id>) -> Result<(Self, Projection), TCError> {
        for column in &columns {
            if !self.columns().any(|name| name == column) {
                return Err(tc_error::not_found!("column not found: {column}"));
            }
        }

        for key in &self.key {
            if !columns.contains(key) {
                return Err(tc_error::bad_request!(
                    "Table selection must include key column {key}"
                ));
            }
        }

        if !columns.starts_with(&self.key) {
            return Err(TCError::bad_request(
                "Table selection must retain the original key prefix",
            ));
        }

        let mut unique = HashSet::with_capacity(columns.len());
        for column in &columns {
            if !unique.insert(column) {
                return Err(tc_error::bad_request!(
                    "duplicate Table selection column {column}"
                ));
            }
        }

        drop(unique);
        let projection = Projection::new(self, &columns);
        let mut selected: Vec<Column> = columns
            .into_iter()
            .zip(projection.indices())
            .map(|(name, index)| Column {
                name,
                dtype: self.primary.column_types[index].clone(),
            })
            .collect();

        let values = selected.split_off(self.key.len());
        let schema = Self::new(selected, values, Vec::new(), *self.storage())?;
        Ok((schema, projection))
    }
}

impl Schema for TableSchema {
    type Id = Id;
    type Error = TCError;
    type Value = Value;
    type Index = TableIndexSchema;

    fn key(&self) -> &[Id] {
        &self.key
    }

    fn values(&self) -> &[Id] {
        &self.values
    }

    fn primary(&self) -> &TableIndexSchema {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, TableIndexSchema)] {
        &self.indices
    }

    fn validate_key(&self, key: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        if key.len() != self.key.len() {
            return Err(tc_error::bad_request!(
                "key arity {} does not match schema key arity {}",
                key.len(),
                self.key.len()
            ));
        }

        let key_types = &self.primary.column_types[..self.key.len()];
        for (i, (value, expected)) in key.iter().zip(key_types.iter()).enumerate() {
            let actual = value.class();
            if !expected.accepts(&actual) {
                return Err(tc_error::bad_request!(
                    "key column {i} expected {:?} but got {:?}",
                    expected,
                    actual
                ));
            }
        }

        Ok(key)
    }

    fn validate_values(&self, values: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        if values.len() != self.values.len() {
            return Err(tc_error::bad_request!(
                "value arity {} does not match schema value arity {}",
                values.len(),
                self.values.len()
            ));
        }

        let value_types = &self.primary.column_types[self.key.len()..];
        for (i, (value, expected)) in values.iter().zip(value_types.iter()).enumerate() {
            let actual = value.class();
            if !expected.accepts(&actual) {
                return Err(tc_error::bad_request!(
                    "value column {i} expected {:?} but got {:?}",
                    expected,
                    actual
                ));
            }
        }

        Ok(values)
    }
}

impl TableSchema {
    /// Return an iterator over all column names (key followed by value columns).
    pub fn columns(&self) -> impl Iterator<Item = &Id> {
        self.key.iter().chain(self.values.iter())
    }

    /// Return all column definitions in row order.
    pub fn column_schema(&self) -> impl Iterator<Item = Column> + '_ {
        self.columns()
            .zip(self.primary.column_types())
            .map(|(name, dtype)| Column {
                name: name.clone(),
                dtype: dtype.clone(),
            })
    }

    /// Return the primary-key column definitions.
    pub fn key_schema(&self) -> impl Iterator<Item = Column> + '_ {
        self.column_schema().take(self.key.len())
    }
}

impl TryFrom<Value> for TableSchema {
    type Error = TCError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let (columns, indices) = <(Value, Value)>::try_cast_from(value, |_| {
            tc_error::bad_request!("expected Table schema [[key, values], indices]")
        })?;

        let (key, values) = <(Value, Value)>::try_cast_from(columns, |_| {
            tc_error::bad_request!("expected Table schema columns [key, values]")
        })?;

        let key = cast_column_list(key, "key")?;
        let values = cast_column_list(values, "value")?;
        let indices = cast_index_list(indices)?;

        Self::new(key, values, indices, StorageConfig::default())
    }
}

fn cast_column_list(value: Value, kind: &str) -> Result<Vec<Column>, TCError> {
    let Value::Tuple(columns) = value else {
        return Err(tc_error::bad_request!("expected Table {kind} column list"));
    };

    columns
        .into_iter()
        .map(|column| {
            Column::try_cast_from(column, |column| {
                tc_error::bad_request!("invalid Table {kind} column: {column:?}")
            })
        })
        .collect()
}

fn cast_index_list(value: Value) -> Result<Vec<(String, Vec<Id>)>, TCError> {
    let indices = match value {
        Value::None => return Ok(Vec::new()),
        Value::Tuple(indices) => indices,
        _ => return Err(tc_error::bad_request!("expected Table index list")),
    };

    indices
        .into_iter()
        .map(|index| {
            let (name, columns) = <(Value, Value)>::try_cast_from(index, |_| {
                tc_error::bad_request!("expected Table index [name, columns]")
            })?;

            let Value::String(name) = name else {
                return Err(tc_error::bad_request!("Table index name must be a string"));
            };

            let columns = Vec::<Id>::try_cast_from(columns, |_| {
                tc_error::bad_request!("invalid columns for Table index {name}")
            })?;

            Ok((name, columns))
        })
        .collect()
}

impl CastFrom<TableSchema> for Value {
    fn cast_from(schema: TableSchema) -> Self {
        let key = schema
            .key
            .iter()
            .zip(schema.primary.column_types().iter().take(schema.key.len()))
            .map(|(name, dtype)| {
                Value::Tuple(vec![
                    Value::String(name.to_string()),
                    Value::cast_from(dtype.clone()),
                ])
            })
            .collect::<Vec<_>>();

        let key = Value::Tuple(key);

        let values = schema
            .values
            .iter()
            .zip(schema.primary.column_types().iter().skip(schema.key.len()))
            .map(|(name, dtype)| {
                Value::Tuple(vec![
                    Value::String(name.to_string()),
                    Value::cast_from(dtype.clone()),
                ])
            })
            .collect::<Vec<_>>();

        let values = Value::Tuple(values);

        let indices = schema
            .indices
            .iter()
            .map(|(name, index_schema)| {
                let col_count = BTreeSchema::len(index_schema) - schema.key.len();
                let cols = index_schema
                    .columns()
                    .iter()
                    .take(col_count)
                    .map(|c| Value::String(c.to_string()))
                    .collect::<Vec<_>>();

                Value::Tuple(vec![Value::String(name.clone()), Value::Tuple(cols)])
            })
            .collect::<Vec<_>>();

        let indices = Value::Tuple(indices);

        Value::Tuple(vec![Value::Tuple(vec![key, values]), indices])
    }
}

impl TryCastFrom<Value> for Column {
    fn can_cast_from(value: &Value) -> bool {
        <(Id, ValueType)>::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        let (name, dtype) = <(Id, ValueType)>::opt_cast_from(value)?;
        Some(Column { name, dtype })
    }
}

impl CastFrom<Column> for Value {
    fn cast_from(col: Column) -> Self {
        Value::Tuple(vec![
            Value::String(col.name.to_string()),
            Value::cast_from(col.dtype),
        ])
    }
}

impl fmt::Display for TableSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TableSchema(key={:?}, values={:?}",
            self.key, self.values
        )?;

        if !self.indices.is_empty() {
            write!(
                f,
                ", indices={:?}",
                self.indices.iter().map(|(n, _)| n).collect::<Vec<_>>()
            )?;
        }

        write!(f, ")")
    }
}
