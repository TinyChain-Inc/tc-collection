use std::collections::HashMap;
use std::fmt;

use b_table::{ColumnRange, IndexSchema, Range, Schema};
use b_tree::Schema as BTreeSchema;
use tc_error::{TCError, TCResult};
use tc_ir::Id;
use tc_value::{Value, ValueType};

use crate::btree::StorageConfig;

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

    fn validate_key(&self, key: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        if key.len() != self.column_names.len() {
            return Err(tc_error::bad_request!(
                "key arity {} does not match index column count {}",
                key.len(),
                self.column_names.len()
            ));
        }

        for (i, (value, expected)) in key.iter().zip(self.column_types.iter()).enumerate() {
            let actual = value.class();
            if &actual != expected {
                return Err(tc_error::bad_request!(
                    "column {i} expected {:?} but got {:?}",
                    expected,
                    actual
                ));
            }
        }

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

        let mut column_map: HashMap<&Id, &Column> = HashMap::with_capacity(key.len() + values.len());
        for col in key.iter().chain(values.iter()) {
            column_map.insert(&col.name, col);
        }

        let mut primary_columns = Vec::with_capacity(key.len() + values.len());
        primary_columns.extend(key.iter().cloned());
        primary_columns.extend(values.iter().cloned());
        let primary = TableIndexSchema::new(primary_columns, storage)?;

        let mut aux_indices = Vec::with_capacity(indices.len());
        for (index_name, column_names) in indices {
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

        Ok(Self {
            key: key_names,
            values: value_names,
            primary,
            indices: aux_indices,
        })
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
            if &actual != expected {
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
            if &actual != expected {
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

    /// Try to construct a [`TableSchema`] from its [`Value`] representation.
    ///
    /// The encoding mirrors the v1 wire format:
    /// ```text
    /// Value::Tuple([
    ///     Value::Tuple([key_columns, value_columns]),
    ///     indices
    /// ])
    /// ```
    /// where each column is `Value::Tuple([name_string, dtype_string])` and
    /// each index is `Value::Tuple([name_string, Value::Tuple([col_name, ...])])`.
    pub fn try_from_value(value: Value) -> TCResult<Self> {
        let (key_values, indices_value) = match &value {
            Value::Tuple(outer) if outer.len() == 2 => (&outer[0], &outer[1]),
            other => return Err(tc_error::bad_request!("invalid table schema: {other:?}")),
        };

        let (key_cols, value_cols) = match key_values {
            Value::Tuple(inner) if inner.len() == 2 => (&inner[0], &inner[1]),
            other => return Err(tc_error::bad_request!("invalid table schema header: {other:?}")),
        };

        let key = parse_columns(key_cols)?;
        let values = parse_columns(value_cols)?;
        let indices = parse_indices(indices_value)?;

        Self::new(key, values, indices, StorageConfig::default())
    }

    /// Encode this schema as a [`Value`] (inverse of [`try_from_value`](Self::try_from_value)).
    pub fn to_value(&self) -> Value {
        let key = self
            .key
            .iter()
            .zip(self.primary.column_types().iter().take(self.key.len()))
            .map(|(name, dtype)| {
                Value::Tuple(vec![
                    Value::String(name.to_string()),
                    Value::String(dtype_to_string(dtype)),
                ])
            })
            .collect::<Vec<_>>();
        let key = Value::Tuple(key);

        let values = self
            .values
            .iter()
            .zip(self.primary.column_types().iter().skip(self.key.len()))
            .map(|(name, dtype)| {
                Value::Tuple(vec![
                    Value::String(name.to_string()),
                    Value::String(dtype_to_string(dtype)),
                ])
            })
            .collect::<Vec<_>>();
        let values = Value::Tuple(values);

        let indices = self
            .indices
            .iter()
            .map(|(name, schema)| {
                let col_count = BTreeSchema::len(schema) - self.key.len();
                let cols = schema
                    .columns()
                    .iter()
                    .take(col_count)
                    .map(|c| Value::String(c.to_string()))
                    .collect::<Vec<_>>();
                Value::Tuple(vec![
                    Value::String(name.clone()),
                    Value::Tuple(cols),
                ])
            })
            .collect::<Vec<_>>();
        let indices = Value::Tuple(indices);

        Value::Tuple(vec![Value::Tuple(vec![key, values]), indices])
    }
}

fn parse_columns(value: &Value) -> TCResult<Vec<Column>> {
    let tuple = match value {
        Value::Tuple(tuple) => tuple,
        other => return Err(tc_error::bad_request!("expected column list, got {other:?}")),
    };

    tuple
        .iter()
        .map(parse_column)
        .collect::<TCResult<Vec<_>>>()
}

fn parse_column(value: &Value) -> TCResult<Column> {
    let pair = match value {
        Value::Tuple(pair) if pair.len() == 2 => pair,
        other => return Err(tc_error::bad_request!("invalid column definition: {other:?}")),
    };

    let name = match &pair[0] {
        Value::String(s) => s.parse::<Id>().map_err(|e| {
            tc_error::bad_request!("invalid column name {s:?}: {e}")
        })?,
        other => return Err(tc_error::bad_request!("column name must be a string, got {other:?}")),
    };

    let dtype = match &pair[1] {
        Value::String(s) => parse_value_type(s)?,
        other => {
            return Err(tc_error::bad_request!(
                "column dtype must be a string, got {other:?}"
            ))
        }
    };

    Ok(Column { name, dtype })
}

fn parse_indices(value: &Value) -> TCResult<Vec<(String, Vec<Id>)>> {
    let tuple = match value {
        Value::Tuple(tuple) => tuple,
        Value::None => return Ok(Vec::new()),
        other => return Err(tc_error::bad_request!("expected index list, got {other:?}")),
    };

    tuple.iter().map(parse_index).collect()
}

fn parse_index(value: &Value) -> TCResult<(String, Vec<Id>)> {
    let pair = match value {
        Value::Tuple(pair) if pair.len() == 2 => pair,
        other => return Err(tc_error::bad_request!("invalid index definition: {other:?}")),
    };

    let name = match &pair[0] {
        Value::String(s) => s.clone(),
        other => return Err(tc_error::bad_request!("index name must be a string, got {other:?}")),
    };

    let cols = match &pair[1] {
        Value::Tuple(cols) => cols
            .iter()
            .map(|c| match c {
                Value::String(s) => s.parse::<Id>().map_err(|e| {
                    tc_error::bad_request!("invalid index column name {s:?}: {e}")
                }),
                other => Err(tc_error::bad_request!(
                    "index column name must be a string, got {other:?}"
                )),
            })
            .collect::<TCResult<Vec<_>>>()?,
        other => return Err(tc_error::bad_request!("expected index column list, got {other:?}")),
    };

    Ok((name, cols))
}

fn parse_value_type(s: &str) -> TCResult<ValueType> {
    match s {
        "Number" => Ok(ValueType::Number),
        "String" => Ok(ValueType::String),
        "Link" => Ok(ValueType::Link),
        "None" => Ok(ValueType::None),
        "Tuple" => Ok(ValueType::Tuple),
        other => Err(tc_error::bad_request!("unknown value type: {other}")),
    }
}

fn dtype_to_string(dtype: &ValueType) -> String {
    match dtype {
        ValueType::Number => "Number",
        ValueType::String => "String",
        ValueType::Link => "Link",
        ValueType::None => "None",
        ValueType::Tuple => "Tuple",
    }
    .to_string()
}

impl fmt::Display for TableSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TableSchema(key={:?}, values={:?}", self.key, self.values)?;
        if !self.indices.is_empty() {
            write!(f, ", indices={:?}", self.indices.iter().map(|(n, _)| n).collect::<Vec<_>>())?;
        }
        write!(f, ")")
    }
}
