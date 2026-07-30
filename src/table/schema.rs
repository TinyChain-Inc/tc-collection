use std::collections::HashMap;
use std::fmt;

use b_table::{ColumnRange, IndexSchema, Range, Schema};
use tc_error::TCError;
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

impl fmt::Display for TableSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TableSchema(key={:?}, values={:?}", self.key, self.values)?;
        if !self.indices.is_empty() {
            write!(f, ", indices={:?}", self.indices.iter().map(|(n, _)| n).collect::<Vec<_>>())?;
        }
        write!(f, ")")
    }
}
