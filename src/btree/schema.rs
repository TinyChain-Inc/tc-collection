use std::sync::Arc;

use b_tree::Schema;
use number_general::Number;
use tc_error::{TCError, TCResult};
use tc_value::{Value, ValueType};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeColumnSchema {
    pub name: String,
    pub dtype: ValueType,
    pub max_size: Option<Number>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StorageConfig {
    pub block_size: usize,
    pub order: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            block_size: 4_096,
            order: 64,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeSchema {
    storage: StorageConfig,
    columns: Arc<Vec<BTreeColumnSchema>>,
}

impl BTreeSchema {
    pub fn new(columns: Vec<BTreeColumnSchema>, storage: StorageConfig) -> TCResult<Self> {
        if columns.is_empty() {
            return Err(TCError::bad_request("BTree schema must contain columns"));
        }

        Ok(Self {
            storage,
            columns: Arc::new(columns),
        })
    }

    pub fn columns(&self) -> &[BTreeColumnSchema] {
        &self.columns
    }
}

impl TryFrom<Vec<BTreeColumnSchema>> for BTreeSchema {
    type Error = TCError;

    fn try_from(columns: Vec<BTreeColumnSchema>) -> TCResult<Self> {
        Self::new(columns, StorageConfig::default())
    }
}

impl Schema for BTreeSchema {
    type Error = TCError;
    type Value = Value;

    fn block_size(&self) -> usize {
        self.storage.block_size
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn order(&self) -> usize {
        self.storage.order
    }

    fn validate_prefix(&self, prefix: &[Value]) -> TCResult<()> {
        if prefix.is_empty() || prefix.len() > self.len() {
            return Err(tc_error::bad_request!(
                "BTree separator must contain 1..={} columns",
                self.len()
            ));
        }

        for (i, (value, column)) in prefix.iter().zip(self.columns.iter()).enumerate() {
            if !column.dtype.accepts(&value.class()) {
                return Err(tc_error::bad_request!(
                    "BTree key column {i} expected {:?} but got {:?}",
                    column.dtype,
                    value.class()
                ));
            }
        }

        Ok(())
    }

    fn validate_key(&self, key: &[Value]) -> TCResult<()> {
        if key.len() != self.len() {
            return Err(tc_error::bad_request!(
                "BTree keys must have arity {}",
                self.len()
            ));
        }

        self.validate_prefix(key)
    }
}
