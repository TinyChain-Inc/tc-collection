//! Selector parsing: `KeyOrRange` and `cast_into_range`.
//!
//! Ports the v1 `KeyOrRange` and `cast_into_range` helpers that decode a
//! request `Value` into an All / Key / Range selector for a table.
//! These are inherently schema-dependent (they need the table's column
//! list to validate), so they remain as functions rather than
//! `TryCastFrom` impls — matching the v1 pattern.

use std::collections::HashMap;
use std::ops::Bound;

use b_table::{ColumnRange, IndexSchema, Range};
use tc_error::{bad_request, not_found, TCResult};
use tc_ir::Id;
use tc_value::Value;

use super::super::file::PersistentTable;

/// A table selector parsed from a request: all rows, a single key, or a range.
///
/// Ported from v1 `KeyOrRange`.
#[derive(Debug)]
pub(crate) enum KeyOrRange {
    All,
    Key(Vec<Value>),
    Range(Range<Id, Value>),
}

impl KeyOrRange {
    /// Parse a selector [`Value`] in the context of the given table's schema.
    ///
    /// - `Value::None` or an empty tuple → `All`
    /// - A tuple of `(column_name, bound)` pairs → `Range`
    /// - A tuple whose length equals the key arity → `Key`
    ///
    /// Matches v1 parse order: range check (all elements are column-bound
    /// pairs) comes before the key-arity check.
    pub(crate) fn try_from_value(
        table: &PersistentTable,
        value: Value,
    ) -> TCResult<Self> {
        let columns = table.schema().primary().columns();

        match value {
            Value::None => Ok(Self::All),
            Value::Tuple(ref tuple) if tuple.is_empty() => Ok(Self::All),
            Value::Tuple(tuple)
                if tuple.iter().all(|v| match v {
                    Value::Tuple(pair) if pair.len() == 2 => match &pair[0] {
                        Value::String(name) => {
                            columns.iter().any(|c| c.as_str() == name.as_str())
                        }
                        _ => false,
                    },
                    _ => false,
                }) =>
            {
                let range = cast_into_range(table, Value::Tuple(tuple))?;
                Ok(Self::Range(range))
            }
            Value::Tuple(key) if key.len() == table.schema().key().len() => Ok(Self::Key(key)),
            other => Err(bad_request!("invalid table selector: {other:?}")),
        }
    }
}

/// Parse a range selector [`Value`] into a [`Range`].
///
/// Ported from v1 `cast_into_range`.  The value is a tuple of
/// `(column_name, bound)` pairs.  If a bound is itself a 2-tuple it is
/// interpreted as `(lower, upper)` inclusive/excluded bounds; otherwise it
/// is an equality match.
pub(crate) fn cast_into_range(
    table: &PersistentTable,
    value: Value,
) -> TCResult<Range<Id, Value>> {
    let tuple = match value {
        Value::Tuple(tuple) => tuple,
        Value::None => return Ok(Range::default()),
        other => return Err(bad_request!("invalid selection bounds: {other:?}")),
    };

    let columns = table.schema().primary().columns();
    let mut ranges = HashMap::new();

    for entry in tuple {
        let pair = match entry {
            Value::Tuple(pair) if pair.len() == 2 => pair,
            other => return Err(bad_request!("invalid range bound: {other:?}")),
        };

        let col_name = match &pair[0] {
            Value::String(s) => s.parse::<Id>().map_err(|e| {
                bad_request!("invalid column name {s:?}: {e}")
            })?,
            other => return Err(bad_request!("column name must be a string, got {other:?}")),
        };

        if !columns.iter().any(|c| c.as_str() == col_name.as_str()) {
            return Err(not_found!("column not found: {col_name}"));
        }

        let col_range = match &pair[1] {
            Value::Tuple(bounds) if bounds.len() == 2 => {
                let lower = match &bounds[0] {
                    Value::None => Bound::Unbounded,
                    other => Bound::Included(other.clone()),
                };
                let upper = match &bounds[1] {
                    Value::None => Bound::Unbounded,
                    other => Bound::Included(other.clone()),
                };
                ColumnRange::In((lower, upper))
            }
            _ => ColumnRange::Eq(pair[1].clone()),
        };

        ranges.insert(col_name, col_range);
    }

    Ok(ranges.into())
}
