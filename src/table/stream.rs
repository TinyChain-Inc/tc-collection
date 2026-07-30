//! Permit-bound row stream with lazy `limit` and `select` transforms.
//!
//! Mirrors the v1 `Rows<'a>` type: a read-permit-bound `Stream` of `Row<Value>`
//! that stays lazy throughout view composition. `limit` and `select` are
//! applied as stream transforms (`take`, column projection) so no full table
//! or intermediate result is ever materialized.
use std::pin::Pin;
use std::task::{Context, Poll};

use b_table::Row;
use futures::stream::{BoxStream, StreamExt, TryStreamExt};
use tc_ir::Id;
use tc_value::Value;

use super::schema::TableSchema;

type Permit = txn_lock::semaphore::PermitRead<txn_lock::set::Range<Vec<Value>>>;

type RowStream = BoxStream<'static, Result<Row<Value>, std::io::Error>>;

/// A transactional read-permit-bound stream of table rows.
///
/// The read permit is held for the lifetime of the stream so that the
/// transactional snapshot remains coherent while rows are consumed.
/// `limit` and `select` return new `Rows` with the transform applied lazily.
///
/// Fields are dropped in declaration order (`stream` before `_permit`), so the
/// stream is released before the permit notifies blocked transactions.
pub struct Rows {
    stream: RowStream,
    _permit: Permit,
}

impl Rows {
    pub(crate) fn new(stream: RowStream, permit: Permit) -> Self {
        Self {
            stream,
            _permit: permit,
        }
    }

    /// Cap this stream to at most `n` rows.
    ///
    /// This is a lazy `take(n)` transform — no rows are consumed until the
    /// returned `Rows` is polled.
    pub fn limit(self, n: u64) -> Self {
        let n = n.try_into().unwrap_or(usize::MAX);
        let stream = self.stream.take(n);
        Self::new(stream.boxed(), self._permit)
    }

    /// Project only the given `columns` from each row.
    ///
    /// Column positions are resolved against `schema` (key columns followed by
    /// value columns). This is a lazy `map_ok` transform — no rows are
    /// consumed until the returned `Rows` is polled.
    pub fn select(self, schema: &TableSchema, columns: &[Id]) -> Self {
        let indices = Self::column_indices(schema, columns);
        let stream = self.stream.map_ok(move |row| {
            let projected: Row<Value> =
                indices.iter().filter_map(|&i| row.get(i).cloned()).collect();
            projected
        });
        Self::new(stream.boxed(), self._permit)
    }

    fn column_indices(schema: &TableSchema, columns: &[Id]) -> Vec<usize> {
        let key = schema.key();
        let values = schema.values();
        let all: Vec<&Id> = key.iter().chain(values.iter()).collect();

        let mut indices = Vec::with_capacity(columns.len());
        for col in columns {
            if let Some(i) = all.iter().position(|name| *name == col) {
                indices.push(i);
            }
        }
        indices
    }
}

impl futures::Stream for Rows {
    type Item = Result<Row<Value>, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}
