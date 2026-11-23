//! Permit-bound row stream with lazy `limit` and `select` transforms.
//!
//! `Rows` stays lazy throughout view composition. `limit` and `select` are
//! applied as stream transforms (`take`, column projection) so no full table
//! or intermediate result is ever materialized.
use std::pin::Pin;
use std::task::{Context, Poll};

use b_table::Row;
use futures::stream::{BoxStream, StreamExt, TryStreamExt};
use tc_ir::Id;
use tc_value::Value;

use super::schema::TableSchema;

pub(crate) type RowStream = BoxStream<'static, Result<Row<Value>, std::io::Error>>;

/// Immutable field positions and final occurrences, prepared before row polling.
#[derive(Clone, Debug)]
pub(super) struct Projection {
    indices: Vec<(usize, bool)>,
}

impl Projection {
    pub(super) fn new(schema: &TableSchema, columns: &[Id]) -> Self {
        let indices: Vec<_> = columns
            .iter()
            .filter_map(|column| schema.columns().position(|name| name == column))
            .collect();

        let mut remaining = vec![0; schema.column_count()];
        for &index in &indices {
            remaining[index] += 1;
        }

        let indices = indices
            .into_iter()
            .map(|index| {
                remaining[index] -= 1;
                (index, remaining[index] == 0)
            })
            .collect();

        Self { indices }
    }

    pub(super) fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().map(|&(index, _)| index)
    }

    fn apply(&self, mut row: Row<Value>) -> Row<Value> {
        self.indices
            .iter()
            .filter_map(|&(index, last)| {
                row.get_mut(index).map(|value| {
                    if last {
                        std::mem::take(value)
                    } else {
                        value.clone()
                    }
                })
            })
            .collect()
    }
}

/// A transactional read-permit-bound stream of table rows.
///
/// The read permit keeps the snapshot coherent until EOF, the first error,
/// or drop. Shared guarded-stream cleanup releases the source before the permit.
/// `limit` and `select` return new `Rows` with the transform applied lazily.
pub struct Rows {
    stream: crate::stream::GuardedStream<
        Result<Row<Value>, std::io::Error>,
        Option<crate::stream::ReadPermit>,
    >,
}

impl Rows {
    pub(crate) fn new(stream: RowStream, permit: crate::stream::ReadPermit) -> Self {
        Self {
            stream: crate::stream::GuardedStream::new(stream, Some(permit)),
        }
    }

    pub(crate) fn local(stream: RowStream) -> Self {
        Self {
            stream: crate::stream::GuardedStream::new(stream, None),
        }
    }

    pub(super) async fn try_count(mut self) -> std::io::Result<u64> {
        let mut count = 0;
        while self.try_next().await?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    pub(super) async fn try_is_empty(mut self) -> std::io::Result<bool> {
        Ok(self.try_next().await?.is_none())
    }

    /// Cap this stream to at most `n` rows.
    ///
    /// This is a lazy `take(n)` transform — no rows are consumed until the
    /// returned `Rows` is polled.
    pub fn limit(self, n: u64) -> Self {
        let n = n.try_into().unwrap_or(usize::MAX);
        Self {
            stream: self.stream.map_stream(|stream| stream.take(n).boxed()),
        }
    }

    /// Project only the given `columns` from each row.
    ///
    /// Column positions are resolved against `schema` (key columns followed by
    /// value columns). This is a lazy `map_ok` transform — no rows are
    /// consumed until the returned `Rows` is polled.
    pub fn select(self, schema: &TableSchema, columns: &[Id]) -> Self {
        self.project(Projection::new(schema, columns))
    }

    pub(super) fn project(self, projection: Projection) -> Self {
        Self {
            stream: self
                .stream
                .map_stream(|stream| stream.map_ok(move |row| projection.apply(row)).boxed()),
        }
    }
}

impl futures::Stream for Rows {
    type Item = Result<Row<Value>, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}
