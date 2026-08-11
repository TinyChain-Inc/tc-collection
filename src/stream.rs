use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::stream::BoxStream;
use tc_value::Value;

pub(crate) type ReadPermit = txn_lock::semaphore::PermitRead<txn_lock::set::Range<Vec<Value>>>;

/// A pull-driven stream which owns the guard that makes its items valid.
///
/// The stream is dropped before the guard, so blocked transactional work is
/// notified only after no further item can be polled.
pub struct GuardedStream<T, Guard> {
    stream: BoxStream<'static, T>,
    _guard: Guard,
}

impl<T, Guard> GuardedStream<T, Guard> {
    pub(crate) fn new(stream: BoxStream<'static, T>, guard: Guard) -> Self {
        Self {
            stream,
            _guard: guard,
        }
    }
}

impl<T, Guard: Unpin> futures::Stream for GuardedStream<T, Guard> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}
