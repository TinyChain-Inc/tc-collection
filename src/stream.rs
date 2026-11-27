use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::stream::BoxStream;
use tc_value::Value;

mod overlay;

pub(crate) use overlay::Overlay;

pub(crate) type ReadPermit = txn_lock::semaphore::PermitRead<txn_lock::set::Range<Vec<Value>>>;
pub(crate) type ReadSnapshot<D> = (ReadPermit, Vec<Arc<D>>);

/// A pull-driven stream which owns the guard that makes its items valid.
///
/// The stream is dropped before the guard, so blocked transactional work is
/// notified only after no further item can be polled. The first error terminates
/// the stream and releases its guards immediately.
pub struct GuardedStream<T, Guard> {
    stream: Option<BoxStream<'static, T>>,
    guard: Option<Guard>,
}

impl<T, Guard> GuardedStream<T, Guard> {
    pub(crate) fn new(stream: BoxStream<'static, T>, guard: Guard) -> Self {
        Self {
            stream: Some(stream),
            guard: Some(guard),
        }
    }

    /// Transform the contents while keeping termination cleanup outside the transform.
    pub(crate) fn map_stream(
        self,
        map: impl FnOnce(BoxStream<'static, T>) -> BoxStream<'static, T>,
    ) -> Self {
        Self {
            stream: self.stream.map(map),
            guard: self.guard,
        }
    }
}

impl<T, E, Guard: Unpin> futures::Stream for GuardedStream<Result<T, E>, Guard> {
    type Item = Result<T, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = match self.stream.as_mut() {
            Some(stream) => stream.as_mut().poll_next(cx),
            None => return Poll::Ready(None),
        };

        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            self.stream = None;
            self.guard = None;
        }

        result
    }
}
