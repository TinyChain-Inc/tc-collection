//! Flat multiway composition of strictly ordered, unique native index streams.

use std::io;

use collate::{Collate, MergeQueue};
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};

struct Source<T> {
    stream: Option<BoxStream<'static, io::Result<T>>>,
    head: Option<T>,
    inserted: bool,
}

pub(crate) struct Overlay<T, C, R> {
    collator: C,
    resolve_insert: R,
    sources: Vec<Source<T>>,
    refill: Vec<usize>,
}

impl<T: Send + 'static, C, R> Overlay<T, C, R>
where
    C: Collate<Value = T> + Send + 'static,
    R: FnMut(&mut T, T, bool) + Send + 'static,
{
    /// Sources are added in visibility order: base, then each delta's deletes
    /// and inserts. The callback chooses the retained insertion representation.
    pub(crate) fn new(
        collator: C,
        base: BoxStream<'static, io::Result<T>>,
        resolve_insert: R,
    ) -> Self {
        let mut result = Self {
            collator,
            resolve_insert,
            sources: Vec::new(),
            refill: Vec::new(),
        };

        result.merge(base);
        result
    }

    pub(crate) fn merge(&mut self, stream: BoxStream<'static, io::Result<T>>) {
        self.push(stream, true);
    }

    pub(crate) fn difference(&mut self, stream: BoxStream<'static, io::Result<T>>) {
        self.push(stream, false);
    }

    fn push(&mut self, stream: BoxStream<'static, io::Result<T>>, inserted: bool) {
        self.refill.push(self.sources.len());
        self.sources.push(Source {
            stream: Some(stream),
            head: None,
            inserted,
        });
    }

    fn take(&mut self, queue: &mut MergeQueue) -> T {
        let source = queue
            .pop(
                |index| self.sources[index].head.as_ref().expect("ready head"),
                &self.collator,
            )
            .expect("selected source");

        let value = self.sources[source].head.take().expect("selected head");

        self.refill.push(source);
        value
    }

    pub(crate) fn into_stream(self) -> BoxStream<'static, io::Result<T>> {
        let queue = MergeQueue::new(self.sources.len());
        stream::try_unfold((self, queue), |(mut state, mut queue)| async move {
            loop {
                // Input order establishes deterministic first-error precedence.
                // Refill only on demand, after delivering the preceding key.
                state.refill.sort_unstable_by(|left, right| right.cmp(left));
                while let Some(source) = state.refill.pop() {
                    tokio::task::coop::consume_budget().await;
                    if let Some(stream) = &mut state.sources[source].stream {
                        state.sources[source].head = stream.try_next().await?;
                        if state.sources[source].head.is_none() {
                            state.sources[source].stream = None;
                        }
                    }

                    if state.sources[source].head.is_some() {
                        queue.push(
                            source,
                            |index| state.sources[index].head.as_ref().expect("ready head"),
                            &state.collator,
                        );
                    }
                }

                let Some(first) = queue.peek() else {
                    return Ok(None);
                };

                let mut inserted = state.sources[first].inserted;
                let mut value = state.take(&mut queue);
                while let Some(source) = queue.peek() {
                    tokio::task::coop::consume_budget().await;
                    if !state
                        .collator
                        .cmp(
                            &value,
                            state.sources[source].head.as_ref().expect("ready head"),
                        )
                        .is_eq()
                    {
                        break;
                    }

                    let incoming = state.take(&mut queue);
                    if state.sources[source].inserted {
                        (state.resolve_insert)(&mut value, incoming, inserted);
                        inserted = true;
                    } else {
                        value = incoming;
                        inserted = false;
                    }
                }

                if inserted {
                    return Ok(Some((value, (state, queue))));
                }
            }
        })
        .boxed()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/stream/overlay.rs"]
mod tests;
