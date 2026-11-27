use std::cmp::Ordering;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::task::Poll;

use collate::Collate;
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};

use super::Overlay;

#[derive(Clone, Copy, Eq, PartialEq)]
struct ByKey(bool);

impl Collate for ByKey {
    type Value = (i32, i32);

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        let order = left.0.cmp(&right.0);
        if self.0 { order.reverse() } else { order }
    }
}

fn input(
    rows: &[(i32, i32)],
    error: Option<&'static str>,
) -> BoxStream<'static, io::Result<(i32, i32)>> {
    let mut rows: Vec<_> = rows.iter().copied().map(Ok).collect();
    if let Some(error) = error {
        rows.push(Err(io::Error::other(error)));
    }

    let mut rows = rows.into_iter();
    let mut pending = true;
    stream::poll_fn(move |cx| {
        pending = !pending;
        if pending {
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(rows.next())
        }
    })
    .boxed()
}

#[tokio::test]
async fn overlays_resolve_identity_in_visibility_order() {
    for reverse in [false, true] {
        for replace in [false, true] {
            let mut expected = std::collections::BTreeMap::from([(1, 10), (3, 30), (5, 50)]);
            let ordered = |mut rows: Vec<_>| {
                if reverse {
                    rows.reverse();
                }

                rows
            };

            let insert: fn(&mut (i32, i32), (i32, i32), bool) = if replace {
                |old, new, _| *old = new
            } else {
                |old, new, present| {
                    if !present {
                        *old = new
                    }
                }
            };

            let mut actual = Overlay::new(
                ByKey(reverse),
                input(
                    &ordered(expected.iter().map(|(k, v)| (*k, *v)).collect()),
                    None,
                ),
                insert,
            );

            for (deletes, inserts) in [
                (vec![(3, 0)], vec![(1, 11), (2, 20), (3, 31)]),
                (vec![(1, 0), (2, 0)], vec![(2, 21), (4, 40)]),
                (vec![(5, 0), (7, 0)], vec![(1, 12), (2, 22)]),
            ] {
                for (key, _) in &deletes {
                    expected.remove(key);
                }

                for (key, value) in &inserts {
                    if replace {
                        expected.insert(*key, *value);
                    } else {
                        expected.entry(*key).or_insert(*value);
                    }
                }

                actual.difference(input(&ordered(deletes), None));
                actual.merge(input(&ordered(inserts), None));
            }

            assert_eq!(
                actual.into_stream().try_collect::<Vec<_>>().await.unwrap(),
                ordered(expected.into_iter().collect())
            );
        }
    }
}

#[derive(Eq, PartialEq)]
struct Counted(ArcIdentity);

#[derive(Clone)]
struct ArcIdentity(Arc<AtomicUsize>);

impl PartialEq for ArcIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ArcIdentity {}

impl Collate for Counted {
    type Value = (i32, i32);

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        self.0.0.fetch_add(1, AtomicOrdering::Relaxed);
        left.0.cmp(&right.0)
    }
}

#[tokio::test]
async fn disjoint_layers_use_logarithmic_merge_selection() {
    let comparisons = Arc::new(AtomicUsize::new(0));
    let mut overlay = Overlay::new(
        Counted(ArcIdentity(comparisons.clone())),
        input(&[], None),
        |old, new, _| *old = new,
    );

    for key in 0..4096 {
        overlay.merge(input(&[(key, key)], None));
    }

    assert_eq!(
        overlay
            .into_stream()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .len(),
        4096
    );

    assert!(comparisons.load(AtomicOrdering::Relaxed) < 4096 * 64);
}

struct Released(Arc<AtomicUsize>);

impl Drop for Released {
    fn drop(&mut self) {
        self.0.fetch_add(1, AtomicOrdering::SeqCst);
    }
}

#[tokio::test]
async fn overlay_releases_inputs_on_error_eof_and_cancellation() {
    for outcome in ["pending", "error", "eof", "unpolled"] {
        let released = Arc::new(AtomicUsize::new(0));
        let source = |mode| {
            let guard = Released(released.clone());
            stream::poll_fn(move |_| {
                let _ = &guard;
                match mode {
                    "error" => Poll::Ready(Some(Err(io::Error::other("source")))),
                    "eof" => Poll::Ready(None),
                    _ => Poll::Pending,
                }
            })
            .boxed()
        };

        let mut overlay = Overlay::new(ByKey(false), source(outcome), |old, new, _| *old = new);
        overlay.merge(source(if outcome == "eof" { "eof" } else { "pending" }));
        let mut stream = overlay.into_stream();
        match outcome {
            "error" => assert!(stream.try_next().await.is_err()),
            "eof" => assert!(stream.try_next().await.unwrap().is_none()),
            "pending" => assert!(futures::poll!(stream.next()).is_pending()),
            _ => {}
        }

        if matches!(outcome, "error" | "eof") {
            assert_eq!(released.load(AtomicOrdering::SeqCst), 2);
        }

        drop(stream);
        assert_eq!(released.load(AtomicOrdering::SeqCst), 2);
    }
}
