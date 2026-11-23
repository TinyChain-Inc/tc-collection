use std::sync::Arc;
use std::sync::atomic::Ordering;

use fensor::TensorSource;
use futures::StreamExt;

use super::{TOUCHED_PAGE_ENTRIES, observation};
use crate::tensor::Data;
use crate::tensor::tests::{create, setup};
use tc_ir::{Transact, Transaction};

// Synthetic layers use real shared visibility state; injected completion models
// canonical application before the finalized frontier is advanced.
fn test_snapshot<T: fensor::TensorElement>(
    canonical: fensor::Tensor<crate::PersistentFile, T>,
    deltas: Vec<Arc<super::Delta<crate::PersistentFile, T>>>,
) -> super::Snapshot<crate::PersistentFile, T>
where
    crate::PersistentFile: fensor::TensorFileEntry<T>,
{
    let owner = super::Owner::<crate::test::TestTxn, T>::new(canonical);
    for (index, delta) in deltas.into_iter().enumerate() {
        owner
            .core
            .write_state()
            .committed
            .insert(crate::test::tx(index as u16 + 1), Some(delta));
    }

    owner.snapshot(crate::test::tx(u16::MAX), None).unwrap()
}

fn assert_length_integrity(error: fensor::Error) {
    let fensor::Error::Io(error) = error else {
        panic!("stored length must return I/O integrity: {error}")
    };

    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(freqfs::IntegrityError::is_marked(&error));
    assert!(error.get_ref().is_some());
    let error = tc_error::TCError::from(error);
    assert!(error.requires_recovery());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn batched_reads_select_sources_and_preserve_materialization_handoff() {
    use fensor::{Layout, StorageGeometry, TensorRead, TensorSchema, TensorTransform, TensorView};
    use futures::{FutureExt, TryStreamExt};

    use crate::StorageContext;
    use crate::persistence::PersistentDelta;

    const ROOT: &str = "00000000-0000-0000-0000-000000000000";

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        Box::pin(async {
            for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
                let (_temp, txn) = setup().await;
                let geometry = StorageGeometry::new(
                    TensorSchema::new(
                        number_general::NumberType::Float(number_general::FloatType::F64),
                        [3, 5].as_slice().into(),
                    )
                    .unwrap(),
                    layout, ([1, 4].as_slice()).iter().copied().product::<u64>() as usize,
                )
                .unwrap();

                let directory = txn.subcontext_unique().context().await.unwrap();
                let canonical =
                    fensor::Tensor::create_with_geometry(directory.clone(), geometry.clone())
                        .await
                        .unwrap();

                for id in 0..geometry.block_count() {
                    let mut block = vec![0.; geometry.block_len(id).unwrap()];
                    block.fill((id + 1) as f64);

                    canonical.replace_logical_block(id, block).await.unwrap();
                }

                if matches!(layout, Layout::Dense) {
                    let blocks = directory.read().await.get_dir("blocks").cloned().unwrap();
                    let first = blocks.read().await.get_file("0").cloned().unwrap();
                    let second = blocks.read().await.get_file("1").cloned().unwrap();
                    let native = TensorView::new(canonical.clone());
                    let transactional = TensorView::new(test_snapshot(canonical.clone(), Vec::new()));

                    let mut native_buffered = false;

                    for (index, mut stream) in [
                        native.read_blocks().unwrap(),
                        transactional.read_blocks().unwrap(),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let held = first.write::<Vec<f64>>().await.unwrap();
                        // Observe storage blocking, independently of how much
                        // cooperative budget the fixture setup consumed.
                        assert!(
                            tokio::task::unconstrained(async {
                                stream.next().now_or_never().is_none()
                            })
                            .await
                        );

                        let buffered = second.try_write::<Vec<f64>>().is_err();
                        if index == 0 {
                            native_buffered = buffered;
                        } else {
                            // Match native prefetch under this host's concurrency
                            // bound, including its one-CPU behavior.
                            assert_eq!(buffered, native_buffered);
                        }

                        drop(stream);
                        drop(held);
                        drop(second.try_write::<Vec<f64>>().unwrap());
                        drop(first.try_write::<Vec<f64>>().unwrap());
                    }
                }

                let first = super::Delta::create(
                    txn.subcontext_unique().context().await.unwrap(),
                    geometry.clone(),
                )
                .await
                .unwrap();

                for (id, value) in [(0, 10.), (1, 11.), (2, 0.), (geometry.block_count() - 1, 14.)] {
                    let mut block = vec![0.; geometry.block_len(id).unwrap()];
                    block.fill(value);

                    first.write(id, block).await.unwrap();
                }

                let second = super::Delta::create(
                    txn.subcontext_unique().context().await.unwrap(),
                    geometry.clone(),
                )
                .await
                .unwrap();

                second.write(0, vec![0.; 4]).await.unwrap();
                let mut replacement = vec![0.; geometry.block_len(1).unwrap()];
                replacement[0] = 21.;
                second.write(1, replacement).await.unwrap();
                let last = geometry.block_count() - 1;
                second.write(last, vec![24.; geometry.block_len(last).unwrap()]).await.unwrap();
                let snapshot = test_snapshot(canonical.clone(), vec![first.clone(), second.clone()]);

                let view = TensorView::new(snapshot.clone());
                let mut expected = vec![0.; 15];
                for (flat, expected_value) in expected.iter_mut().enumerate() {
                    let coord = [flat as u64 / 5, flat as u64 % 5];
                    let (id, offset) = geometry.block_position(&coord).unwrap();
                    *expected_value = if id == 0 { 0. } else if id == last { 24. } else if id == 1 { if offset == 0 { 21. } else { 0. } } else if id == 2 { 0. } else { (id + 1) as f64 };
                }

                let blocks: Vec<_> = view.read_blocks().unwrap().try_collect().await.unwrap();
                assert_eq!(blocks.concat(), expected);
                assert_eq!(view.read_value(&[0, 4]).await.unwrap(), 21.);
                let repeated = view
                    .clone()
                    .slice(
                        vec![
                            fensor::AxisRange::Of(vec![2, 0, 2]),
                            fensor::AxisRange::Of(vec![4, 0, 4]),
                        ]
                        .into(),
                    )
                    .unwrap();

                let blocks: Vec<_> = repeated.read_blocks().unwrap().try_collect().await.unwrap();
                assert_eq!(blocks.concat(), [expected[14], expected[10], expected[14], expected[4], expected[0], expected[4], expected[14], expected[10], expected[14]]);

                let payload = second._workspace.read().await.get_dir("payload").cloned().unwrap();
                let values = payload.read().await.get_dir("values").cloned().unwrap();
                let primary = values.read().await.get_dir("primary").cloned().unwrap();
                let root = primary.read().await.get_file(ROOT).cloned().unwrap();
                for extra in [1., f64::NAN, f64::INFINITY] {
                    let saved = {
                        let mut node = root.write::<fensor::SparseNode<f64>>().await.unwrap();
                        let saved = node.clone();
                        let b_tree::Node::Leaf(rows) = &mut *node else { panic!("small fixture leaf") };
                        let row = rows.iter_mut().find(|row| row[0] == fensor::SparseCell::Key(1)).unwrap();
                        let fensor::SparseCell::Payload(values) = &mut row[1] else { panic!("payload") };
                        values.push(extra);
                        saved
                    };

                    assert_length_integrity(view.read_value(&[0, 4]).await.unwrap_err());
                    assert_length_integrity(view.read_blocks().unwrap().try_concat().await.unwrap_err());
                    assert_length_integrity(snapshot.read_logical_block(1).await.unwrap_err());
                    *root.write::<fensor::SparseNode<f64>>().await.unwrap() = saved;
                }

                // A numerical read paused in payload I/O must reject obsolete data
                // and errors after successful materialization publishes its handoff.
                let payload = second
                    ._workspace
                    .read()
                    .await
                    .get_dir("payload")
                    .cloned()
                    .unwrap();

                let values = payload.read().await.get_dir("values").cloned().unwrap();
                let primary = values.read().await.get_dir("primary").cloned().unwrap();
                let root = primary.read().await.get_file(ROOT).cloned().unwrap();
                let guard = root.write::<fensor::SparseNode<f64>>().await.unwrap();
                let mut read = Box::pin(view.read_value(&[0, 4]));
                futures::future::poll_fn(|cx| {
                    assert!(std::future::Future::poll(read.as_mut(), cx).is_pending());
                    std::task::Poll::Ready(())
                })
                .await;

                drop(guard);
                first.apply_to(&canonical).await.unwrap();
                let (value, applied) = futures::join!(read, second.apply_to(&canonical));
                applied.unwrap();
                snapshot.visibility.0.write().unwrap().finalized = Some(crate::test::tx(2));
                assert_eq!(value.unwrap(), 21.);
                *root.write::<fensor::SparseNode<f64>>().await.unwrap() = b_tree::Node::Leaf(vec![
                    vec![fensor::SparseCell::Key(1), fensor::SparseCell::Key(99)],
                ]);

                let blocks: Vec<_> = view.read_blocks().unwrap().try_collect().await.unwrap();
                assert_eq!(blocks.concat(), expected);

                // Source selection finishes before native payload reads. An invalid
                // later membership row wins over an earlier malformed payload.
                let bad = super::Delta::create(
                    txn.subcontext_unique().context().await.unwrap(),
                    geometry.clone(),
                )
                .await
                .unwrap();

                bad.write(0, vec![1.; 4]).await.unwrap();
                let payload = bad
                    ._workspace
                    .read()
                    .await
                    .get_dir("payload")
                    .cloned()
                    .unwrap();

                let values = payload.read().await.get_dir("values").cloned().unwrap();
                let primary = values.read().await.get_dir("primary").cloned().unwrap();
                let root = primary.read().await.get_file(ROOT).cloned().unwrap();
                *root.write::<fensor::SparseNode<f64>>().await.unwrap() = b_tree::Node::Leaf(vec![
                    vec![fensor::SparseCell::Key(0), fensor::SparseCell::Key(99)],
                ]);

                let touched = bad
                    ._workspace
                    .read()
                    .await
                    .get_dir("touched")
                    .cloned()
                    .unwrap();

                let root = touched.read().await.get_file(ROOT).cloned().unwrap();
                *root.write::<crate::TensorIndexNode>().await.unwrap() =
                    b_tree::Node::Leaf(vec![vec![0], vec![4, 4]]);

                let snapshot = test_snapshot(canonical, vec![bad]);

                let view = TensorView::new(snapshot);
                let error = view
                    .read_blocks()
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap_err();

                assert!(
                    error
                        .to_string()
                        .contains("stored row does not match the BTree schema"),
                    "{error}"
                );

                assert!(matches!(&error, fensor::Error::Io(error) if freqfs::IntegrityError::is_marked(error)));
                // Native scans validate every row in the containing page. Restore
                // membership so the independent malformed-payload assertion is reached.
                *root.write::<crate::TensorIndexNode>().await.unwrap() =
                    b_tree::Node::Leaf(vec![vec![0], vec![4]]);

                let error = view.read_value(&[0, 0]).await.unwrap_err();
                assert!(
                    error.to_string().contains("invalid sparse payload"),
                    "{error}"
                );
            }
        }),
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn owned_occupied_scan_retains_layers_and_releases_leases() {
    Box::pin(observation::scope(async {
        use futures::{StreamExt, TryStreamExt};

        let (_, txn) = setup().await;
        let len = TOUCHED_PAGE_ENTRIES + 1;
        let (tensor, _) = create(&txn, vec![len as u64].into(), true, 4096).await;
        let Data::F64(value) = &tensor.data else {
            panic!("dtype")
        };

        let owner = value.owner().unwrap();
        let delta = owner.stage(&txn).await.unwrap();
        observation::watch(&delta.generation);
        for block in 0..owner.geometry().block_count() {
            let mut values = vec![0.; owner.geometry().block_len(block).unwrap()];
            values.fill(1.);
            delta.write(block, values).await.unwrap();
        }

        owner.publish(&txn, delta.clone()).await.unwrap();
        {
            let snapshot = owner
                .try_snapshot(txn.id(), 0..owner.geometry().block_count())
                .unwrap();

            let lease = Arc::downgrade(snapshot._permit.as_ref().unwrap());
            let mut scan = snapshot.occupied_blocks(0..len as u64);
            drop(snapshot);
            let page: Vec<_> = scan
                .by_ref()
                .take(TOUCHED_PAGE_ENTRIES)
                .try_collect()
                .await
                .unwrap();

            assert_eq!(page.len(), TOUCHED_PAGE_ENTRIES);
            assert!(lease.upgrade().is_some());
            assert_eq!(scan.try_next().await.unwrap().unwrap(), len as u64 - 1);
            assert!(scan.try_next().await.unwrap().is_none());
            assert!(
                lease.upgrade().is_none(),
                "EOF releases lease before stream drop"
            );

            assert_eq!(observation::counts().overlay_starts, 1);
        }

        for consume in [false, true] {
            let snapshot = owner
                .try_snapshot(txn.id(), 0..owner.geometry().block_count())
                .unwrap();

            let lease = Arc::downgrade(snapshot._permit.as_ref().unwrap());
            let mut scan = snapshot.occupied_blocks(0..len as u64);
            drop(snapshot);
            if consume {
                assert!(scan.try_next().await.unwrap().is_some());
            }

            drop(scan);
            assert!(lease.upgrade().is_none());
        }
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn overlay_suppresses_blocks_and_discards_prefetched_workspace_keys() {
    Box::pin(observation::scope(async {
        use futures::{StreamExt, TryStreamExt};

        use crate::persistence::PersistentDelta;

        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![600].into(), true, 2).await;
        let Data::F64(value) = &tensor.data else {
            unreachable!()
        };

        let owner = value.owner().unwrap();
        let canonical = owner.canonical().unwrap();
        for id in 0..owner.geometry().block_count() {
            canonical
                .replace_logical_block(id, vec![1.; owner.geometry().block_len(id).unwrap()])
                .await
                .unwrap();
        }

        let first = owner.stage(&txn).await.unwrap();
        observation::watch(&first.generation);
        first
            .write(0, vec![0.; owner.geometry().block_len(0).unwrap()])
            .await
            .unwrap();

        first
            .write(1, vec![2.; owner.geometry().block_len(1).unwrap()])
            .await
            .unwrap();

        let second = owner.stage(&txn).await.unwrap();
        second
            .write(1, vec![0.; owner.geometry().block_len(1).unwrap()])
            .await
            .unwrap();

        let lower = canonical.occupied_blocks(0..600);
        let snapshot = test_snapshot(canonical.clone(), vec![first.clone(), second.clone()]);
        let mut stream = super::overlay_keys(
            lower,
            snapshot.deltas.iter().cloned(),
            snapshot.visibility.clone(),
            canonical.clone(),
            0..600,
        );

        let key = stream.try_next().await.unwrap().unwrap();
        assert!(key >= 2);
        first.apply_to(&canonical).await.unwrap();
        snapshot.visibility.0.write().unwrap().finalized = Some(crate::test::tx(1));
        first
            ._workspace
            .write()
            .await
            .truncate_and_sync()
            .await
            .unwrap();

        second.apply_to(&canonical).await.unwrap();
        snapshot.visibility.0.write().unwrap().finalized = Some(crate::test::tx(2));
        second
            ._workspace
            .write()
            .await
            .truncate_and_sync()
            .await
            .unwrap();

        let rest: Vec<_> = stream.try_collect().await.unwrap();
        let expected: Vec<_> = canonical
            .occupied_blocks(key + 1..600)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(rest, expected);
        assert!(observation::counts().touched_lookups > 0);
        // Payload keys need no touched or occupied-membership reprobe.
        let own = owner.stage(&txn).await.unwrap();
        observation::watch(&own.generation);
        own.write(0, vec![3.; owner.geometry().block_len(0).unwrap()])
            .await
            .unwrap();

        let snapshot = test_snapshot(canonical.clone(), vec![own.clone()]);
        let keys: Vec<_> = super::overlay_keys(
            futures::stream::empty().boxed(),
            snapshot.deltas.iter().cloned(),
            snapshot.visibility,
            canonical,
            0..600,
        )
        .try_collect()
        .await
        .unwrap();

        assert!(!keys.is_empty());
        assert_eq!(observation::counts().touched_lookups, 0);
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn later_snapshots_survive_finalized_workspace_cleanup() {
    Box::pin(async {
        use fensor::TensorSource;
        use futures::TryStreamExt;

        use tc_ir::{NetworkTime, TxnId};

        for sparse in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![4].into(), sparse, 2).await;
            {
                let Data::F64(value) = &tensor.data else {
                    unreachable!()
                };

                fensor::TensorWrite::write_value(
                    &value.owner().unwrap().canonical().unwrap(),
                    &[1],
                    9.0,
                )
                .await
                .unwrap();
            }

            tensor.write_value(&txn, &[0], 2.0.into()).await.unwrap();
            tensor.write_value(&txn, &[1], 0.0.into()).await.unwrap();
            tensor.write_value(&txn, &[3], 4.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
            let later = TxnId::from_parts(NetworkTime::from_nanos(1), 11);
            let expression = tensor.mul_scalar(later, 2.0.into()).await.unwrap();
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };

            let owner = value.owner().unwrap();
            let snapshot = owner
                .try_snapshot(later, 0..owner.geometry().block_count())
                .unwrap();

            let delta = snapshot.deltas[0].1.clone();
            let entries = if sparse {
                Some(tensor.sparse_elements(later).await.unwrap())
            } else {
                None
            };

            tensor.finalize(&txn.id()).await.unwrap();
            // The caller removes an expired workspace after recursive
            // finalization even though a later expression is still live.
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                delta
                    ._workspace
                    .write()
                    .await
                    .truncate_and_sync()
                    .await
                    .unwrap();

                let values: Vec<_> = expression
                    .elements(later)
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();

                assert_eq!(
                    values,
                    [4.0, 0.0, 0.0, 8.0].map(number_general::Number::from)
                );

                let mut expected = vec![0.0; owner.geometry().block_len(0).unwrap()];
                expected[0] = 2.0;
                assert_eq!(snapshot.read_logical_block(0).await.unwrap(), expected);
                if let Some(entries) = entries {
                    let entries: Vec<_> = entries.try_collect().await.unwrap();
                    assert_eq!(entries, [(vec![0], 2.0.into()), (vec![3], 4.0.into())]);
                    let first = owner.geometry().block_position(&[0]).unwrap().0;
                    let last = owner.geometry().block_position(&[3]).unwrap().0;
                    let keys: Vec<_> = snapshot
                        .occupied_blocks(0..owner.geometry().block_count())
                        .take(TOUCHED_PAGE_ENTRIES)
                        .try_collect()
                        .await
                        .unwrap();

                    assert_eq!(keys, [first, last]);
                    // Each delta cursor resumes on native storage after
                    // the handoff, preserving unvisited occupied keys.
                    let keys: Vec<_> = super::overlay_keys(
                        owner
                            .canonical()
                            .unwrap()
                            .occupied_blocks(first + 1..owner.geometry().block_count()),
                        [(txn.id(), delta.clone())],
                        snapshot.visibility.clone(),
                        owner.canonical().unwrap(),
                        first + 1..owner.geometry().block_count(),
                    )
                    .try_collect()
                    .await
                    .unwrap();

                    assert_eq!(keys, [last]);
                }
            })
            .await
            .unwrap();
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn replacement_pages_reuse_keys_and_validate_entries() {
    Box::pin(observation::scope(async {
        use fensor::{Layout, StorageGeometry, TensorSchema};
        use futures::TryStreamExt;

        use super::Delta;
        use crate::StorageContext;

        const ROOT: &str = "00000000-0000-0000-0000-000000000000";

        // Payload ownership follows canonical chunks, including a dense
        // owner's private matrix and sparse dense suffixes spanning chunks.
        for (layout, shape, block_shape) in [
            (Layout::Dense, vec![2, 8192], vec![1, 4096]),
            (
                Layout::Sparse { axis: Some(0) },
                vec![2, 8192],
                vec![1, 4096],
            ),
            (
                Layout::Sparse { axis: Some(1) },
                vec![2, 2, 8192],
                vec![1, 1, 4096],
            ),
            (Layout::Dense, vec![2, 5], vec![1, 4]),
            (Layout::Sparse { axis: Some(0) }, vec![2, 5], vec![1, 4]),
        ] {
            let (_, txn) = setup().await;
            let geometry = StorageGeometry::new(
                TensorSchema::new(
                    number_general::NumberType::Float(number_general::FloatType::F64),
                    shape.into(),
                )
                .unwrap(),
                layout,
                (block_shape).iter().copied().product::<u64>() as usize,
            )
            .unwrap();

            let patch = Delta::<_, f64>::create(
                txn.subcontext_unique().context().await.unwrap(),
                geometry.clone(),
            )
            .await
            .unwrap();

            let occupied = || {
                patch
                    .payload
                    .occupied_blocks(0..geometry.block_count())
                    .try_collect::<Vec<_>>()
            };

            let invalid_length = patch
                .write(1, vec![1.; geometry.block_capacity() - 1])
                .await
                .unwrap_err();

            assert!(!freqfs::IntegrityError::is_marked(&invalid_length));
            assert!(!tc_error::TCError::from(invalid_length).requires_recovery());
            // Invalid IDs reject before waiting for the touched-index guard.
            let touched = patch.touched.write().await;
            let invalid_read = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                patch.read(geometry.block_count()),
            )
            .await
            .unwrap()
            .unwrap_err();

            assert!(
                invalid_read
                    .to_string()
                    .contains("logical block out of bounds")
            );

            drop(touched);
            let invalid_id = patch
                .write(geometry.block_count(), Vec::new())
                .await
                .unwrap_err();

            assert_eq!(invalid_id.kind(), std::io::ErrorKind::InvalidData);
            assert!(
                invalid_id
                    .to_string()
                    .contains("logical block out of bounds")
            );

            assert!(!freqfs::IntegrityError::is_marked(&invalid_id));
            assert!(!tc_error::TCError::from(invalid_id).requires_recovery());
            for len in [
                geometry.block_len(1).unwrap() - 1,
                geometry.block_len(1).unwrap() + 1,
            ] {
                let error = patch.write(1, vec![1.; len]).await.unwrap_err();
                assert!(!freqfs::IntegrityError::is_marked(&error));
                assert!(!tc_error::TCError::from(error).requires_recovery());
            }

            assert_eq!(patch.generation.load(Ordering::Acquire), 0);
            assert!(patch.touched_page(None).await.unwrap().is_empty());
            assert!(occupied().await.unwrap().is_empty());

            let mut values = vec![0.; geometry.block_len(1).unwrap()];
            values[0] = 7.;
            patch.write(1, values.clone()).await.unwrap();
            assert_eq!(patch.touched_page(None).await.unwrap(), [1]);
            assert_eq!(occupied().await.unwrap(), [1]);
            assert_eq!(patch.read(1).await.unwrap(), Some(values));
            for id in [0, 2, geometry.block_count() - 1] {
                assert!(patch.read(id).await.unwrap().is_none());
            }

            // No zero-valued sibling is allocated, and deleting the one
            // row retains explicit-zero replacement membership.
            let zeros = vec![0.; geometry.block_len(1).unwrap()];
            patch.write(1, zeros.clone()).await.unwrap();
            assert_eq!(patch.touched_page(None).await.unwrap(), [1]);
            assert!(occupied().await.unwrap().is_empty());
            assert_eq!(patch.read(1).await.unwrap(), Some(zeros));
        }

        for sparse in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![6].into(), sparse, 2).await;
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };

            let owner = value.owner().unwrap();
            let patch = owner.stage(&txn).await.unwrap();
            observation::watch(&patch.generation);
            assert!(patch.read(0).await.unwrap().is_none());
            observation::watch(&patch.generation);
            for id in 0..owner.geometry().block_count() {
                patch
                    .write(id, vec![id as f64; owner.geometry().block_len(id).unwrap()])
                    .await
                    .unwrap();
            }

            // Repeated membership survives both zero/nonzero transitions.
            for value in [0., 2.] {
                patch
                    .write(1, vec![value; owner.geometry().block_len(1).unwrap()])
                    .await
                    .unwrap();
            }

            assert!(!patch.touched.write().await.insert(vec![1]).await.unwrap());

            assert!(
                patch
                    .touched
                    .write()
                    .await
                    .insert(vec![0, 1])
                    .await
                    .is_err()
            );

            let rows: Vec<_> = patch.replacements().try_collect().await.unwrap();
            assert_eq!(rows.len(), owner.geometry().block_count() as usize);
            assert_eq!(observation::counts().touched_lookups, 0);
            assert_eq!(rows[0].1, vec![0.; owner.geometry().block_len(0).unwrap()]);
            assert_eq!(rows[1].1, vec![2.; owner.geometry().block_len(1).unwrap()]);
            assert_eq!(patch.read(0).await.unwrap(), Some(rows[0].1.clone()));
            assert_eq!(observation::counts().touched_lookups, 1);

            // Membership cannot hide malformed native replacement storage.
            let payload = patch
                ._workspace
                .read()
                .await
                .get_dir("payload")
                .cloned()
                .unwrap();

            let values = payload.read().await.get_dir("values").cloned().unwrap();
            let primary = values.read().await.get_dir("primary").cloned().unwrap();
            let root = primary.read().await.get_file(ROOT).cloned().unwrap();
            {
                let mut node = root.write::<fensor::SparseNode<f64>>().await.unwrap();
                let b_tree::Node::Leaf(rows) = &mut *node else {
                    unreachable!()
                };

                let row = rows
                    .iter_mut()
                    .find(|row| row[0] == fensor::SparseCell::Key(1))
                    .unwrap();

                row[1] = fensor::SparseCell::Key(99);
            }

            assert!(patch.read(1).await.is_err());
            assert!(patch.replacements().try_collect::<Vec<_>>().await.is_err());

            // A selected malformed key must error, never masquerade as an
            // absent replacement and expose an older block.
            let touched = patch
                ._workspace
                .read()
                .await
                .get_dir("touched")
                .cloned()
                .unwrap();

            let root = touched.read().await.get_file(ROOT).cloned().unwrap();
            *root.write::<crate::TensorIndexNode>().await.unwrap() =
                b_tree::Node::Leaf(vec![vec![0, 0, 0]]);

            assert!(patch.read(0).await.is_err());
            assert!(patch.touches(0).await.is_err());
            assert!(patch.replacements().try_collect::<Vec<_>>().await.is_err());
            *root.write::<crate::TensorIndexNode>().await.unwrap() =
                b_tree::Node::Leaf(vec![vec![owner.geometry().block_count()]]);

            let error = patch
                .replacements()
                .try_collect::<Vec<_>>()
                .await
                .unwrap_err();

            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert!(freqfs::IntegrityError::is_marked(&error));
            let cause = error.get_ref().unwrap().source().unwrap();
            let cause = cause.downcast_ref::<std::io::Error>().unwrap();

            assert!(matches!(
                cause.get_ref().unwrap().downcast_ref::<fensor::Error>(),
                Some(fensor::Error::InvalidCoord(_))
            ));

            assert!(tc_error::TCError::from(error).requires_recovery());
            drop(patch.touched.try_write().unwrap());
        }
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn touched_pages_are_bounded_and_reload_single_column_nodes() {
    Box::pin(async {
        use futures::TryStreamExt;

        use super::{Delta, TouchedSchema};
        use crate::StorageContext;

        let len = TOUCHED_PAGE_ENTRIES as u64 + 1;
        for layout in [
            fensor::Layout::Dense,
            fensor::Layout::Sparse { axis: Some(0) },
        ] {
            let (_, txn) = setup().await;
            let geometry = fensor::StorageGeometry::new(
                fensor::TensorSchema::new(
                    number_general::NumberType::Float(number_general::FloatType::F64),
                    vec![len].into(),
                )
                .unwrap(),
                layout,
                [1].iter().copied().product::<u64>() as usize,
            )
            .unwrap();

            let mut patch =
                Delta::<_, f64>::create(txn.subcontext_unique().context().await.unwrap(), geometry)
                    .await
                    .unwrap();

            // Missing native payloads are valid explicit-zero replacements.
            patch
                .touched
                .write()
                .await
                .insert_sorted(futures::stream::iter((0..len).map(|id| Ok(vec![id]))))
                .await
                .unwrap();

            patch.touched.sync().await.unwrap();
            let touched = patch
                ._workspace
                .read()
                .await
                .get_dir("touched")
                .cloned()
                .unwrap();

            let path = touched.read().await.path().to_owned();
            let cache = freqfs::Cache::<crate::PersistentFile>::new(
                32 * 1024,
                None,
                0,
                std::time::Duration::from_secs(3),
            );

            Arc::get_mut(&mut patch).unwrap().touched = b_tree::BTreeLock::load(
                TouchedSchema,
                b_tree::IndexOrder::new(b_tree::collate::Collator::default()),
                cache.load(path).unwrap(),
            )
            .unwrap();

            patch.touched.validate().await.unwrap();
            let first = patch.touched_page(None).await.unwrap();
            assert_eq!(first, (0..len - 1).collect::<Vec<_>>());
            assert_eq!(
                patch.touched_page(first.last().copied()).await.unwrap(),
                [len - 1]
            );

            assert!(patch.touched_page(Some(len - 1)).await.unwrap().is_empty());
            // A returned page retains no native index guard.
            drop(
                tokio::time::timeout(std::time::Duration::from_secs(2), patch.touched.write())
                    .await
                    .unwrap(),
            );

            let mut replacements = Box::pin(patch.replacements());
            let mut expected = 0;
            while let Some((id, values)) = replacements.try_next().await.unwrap() {
                assert_eq!(id, expected);
                assert_eq!(values, [0.]);
                expected += 1;
            }

            assert_eq!(expected, len);
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn vacant_publication_adopts_staged_ownership_without_replaying_blocks() {
    Box::pin(async {
        for checked in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![4].into(), true, 2).await;
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };

            let owner = value.owner().unwrap();
            let patch = owner.stage(&txn).await.unwrap();
            assert_eq!(std::mem::size_of_val(&patch), std::mem::size_of::<usize>());
            let copy = patch.clone();
            assert!(Arc::ptr_eq(&copy, &patch));
            assert_eq!(Arc::strong_count(&patch), 2);
            assert_eq!(Arc::strong_count(&patch.generation), 1);
            drop(copy);
            assert_eq!(Arc::strong_count(&patch), 1);
            assert!(owner.snapshot(txn.id(), None).unwrap()._permit.is_none());
            patch
                .write(0, vec![7.; owner.geometry().block_len(0).unwrap()])
                .await
                .unwrap();

            let generation = patch.generation.clone();
            if checked {
                owner.publish_if_unchanged(&txn, patch, None).await.unwrap();
            } else {
                owner.publish(&txn, patch).await.unwrap();
            }

            let pending = owner.core.state.read().unwrap().pending[&txn.id()].clone();
            assert!(Arc::ptr_eq(&generation, &pending.generation));
            assert_eq!(pending.read(0).await.unwrap().unwrap()[0], 7.);
            let patch = owner.stage(&txn).await.unwrap();
            patch
                .write(1, vec![0.; owner.geometry().block_len(1).unwrap()])
                .await
                .unwrap();

            owner.publish(&txn, patch).await.unwrap();
            assert_eq!(pending.read(0).await.unwrap().unwrap()[0], 7.);
            assert_eq!(pending.read(1).await.unwrap().unwrap()[0], 0.);

            // Retaining a generation stamp does not retain an unpublished workspace owner.
            let unpublished = owner.stage(&txn).await.unwrap();
            let released = Arc::downgrade(&unpublished);
            let stamp = unpublished.generation.clone();
            drop(unpublished);
            assert!(released.upgrade().is_none());
            assert_eq!(stamp.load(Ordering::Acquire), 0);

            tensor.commit(txn.id()).await.unwrap();
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn staged_publication_cancellation_rolls_back_and_stale_generations_conflict() {
    Box::pin(async {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![4].into(), true, 2).await;
            tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };

            let owner = value.owner().unwrap();
            let stamp = owner.pending_stamp(txn.id()).unwrap();
            let patch = owner.stage(&txn).await.unwrap();
            patch
                .write(0, vec![3.0; owner.geometry().block_len(0).unwrap()])
                .await
                .unwrap();

            tensor.write_value(&txn, &[0], 2.0.into()).await.unwrap();
            assert!(
                owner
                    .publish_if_unchanged(&txn, patch, stamp)
                    .await
                    .is_err()
            );

            assert_eq!(tensor.read_value(txn.id(), &[0]).await.unwrap(), 2.0.into());
            let delta = owner.core.state.read().unwrap().pending[&txn.id()].clone();
            let patch = owner.stage(&txn).await.unwrap();
            patch
                .write(0, vec![9.0; owner.geometry().block_len(0).unwrap()])
                .await
                .unwrap();

            let occupied = delta.touched.write().await;
            let mut publication = Box::pin(owner.publish(&txn, patch));
            tokio::select! {
                result = &mut publication => panic!("publication should wait for the occupied index: {result:?}"),
                result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                    loop {
                        if delta.payload.read(0).await.unwrap()[0] == 9.0 { break; }
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                }) => result.unwrap(),
            }

            drop(publication);
            drop(occupied);
            tensor.rollback(&txn.id()).await.unwrap();
            tensor.fill(&txn, 5.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
    }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn aliased_sources_reserve_one_combined_interval() {
    Box::pin(async {
        use fensor::AxisRange;

        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![8].into(), false, 2).await;
        let left = tensor
            .clone()
            .slice(vec![AxisRange::In(0, 2, 1)].into())
            .unwrap();

        let right = tensor
            .clone()
            .slice(vec![AxisRange::In(4, 6, 1)].into())
            .unwrap();

        let (Data::F64(left), Data::F64(right)) = (&left.data, &right.data) else {
            unreachable!()
        };

        let [left, right] = crate::tensor::Typed::expressions([left, right], txn.id()).unwrap();
        assert!(tensor.write_value(&txn, &[3], 1.0.into()).await.is_err());
        tensor.write_value(&txn, &[7], 2.0.into()).await.unwrap();
        drop(left);
        drop(right);
        tensor.write_value(&txn, &[3], 1.0.into()).await.unwrap();
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cancellation_after_payload_mutation_requires_rollback() {
    Box::pin(async {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![4].into(), true, 2).await;
            tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };

            let delta = value
                .owner()
                .unwrap()
                .core
                .state
                .read()
                .unwrap()
                .pending
                .get(&txn.id())
                .cloned()
                .unwrap();

            let touched = delta.touched.write().await;
            let mut mutation = Box::pin(delta.write(0, vec![2.; delta.geometry.block_len(0).unwrap()]));
            tokio::time::timeout(std::time::Duration::from_secs(5),async {
                tokio::select! {
                    result=&mut mutation => panic!("publication should wait for touched index: {result:?}"),
                    _=async {
                        loop {
                            if delta.payload.read(0).await.unwrap()[0]==2. {break;}
                            tokio::task::yield_now().await;
                        }
                    } => {}
                }
            }).await.unwrap();

            drop(mutation);
            drop(touched);
            tensor.rollback(&txn.id()).await.unwrap();
            assert_eq!(
                tensor.read_value(txn.id(), &[0]).await.unwrap(),
                number_general::Number::from(0.0)
            );
    }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn deep_overlay_polling_and_cancellation_use_normal_worker_stacks() {
    use std::time::Duration;

    use futures::TryStreamExt;

    let task = tokio::spawn(async {
        for occupied in [false, true] {
            let (_temp, txn) = setup().await;
            let (tensor, directory) = create(&txn, vec![1].into(), true, 1).await;
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };

            let owner = value.owner().unwrap();
            let delta = owner.stage(&txn).await.unwrap();
            delta
                .write(0, vec![if occupied { 1. } else { 0. }])
                .await
                .unwrap();

            // Share the native fixture to isolate traversal depth from storage setup.
            for pending in [false, true] {
                let mut snapshot =
                    test_snapshot(owner.canonical().unwrap(), vec![delta.clone(); 1024]);

                snapshot._permit = Some(Arc::new(
                    owner
                        .core
                        .acquire_read_permit(txn.id(), 0..1)
                        .await
                        .unwrap(),
                ));

                let lease = Arc::downgrade(snapshot._permit.as_ref().unwrap());
                let mut scan = snapshot.occupied_blocks(0..1);
                drop(snapshot);
                if pending {
                    let values = directory.read().await.get_dir("values").cloned().unwrap();
                    let primary = values.read().await.get_dir("primary").cloned().unwrap();
                    let root = primary
                        .read()
                        .await
                        .get_file("00000000-0000-0000-0000-000000000000")
                        .cloned()
                        .unwrap();

                    let guard = root.write::<fensor::SparseNode<f64>>().await.unwrap();
                    assert!(
                        tokio::time::timeout(std::time::Duration::from_millis(20), scan.try_next())
                            .await
                            .is_err()
                    );

                    drop(scan);
                    assert!(lease.upgrade().is_none());
                    drop(guard);
                    owner
                        .canonical()
                        .unwrap()
                        .replace_logical_block(0, vec![0.])
                        .await
                        .unwrap();
                } else {
                    // A fully cached deep scan must cooperate before reaching EOF.
                    if !occupied {
                        let mut next = std::pin::pin!(scan.try_next());
                        futures::future::poll_fn(|cx| {
                            assert!(std::future::Future::poll(next.as_mut(), cx).is_pending());
                            std::task::Poll::Ready(())
                        })
                        .await;
                    }

                    assert_eq!(scan.try_next().await.unwrap(), occupied.then_some(0));
                    assert!(scan.try_next().await.unwrap().is_none());
                    assert!(lease.upgrade().is_none());
                }
            }
        }
    });

    tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn batched_materialization_discards_partial_output_and_obsolete_errors() {
    use fensor::{Layout, StorageGeometry, TensorRead, TensorSchema, TensorView};
    use futures::TryStreamExt;
    use tokio::sync::oneshot;

    use crate::StorageContext;

    Box::pin(observation::scope(async {
        for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
            for (publish, marked) in [(false, false), (false, true), (true, false), (true, true)] {
                let (_temp, txn) = setup().await;
                let geometry = StorageGeometry::new(
                    TensorSchema::new(
                        number_general::NumberType::Float(number_general::FloatType::F64),
                        [2, 3].as_slice().into(),
                    )
                    .unwrap(),
                    layout,
                    ([1, 2].as_slice()).iter().copied().product::<u64>() as usize,
                )
                .unwrap();

                let canonical = fensor::Tensor::create_with_geometry(
                    txn.subcontext_unique().context().await.unwrap(),
                    geometry.clone(),
                )
                .await
                .unwrap();

                let delta = super::Delta::create(
                    txn.subcontext_unique().context().await.unwrap(),
                    geometry.clone(),
                )
                .await
                .unwrap();

                for id in 0..geometry.block_count() {
                    let mut values = vec![0.; geometry.block_len(id).unwrap()];
                    values.fill(7.);

                    delta.write(id, values).await.unwrap();
                }

                let snapshot = test_snapshot(canonical, vec![delta.clone()]);
                let view = TensorView::new(snapshot.clone());
                let materialized = snapshot.visibility.clone();
                let cutoff = snapshot.deltas[0].0;
                let (observed, mut receipt) = oneshot::channel();
                observation::on_scatter(&delta.generation, move || {
                    // Model canonical application completed before publication,
                    // with the obsolete workspace still holding nonzero payloads.
                    // Sparse canonical absence cannot overwrite partial output:
                    // the handoff must explicitly reset the scattered positions.
                    if publish {
                        materialized.0.write().unwrap().finalized = Some(cutoff);
                    }

                    observed.send(()).unwrap();
                    let error = std::io::Error::other("obsolete payload read");
                    Err(if marked {
                        freqfs::IntegrityError::mark(error)
                    } else {
                        error
                    }
                    .into())
                });

                let values = view.read_blocks().unwrap().try_concat().await;
                receipt.try_recv().unwrap();
                if publish {
                    assert_eq!(values.unwrap(), vec![0.; 6]);
                } else {
                    let fensor::Error::Io(error) = values.unwrap_err() else {
                        panic!("expected original payload I/O error")
                    };

                    assert_eq!(error.to_string(), "obsolete payload read");
                    assert_eq!(freqfs::IntegrityError::is_marked(&error), marked);
                    assert_eq!(tc_error::TCError::from(error).requires_recovery(), marked);
                }

                // The failed visitor releases its BTree/file guards before either
                // returning the error or re-entering canonical storage.
                delta
                    .payload
                    .replace(0, vec![0.; geometry.block_len(0).unwrap()])
                    .await
                    .unwrap();
            }
        }
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn batched_membership_preserves_unresolved_runs_and_handoff() {
    use fensor::{Layout, StorageGeometry, TensorSchema};
    use tokio::sync::oneshot;

    use crate::StorageContext;

    const ROOT: &str = "00000000-0000-0000-0000-000000000000";

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        Box::pin(observation::scope(async {
            let (_temp, txn) = setup().await;
            let geometry = StorageGeometry::new(
                TensorSchema::new(
                    number_general::NumberType::Float(number_general::FloatType::F64),
                    [768].as_slice().into(),
                )
                .unwrap(),
                Layout::Sparse { axis: Some(0) },
                ([1].as_slice()).iter().copied().product::<u64>() as usize,
            )
            .unwrap();

            let canonical = fensor::Tensor::<_, f64>::create_with_geometry(
                txn.subcontext_unique().context().await.unwrap(),
                geometry.clone(),
            )
            .await
            .unwrap();

            let older = super::Delta::create(
                txn.subcontext_unique().context().await.unwrap(),
                geometry.clone(),
            )
            .await
            .unwrap();

            older
                .touched
                .write()
                .await
                .insert_sorted(futures::stream::iter((0..640).map(|id| Ok(vec![id]))))
                .await
                .unwrap();

            let newer =
                super::Delta::create(txn.subcontext_unique().context().await.unwrap(), geometry)
                    .await
                    .unwrap();

            newer
                .touched
                .write()
                .await
                .insert_sorted(futures::stream::iter(
                    [129, 130, 258].map(|id| Ok(vec![id])),
                ))
                .await
                .unwrap();

            let snapshot = test_snapshot(canonical, vec![older.clone(), newer.clone()]);

            // One run spans native leaves; newer selections split only the
            // older scan. Missing payload rows still select explicit zeros.
            let ids: Vec<_> = (0..768).collect();
            let mut sources = vec![None; ids.len()];
            for (delta, scans) in [(&newer, 1), (&older, 3)] {
                observation::watch(&delta.generation);
                snapshot.resolve_sources(&ids, &mut sources).await.unwrap();
                assert_eq!(observation::counts().membership_scans, scans);
                assert_eq!(observation::counts().touched_lookups, 0);
                drop(delta.touched.try_write().unwrap());
                for (&id, &source) in ids.iter().zip(&sources) {
                    let expected = if [129, 130, 258].contains(&id) {
                        Some(1)
                    } else if id < 640 {
                        Some(0)
                    } else {
                        None
                    };

                    assert_eq!(source, expected, "block {id}");
                }
            }

            let ids = [0, 1, 2, 129, 130, 131, 257, 258, 259, 700, 767];
            let mut sources = [None; 11];
            for (delta, scans) in [(&newer, 5), (&older, 6)] {
                observation::watch(&delta.generation);
                snapshot.resolve_sources(&ids, &mut sources).await.unwrap();
                assert_eq!(observation::counts().membership_scans, scans);
                assert_eq!(
                    sources,
                    [
                        Some(0),
                        Some(0),
                        Some(0),
                        Some(1),
                        Some(1),
                        Some(0),
                        Some(0),
                        Some(1),
                        Some(0),
                        None,
                        None
                    ],
                );
            }

            observation::watch(&older.generation);
            let mut sources = [None; 2];
            fn send<T: Send>(value: T) -> T {
                value
            }

            send(snapshot.resolve_sources(&[129, 130], &mut sources))
                .await
                .unwrap();

            assert_eq!(sources, [Some(1), Some(1)]);
            assert_eq!(observation::counts().membership_scans, 0);

            for delta in [&newer, &older] {
                observation::watch(&delta.generation);
                let mut source = [None];
                snapshot.resolve_sources(&[700], &mut source).await.unwrap();
                assert_eq!(source, [None]);
                assert_eq!(observation::counts().membership_scans, 1);
            }

            let touched = older
                ._workspace
                .read()
                .await
                .get_dir("touched")
                .cloned()
                .unwrap();

            let root = touched.read().await.get_file(ROOT).cloned().unwrap();
            let last = match &*root.read::<crate::TensorIndexNode>().await.unwrap() {
                b_tree::Node::Index(_, children) => *children.last().unwrap(),
                b_tree::Node::Leaf(_) => panic!("fixture must span native leaves"),
            };

            let last = touched
                .read()
                .await
                .get_file(&last.to_string())
                .cloned()
                .unwrap();

            // The first older run assigns block 0 before a later leaf blocks.
            // Cancellation must release every retained index/file read guard.
            for publish in [false, true] {
                let mut held = last.write::<crate::TensorIndexNode>().await.unwrap();
                let (completed, receipt) = oneshot::channel();
                observation::on_membership(&older.generation, 1, move || {
                    completed.send(()).unwrap();
                    Ok(())
                });

                let mut sources = [None; 3];
                let mut read = Box::pin(tokio::task::unconstrained(
                    snapshot.resolve_sources(&[0, 129, 639], &mut sources),
                ));

                tokio::select! {
                    biased;
                    result = &mut read => panic!("later leaf must block: {result:?}"),
                    result = receipt => result.unwrap(),
                }

                assert_eq!(observation::counts().membership_scans, 2);

                if publish {
                    // Completed canonical application makes both the earlier
                    // assignment and this malformed later workspace obsolete.
                    snapshot.visibility.0.write().unwrap().finalized = Some(snapshot.deltas[0].0);
                    *held = b_tree::Node::Leaf(vec![vec![639, 639]]);
                    drop(held);
                    read.await.unwrap();
                    assert_eq!(sources, [None, Some(1), None]);
                } else {
                    drop(read);
                    drop(held);
                    assert_eq!(sources, [Some(0), Some(1), None]);
                }

                drop(older.touched.try_write().unwrap());
                drop(last.try_write::<crate::TensorIndexNode>().unwrap());
            }
        })),
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn batched_membership_reports_newest_errors_and_releases_sources() {
    use fensor::{Layout, StorageGeometry, TensorSchema};

    use crate::StorageContext;

    const ROOT: &str = "00000000-0000-0000-0000-000000000000";

    Box::pin(observation::scope(async {
        let (_temp, txn) = setup().await;
        let geometry = StorageGeometry::new(
            TensorSchema::new(
                number_general::NumberType::Float(number_general::FloatType::F64),
                [4].as_slice().into(),
            )
            .unwrap(),
            Layout::Sparse { axis: Some(0) },
            ([1].as_slice()).iter().copied().product::<u64>() as usize,
        )
        .unwrap();

        let canonical = fensor::Tensor::<_, f64>::create_with_geometry(
            txn.subcontext_unique().context().await.unwrap(),
            geometry.clone(),
        )
        .await
        .unwrap();

        let mut deltas = Vec::new();
        let mut roots = Vec::new();
        for keys in [vec![vec![0, 0]], vec![vec![3], vec![3]]] {
            let delta = super::Delta::create(
                txn.subcontext_unique().context().await.unwrap(),
                geometry.clone(),
            )
            .await
            .unwrap();

            let touched = delta
                ._workspace
                .read()
                .await
                .get_dir("touched")
                .cloned()
                .unwrap();

            let root = touched.read().await.get_file(ROOT).cloned().unwrap();
            *root.write::<crate::TensorIndexNode>().await.unwrap() = b_tree::Node::Leaf(keys);
            roots.push(root);
            deltas.push(delta);
        }

        let snapshot = test_snapshot(canonical, deltas);

        // The newer layer's later block is visited before the older layer's
        // earlier block. Strict ordering includes duplicate membership rows.
        let mut sources = [None; 2];
        let error = snapshot
            .resolve_sources(&[0, 3], &mut sources)
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid Tensor touched-block order")
        );

        *roots[1].write::<crate::TensorIndexNode>().await.unwrap() = b_tree::Node::Leaf(Vec::new());
        let error = snapshot
            .resolve_sources(&[0, 3], &mut sources)
            .await
            .unwrap_err();

        // Native BTree validation rejects the malformed row width before
        // Tensor membership consumption, retaining its integrity provenance.
        assert!(matches!(error, fensor::Error::Io(error)
            if error.kind() == std::io::ErrorKind::InvalidData
                && freqfs::IntegrityError::is_marked(&error)));

        drop(snapshot.deltas[0].1.touched.try_write().unwrap());
        drop(snapshot.deltas[1].1.touched.try_write().unwrap());

        // Materialization after a completed earlier run discards all of this
        // layer's assignments, including when the later completion returns an
        // error. Without publication the same error must remain observable.
        for publish in [false, true] {
            *roots[0].write::<crate::TensorIndexNode>().await.unwrap() =
                b_tree::Node::Leaf(vec![vec![0], vec![3]]);

            let delta = &snapshot.deltas[0].1;
            let materialized = snapshot.visibility.clone();
            let cutoff = snapshot.deltas[0].0;
            observation::on_membership(&delta.generation, 2, move || {
                if publish {
                    materialized.0.write().unwrap().finalized = Some(cutoff);
                }

                Err(std::io::Error::other("membership completion failed").into())
            });

            let result = snapshot.resolve_sources(&[0, 3], &mut sources).await;
            assert_eq!(observation::counts().membership_scans, 2);
            if publish {
                result.unwrap();
                assert_eq!(sources, [None, None]);
            } else {
                assert!(matches!(result, Err(fensor::Error::Io(error))
                    if error.to_string() == "membership completion failed"));
            }

            drop(delta.touched.try_write().unwrap());
            drop(roots[0].try_write::<crate::TensorIndexNode>().unwrap());
        }
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn full_fills_replace_compact_blocks_without_reading_old_payloads() {
    use crate::tensor::Tensor;
    use crate::tensor::typed::Typed;
    use crate::test::tensor::{id, values};

    observation::scope(Box::pin(async {
        for sparse in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, dir) = create(&txn, vec![5].into(), sparse, 2).await;
            tensor.write_value(&txn, &[0], 7.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
            tensor.finalize(&txn.id()).await.unwrap();
            let later = txn.at(11);
            tensor.write_value(&later, &[4], 9.0.into()).await.unwrap();
            let Data::F64(Typed::Owner(owner)) = &tensor.data else {
                panic!("f64 owner")
            };

            let previous = owner
                .core
                .read_state()
                .pending
                .get(&later.id())
                .unwrap()
                .clone();

            assert_eq!(owner.geometry().block_len(2).unwrap(), 1);
            observation::watch(&previous.generation);
            tensor.fill(&later, 3.0.into()).await.unwrap();
            assert_eq!(
                observation::counts().payload_reads,
                0,
                "full replacement does not read old delta payloads"
            );

            assert_eq!(values(&tensor, later.id()).await, [3.; 5]);
            tensor.rollback(&later.id()).await.unwrap();
            assert_eq!(values(&tensor, id(12)).await, [7., 0., 0., 0., 0.]);

            if !sparse {
                let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
                let file = blocks.read().await.get_file("0").unwrap().clone();
                let held = file.write::<Vec<f64>>().await.unwrap();
                tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    Box::pin(tensor.fill(&txn.at(12), 0.0.into())),
                )
                .await
                .expect("full fill never waits for old canonical payloads")
                .unwrap();

                drop(held);
            } else {
                tensor.fill(&txn.at(12), 0.0.into()).await.unwrap();
            }

            tensor.commit(id(12)).await.unwrap();
            tensor.finalize(&id(12)).await.unwrap();
            tensor.sync_all().await.unwrap();
            let schema = tensor.schema().unwrap();
            let path = dir.read().await.path().to_owned();
            drop(previous);
            drop(tensor);
            drop(dir);
            let cache = freqfs::Cache::<crate::PersistentFile>::new(
                256 * 1024,
                None,
                0,
                std::time::Duration::from_secs(3),
            );

            let reopened =
                Tensor::<crate::test::tensor::Txn>::load(cache.load(path).unwrap(), schema)
                    .await
                    .unwrap();

            assert_eq!(values(&reopened, id(13)).await, [0.; 5]);
        }
    }))
    .await;

    #[cfg(feature = "complex")]
    Box::pin(async {
        use fensor::complex::Complex64;
        use futures::TryStreamExt;

        use crate::StorageContext;
        use crate::tensor::TensorSchema;

        for sparse_axis in [None, Some(0)] {
            let (_, txn) = setup().await;
            let schema = TensorSchema::new(
                number_general::NumberType::Complex(number_general::ComplexType::C64),
                vec![2, 5].into(),
                sparse_axis,
            )
            .unwrap();

            let tensor = Tensor::<crate::test::tensor::Txn>::create(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
                2,
            )
            .await
            .unwrap();

            tensor
                .write_value(&txn, &[1, 4], Complex64::new(9., -3.).into())
                .await
                .unwrap();

            let expected = number_general::Number::from(Complex64::new(2., -1.));
            tensor.fill(&txn, expected).await.unwrap();
            assert_eq!(
                tensor
                    .elements(txn.id())
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap(),
                vec![expected; 10]
            );

            tensor
                .fill(&txn, Complex64::new(0., 0.).into())
                .await
                .unwrap();

            tensor.commit(txn.id()).await.unwrap();
            tensor.finalize(&txn.id()).await.unwrap();
            assert_eq!(
                tensor
                    .elements(crate::test::tensor::id(11))
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap(),
                vec![number_general::Number::from(Complex64::new(0., 0.)); 10]
            );
        }
    })
    .await;
}
