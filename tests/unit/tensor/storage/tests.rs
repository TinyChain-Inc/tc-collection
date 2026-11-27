use std::sync::Arc;
use std::sync::atomic::Ordering;

use fensor::TensorSource;
use futures::{FutureExt, StreamExt};
use tc_ir::{Transact, Transaction};

use super::BlockSource::{Canonical, Zero};
use super::{Delta, ZERO_BATCH_SIZE, observation};
use crate::tensor::Tensor;
use crate::tensor::tests::{create, setup};
use crate::{Restore, StorageContext};

impl<F: crate::CollectionFile, T: fensor::TensorElement> super::Delta<F, T> {
    fn zeros(&self) -> &super::Zeros<F> {
        self.zeros.as_ref().expect("partial delta fixture")
    }
}

impl<Txn: crate::StorageContext, T: fensor::TensorElement> super::TensorBase<Txn, T>
where
    Txn::File: fensor::TensorFileEntry<T>,
{
    pub(in crate::tensor) fn canonical(&self) -> &fensor::Tensor<Txn::File, T> {
        &self.canonical
    }
}

pub(in crate::tensor) async fn corrupt_block(
    owner: &super::TensorBase<crate::test::TestTxn, f64>,
    id: tc_ir::TxnId,
    coord: &[u64],
) {
    let block = owner.geometry().block_position(coord).unwrap().0;
    let snapshot = owner.versions.visible_snapshot(id).unwrap();
    let delta = snapshot.last().unwrap();
    let name = if delta.is_whole() {
        "replacement"
    } else {
        "payload"
    };
    let payload = delta
        .storage
        .dir()
        .read()
        .await
        .get_dir(name)
        .cloned()
        .unwrap();

    let values = payload.read().await.get_dir("values").cloned().unwrap();
    let primary = values.read().await.get_dir("primary").cloned().unwrap();
    let files: Vec<_> = primary.read().await.files().cloned().collect();
    let mut corrupted = false;
    for file in files {
        let mut node = file.write::<crate::TensorSparseNode<f64>>().await.unwrap();
        let b_tree::Node::Leaf(rows) = &mut *node else {
            continue;
        };

        if let Some(row) = rows
            .iter_mut()
            .find(|row| row[0] == fensor::SparseCell::Key(block))
        {
            row[1] = fensor::SparseCell::Key(99);
            corrupted = true;
        }
    }

    assert!(corrupted);
}

// Standalone delta fixtures are never published into a collection's store.
fn allocation(
    dir: freqfs::DirLock<crate::PersistentFile>,
) -> Arc<crate::persistence::storage::Allocation<crate::PersistentFile>> {
    Arc::new(crate::persistence::storage::Allocation::fixture(dir))
}

fn test_snapshot<T: fensor::TensorElement>(
    canonical: fensor::Tensor<crate::PersistentFile, T>,
    deltas: Vec<Arc<super::Delta<crate::PersistentFile, T>>>,
) -> super::Snapshot<crate::PersistentFile, T>
where
    crate::PersistentFile: fensor::TensorFileEntry<T>,
{
    super::Snapshot {
        canonical: Arc::new(canonical),
        deltas: deltas.into(),
        _permit: None,
    }
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
async fn reopen_rejects_overlapping_payload_and_zero_override() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let (tensor, dir) = create(&txn, vec![4].into(), true, 2).await;
        let schema = tensor.schema().unwrap();
        tensor.write_value(&txn, &[1], 7.0.into()).await.unwrap();
        tensor.commit(txn.id()).await.unwrap();

        let Tensor::F64(value) = &tensor else {
            unreachable!()
        };

        let owner = value.base().unwrap();
        let versions = owner.versions.visible_snapshot(txn.id()).unwrap();
        let delta = versions.last().unwrap();
        assert!(delta.zeros().write().await.insert(vec![1]).await.unwrap());
        delta.zeros().sync().await.unwrap();

        let path = dir.read().await.path().to_owned();
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            128 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );
        let reopened =
            Tensor::<crate::test::TestTxn>::load(cache.load(path).unwrap(), schema).await;
        assert!(reopened.err().unwrap().requires_recovery());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn immutable_access_does_not_wait_for_transaction_state() {
    let (_root, txn) = setup().await;
    let (tensor, _) = create(&txn, vec![4].into(), true, 2).await;
    let Tensor::F64(value) = &tensor else {
        unreachable!()
    };

    let owner = value.base().unwrap();
    let alias = owner.clone();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let (worker, result) = {
        let _state = owner.versions.hold_state_for_test();
        let worker = tokio::spawn(async move {
            assert_eq!(alias.geometry().schema().shape().as_slice(), &[4]);
            alias.canonical().sync().await.unwrap();
            alias.canonical().sync_all().await.unwrap();
            sender.send(()).unwrap();
        });

        let result = receiver.recv_timeout(std::time::Duration::from_secs(3));
        (worker, result)
    };

    worker.await.unwrap();
    assert!(result.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn batched_reads_select_sources_and_preserve_errors() {
    use fensor::{Layout, StorageGeometry, TensorRead, TensorSchema, TensorTransform, TensorView};
    use futures::{FutureExt, TryStreamExt};

    use crate::StorageContext;

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

                let first = super::Delta::create(allocation(txn.subcontext_unique().context().await.unwrap()),
                    geometry.clone(), false,
                    )
                .await
                .unwrap();

                for (id, value) in [(0, 10.), (1, 11.), (2, 0.), (geometry.block_count() - 1, 14.)] {
                    let mut block = vec![0.; geometry.block_len(id).unwrap()];
                    block.fill(value);

                    first.write(id, Some(block)).await.unwrap();
                }

                let second = super::Delta::create(allocation(txn.subcontext_unique().context().await.unwrap()),
                    geometry.clone(), false,
                    )
                .await
                .unwrap();

                second.write(0, Some(vec![0.; 4])).await.unwrap();
                let mut replacement = vec![0.; geometry.block_len(1).unwrap()];
                replacement[0] = 21.;
                second.write(1, Some(replacement)).await.unwrap();
                let last = geometry.block_count() - 1;
                second
                    .write(last, Some(vec![24.; geometry.block_len(last).unwrap()]))
                    .await
                    .unwrap();
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

                let payload = second.storage.dir().read().await.get_dir("payload").cloned().unwrap();
                let values = payload.read().await.get_dir("values").cloned().unwrap();
                let primary = values.read().await.get_dir("primary").cloned().unwrap();
                let root = primary.read().await.get_file(ROOT).cloned().unwrap();
                for extra in [1., f64::NAN, f64::INFINITY] {
                    let saved = {
                        let mut node = root.write::<crate::TensorSparseNode<f64>>().await.unwrap();
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
                    *root.write::<crate::TensorSparseNode<f64>>().await.unwrap() = saved;
                }

                // The zero index is scanned before the payload tree in a run.
                let bad = super::Delta::create(allocation(txn.subcontext_unique().context().await.unwrap()),
                    geometry.clone(), false,
                    )
                .await
                .unwrap();

                bad.write(0, Some(vec![1.; 4])).await.unwrap();
                let payload = bad
                    .storage.dir()
                    .read()
                    .await
                    .get_dir("payload")
                    .cloned()
                    .unwrap();

                let values = payload.read().await.get_dir("values").cloned().unwrap();
                let primary = values.read().await.get_dir("primary").cloned().unwrap();
                let root = primary.read().await.get_file(ROOT).cloned().unwrap();
                *root.write::<crate::TensorSparseNode<f64>>().await.unwrap() = b_tree::Node::Leaf(vec![
                    vec![fensor::SparseCell::Key(0), fensor::SparseCell::Key(99)],
                ]);

                let zeros = bad
                    .storage.dir()
                    .read()
                    .await
                    .get_dir("zeros")
                    .cloned()
                    .unwrap();

                let root = zeros.read().await.get_file(ROOT).cloned().unwrap();
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
                // Native scans validate every row in the containing node. Restore
                // the zero index so the malformed payload can be read.
                *root.write::<crate::TensorIndexNode>().await.unwrap() =
                    b_tree::Node::Leaf(Vec::new());

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
        let len = ZERO_BATCH_SIZE + 1;
        let (tensor, _) = create(&txn, vec![len as u64].into(), true, 4096).await;
        let Tensor::F64(value) = &tensor else {
            panic!("dtype")
        };

        let owner = value.base().unwrap();
        let delta = owner.stage(&txn, false).await.unwrap();
        observation::watch(&delta.generation);
        for block in 0..owner.geometry().block_count() {
            let mut values = vec![0.; owner.geometry().block_len(block).unwrap()];
            values.fill(1.);
            delta.write(block, Some(values)).await.unwrap();
        }

        owner
            .publish(&txn, delta.clone(), |_| Ok(()))
            .await
            .unwrap();
        {
            let snapshot = owner
                .try_snapshot(txn.id(), 0..owner.geometry().block_count())
                .unwrap();

            let lease = Arc::downgrade(snapshot._permit.as_ref().unwrap());
            let mut scan = snapshot.occupied_blocks(0..len as u64);
            drop(snapshot);
            let ids: Vec<_> = scan
                .by_ref()
                .take(ZERO_BATCH_SIZE)
                .try_collect()
                .await
                .unwrap();

            assert_eq!(ids.len(), ZERO_BATCH_SIZE);
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
async fn overlay_suppresses_zero_replacements_without_membership_reprobes() {
    Box::pin(observation::scope(async {
        use futures::TryStreamExt;

        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![600].into(), true, 2).await;
        let Tensor::F64(value) = &tensor else {
            unreachable!()
        };

        let owner = value.base().unwrap();
        let canonical = owner.canonical().clone();
        for id in 0..owner.geometry().block_count() {
            canonical
                .replace_logical_block(id, vec![1.; owner.geometry().block_len(id).unwrap()])
                .await
                .unwrap();
        }

        let first = owner.stage(&txn, false).await.unwrap();
        observation::watch(&first.generation);
        first
            .write(0, Some(vec![0.; owner.geometry().block_len(0).unwrap()]))
            .await
            .unwrap();

        first
            .write(1, Some(vec![2.; owner.geometry().block_len(1).unwrap()]))
            .await
            .unwrap();

        let second = owner.stage(&txn, false).await.unwrap();
        second
            .write(1, Some(vec![0.; owner.geometry().block_len(1).unwrap()]))
            .await
            .unwrap();

        let snapshot = test_snapshot(canonical.clone(), vec![first, second.clone()]);
        let keys: Vec<_> = snapshot
            .occupied_blocks(0..600)
            .try_collect()
            .await
            .unwrap();
        let expected: Vec<_> = canonical
            .occupied_blocks(2..600)
            .try_collect()
            .await
            .unwrap();
        assert_eq!(keys, expected);
        assert_eq!(observation::counts().zero_lookups, 0);

        // The shared overlay checks zero streams even when there is no older key.
        let zeros = second
            .storage
            .dir()
            .read()
            .await
            .get_dir("zeros")
            .unwrap()
            .clone();
        let root = zeros
            .read()
            .await
            .get_file("00000000-0000-0000-0000-000000000000")
            .unwrap()
            .clone();

        *root.write::<crate::TensorIndexNode>().await.unwrap() =
            b_tree::Node::Leaf(vec![vec![1, 2]]);

        let empty_base = fensor::Tensor::<_, f64>::create_with_geometry(
            txn.subcontext_unique().context().await.unwrap(),
            second.payload.geometry().clone(),
        )
        .await
        .unwrap();

        let error = test_snapshot(empty_base, vec![second])
            .occupied_blocks(0..600)
            .try_next()
            .await
            .unwrap_err();

        assert!(matches!(error, fensor::Error::Io(error)
            if freqfs::IntegrityError::is_marked(&error)));
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
                let Tensor::F64(value) = &tensor else {
                    unreachable!()
                };

                fensor::TensorWrite::write_value(
                    &value.base().unwrap().canonical().clone(),
                    &[1],
                    9.0,
                )
                .await
                .unwrap();
            }

            let writes = txn.subcontext_unique();
            tensor.write_value(&writes, &[0], 2.0.into()).await.unwrap();
            tensor.write_value(&writes, &[1], 0.0.into()).await.unwrap();
            tensor.write_value(&writes, &[3], 4.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
            let later = TxnId::from_parts(NetworkTime::from_nanos(1), 11);
            let expression = tensor.mul_scalar(later, 2.0.into()).await.unwrap();
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };

            let owner = value.base().unwrap();
            let snapshot = owner
                .try_snapshot(later, 0..owner.geometry().block_count())
                .unwrap();

            let entries = if sparse {
                Some(tensor.sparse_elements(later).await.unwrap())
            } else {
                None
            };

            tensor.finalize(&txn.id()).await.unwrap();
            // The caller removes an expired workspace after recursive
            // finalization even though a later expression is still live.
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                writes
                    .context()
                    .await
                    .unwrap()
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
                        .take(ZERO_BATCH_SIZE)
                        .try_collect()
                        .await
                        .unwrap();

                    assert_eq!(keys, [first, last]);
                    // Each delta cursor resumes on native storage after
                    // the handoff, preserving unvisited occupied keys.
                    let keys: Vec<_> = snapshot
                        .occupied_blocks(first + 1..owner.geometry().block_count())
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
async fn replacement_batches_reuse_keys_and_validate_entries() {
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
                allocation(txn.subcontext_unique().context().await.unwrap()),
                geometry.clone(),
                false,
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
                .write(1, Some(vec![1.; geometry.block_capacity() - 1]))
                .await
                .unwrap_err();

            assert!(!freqfs::IntegrityError::is_marked(&invalid_length));
            assert!(!tc_error::TCError::from(invalid_length).requires_recovery());
            // Invalid IDs reject before waiting for the zeros-index guard.
            let zeros = patch.zeros().write().await;
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

            drop(zeros);
            let invalid_id = patch.write(geometry.block_count(), None).await.unwrap_err();

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
                let error = patch.write(1, Some(vec![1.; len])).await.unwrap_err();
                assert!(!freqfs::IntegrityError::is_marked(&error));
                assert!(!tc_error::TCError::from(error).requires_recovery());
            }

            assert_eq!(patch.generation.load(Ordering::Acquire), 0);
            assert!(
                patch
                    .zero_ids(0..geometry.block_count())
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(occupied().await.unwrap().is_empty());

            let mut values = vec![0.; geometry.block_len(1).unwrap()];
            values[0] = 7.;
            patch.write(1, Some(values.clone())).await.unwrap();
            assert!(
                patch
                    .zero_ids(0..geometry.block_count())
                    .try_next()
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(occupied().await.unwrap(), [1]);
            assert_eq!(patch.read(1).await.unwrap(), Some(values));
            for id in [0, 2, geometry.block_count() - 1] {
                assert!(patch.read(id).await.unwrap().is_none());
            }

            // Deleting the payload leaves an explicit zero override.
            let zeros = vec![0.; geometry.block_len(1).unwrap()];
            patch.write(1, None).await.unwrap();
            assert_eq!(
                patch
                    .zero_ids(0..geometry.block_count())
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap(),
                [1]
            );
            assert!(occupied().await.unwrap().is_empty());
            assert_eq!(patch.read(1).await.unwrap(), Some(zeros));

            patch
                .write(1, Some(vec![-0.; geometry.block_len(1).unwrap()]))
                .await
                .unwrap();
            let block = patch.read(1).await.unwrap().unwrap();
            if matches!(geometry.layout(), Layout::Dense) {
                assert!(block.iter().all(|value| value.is_sign_negative()));
                assert!(
                    patch
                        .zero_ids(0..geometry.block_count())
                        .try_next()
                        .await
                        .unwrap()
                        .is_none()
                );
                assert_eq!(occupied().await.unwrap(), [1]);
            } else {
                assert!(block.iter().all(|value| value.is_sign_positive()));
                assert_eq!(
                    patch
                        .zero_ids(0..geometry.block_count())
                        .try_next()
                        .await
                        .unwrap(),
                    Some(1)
                );
                assert!(occupied().await.unwrap().is_empty());
            }
        }

        for sparse in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![6].into(), sparse, 2).await;
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };

            let owner = value.base().unwrap();
            let patch = owner.stage(&txn, false).await.unwrap();
            observation::watch(&patch.generation);
            assert!(patch.read(0).await.unwrap().is_none());
            observation::watch(&patch.generation);
            for id in 0..owner.geometry().block_count() {
                patch
                    .write(
                        id,
                        Some(vec![id as f64; owner.geometry().block_len(id).unwrap()]),
                    )
                    .await
                    .unwrap();
            }

            // Repeated writes reconcile both directions of the transition.
            for value in [0., 2.] {
                patch
                    .write(1, Some(vec![value; owner.geometry().block_len(1).unwrap()]))
                    .await
                    .unwrap();
            }

            assert_eq!(
                patch
                    .zero_ids(0..owner.geometry().block_count())
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap(),
                [0]
            );
            assert!(
                patch
                    .zeros()
                    .write()
                    .await
                    .insert(vec![0, 1])
                    .await
                    .is_err()
            );

            let rows: Vec<_> = patch.replacements().try_collect().await.unwrap();
            assert_eq!(rows.len(), owner.geometry().block_count() as usize);
            assert_eq!(observation::counts().zero_lookups, 0);
            assert_eq!(rows[0].1, None);
            assert_eq!(
                rows[1].1,
                Some(vec![2.; owner.geometry().block_len(1).unwrap()])
            );
            assert_eq!(
                patch.read(0).await.unwrap(),
                Some(vec![0.; owner.geometry().block_len(0).unwrap()])
            );
            assert_eq!(observation::counts().zero_lookups, 1);
            assert_eq!(observation::counts().payload_reads, rows.len() - 1);

            // Membership cannot hide malformed native replacement storage.
            let payload = patch
                .storage
                .dir()
                .read()
                .await
                .get_dir("payload")
                .cloned()
                .unwrap();

            let values = payload.read().await.get_dir("values").cloned().unwrap();
            let primary = values.read().await.get_dir("primary").cloned().unwrap();
            let root = primary.read().await.get_file(ROOT).cloned().unwrap();
            {
                let mut node = root.write::<crate::TensorSparseNode<f64>>().await.unwrap();
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
            let zeros = patch
                .storage
                .dir()
                .read()
                .await
                .get_dir("zeros")
                .cloned()
                .unwrap();

            let root = zeros.read().await.get_file(ROOT).cloned().unwrap();
            *root.write::<crate::TensorIndexNode>().await.unwrap() =
                b_tree::Node::Leaf(vec![vec![0, 0, 0]]);

            assert!(patch.read(0).await.is_err());
            assert!(patch.replacements().try_collect::<Vec<_>>().await.is_err());
            *root.write::<crate::TensorIndexNode>().await.unwrap() =
                b_tree::Node::Leaf(vec![vec![owner.geometry().block_count()]]);

            let error = patch.zeros().validate().await.unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert!(freqfs::IntegrityError::is_marked(&error));
            assert!(tc_error::TCError::from(error).requires_recovery());
            drop(patch.zeros().try_write().unwrap());
        }
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn zero_batches_are_bounded_and_reload_single_column_nodes() {
    Box::pin(async {
        use futures::TryStreamExt;

        use super::ZeroSchema;
        use crate::StorageContext;

        let len = ZERO_BATCH_SIZE as u64 + 1;
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

            let mut patch = Delta::<_, f64>::create(
                allocation(txn.subcontext_unique().context().await.unwrap()),
                geometry,
                false,
            )
            .await
            .unwrap();

            // Missing native payloads are valid explicit-zero replacements.
            patch
                .zeros()
                .write()
                .await
                .insert_sorted(futures::stream::iter((0..len).map(|id| Ok(vec![id]))))
                .await
                .unwrap();

            patch.zeros().sync().await.unwrap();
            let zeros = patch
                .storage
                .dir()
                .read()
                .await
                .get_dir("zeros")
                .cloned()
                .unwrap();

            let path = zeros.read().await.path().to_owned();
            let cache = freqfs::Cache::<crate::PersistentFile>::new(
                32 * 1024,
                None,
                0,
                std::time::Duration::from_secs(3),
            );

            Arc::get_mut(&mut patch).unwrap().zeros = Some(
                b_tree::BTreeLock::load(
                    ZeroSchema(len),
                    b_tree::collate::Collator::default(),
                    cache.load(path).unwrap(),
                )
                .unwrap(),
            );

            patch.zeros().validate().await.unwrap();
            let first = Delta::<_, f64>::zero_batch(patch.zeros(), 0..len)
                .await
                .unwrap();
            assert_eq!(first, (0..len - 1).collect::<Vec<_>>());
            let mut visited = 0;
            Delta::<_, f64>::visit_zeros(&patch.zeros().read().await, 0..len, |id| {
                assert_eq!(id, visited);
                visited += 1;
                std::ops::ControlFlow::Continue(())
            })
            .await
            .unwrap();

            assert_eq!(visited, len);
            assert_eq!(
                Delta::<_, f64>::zero_batch(patch.zeros(), len - 1..len)
                    .await
                    .unwrap(),
                [len - 1]
            );

            assert!(
                Delta::<_, f64>::zero_batch(patch.zeros(), len..len)
                    .await
                    .unwrap()
                    .is_empty()
            );
            // Returning a batch releases the index guard.
            drop(
                tokio::time::timeout(std::time::Duration::from_secs(2), patch.zeros().write())
                    .await
                    .unwrap(),
            );

            let mut replacements = Box::pin(patch.replacements());
            let mut expected = 0;
            while let Some((id, values)) = replacements.try_next().await.unwrap() {
                assert_eq!(id, expected);
                assert!(values.is_none());
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
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };

            let owner = value.base().unwrap();
            let patch = owner.stage(&txn, false).await.unwrap();
            assert_eq!(std::mem::size_of_val(&patch), std::mem::size_of::<usize>());
            let copy = patch.clone();
            assert!(Arc::ptr_eq(&copy, &patch));
            assert_eq!(Arc::strong_count(&patch), 2);
            assert_eq!(Arc::strong_count(&patch.generation), 1);
            drop(copy);
            assert_eq!(Arc::strong_count(&patch), 1);
            assert!(owner.snapshot(txn.id(), None).unwrap()._permit.is_none());
            patch
                .write(0, Some(vec![7.; owner.geometry().block_len(0).unwrap()]))
                .await
                .unwrap();

            let generation = patch.generation.clone();
            if checked {
                owner.publish_if_unchanged(&txn, patch, None).await.unwrap();
            } else {
                owner.publish(&txn, patch, |_| Ok(())).await.unwrap();
            }

            let pending = owner.versions.pending_for_test(txn.id());
            assert!(Arc::ptr_eq(&generation, &pending.generation));
            assert_eq!(pending.read(0).await.unwrap().unwrap()[0], 7.);
            let patch = owner.stage(&txn, false).await.unwrap();
            patch
                .write(1, Some(vec![0.; owner.geometry().block_len(1).unwrap()]))
                .await
                .unwrap();

            owner.publish(&txn, patch, |_| Ok(())).await.unwrap();
            assert_eq!(pending.read(0).await.unwrap().unwrap()[0], 7.);
            assert_eq!(pending.read(1).await.unwrap().unwrap()[0], 0.);

            // Retaining a generation stamp does not retain an unpublished workspace owner.
            let unpublished = owner.stage(&txn, false).await.unwrap();
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
async fn sparse_restoration_errors_and_cancellation_preserve_pending_data() {
    Box::pin(async {
        let (_root, txn) = setup().await;
        let (source, _) = create(&txn, vec![4].into(), true, 2).await;
        let (target, _) = create(&txn, vec![4].into(), true, 2).await;
        source.write_value(&txn, &[0], 3.0.into()).await.unwrap();
        source.write_value(&txn, &[3], 5.0.into()).await.unwrap();
        target.write_value(&txn, &[1], 7.0.into()).await.unwrap();
        let (Tensor::F64(source_access), Tensor::F64(target_access)) = (&source, &target) else {
            unreachable!()
        };
        let source_base = source_access.base().unwrap();
        let target_base = target_access.base().unwrap();
        let deltas = source_base.versions.visible_snapshot(txn.id()).unwrap();
        {
            // Stall reads after admission while both collection reservations
            // are held. A reciprocal restore must reject the conflict immediately.
            let _membership = deltas[0].zeros().write().await;
            let mut restoring = Box::pin(target.restore(&txn, &source));
            assert!(
                tokio::task::unconstrained(restoring.as_mut())
                    .now_or_never()
                    .is_none()
            );
            assert!(
                source
                    .restore(&txn, &target)
                    .now_or_never()
                    .expect("reciprocal restore returns without waiting")
                    .is_err()
            );
        }

        assert_eq!(target.read_value(txn.id(), &[1]).await.unwrap(), 7.0.into());
        for owner in [source_base, target_base] {
            assert!(owner.try_acquire_write_permit(txn.id(), 0..4).is_ok());
        }

        corrupt_block(source_base, txn.id(), &[3]).await;
        let error = target.restore(&txn, &source).await.unwrap_err();
        assert!(error.requires_recovery());
        assert_eq!(target.read_value(txn.id(), &[0]).await.unwrap(), 0.0.into());
        assert_eq!(target.read_value(txn.id(), &[1]).await.unwrap(), 7.0.into());
        for owner in [source_base, target_base] {
            assert!(owner.try_acquire_write_permit(txn.id(), 0..4).is_ok());
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn staged_publication_cancellation_rolls_back_and_stale_generations_conflict() {
    Box::pin(async {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![2].into(), true, 1).await;
            tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };

            let owner = value.base().unwrap();
            for complete in [false, true] {
                let stamp = owner.pending_stamp(txn.id()).unwrap();
                let patch = owner.stage(&txn, complete).await.unwrap();
                patch
                    .write(0, Some(vec![3.0; owner.geometry().block_len(0).unwrap()]))
                    .await
                    .unwrap();

                tensor.write_value(&txn, &[0], 2.0.into()).await.unwrap();
                let error = owner
                    .publish_if_unchanged(&txn, patch, stamp)
                    .await
                    .unwrap_err();
                assert_eq!(error.code(), tc_error::ErrorKind::Conflict);
                assert_eq!(tensor.read_value(txn.id(), &[0]).await.unwrap(), 2.0.into());
            }

            tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
            tensor.write_value(&txn, &[1], 2.0.into()).await.unwrap();
            let stamp = owner.pending_stamp(txn.id()).unwrap();
            let left = owner.stage(&txn, false).await.unwrap();
            left.write(0, Some(vec![2.0])).await.unwrap();
            let right = owner.stage(&txn, false).await.unwrap();
            right.write(1, Some(vec![1.0])).await.unwrap();
            {
                let _left_write = owner.try_acquire_write_permit(txn.id(), 0..1).unwrap();
                let _right_write = owner.try_acquire_write_permit(txn.id(), 1..2).unwrap();
                let staged_index = left.zeros().write().await;
                let mut first =
                    Box::pin(owner.publish_if_unchanged(&txn, left.clone(), stamp.clone()));
                assert!(tokio::task::unconstrained(first.as_mut()).now_or_never().is_none());
                let error = owner.publish_if_unchanged(&txn, right, stamp).await.unwrap_err();
                assert_eq!(error.code(), tc_error::ErrorKind::Conflict);
                drop(staged_index);
                first.await.unwrap();
            }
            assert_eq!(tensor.read_value(txn.id(), &[0]).await.unwrap(), 2.0.into());
            assert_eq!(tensor.read_value(txn.id(), &[1]).await.unwrap(), 2.0.into());

            let delta = owner.versions.pending_for_test(txn.id());
            let patch = owner.stage(&txn, false).await.unwrap();
            patch
                .write(0, Some(vec![9.0; owner.geometry().block_len(0).unwrap()]))
                .await
                .unwrap();

            let occupied = delta.zeros().write().await;
            let mut publication = Box::pin(owner.publish(&txn, patch, |_| Ok(())));
            tokio::select! {
                result = &mut publication => panic!("publication should wait for the occupied index: {result:?}"),
                result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                    loop {
                        if delta.payload.read(0).await.unwrap().unwrap()[0] == 9.0 { break; }
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

        let (Tensor::F64(left), Tensor::F64(right)) = (&left, &right) else {
            unreachable!()
        };

        let [left, right] =
            crate::tensor::TensorAccess::expressions([left, right], txn.id()).unwrap();

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
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };

            let delta = value
                .base()
                .unwrap()
                .versions.pending_for_test(txn.id());

            let zeros = delta.zeros().write().await;
            let len = delta.payload.geometry().block_len(0).unwrap();
            let mut mutation = Box::pin(delta.write(0, Some(vec![2.; len])));
            tokio::time::timeout(std::time::Duration::from_secs(5),async {
                tokio::select! {
                    result=&mut mutation => panic!("publication should wait for zeros index: {result:?}"),
                    _=async {
                        loop {
                            if delta.payload.read(0).await.unwrap().unwrap()[0] == 2. { break; }
                            tokio::task::yield_now().await;
                        }
                    } => {}
                }
            }).await.unwrap();

            drop(mutation);
            drop(zeros);
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
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };

            let owner = value.base().unwrap();
            let delta = owner.stage(&txn, false).await.unwrap();
            delta.write(0, occupied.then(|| vec![1.])).await.unwrap();

            // Share the native fixture to isolate traversal depth from storage setup.
            for pending in [false, true] {
                let mut snapshot =
                    test_snapshot(owner.canonical().clone(), vec![delta.clone(); 1024]);

                snapshot._permit = owner.read_snapshot(txn.id(), 0..1).await.unwrap()._permit;

                let lease = Arc::downgrade(snapshot._permit.as_ref().unwrap());
                let mut scan = snapshot.occupied_blocks(0..1);
                drop(snapshot);
                if pending {
                    let values = directory
                        .read()
                        .await
                        .get_dir("base")
                        .cloned()
                        .unwrap()
                        .read()
                        .await
                        .get_dir("values")
                        .cloned()
                        .unwrap();

                    let primary = values.read().await.get_dir("primary").cloned().unwrap();
                    let root = primary
                        .read()
                        .await
                        .get_file("00000000-0000-0000-0000-000000000000")
                        .cloned()
                        .unwrap();

                    let guard = root.write::<crate::TensorSparseNode<f64>>().await.unwrap();
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
                        .clone()
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
async fn batched_payload_errors_preserve_provenance_and_release_guards() {
    use fensor::{Layout, StorageGeometry, TensorRead, TensorSchema, TensorView};
    use futures::TryStreamExt;
    use tokio::sync::oneshot;

    use crate::StorageContext;

    Box::pin(observation::scope(async {
        for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
            for marked in [false, true] {
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
                    allocation(txn.subcontext_unique().context().await.unwrap()),
                    geometry.clone(),
                    false,
                )
                .await
                .unwrap();

                for id in 0..geometry.block_count() {
                    let mut values = vec![0.; geometry.block_len(id).unwrap()];
                    values.fill(7.);

                    delta.write(id, Some(values)).await.unwrap();
                }

                let snapshot = test_snapshot(canonical, vec![delta.clone()]);
                let view = TensorView::new(snapshot.clone());
                let (observed, mut receipt) = oneshot::channel();
                observation::on_scatter(&delta.generation, move || {
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
                {
                    let fensor::Error::Io(error) = values.unwrap_err() else {
                        panic!("expected original payload I/O error")
                    };

                    assert_eq!(error.to_string(), "obsolete payload read");
                    assert_eq!(freqfs::IntegrityError::is_marked(&error), marked);
                    assert_eq!(tc_error::TCError::from(error).requires_recovery(), marked);
                }

                // The failed visitor releases its BTree/file guards before either
                // returning the error or re-entering canonical storage.
                delta.payload.replace(0, None).await.unwrap();
            }
        }
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn batched_membership_groups_unresolved_runs_and_cancels_cleanly() {
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
                allocation(txn.subcontext_unique().context().await.unwrap()),
                geometry.clone(),
                false,
            )
            .await
            .unwrap();

            older
                .zeros()
                .write()
                .await
                .insert_sorted(futures::stream::iter((0..640).map(|id| Ok(vec![id]))))
                .await
                .unwrap();

            let newer = super::Delta::create(
                allocation(txn.subcontext_unique().context().await.unwrap()),
                geometry,
                false,
            )
            .await
            .unwrap();

            newer
                .zeros()
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
            let mut sources = vec![Canonical; ids.len()];
            for (delta, scans) in [(&newer, 1), (&older, 3)] {
                observation::watch(&delta.generation);
                snapshot.resolve_sources(&ids, &mut sources).await.unwrap();
                assert_eq!(observation::counts().membership_scans, scans);
                assert_eq!(observation::counts().zero_lookups, 0);
                drop(delta.zeros().try_write().unwrap());
                for (&id, &source) in ids.iter().zip(&sources) {
                    let expected = if id < 640 { Zero } else { Canonical };
                    assert_eq!(source, expected, "block {id}");
                }
            }

            let ids = [0, 1, 2, 129, 130, 131, 257, 258, 259, 700, 767];
            let mut sources = [Canonical; 11];
            for (delta, scans) in [(&newer, 5), (&older, 6)] {
                observation::watch(&delta.generation);
                snapshot.resolve_sources(&ids, &mut sources).await.unwrap();
                assert_eq!(observation::counts().membership_scans, scans);
                assert_eq!(
                    sources,
                    [
                        Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Canonical, Canonical
                    ],
                );
            }

            observation::watch(&older.generation);
            let mut sources = [Canonical; 2];
            fn send<T: Send>(value: T) -> T {
                value
            }

            send(snapshot.resolve_sources(&[129, 130], &mut sources))
                .await
                .unwrap();

            assert_eq!(sources, [Zero, Zero]);
            assert_eq!(observation::counts().membership_scans, 0);

            for delta in [&newer, &older] {
                observation::watch(&delta.generation);
                let mut source = [Canonical];
                snapshot.resolve_sources(&[700], &mut source).await.unwrap();
                assert_eq!(source, [Canonical]);
                assert_eq!(observation::counts().membership_scans, 1);
            }

            let zeros = older
                .storage
                .dir()
                .read()
                .await
                .get_dir("zeros")
                .cloned()
                .unwrap();

            let root = zeros.read().await.get_file(ROOT).cloned().unwrap();
            let last = match &*root.read::<crate::TensorIndexNode>().await.unwrap() {
                b_tree::Node::Index(_, children) => *children.last().unwrap(),
                b_tree::Node::Leaf(_) => panic!("fixture must span native leaves"),
            };

            let last = zeros
                .read()
                .await
                .get_file(&last.to_string())
                .cloned()
                .unwrap();

            // The first older run assigns block 0 before a later leaf blocks.
            // Cancellation must release every retained index/file read guard.
            {
                let held = last.write::<crate::TensorIndexNode>().await.unwrap();
                let (completed, receipt) = oneshot::channel();
                observation::on_membership(&older.generation, 1, move || {
                    completed.send(()).unwrap();
                    Ok(())
                });

                let mut sources = [Canonical; 3];
                let mut read = Box::pin(tokio::task::unconstrained(
                    snapshot.resolve_sources(&[0, 129, 639], &mut sources),
                ));

                tokio::select! {
                    biased;
                    result = &mut read => panic!("later leaf must block: {result:?}"),
                    result = receipt => result.unwrap(),
                }

                assert_eq!(observation::counts().membership_scans, 2);

                drop(read);
                drop(held);
                assert_eq!(sources, [Zero, Zero, Canonical]);

                drop(older.zeros().try_write().unwrap());
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
            let delta = super::Delta::create(allocation(txn.subcontext_unique().context().await.unwrap()),
                geometry.clone(), false,
                )
            .await
            .unwrap();

            let zeros = delta
                .storage.dir()
                .read()
                .await
                .get_dir("zeros")
                .cloned()
                .unwrap();

            let root = zeros.read().await.get_file(ROOT).cloned().unwrap();
            *root.write::<crate::TensorIndexNode>().await.unwrap() = b_tree::Node::Leaf(keys);
            roots.push(root);
            deltas.push(delta);
        }

        let snapshot = test_snapshot(canonical, deltas);

        // The newer layer's later block is visited before the older layer's
        // earlier block. Strict ordering includes duplicate membership rows.
        observation::watch(&snapshot.deltas[0].generation);
        let mut sources = [Canonical; 2];
        let error = snapshot
            .resolve_sources(&[0, 3], &mut sources)
            .await
            .unwrap_err();

        assert!(matches!(error, fensor::Error::Io(error)
            if error.kind() == std::io::ErrorKind::InvalidData
                && freqfs::IntegrityError::is_marked(&error)));
        assert_eq!(observation::counts().membership_scans, 0);

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

        drop(snapshot.deltas[0].zeros().try_write().unwrap());
        drop(snapshot.deltas[1].zeros().try_write().unwrap());

        // A failed membership visitor preserves its native cause and releases guards.
        *roots[0].write::<crate::TensorIndexNode>().await.unwrap() =
            b_tree::Node::Leaf(vec![vec![0], vec![3]]);

        let delta = &snapshot.deltas[0];
        observation::on_membership(&delta.generation, 2, || {
            Err(std::io::Error::other("membership completion failed").into())
        });

        let error = snapshot.resolve_sources(&[0, 3], &mut sources).await.unwrap_err();
        assert!(matches!(error, fensor::Error::Io(error) if error.to_string() == "membership completion failed"));
        drop(delta.zeros().try_write().unwrap());
        drop(roots[0].try_write::<crate::TensorIndexNode>().unwrap());
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn whole_replacement_retains_coverage_through_patches_and_compaction() {
    use futures::TryStreamExt;

    use crate::test::tensor::{id, values};

    Box::pin(async {
        let (_, txn) = setup().await;
        let (tensor, dir) = create(&txn, vec![3].into(), true, 1).await;
        tensor.fill(&txn, 0.0.into()).await.unwrap();
        tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
        let range = vec![fensor::AxisRange::In(1, 3, 1)].into();
        tensor.fill_range(&txn, range, 2.0.into()).await.unwrap();
        assert_eq!(values(&tensor, txn.id()).await, [1., 2., 2.]);

        let Tensor::F64(access) = &tensor else {
            unreachable!()
        };
        let owner = access.base().unwrap();
        let pending = owner.versions.pending_for_test(txn.id());
        {
            let dir = pending.storage.dir().read().await;
            assert_eq!(dir.len(), 1);
            assert!(dir.get_dir("replacement").is_some());
        }
        drop(pending);
        tensor.commit(txn.id()).await.unwrap();

        let next = txn.at(11);
        for (i, value) in [0., 5., 0.].into_iter().enumerate() {
            tensor
                .write_value(&next, &[i as u64], value.into())
                .await
                .unwrap();
        }
        tensor.commit(next.id()).await.unwrap();
        assert_eq!(tensor.read_value(id(12), &[1]).await.unwrap(), 5.0.into());

        let snapshot = owner.snapshot(id(12), None).unwrap();
        let mut keys = snapshot.occupied_blocks(0..3);
        drop(snapshot);
        tensor.finalize(&next.id()).await.unwrap();
        assert_eq!(keys.try_next().await.unwrap(), Some(1));
        assert_eq!(keys.try_next().await.unwrap(), None);

        let schema = tensor.schema().unwrap();
        let path = dir.read().await.path().to_owned();
        drop(tensor);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            128 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );
        let reopened = Tensor::load(cache.load(path).unwrap(), schema)
            .await
            .unwrap();
        assert_eq!(values(&reopened, id(12)).await, [0., 5., 0.]);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cutoff_reuses_whole_allocation_and_preserves_snapshots() {
    use crate::test::tensor::{id, values};

    Box::pin(async {
        for fill in [0., 2.] {
            let (_, txn) = setup().await;
            let (tensor, dir) = create(&txn, vec![4].into(), true, 1).await;
            let schema = tensor.schema().unwrap();
            let Tensor::F64(access) = &tensor else {
                unreachable!()
            };
            let owner = access.base().unwrap();
            tensor.write_value(&txn, &[0], 7.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
            let old = owner.snapshot(txn.id(), None).unwrap();
            let replacement = txn.at(11);
            tensor.fill(&replacement, fill.into()).await.unwrap();
            tensor.commit(replacement.id()).await.unwrap();

            let expected = owner.snapshot(replacement.id(), None).unwrap().deltas[0]
                .storage
                .id()
                .clone();

            tensor.finalize(&replacement.id()).await.unwrap();
            assert_eq!(old.read_logical_block(0).await.unwrap(), [7.]);
            drop(old);
            let path = dir.read().await.path().to_owned();
            drop(tensor);

            let cache = freqfs::Cache::<crate::PersistentFile>::new(
                128 * 1024,
                None,
                0,
                std::time::Duration::from_secs(3),
            );
            let reopened = Tensor::load(cache.load(path).unwrap(), schema)
                .await
                .unwrap();
            let Tensor::F64(access) = &reopened else {
                unreachable!()
            };
            let snapshot = access.base().unwrap().snapshot(id(12), None).unwrap();
            assert_eq!(snapshot.deltas.len(), 1);
            assert_eq!(snapshot.deltas[0].storage.id(), &expected);
            assert_eq!(values(&reopened, id(12)).await, [fill; 4]);
            reopened
                .write_value(&txn.at(12), &[1], 9.0.into())
                .await
                .unwrap();
            reopened.commit(id(12)).await.unwrap();
            reopened.finalize(&id(12)).await.unwrap();
            assert_eq!(values(&reopened, id(13)).await, [fill, 9., fill, fill]);
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn reopen_classifies_invalid_delta_directories_as_recovery_required() {
    Box::pin(async {
        for (whole, parent, remove, add) in [
            (false, [].as_slice(), Some("zeros"), None),
            (false, [].as_slice(), Some("zeros"), Some("touched")),
            (true, [].as_slice(), None, Some("zeros")),
            (true, [].as_slice(), Some("replacement"), None),
            (
                false,
                ["payload", "values"].as_slice(),
                Some("primary"),
                None,
            ),
        ] {
            let (_, txn) = setup().await;
            let (tensor, dir) = create(&txn, vec![4].into(), true, 1).await;
            let schema = tensor.schema().unwrap();
            if whole {
                tensor.fill(&txn, 0.0.into()).await.unwrap();
            } else {
                tensor.write_value(&txn, &[1], 7.0.into()).await.unwrap();
            }
            tensor.commit(txn.id()).await.unwrap();

            let Tensor::F64(access) = &tensor else {
                unreachable!()
            };
            let versions = access
                .base()
                .unwrap()
                .versions
                .visible_snapshot(txn.id())
                .unwrap();
            {
                let mut target = versions.last().unwrap().storage.dir().clone();
                for child in parent {
                    let next = target.read().await.get_dir(child).cloned().unwrap();
                    target = next;
                }

                let mut version = target.write().await;
                if let Some(name) = remove {
                    version.delete(name).await;
                }
                if let Some(name) = add {
                    version.create_dir(name.into()).unwrap();
                }
                version.sync_all().await.unwrap();
            }

            let path = dir.read().await.path().to_owned();
            let cache = freqfs::Cache::<crate::PersistentFile>::new(
                128 * 1024,
                None,
                0,
                std::time::Duration::from_secs(3),
            );
            let result =
                Tensor::<crate::test::TestTxn>::load(cache.load(path).unwrap(), schema).await;
            assert!(result.unwrap_err().requires_recovery());
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn full_sparse_assignment_visits_occupied_blocks_and_replaces_pending() {
    observation::scope(Box::pin(async {
        let (_, txn) = setup().await;
        let size = 3 * fensor::MAX_BATCH_ELEMENTS as u64;
        let (source, _) = create(&txn, vec![size].into(), true, 2).await;
        let (target, _) = create(&txn, vec![size].into(), true, 2).await;
        source
            .write_value(&txn, &[size - 1], 7.0.into())
            .await
            .unwrap();
        target.write_value(&txn, &[0], 9.0.into()).await.unwrap();

        let Tensor::F64(value) = &source else {
            panic!("f64 owner")
        };
        let pending = value.base().unwrap().versions.pending_for_test(txn.id());
        observation::watch(&pending.generation);
        let doubled = source.mul_scalar(txn.id(), 2.0.into()).await.unwrap();
        target.write_tensor(&txn, doubled).await.unwrap();

        assert!(observation::counts().membership_scans <= 1);
        assert_eq!(target.read_value(txn.id(), &[0]).await.unwrap(), 0.0.into());
        assert_eq!(
            target.read_value(txn.id(), &[size - 1]).await.unwrap(),
            14.0.into()
        );
    }))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn full_fills_replace_compact_blocks_without_reading_old_payloads() {
    use crate::tensor::Tensor;
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
            let Tensor::F64(value) = &tensor else {
                panic!("f64 owner")
            };

            let owner = value.base().unwrap();

            let previous = owner.versions.pending_for_test(later.id());

            if !sparse {
                let snapshot = owner.snapshot(later.id(), None).unwrap();
                assert!(matches!(
                    snapshot.occupied_blocks(0..3).next().await,
                    Some(Err(fensor::Error::InvalidLayout(_)))
                ));
            }

            assert_eq!(owner.geometry().block_len(2).unwrap(), 1);
            observation::watch(&previous.generation);
            tensor.fill(&later, 3.0.into()).await.unwrap();
            assert_eq!(
                observation::counts().payload_reads,
                0,
                "full replacement does not read old delta payloads"
            );

            assert_eq!(values(&tensor, later.id()).await, [3.; 5]);
            if !sparse {
                let snapshot = owner.snapshot(later.id(), None).unwrap();
                assert!(matches!(
                    snapshot.occupied_blocks(0..3).next().await,
                    Some(Err(fensor::Error::InvalidLayout(_)))
                ));
            }

            tensor.rollback(&later.id()).await.unwrap();
            assert_eq!(values(&tensor, id(12)).await, [7., 0., 0., 0., 0.]);

            if !sparse {
                let blocks = dir
                    .read()
                    .await
                    .get_dir("base")
                    .cloned()
                    .unwrap()
                    .read()
                    .await
                    .get_dir("blocks")
                    .unwrap()
                    .clone();

                let file = blocks.read().await.get_file("0").unwrap().clone();
                let held = file.write::<Vec<f64>>().await.unwrap();
                let range = vec![fensor::AxisRange::In(0, 2, 1)].into();
                tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    Box::pin(tensor.fill_range(&txn.at(12), range, 3.0.into())),
                )
                .await
                .expect("an aligned partial fill does not read replaced payloads")
                .unwrap();

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
