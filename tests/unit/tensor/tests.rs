use std::time::Duration;

use fensor::TensorSource;
use futures::TryStreamExt;
use number_general::{FloatType, Number, UIntType};
use safecast::CastInto;
use tc_ir::{IntoView, Transact, Transaction};

use super::*;
use crate::test::tensor as support;
use crate::{Collection, CopyFrom, PersistentFile, Restore};
pub(super) use support::{Txn, create, setup};
use support::{id, values};

macro_rules! test {
    ($name:ident, $body:expr) => {
        #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
        async fn $name() {
            Box::pin($body).await;
        }
    };
}

test!(decisions_ordered_visibility_and_cutoff, async {
    let (_, txn) = setup().await;
    let (tensor, _) = create(&txn, vec![8].into(), false, 2).await;
    tensor.write_value(&txn, &[0], 7.0.into()).await.unwrap();
    assert_eq!(
        tensor.read_value(id(9), &[0]).await.unwrap(),
        Number::from(0.0)
    );

    assert_eq!(
        tensor.read_value(id(10), &[0]).await.unwrap(),
        Number::from(7.0)
    );

    assert!(
        tokio::time::timeout(Duration::from_millis(20), tensor.read_value(id(11), &[0]))
            .await
            .is_err()
    );

    tensor.commit(id(10)).await.unwrap();
    tensor.commit(id(10)).await.unwrap();
    assert_eq!(
        tensor
            .write_value(&txn, &[0], 99.0.into())
            .await
            .unwrap_err()
            .message(),
        txn_lock::Error::Committed.to_string()
    );

    assert_eq!(
        tokio::time::timeout(Duration::from_millis(200), tensor.read_value(id(11), &[0]))
            .await
            .expect("rejected committed write must not reserve its block")
            .unwrap(),
        Number::from(7.0)
    );

    assert_eq!(
        tensor.read_value(id(11), &[0]).await.unwrap(),
        Number::from(7.0)
    );

    assert_eq!(
        tensor.read_value(id(9), &[0]).await.unwrap(),
        Number::from(0.0)
    );

    tensor
        .write_value(&txn.at(12), &[0], 8.0.into())
        .await
        .unwrap();

    tensor.rollback(&id(12)).await.unwrap();
    tensor.rollback(&id(12)).await.unwrap();
    assert_eq!(
        tensor.read_value(id(13), &[0]).await.unwrap(),
        Number::from(7.0)
    );

    tensor
        .write_value(&txn.at(14), &[1], 3.0.into())
        .await
        .unwrap();

    tensor.commit(id(14)).await.unwrap();
    for coord in [[0], [2]] {
        assert_eq!(
            tensor
                .write_value(&txn.at(13), &coord, 99.0.into())
                .await
                .unwrap_err()
                .code(),
            tc_error::ErrorKind::Conflict,
        );
    }

    assert_eq!(
        tensor.read_value(id(13), &[1]).await.unwrap(),
        Number::from(0.0)
    );

    tensor.finalize(&id(10)).await.unwrap();
    tensor.finalize(&id(10)).await.unwrap();
    assert!(tensor.commit(id(10)).await.is_err());
    assert!(tensor.read_value(id(10), &[0]).await.is_err());
    assert_eq!(
        tensor
            .write_value(&txn, &[0], 99.0.into())
            .await
            .unwrap_err()
            .message(),
        txn_lock::Error::Outdated.to_string()
    );

    assert_eq!(
        tokio::time::timeout(Duration::from_millis(200), tensor.read_value(id(15), &[0]))
            .await
            .expect("rejected finalized write must not reserve its block")
            .unwrap(),
        Number::from(7.0)
    );

    assert_eq!(
        values(&tensor, id(15)).await,
        [7.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    tensor.finalize(&id(14)).await.unwrap();
});

test!(
    conflicts_are_logical_blocks_and_views_resolve_overlaps,
    async {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![2, 4].into(), false, 2).await;
        tensor.write_value(&txn, &[0, 0], 1.0.into()).await.unwrap();
        assert!(
            tensor
                .write_value(&txn.at(11), &[0, 1], 2.0.into())
                .await
                .is_err()
        );

        tensor
            .write_value(&txn.at(11), &[1, 3], 3.0.into())
            .await
            .unwrap();

        let transposed = tensor.clone().transpose(None).unwrap();
        assert!(
            transposed
                .write_value(&txn.at(12), &[1, 0], 4.0.into())
                .await
                .is_err()
        );

        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                tensor.read_value(id(13), &[0, 0])
            )
            .await
            .is_err()
        );

        tensor.finalize(&id(11)).await.unwrap(); // cutoff discards both pending deltas
        assert_eq!(
            tensor.read_value(id(13), &[0, 0]).await.unwrap(),
            Number::from(0.0)
        );

        tensor
            .write_value(&txn.at(14), &[0, 0], 5.0.into())
            .await
            .unwrap();
    }
);

test!(
    stale_assignment_rejects_before_reading_source_payload,
    async {
        let (_, txn) = setup().await;
        let (source, dir) = create(&txn, vec![2].into(), false, 2).await;
        let (target, _) = create(&txn, vec![2].into(), false, 2).await;
        let newer = txn.at(12);
        target.write_value(&newer, &[1], 8.0.into()).await.unwrap();
        target.commit(newer.id()).await.unwrap();

        let base = dir.read().await.get_dir("base").cloned().unwrap();
        let blocks = base.read().await.get_dir("blocks").cloned().unwrap();
        blocks.write().await.delete("0").await;

        let older = txn.at(11);
        assert_eq!(
            target
                .write_tensor(&older, source)
                .await
                .unwrap_err()
                .code(),
            tc_error::ErrorKind::Conflict,
        );
        assert_eq!(values(&target, id(13)).await, [0.0, 8.0]);
    }
);

test!(self_assignment_stages_before_publishing, async {
    let (_, txn) = setup().await;
    let tensor = Tensor::dense_f64(&txn, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .await
        .unwrap();

    let transposed = tensor.clone().transpose(None).unwrap();
    tensor.write_tensor(&txn, transposed).await.unwrap();
    assert_eq!(values(&tensor, txn.id()).await, [1.0, 3.0, 2.0, 4.0]);
    assert_eq!(values(&tensor, id(9)).await, [1.0, 2.0, 3.0, 4.0]);
    tensor.rollback(&txn.id()).await.unwrap();
    assert_eq!(values(&tensor, id(11)).await, [1.0, 2.0, 3.0, 4.0]);
    let source = tensor
        .clone()
        .slice(vec![AxisRange::At(1), AxisRange::In(0, 2, 1)].into())
        .unwrap();

    tensor
        .write_tensor_range(
            &txn.at(11),
            vec![AxisRange::At(0), AxisRange::In(0, 2, 1)].into(),
            source,
        )
        .await
        .unwrap();

    assert_eq!(values(&tensor, id(11)).await, [3.0, 4.0, 3.0, 4.0]);
});

test!(transformed_views_cannot_regain_mutation_authority, async {
    for sparse in [false, true] {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![2, 2].into(), sparse, 2).await;
        tensor.write_value(&txn, &[0, 1], 3.0.into()).await.unwrap();
        let full: Range = vec![AxisRange::In(0, 2, 1); 2].into();
        let views = [
            tensor.clone().into_read_only(),
            tensor.clone().slice(full.clone()).unwrap(),
            tensor.clone().reshape(vec![2, 2].into()).unwrap(),
            tensor
                .clone()
                .reshape(vec![4].into())
                .unwrap()
                .reshape(vec![2, 2].into())
                .unwrap(),
            tensor.clone().transpose(Some(vec![0, 1].into())).unwrap(),
            tensor
                .clone()
                .transpose(None)
                .unwrap()
                .transpose(None)
                .unwrap(),
            tensor.clone().flip(0).unwrap().flip(0).unwrap(),
            tensor
                .clone()
                .unsqueeze(vec![0].into())
                .unwrap()
                .squeeze(vec![0].into())
                .unwrap(),
            tensor.clone().broadcast(vec![2, 2].into()).unwrap(),
            tensor.as_dense(txn.id()).await.unwrap(),
        ];

        for view in views {
            // Cloning a public value cannot recover mutation authority.
            let view = view.clone();
            assert!(!view.is_persistent());
            assert!(dispatch!(&view, value => value.base().is_err()));
            assert!(view.write_value(&txn, &[0, 1], 9.0.into()).await.is_err());
            assert!(view.fill(&txn, 9.0.into()).await.is_err());
            assert!(
                view.fill_range(&txn, full.clone(), 9.0.into())
                    .await
                    .is_err()
            );

            assert!(view.write_tensor(&txn, tensor.clone()).await.is_err());
            assert!(
                view.write_tensor_range(&txn, full.clone(), tensor.clone())
                    .await
                    .is_err()
            );

            assert!(view.restore(&txn, &tensor).await.is_err());
            assert!(view.commit(txn.id()).await.is_err());
            assert!(view.rollback(&txn.id()).await.is_err());
            assert!(view.finalize(&txn.id()).await.is_err());
            assert_eq!(values(&view, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
        }
        assert_eq!(values(&tensor, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
        for bound in [
            AxisRange::In(1, 1, 1),
            AxisRange::In(0, 3, 1),
            AxisRange::In(0, 2, 0),
        ] {
            let range = vec![bound, AxisRange::In(0, 2, 1)].into();
            let error = tensor
                .fill_range(&txn, range, 9.0.into())
                .await
                .unwrap_err();
            assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
        }

        // Explicit gather selections remain read-only even when their indices
        // are unique. Repeated indices must not acquire last-writer semantics.
        for indices in [vec![0, 1], vec![1, 0], vec![0, 0]] {
            let selection: Range = vec![AxisRange::Of(indices), AxisRange::In(0, 2, 1)].into();
            assert_eq!(
                tensor
                    .fill_range(&txn, selection.clone(), 9.0.into())
                    .await
                    .unwrap_err()
                    .code(),
                tc_error::ErrorKind::BadRequest
            );

            assert_eq!(
                tensor
                    .write_tensor_range(&txn, selection, tensor.clone())
                    .await
                    .unwrap_err()
                    .code(),
                tc_error::ErrorKind::BadRequest
            );

            assert_eq!(values(&tensor, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
        }
        let copied = crate::tensor::Tensor::copy_from(
            &txn,
            txn.subcontext_unique().context().await.unwrap(),
            &tensor.clone().into_read_only(),
        )
        .await
        .unwrap();

        assert!(copied.is_persistent());
        copied.fill(&txn, 5.0.into()).await.unwrap();
        copied.commit(txn.id()).await.unwrap();
        assert_eq!(values(&copied, id(11)).await, [5.0; 4]);
        assert_eq!(values(&tensor, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
    }
});

test!(
    immutable_results_survive_commit_visibility_publication,
    async {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![4].into(), false, 2).await;
        tensor.write_value(&txn, &[1], (-3.).into()).await.unwrap();
        let expression = tensor.abs(txn.id()).await.unwrap();
        let mut stored = tensor.elements(txn.id()).await.unwrap();
        let mut computed = expression.elements(txn.id()).await.unwrap();

        tensor.commit(txn.id()).await.unwrap();
        tensor
            .write_value(&txn.at(12), &[1], 9.0.into())
            .await
            .unwrap();

        tensor.commit(id(12)).await.unwrap();

        for value in [0.0_f64, -3., 0., 0.] {
            assert_eq!(stored.try_next().await.unwrap(), Some(value.into()));
            assert_eq!(computed.try_next().await.unwrap(), Some(value.abs().into()));
        }
        assert!(stored.try_next().await.unwrap().is_none());
        assert!(computed.try_next().await.unwrap().is_none());
        drop(stored);
        drop(computed);
        drop(expression);
        tensor.finalize(&id(12)).await.unwrap();
        assert_eq!(tensor.read_value(id(13), &[1]).await.unwrap(), 9.0.into());
    }
);

test!(expressions_and_streams_retain_and_release_leases, async {
    let (_, txn) = setup().await;
    let tensor = Tensor::dense_f64(&txn, vec![4], vec![1.0, 2.0, 3.0, 4.0])
        .await
        .unwrap();

    let expr = tensor.add(txn.id(), &tensor).await.unwrap();
    assert!(tensor.write_value(&txn, &[0], 5.0.into()).await.is_err());
    assert!(expr.write_value(&txn, &[0], 5.0.into()).await.is_err());
    assert_eq!(values(&expr, txn.id()).await, [2.0, 4.0, 6.0, 8.0]);
    for view in [expr.clone(), expr.clone().flip(0).unwrap()] {
        assert!(view.read_value(id(11), &[0]).await.is_err());
        assert!(view.elements(id(11)).await.is_err());
        assert!(view.add(id(11), &tensor).await.is_err());
    }

    drop(expr);
    let stream = tensor.elements(txn.id()).await.unwrap();
    assert!(tensor.write_value(&txn, &[0], 5.0.into()).await.is_err());
    drop(stream);
    tensor.write_value(&txn, &[0], 5.0.into()).await.unwrap();
    let mut stream = tensor.elements(txn.id()).await.unwrap();
    while stream.try_next().await.unwrap().is_some() {}
    tensor.write_value(&txn, &[0], 6.0.into()).await.unwrap();
    assert!(stream.try_next().await.unwrap().is_none());

    for sparse in [false, true] {
        for broad_first in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![8].into(), sparse, 2).await;
            let narrow = tensor
                .clone()
                .slice(vec![AxisRange::In(2, 4, 1)].into())
                .unwrap();

            let (first, second) = if broad_first {
                (&tensor, &narrow)
            } else {
                (&narrow, &tensor)
            };

            let first = first.elements(txn.id()).await.unwrap();
            let second = second.elements(txn.id()).await.unwrap();
            drop(second);
            assert!(
                tensor.write_value(&txn, &[2], 5.0.into()).await.is_err(),
                "retained overlap must exclude a point write (sparse={sparse}, broad_first={broad_first})"
            );

            assert!(tensor.fill(&txn, 9.0.into()).await.is_err());
            if !broad_first {
                tensor.write_value(&txn, &[6], 7.0.into()).await.unwrap();
            }
            drop(first);
            tensor.write_value(&txn, &[2], 5.0.into()).await.unwrap();
            assert_eq!(
                tensor.read_value(txn.id(), &[2]).await.unwrap(),
                Number::from(5.0)
            );

            tensor.fill(&txn, 9.0.into()).await.unwrap();
            assert_eq!(values(&tensor, txn.id()).await, [9.0; 8]);
        }
    }
});

// Exercise deep expressions on ordinary Tokio worker stacks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deep_computed_streams_release_transaction_leases() {
    const CHILD: &str = "TC_COLLECTION_EXPRESSION_DEPTH_CHILD";
    assert!(std::env::var_os("RUST_MIN_STACK").is_none());
    if std::env::var_os(CHILD).is_some() {
        tokio::spawn(async {
            for sparse in [false, true] {
                let (_, txn) = setup().await;
                let (tensor, _) = create(&txn, vec![2].into(), sparse, 2).await;
                tensor.write_value(&txn, &[1], (-2.).into()).await.unwrap();
                let mut expression = tensor.clone();
                for _ in 0..1024 {
                    expression = expression.abs(txn.id()).await.unwrap();
                }

                assert_eq!(
                    expression.read_value(txn.id(), &[1]).await.unwrap(),
                    2.0.into()
                );

                drop(expression.elements(txn.id()).await.unwrap());
                let mut stream = expression.elements(txn.id()).await.unwrap();
                drop(expression);
                assert!(tensor.write_value(&txn, &[1], 3.0.into()).await.is_err());
                assert_eq!(stream.try_next().await.unwrap(), Some(0.0.into()));
                assert_eq!(stream.try_next().await.unwrap(), Some(2.0.into()));
                assert!(stream.try_next().await.unwrap().is_none());
                // EOF releases the entire retained graph and its read permit,
                // even while the exhausted stream remains in scope.
                tensor.write_value(&txn, &[1], 3.0.into()).await.unwrap();
                assert!(stream.try_next().await.unwrap().is_none());
            }

            eprintln!("deep collection expressions completed both layouts");
        })
        .await
        .unwrap();

        return;
    }

    let log = std::env::temp_dir().join(format!(
        "tc-collection-expression-depth-{}.log",
        std::process::id()
    ));

    let output = std::fs::File::create(&log).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tensor::tests::deep_computed_streams_release_transaction_leases",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .stdout(output.try_clone().unwrap())
        .stderr(output)
        .spawn()
        .unwrap();

    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }

        if started.elapsed() > Duration::from_secs(60) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "deep collection expression timed out; log: {}",
                log.display()
            );
        }

        std::thread::sleep(Duration::from_millis(10));
    };

    let transcript = std::fs::read_to_string(&log).unwrap();
    assert!(
        status.success()
            && transcript.contains("completed both layouts")
            && transcript.contains("1 passed"),
        "{status}; log: {}\n{}",
        log.display(),
        transcript
    );

    std::fs::remove_file(log).unwrap();
}

test!(
    multi_owner_conflicts_release_previously_acquired_leases,
    async {
        let (_, txn) = setup().await;
        let (left, _) = create(&txn, vec![4].into(), false, 2).await;
        let (right, _) = create(&txn, vec![4].into(), false, 2).await;
        right
            .write_value(&txn.at(9), &[0], 1.0.into())
            .await
            .unwrap();

        assert!(left.add(id(10), &right).await.is_err());
        left.write_value(&txn, &[0], 2.0.into()).await.unwrap();
    }
);

test!(sparse_zero_versions_suppress_older_occupancy, async {
    let (_, txn) = setup().await;
    let (tensor, dir) = create(&txn, vec![2, 4].into(), true, 2).await;
    tensor.write_value(&txn, &[0, 0], 4.0.into()).await.unwrap();
    tensor.write_value(&txn, &[1, 3], 7.0.into()).await.unwrap();
    tensor.commit(txn.id()).await.unwrap();
    tensor
        .write_value(&txn.at(12), &[0, 0], 0.0.into())
        .await
        .unwrap();

    tensor
        .write_value(&txn.at(12), &[1, 3], 8.0.into())
        .await
        .unwrap();

    tensor.commit(id(12)).await.unwrap();
    assert_eq!(
        values(&tensor, id(11)).await,
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    );

    assert_eq!(
        values(&tensor, id(13)).await,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]
    );

    {
        let path = dir.read().await.path().to_owned();
        let cache =
            freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
        let reopened = Tensor::<Txn>::load(cache.load(path).unwrap(), tensor.schema().unwrap())
            .await
            .unwrap();

        assert_eq!(
            reopened.read_value(id(11), &[0, 0]).await.unwrap(),
            4.0.into()
        );
        assert_eq!(
            reopened.read_value(id(13), &[0, 0]).await.unwrap(),
            0.0.into()
        );
        assert_eq!(
            reopened.read_value(id(13), &[1, 3]).await.unwrap(),
            8.0.into()
        );
    }

    let Tensor::F64(typed) = &tensor else {
        unreachable!()
    };

    let owner = typed.base().unwrap();
    let snapshot = owner
        .try_snapshot(id(13), 0..owner.geometry().block_count())
        .unwrap();

    let keys: Vec<_> = snapshot
        .occupied_blocks(0..owner.geometry().block_count())
        .take(2)
        .try_collect()
        .await
        .unwrap();

    assert_eq!(keys.len(), 1);
    drop(snapshot);
    tensor.finalize(&id(12)).await.unwrap();
    tensor
        .write_value(&txn.at(14), &[1, 3], 0.0.into())
        .await
        .unwrap();

    tensor.commit(id(14)).await.unwrap();
    tensor.finalize(&id(14)).await.unwrap();
    let snapshot = owner
        .try_snapshot(id(15), 0..owner.geometry().block_count())
        .unwrap();

    assert!(
        snapshot
            .occupied_blocks(0..owner.geometry().block_count())
            .try_next()
            .await
            .unwrap()
            .is_none()
    );
});

test!(restart_loads_committed_versions_without_replay, async {
    let (_, txn) = setup().await;
    let (tensor, dir) = create(&txn, vec![4].into(), true, 2).await;
    let schema = tensor.schema().unwrap();
    tensor.write_value(&txn, &[0], 3.0.into()).await.unwrap();
    tensor.write_value(&txn, &[1], 7.0.into()).await.unwrap();
    tensor.commit(id(10)).await.unwrap();
    tensor.finalize(&id(10)).await.unwrap();
    tensor
        .write_value(&txn.at(11), &[1], 0.0.into())
        .await
        .unwrap();
    tensor
        .write_value(&txn.at(11), &[3], 9.0.into())
        .await
        .unwrap();

    tensor.commit(id(11)).await.unwrap();
    let path = dir.read().await.path().to_owned();
    let cache = freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));

    let reopened = Tensor::<Txn>::load(cache.load(path.clone()).unwrap(), schema.clone())
        .await
        .unwrap();

    reopened.commit(id(11)).await.unwrap();
    let expected = [3.0, 0.0, 0.0, 9.0];
    assert_eq!(values(&reopened, id(12)).await, expected);
    reopened.finalize(&id(11)).await.unwrap();
    drop(reopened);

    let cache = freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
    let compacted = Tensor::<Txn>::load(cache.load(path).unwrap(), schema)
        .await
        .unwrap();
    assert_eq!(values(&compacted, id(12)).await, expected);
});

test!(
    large_computed_sparse_projection_streams_logical_entries,
    async {
        let (_, txn) = setup().await;
        let width = fensor::MAX_BATCH_ELEMENTS as u64 + 1;
        let last = width - 1;
        let (tensor, _) = create(&txn, vec![width, width].into(), true, 64).await;
        tensor
            .write_value(&txn, &[last, last], 3.0.into())
            .await
            .unwrap();
        let diagonal = tensor.diag(txn.id()).await.unwrap();
        let view = diagonal.into_view(txn.clone()).await.unwrap();
        let encoded = destream_json::encode(view)
            .unwrap()
            .map_err(std::io::Error::other);
        let decoded: Tensor<Txn> = destream_json::try_decode(txn.clone(), encoded)
            .await
            .unwrap();
        assert_eq!(decoded.shape(), [width]);
        let entries: Vec<_> = decoded
            .sparse_elements(txn.id())
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(entries, [(vec![last], Number::from(3.0))]);
    }
);

test!(dense_restore_publishes_complete_reblocked_values, async {
    let (_, txn) = setup().await;
    let (source, _) = create(&txn, vec![2, 5].into(), false, 3).await;
    let (target, dir) = create(&txn, vec![2, 5].into(), false, 4).await;
    let schema = target.schema().unwrap();
    source.write_value(&txn, &[0, 3], 3.0.into()).await.unwrap();
    source
        .write_value(&txn, &[1, 4], (-0.0_f64).into())
        .await
        .unwrap();
    source.commit(txn.id()).await.unwrap();
    target.fill(&txn, 5.0.into()).await.unwrap();
    target.commit(txn.id()).await.unwrap();

    let restore = txn.at(11);
    target.restore(&restore, &source).await.unwrap();
    assert_eq!(values(&target, txn.id()).await, [5.0; 10]);
    target.commit(restore.id()).await.unwrap();
    target.finalize(&restore.id()).await.unwrap();

    let path = dir.read().await.path().to_owned();
    let cache = freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
    let reopened = Tensor::<Txn>::load(cache.load(path).unwrap(), schema)
        .await
        .unwrap();
    let restored = values(&reopened, id(12)).await;
    assert_eq!(
        restored,
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0]
    );
    assert_eq!(restored[9].to_bits(), (-0.0_f64).to_bits());
});

test!(copy_restore_and_hash_ignore_physical_layout, async {
    let (_, txn) = setup().await;
    let (left, _) = create(&txn, vec![2, 3].into(), true, 2).await;
    let (right, _) = create(&txn, vec![2, 3].into(), true, 4).await;
    left.write_value(&txn, &[1, 2], 7.0.into()).await.unwrap();
    right.restore(&txn, &left).await.unwrap();
    let left_collection = Collection::Tensor(left.clone());
    let right_collection = Collection::Tensor(right.clone());
    assert_eq!(
        left_collection.hash(txn.id()).await.unwrap(),
        right_collection.hash(txn.id()).await.unwrap()
    );

    let copy = crate::tensor::Tensor::copy_from(
        &txn,
        txn.subcontext_unique().context().await.unwrap(),
        &left,
    )
    .await
    .unwrap();

    left.fill(&txn, 0.0.into()).await.unwrap();
    assert_eq!(
        values(&copy, txn.id()).await,
        [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    );

    right.rollback(&txn.id()).await.unwrap();
    assert_eq!(values(&right, id(11)).await, [0.0; 6]);
});

test!(
    copy_retains_sparse_axis_hints_and_uses_schema_default,
    async {
        let (_, txn) = setup().await;
        let layout = Layout::Sparse { axis: Some(0) };
        let store = crate::persistence::storage::Store::create(
            txn.subcontext_unique().context().await.unwrap(),
        )
        .await
        .unwrap();
        let native = fensor::Tensor::<PersistentFile, f64>::from_sparse_elements(
            store.base().clone(),
            fensor::TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 3].into())
                .unwrap(),
            layout,
            3,
            futures::stream::iter([Ok::<_, fensor::Error>((vec![1, 2], 2.5))]),
        )
        .await
        .unwrap();

        let tensor = Tensor::<Txn>::F64(TensorAccess::from_base(
            super::storage::TensorBase::from_native(native, store)
                .await
                .unwrap(),
        ));

        let cases = [
            (tensor.clone(), layout),
            (tensor.round(txn.id()).await.unwrap(), layout),
            (
                tensor
                    .cast(txn.id(), NumberType::Float(FloatType::F32))
                    .await
                    .unwrap(),
                layout,
            ),
            (
                tensor.clone().transpose(Some(vec![1, 0].into())).unwrap(),
                Layout::Sparse { axis: None },
            ),
            (tensor.as_dense(txn.id()).await.unwrap(), Layout::Dense),
        ];

        for (source, expected_layout) in cases {
            let dir = txn.subcontext_unique().context().await.unwrap();
            let copy = crate::tensor::Tensor::copy_from(&txn, dir.clone(), &source)
                .await
                .unwrap();

            assert_eq!(copy.layout(), expected_layout);
            assert_eq!(
                values(&copy, txn.id()).await,
                values(&source, txn.id()).await
            );

            let schema = copy.schema().unwrap();
            drop(copy);
            let reopened = Tensor::<Txn>::load(dir, schema).await.unwrap();
            assert_eq!(
                values(&reopened, txn.id()).await,
                values(&source, txn.id()).await
            );
        }

        let (source, _) = create(&txn, vec![2, 4097].into(), true, 512).await;
        source
            .write_value(&txn, &[1, 4096], 7.0.into())
            .await
            .unwrap();

        let view = source
            .round(txn.id())
            .await
            .unwrap()
            .transpose(None)
            .unwrap();

        let copy = crate::tensor::Tensor::copy_from(
            &txn,
            txn.subcontext_unique().context().await.unwrap(),
            &view,
        )
        .await
        .unwrap();
        drop(view);
        source
            .write_value(&txn, &[1, 4096], 3.0.into())
            .await
            .unwrap();

        assert_eq!(
            copy.read_value(txn.id(), &[4096, 1]).await.unwrap(),
            7.0.into()
        );
    }
);

test!(typed_tensor_rejects_invalid_implicit_conversions, async {
    let (_, txn) = setup().await;
    let tensor = Tensor::dense_u8(&txn, vec![2], vec![1, 2]).await.unwrap();
    tensor.write_value(&txn, &[0], 7_u8.into()).await.unwrap();

    for invalid in [256_u16.into(), 0.5_f64.into()] {
        let write = tensor.write_value(&txn, &[1], invalid).await;
        let fill = tensor.fill(&txn, invalid).await;
        let add = tensor.add_scalar(txn.id(), invalid).await;
        for error in [write.unwrap_err(), fill.unwrap_err(), add.unwrap_err()] {
            assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
        }
    }

    assert_eq!(values(&tensor, txn.id()).await, [7., 2.]);

    let dtype = tc_value::number_type_path(&NumberType::UInt(UIntType::U8)).to_string();
    for contents in ["[0.5]", "[[[0],256]]"] {
        let payload = format!("[[{dtype:?},[1]],{contents}]");
        let input = futures::stream::iter([Ok::<_, std::io::Error>(payload.into_bytes().into())]);
        assert!(
            destream_json::try_decode::<_, _, Tensor<Txn>>(txn.clone(), input)
                .await
                .is_err()
        );
    }
});

test!(dense_and_sparse_codecs_preserve_semantic_schema, async {
    use tc_value::Value;

    let (_root, txn) = Txn::owned(id(10), 256 * 1024, 0);
    let dtype = NumberType::Float(FloatType::F64);
    let mut sparse_wire = None;
    for (axis, capacity) in [(None, 2), (Some(0), 2), (Some(1), 5)] {
        let schema = TensorSchema::new(dtype, vec![2, 3].into(), axis).unwrap();
        assert_eq!(
            TensorSchema::try_from(Value::from(schema.clone())).unwrap(),
            schema
        );
        let tensor = Tensor::<Txn>::create(
            txn.subcontext_unique().context().await.unwrap(),
            schema.clone(),
            capacity,
        )
        .await
        .unwrap();

        tensor.write_value(&txn, &[0, 1], 3.0.into()).await.unwrap();
        tensor.write_value(&txn, &[1, 2], 7.0.into()).await.unwrap();
        let expected = [0.0, 3.0, 0.0, 0.0, 0.0, 7.0];
        let hash = Collection::from(tensor.clone())
            .hash(txn.id())
            .await
            .unwrap();

        let view = tensor.clone().into_view(txn.clone()).await.unwrap();
        let json = destream_json::encode(view)
            .unwrap()
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .unwrap();
        let view = tensor.clone().into_view(txn.clone()).await.unwrap();
        let tbon = tbon::en::encode(view)
            .unwrap()
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .unwrap();

        if axis.is_some() {
            if let Some(previous) = &sparse_wire {
                assert_eq!(previous, &(json.clone(), tbon.clone(), hash));
            } else {
                sparse_wire = Some((json.clone(), tbon.clone(), hash));
            }
        }

        let json = futures::stream::iter([Ok::<_, std::io::Error>(json.into())]);
        let tbon = futures::stream::iter([Ok::<_, std::io::Error>(tbon.into())]);
        let from_json: Tensor<Txn> = destream_json::try_decode(txn.clone(), json).await.unwrap();
        let from_tbon: Tensor<Txn> = tbon::de::try_decode(txn.clone(), tbon).await.unwrap();
        let logical = schema.logical_value();
        assert_eq!(
            TensorSchema::try_from(logical).unwrap().shape(),
            from_json.shape()
        );
        for decoded in [from_json, from_tbon] {
            assert_eq!(
                decoded.layout(),
                if axis.is_some() {
                    Layout::Sparse { axis: None }
                } else {
                    Layout::Dense
                }
            );
            assert_eq!(values(&decoded, txn.id()).await, expected);
            assert_eq!(
                Collection::from(decoded.clone())
                    .hash(txn.id())
                    .await
                    .unwrap(),
                hash
            );
            tensor.fill(&txn, 0.0.into()).await.unwrap();
            tensor.restore(&txn, &decoded).await.unwrap();
            assert_eq!(values(&tensor, txn.id()).await, expected);
            assert_eq!(tensor.schema().unwrap(), schema);
        }
    }
});

test!(
    finalization_completes_without_cross_domain_deadlock,
    async {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![8].into(), true, 2).await;
        tensor.fill(&txn, 3.0.into()).await.unwrap();
        tensor.commit(txn.id()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            let cutoff = id(10);
            tensor.finalize(&cutoff).await.unwrap();
        })
        .await
        .unwrap();

        assert_eq!(values(&tensor, id(11)).await, [3.0; 8]);
    }
);

test!(
    large_sparse_reduction_uses_bounded_occupied_traversal,
    async {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![1_000_000_000_000].into(), true, 64).await;
        tensor.write_value(&txn, &[2], 3.0.into()).await.unwrap();
        tensor
            .write_value(&txn, &[999_999_999_999], 7.0.into())
            .await
            .unwrap();

        let sum = tokio::time::timeout(Duration::from_secs(5), tensor.sum(txn.id(), None, false))
            .await
            .unwrap()
            .unwrap();

        let TensorReduceResult::Scalar(sum) = sum else {
            panic!("scalar sum")
        };

        assert_eq!(sum, Number::from(10.0));
    }
);

test!(
    sparse_restoration_replaces_pending_in_destination_geometry,
    async {
        let (_root, txn) = setup().await;
        let last = 127;
        let shape = vec![last + 1, 11];
        let schema =
            TensorSchema::new(FloatType::F64.into(), shape.clone().into(), Some(0)).unwrap();
        let source = Tensor::create(txn.subcontext_unique().context().await.unwrap(), schema, 4)
            .await
            .unwrap();
        let (target, dir) = create(&txn, shape.into(), true, 4).await;
        let schema = target.schema().unwrap();
        let restore = txn.at(11);

        source.write_value(&txn, &[1, 4], 7.0.into()).await.unwrap();
        source.commit(txn.id()).await.unwrap();
        source
            .write_value(&restore, &[last, 10], 9.0.into())
            .await
            .unwrap();

        target.write_value(&txn, &[0, 0], 1.0.into()).await.unwrap();
        target.write_value(&txn, &[1, 4], 2.0.into()).await.unwrap();
        target.commit(txn.id()).await.unwrap();
        target
            .write_value(&restore, &[0, 0], 0.0.into())
            .await
            .unwrap();
        target
            .write_value(&restore, &[3, 6], 99.0.into())
            .await
            .unwrap();

        target.restore(&restore, &source).await.unwrap();

        let restored = [
            ([0, 0], 0.0),
            ([1, 4], 7.0),
            ([3, 6], 0.0),
            ([last, 10], 9.0),
        ];
        for (coord, expected) in restored {
            assert_eq!(
                target.read_value(restore.id(), &coord).await.unwrap(),
                expected.into()
            );
        }

        target.rollback(&restore.id()).await.unwrap();
        for (coord, expected) in [([0, 0], 1.0), ([1, 4], 2.0), ([last, 10], 0.0)] {
            assert_eq!(
                target.read_value(id(12), &coord).await.unwrap(),
                expected.into()
            );
        }

        source.commit(restore.id()).await.unwrap();
        let restore = txn.at(12);
        target.restore(&restore, &source).await.unwrap();
        target.commit(restore.id()).await.unwrap();
        target.finalize(&restore.id()).await.unwrap();

        let path = dir.read().await.path().to_owned();
        let cache =
            freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
        let reopened = Tensor::<Txn>::load(cache.load(path).unwrap(), schema)
            .await
            .unwrap();
        for (coord, expected) in restored {
            assert_eq!(
                reopened.read_value(id(13), &coord).await.unwrap(),
                expected.into()
            );
        }
    }
);

test!(sparse_restoration_masks_accepted_older_replay, async {
    let (_root, txn) = setup().await;
    let (source, _) = create(&txn, vec![4].into(), true, 4).await;
    let (target, dir) = create(&txn, vec![4].into(), true, 4).await;
    let schema = target.schema().unwrap();
    let restore = txn.at(12);
    target.restore(&restore, &source).await.unwrap();
    target.commit(restore.id()).await.unwrap();
    drop(target);

    let path = dir.read().await.path().to_owned();
    let cache = freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
    let target = Tensor::<Txn>::load(cache.load(path.clone()).unwrap(), schema.clone())
        .await
        .unwrap();
    let older = txn.at(11);
    assert_eq!(
        target
            .write_value(&older, &[3], 3.0.into())
            .await
            .unwrap_err()
            .code(),
        tc_error::ErrorKind::Conflict,
    );
    target
        .recover(older.id(), || target.write_value(&older, &[3], 3.0.into()))
        .await
        .unwrap();

    assert_eq!(
        target.read_value(older.id(), &[3]).await.unwrap(),
        3.0.into()
    );
    assert_eq!(
        target.read_value(restore.id(), &[3]).await.unwrap(),
        0.0.into()
    );
    target.finalize(&restore.id()).await.unwrap();
    assert_eq!(target.read_value(id(13), &[3]).await.unwrap(), 0.0.into());
    drop(target);

    let cache = freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
    let target = Tensor::<Txn>::load(cache.load(path).unwrap(), schema)
        .await
        .unwrap();
    assert_eq!(target.read_value(id(13), &[3]).await.unwrap(), 0.0.into());
});

test!(
    whole_fill_replaces_pending_and_preserves_zero_semantics_after_reopen,
    async {
        for (sparse, fill, expected) in [
            (false, 0.0f64, 0.0f64),
            (false, -0.0, -0.0),
            (true, -0.0, 0.0),
        ] {
            let (_root, txn) = setup().await;
            let (tensor, dir) = create(&txn, vec![4].into(), sparse, 2).await;
            let schema = tensor.schema().unwrap();
            tensor.fill(&txn, 1.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();

            let next = txn.at(11);
            tensor.write_value(&next, &[0], 7.0.into()).await.unwrap();
            tensor.fill(&next, fill.into()).await.unwrap();
            for value in values(&tensor, next.id()).await {
                assert_eq!(value.to_bits(), expected.to_bits());
            }

            tensor.rollback(&next.id()).await.unwrap();
            assert_eq!(values(&tensor, id(12)).await, [1.0; 4]);
            tensor.fill(&txn.at(12), fill.into()).await.unwrap();
            tensor.commit(id(12)).await.unwrap();
            tensor.finalize(&id(12)).await.unwrap();
            drop(tensor);

            let path = dir.read().await.path().to_owned();
            let cache =
                freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
            let reopened = Tensor::<Txn>::load(cache.load(path).unwrap(), schema)
                .await
                .unwrap();
            for value in values(&reopened, id(13)).await {
                assert_eq!(value.to_bits(), expected.to_bits());
            }
        }
    }
);

test!(failed_compaction_preserves_published_versions, async {
    for (sparse, whole, corrupt_newer) in [
        (false, false, false),
        (true, false, false),
        (true, true, false),
        (true, true, true),
    ] {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![4].into(), sparse, 2).await;
        tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
        tensor.write_value(&txn, &[2], 1.0.into()).await.unwrap();
        tensor.commit(txn.id()).await.unwrap();
        tensor.finalize(&txn.id()).await.unwrap();
        let Tensor::F64(value) = &tensor else {
            unreachable!()
        };
        let old = value.base().unwrap().snapshot(id(12), None).unwrap();
        let next = txn.at(11);
        if whole {
            if !corrupt_newer {
                super::storage::tests::corrupt_block(value.base().unwrap(), id(12), &[2]).await;
            }
            let fill = if corrupt_newer { 2.0 } else { 0.0 };
            tensor.fill(&next, fill.into()).await.unwrap();
        } else {
            tensor.write_value(&next, &[0], 3.0.into()).await.unwrap();
            tensor.write_value(&next, &[2], 7.0.into()).await.unwrap();
        }
        tensor.commit(next.id()).await.unwrap();
        if !whole || corrupt_newer {
            super::storage::tests::corrupt_block(value.base().unwrap(), id(12), &[2]).await;
        }
        assert!(
            tensor
                .finalize(&next.id())
                .await
                .unwrap_err()
                .requires_recovery()
        );
        assert_eq!(old.read_logical_block(0).await.unwrap()[0], 1.0);
        assert!(
            tensor
                .read_value(id(12), &[0])
                .await
                .unwrap_err()
                .requires_recovery()
        );
    }
});

test!(
    strict_reopen_requires_correct_payloads_and_metadata,
    async {
        let (_, txn) = setup().await;
        for corruption in 0..4 {
            let (tensor, dir) = create(&txn, vec![4].into(), corruption == 3, 2).await;
            let schema = tensor.schema().unwrap();
            let blocks = dir
                .read()
                .await
                .get_dir("base")
                .cloned()
                .unwrap()
                .read()
                .await
                .get_dir("blocks")
                .cloned()
                .unwrap();
            match corruption {
                0 => {
                    blocks.write().await.delete("metadata").await;
                }
                1 => {
                    blocks.write().await.delete("0").await;
                    blocks
                        .write()
                        .await
                        .create_file("0".into(), vec![0u64; 2])
                        .await
                        .unwrap();
                }
                2 => {
                    let block = blocks.read().await.get_file("0").cloned().unwrap();
                    *block.write::<Vec<f64>>().await.unwrap() = vec![0.0; 1];
                }
                _ => {
                    dir.read()
                        .await
                        .get_dir("base")
                        .cloned()
                        .unwrap()
                        .write()
                        .await
                        .delete("values")
                        .await;
                }
            }
            let error = Tensor::<Txn>::load(dir.clone(), schema).await.unwrap_err();
            assert!(error.requires_recovery());
            if corruption == 0 {
                assert!(!blocks.read().await.contains("metadata"));
            } else if corruption == 3 {
                assert!(
                    !dir.read()
                        .await
                        .get_dir("base")
                        .cloned()
                        .unwrap()
                        .read()
                        .await
                        .contains("values")
                );
            }
        }
    }
);

test!(all_real_dtypes_evaluate_stream_and_roundtrip, async {
    let (_, txn) = setup().await;
    use number_general::{IntType, UIntType};
    let dtypes = [
        NumberType::UInt(UIntType::U8),
        NumberType::UInt(UIntType::U16),
        NumberType::UInt(UIntType::U32),
        NumberType::UInt(UIntType::U64),
        NumberType::Int(IntType::I8),
        NumberType::Int(IntType::I16),
        NumberType::Int(IntType::I32),
        NumberType::Int(IntType::I64),
        NumberType::Float(FloatType::F32),
        NumberType::Float(FloatType::F64),
    ];

    for dtype in dtypes {
        for sparse in [false, true] {
            let schema = TensorSchema::new(dtype, vec![2, 2].into(), sparse.then_some(1)).unwrap();
            let tensor =
                Tensor::<Txn>::create(txn.subcontext_unique().context().await.unwrap(), schema, 2)
                    .await
                    .unwrap();

            tensor.write_value(&txn, &[0, 0], 2u8.into()).await.unwrap();
            tensor.write_value(&txn, &[1, 1], 3u8.into()).await.unwrap();
            let doubled = tensor.add(txn.id(), &tensor).await.unwrap();
            assert_eq!(values(&doubled, txn.id()).await, [4.0, 0.0, 0.0, 6.0]);
            drop(doubled);
            let encoded = destream_json::encode(tensor.into_view(txn.clone()).await.unwrap())
                .unwrap()
                .map_err(std::io::Error::other);

            let decoded: Tensor<Txn> = destream_json::try_decode(txn.clone(), encoded)
                .await
                .unwrap();

            assert_eq!(decoded.number_type(), dtype);
            assert_eq!(values(&decoded, txn.id()).await, [2.0, 0.0, 0.0, 3.0]);
        }
    }
    for (dtype, expected) in [
        (NumberType::Float(FloatType::F64), vec![1.5, 2.5, 3.5, 4.5]),
        (NumberType::UInt(UIntType::U64), vec![1.0, 2.0, 3.0, 4.0]),
    ] {
        let tensor = if dtype == NumberType::Float(FloatType::F64) {
            Tensor::dense_f64(&txn, vec![2, 2], expected.clone())
                .await
                .unwrap()
        } else {
            Tensor::dense_u64(&txn, vec![2, 2], vec![1, 2, 3, 4])
                .await
                .unwrap()
        };

        let encoded = destream_json::encode(tensor.into_view(txn.clone()).await.unwrap())
            .unwrap()
            .map_err(std::io::Error::other);

        let decoded: Tensor<Txn> = destream_json::try_decode(txn.clone(), encoded)
            .await
            .unwrap();

        assert_eq!(decoded.number_type(), dtype);
        assert_eq!(decoded.shape(), &[2, 2]);
        assert_eq!(values(&decoded, txn.id()).await, expected);
    }
});

#[cfg(feature = "complex")]
test!(complex_fourier_and_codec_use_native_semantics, async {
    use fensor::complex::Complex64;
    let (_, txn) = setup().await;
    let tensor = Tensor::dense_c64(&txn, vec![2, 2], vec![Complex64::new(1.0, 2.0); 4])
        .await
        .unwrap();

    let roundtrip = tensor
        .fft2(txn.id())
        .await
        .unwrap()
        .ifft2(txn.id())
        .await
        .unwrap();

    let value = roundtrip.read_value(txn.id(), &[0, 0]).await.unwrap();
    let value: Complex64 = value.cast_into();
    assert!((value - Complex64::new(4.0, 8.0)).norm() < 1e-12);
    let encoded = destream_json::encode(tensor.into_view(txn.clone()).await.unwrap())
        .unwrap()
        .map_err(std::io::Error::other);

    let decoded: Tensor<Txn> = destream_json::try_decode(txn.clone(), encoded)
        .await
        .unwrap();

    let value: Complex64 = decoded
        .read_value(txn.id(), &[1, 1])
        .await
        .unwrap()
        .cast_into();

    assert_eq!(value, Complex64::new(1.0, 2.0));
});

#[cfg(feature = "complex")]
test!(
    complex32_tbon_preserves_components_in_dense_and_sparse_tensors,
    async {
        use fensor::complex::Complex32;

        let (_root, txn) = setup().await;
        for sparse in [false, true] {
            let schema = TensorSchema::new(
                number_general::ComplexType::C32.into(),
                vec![2].into(),
                sparse.then_some(0),
            )
            .unwrap();
            let tensor =
                Tensor::<Txn>::create(txn.subcontext_unique().context().await.unwrap(), schema, 1)
                    .await
                    .unwrap();
            tensor
                .write_value(&txn, &[0], Complex32::new(1.0, 2.0).into())
                .await
                .unwrap();
            tensor
                .write_value(&txn, &[1], Complex32::new(0.0, -0.0).into())
                .await
                .unwrap();
            let encoded = tbon::en::encode(tensor.into_view(txn.clone()).await.unwrap()).unwrap();
            let decoded: Tensor<Txn> = tbon::de::try_decode(txn.clone(), encoded).await.unwrap();
            let first: Complex32 = decoded
                .read_value(txn.id(), &[0])
                .await
                .unwrap()
                .cast_into();
            let second: Complex32 = decoded
                .read_value(txn.id(), &[1])
                .await
                .unwrap()
                .cast_into();
            assert_eq!(first, Complex32::new(1.0, 2.0));
            assert_eq!(second, Complex32::new(0.0, 0.0));
            assert_eq!(second.im.is_sign_negative(), !sparse);
        }
    }
);

test!(oversized_gathers_are_rejected_at_construction, async {
    let (_, txn) = setup().await;
    let tensor = Tensor::dense_u8(&txn, vec![1; u64::BITS as usize], vec![1])
        .await
        .unwrap();
    let gather = (0..u64::BITS).map(|_| AxisRange::Of(vec![0, 0])).collect();
    let error = tensor.slice(gather).unwrap_err();
    assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
});

test!(computed_self_assignment_consumes_its_input_leases, async {
    for sparse in [false, true] {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![2].into(), sparse, 1).await;
        tensor.fill(&txn, 2.0.into()).await.unwrap();
        tensor.commit(txn.id()).await.unwrap();

        let txn = txn.at(11);
        tensor.write_value(&txn, &[0], 3.0.into()).await.unwrap();
        let expression = tensor
            .as_dense(txn.id())
            .await
            .unwrap()
            .sub_scalar(txn.id(), 2.0.into())
            .await
            .unwrap();
        tensor.write_tensor(&txn, expression).await.unwrap();
        assert_eq!(values(&tensor, txn.id()).await, [1.0, 0.0]);

        let expression = tensor.mul_scalar(txn.id(), 2.0.into()).await.unwrap();
        let retained = expression.clone();
        assert!(tensor.write_tensor(&txn, expression).await.is_err());
        assert_eq!(values(&retained, txn.id()).await, [2.0, 0.0]);
        drop(retained);
        assert_eq!(values(&tensor, txn.id()).await, [1.0, 0.0]);
    }
});

test!(
    stream_errors_release_permits_without_requiring_drop,
    async {
        let (_, txn) = setup().await;
        let (tensor, dir) = create(&txn, vec![4].into(), false, 2).await;
        let mut values = tensor.elements(txn.id()).await.unwrap();
        let blocks = dir
            .read()
            .await
            .get_dir("base")
            .cloned()
            .unwrap()
            .read()
            .await
            .get_dir("blocks")
            .cloned()
            .unwrap();
        blocks.write().await.delete("0").await;
        assert!(values.try_next().await.is_err());
        tensor.write_value(&txn, &[2], 3.0.into()).await.unwrap();
        assert!(values.try_next().await.unwrap().is_none());
    }
);

test!(
    disjoint_selection_publication_merges_pending_blocks,
    async {
        for sparse in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![8].into(), sparse, 2).await;
            let left: Range = vec![AxisRange::In(0, 2, 1)].into();
            let right: Range = vec![AxisRange::In(6, 8, 1)].into();
            assert!(
                tensor
                    .clone()
                    .slice(left.clone())
                    .unwrap()
                    .fill(&txn, 3.0.into())
                    .await
                    .is_err()
            );

            let (l, r) = tokio::join!(
                tensor.fill_range(&txn, left.clone(), 3.0.into()),
                tensor.fill_range(&txn, right.clone(), 7.0.into()),
            );

            l.unwrap();
            r.unwrap();
            assert_eq!(
                values(&tensor, txn.id()).await,
                [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 7.0, 7.0]
            );

            tensor.rollback(&txn.id()).await.unwrap();
            tensor
                .fill_range(&txn.at(11), left.clone(), 2.0.into())
                .await
                .unwrap();

            tensor
                .fill_range(&txn.at(12), right, 8.0.into())
                .await
                .unwrap();

            assert!(
                tensor
                    .fill_range(&txn.at(12), left, 4.0.into())
                    .await
                    .is_err()
            );

            tensor.commit(id(12)).await.unwrap();
            tensor.commit(id(11)).await.unwrap();
            assert_eq!(
                values(&tensor, id(13)).await,
                [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 8.0, 8.0]
            );
        }
    }
);

test!(native_sparse_statistics_include_implicit_zeros, async {
    let (_, txn) = setup().await;
    let (tensor, _) = create(&txn, vec![2, 3].into(), true, 2).await;
    {
        let Tensor::F64(value) = &tensor else {
            unreachable!()
        };

        fensor::TensorWrite::write_value(&value.base().unwrap().canonical().clone(), &[0, 0], 2.0)
            .await
            .unwrap();
    }
    {
        let Tensor::F64(value) = &tensor else {
            unreachable!()
        };

        fensor::TensorWrite::write_value(&value.base().unwrap().canonical().clone(), &[0, 2], 4.0)
            .await
            .unwrap();
    }
    let error = tensor.sub_scalar(txn.id(), 2.0.into()).await.unwrap_err();
    assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
    assert!(error.to_string().contains("dense"));
    let dense = tensor.as_dense(txn.id()).await.unwrap();
    let shifted = dense.sub_scalar(txn.id(), 2.0.into()).await.unwrap();
    for (result, expected) in [
        (shifted.mean(txn.id(), None, false).await.unwrap(), -1.0),
        (
            shifted.std(txn.id(), None, false).await.unwrap(),
            (7.0_f64 / 3.0).sqrt(),
        ),
        (
            shifted.norm(txn.id(), None, false).await.unwrap(),
            20.0_f64.sqrt(),
        ),
    ] {
        let TensorReduceResult::Scalar(value) = result else {
            panic!("scalar reduction")
        };

        let actual: f64 = safecast::CastInto::cast_into(value);
        assert!((actual - expected).abs() < 1e-12);
    }
    let TensorReduceResult::Tensor(means) =
        shifted.mean(txn.id(), Some(vec![1]), true).await.unwrap()
    else {
        panic!("axis reduction")
    };

    let output = values(&means, txn.id()).await;
    assert_eq!(output, [0.0, -2.0]);
    let TensorReduceResult::Scalar(mean) = tensor.mean(txn.id(), None, false).await.unwrap() else {
        panic!("mean");
    };

    assert_eq!(mean, 1.0.into());
});

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn typed_literal_declarations_preserve_dtype_and_validation() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let (_root, limited) = Txn::owned(id(10), 256 * 1024, 8);
        let exact = Tensor::dense_u64(&limited, vec![1], vec![1]).await.unwrap();
        assert_eq!(
            exact.read_value(limited.id(), &[0]).await.unwrap(),
            Number::from(1u64)
        );

        let error = Tensor::dense_u64(&limited, vec![2], vec![1, 2])
            .await
            .unwrap_err();

        assert_eq!(
            error.message(),
            "Tensor buffer exceeds the delegated allocation limit"
        );

        macro_rules! check {
            ($method:ident, $ty:ty, $one:expr) => {{
                let one: $ty = $one;
                let tensor = Tensor::$method(&txn, vec![1], vec![one]).await.unwrap();
                assert_eq!(
                    tensor.number_type(),
                    <$ty as number_general::DType>::dtype()
                );
                assert_eq!(
                    tensor.read_value(txn.id(), &[0]).await.unwrap(),
                    Number::from(one)
                );
                assert!(Tensor::$method(&txn, vec![2], vec![one]).await.is_err());
            }};
        }

        check!(dense_u8, u8, 1);
        check!(dense_u16, u16, 1);
        check!(dense_u32, u32, 1);
        check!(dense_u64, u64, 1);
        check!(dense_i8, i8, 1);
        check!(dense_i16, i16, 1);
        check!(dense_i32, i32, 1);
        check!(dense_i64, i64, 1);
        check!(dense_f32, f32, 1.);
        check!(dense_f64, f64, 1.);
        #[cfg(feature = "complex")]
        {
            check!(
                dense_c32,
                fensor::complex::Complex32,
                fensor::complex::Complex32::new(1., 2.)
            );

            check!(
                dense_c64,
                fensor::complex::Complex64,
                fensor::complex::Complex64::new(1., 2.)
            );
        }
    })
    .await;
}

test!(streamed_schema_rejects_invalid_fields, async {
    use tc_value::Value;

    let dtype = tc_value::number_type_path(&NumberType::Float(FloatType::F64)).to_string();
    for extra in [Value::from(0u64), Value::from("sparse"), Value::None] {
        let schema = Value::Tuple(vec![
            Value::String(dtype.clone()),
            Value::Tuple(vec![Value::from(2u64), Value::from(3u64)]),
            extra,
        ]);

        let json = destream_json::encode(&schema)
            .unwrap()
            .map_err(std::io::Error::other);

        assert!(
            destream_json::try_decode::<_, _, TensorSchema>((), json)
                .await
                .is_err()
        );

        assert!(
            tbon::de::try_decode::<_, _, TensorSchema>((), tbon::en::encode(&schema).unwrap())
                .await
                .is_err()
        );
    }
    for rank in [MAX_TENSOR_METADATA_RANK, MAX_TENSOR_METADATA_RANK + 1] {
        let schema = (dtype.clone(), vec![1u64; rank]);
        let json = destream_json::try_decode::<_, _, TensorSchema>(
            (),
            destream_json::encode(&schema)
                .unwrap()
                .map_err(std::io::Error::other),
        )
        .await;

        let tbon =
            tbon::de::try_decode::<_, _, TensorSchema>((), tbon::en::encode(&schema).unwrap())
                .await;

        assert_eq!(json.is_ok(), rank == MAX_TENSOR_METADATA_RANK);
        assert_eq!(tbon.is_ok(), rank == MAX_TENSOR_METADATA_RANK);
    }
    // Decode schema boundaries without allocating collection storage. Error
    // precedence still validates dtype before shape and consumes extra fields.
    for (name, fields, expected) in [
        (
            "empty",
            "".to_string(),
            "expected sequence element is missing",
        ),
        (
            "missing shape",
            format!("{dtype:?}"),
            "expected sequence element is missing",
        ),
        (
            "invalid dtype first",
            "\"invalid\",false".to_string(),
            "Tensor dtype",
        ),
        (
            "extra schema field",
            format!("{dtype:?},[1],\"sparse\",0"),
            "unexpected Tensor schema field",
        ),
    ] {
        let input = format!("[{fields}]");
        let encoded = futures::stream::iter([Ok::<_, std::io::Error>(input.into_bytes().into())]);
        let error = destream_json::try_decode::<_, _, TensorSchema>((), encoded)
            .await
            .unwrap_err();

        assert!(error.to_string().contains(expected), "{name}: {error}");
    }
});

test!(streamed_codec_rejects_invalid_payloads, async {
    let (_root, txn) = setup().await;
    let dtype = tc_value::number_type_path(&NumberType::Float(FloatType::F64)).to_string();
    let entries = [
        ("[]", "expected sequence element is missing"),
        ("[[0]]", "expected sequence element is missing"),
        ("[[],1]", "sparse coordinate rank mismatch"),
        ("[[0,0],1]", "too many Tensor dimensions"),
        ("[[0],1,0]", "unexpected Tensor pair field"),
    ]
    .into_iter()
    .map(|(entry, expected)| (format!("[[{dtype:?},[2]],[{entry}]]"), Some(expected)));
    let payloads = [
        format!("[[{dtype:?},[2],0],[[0,1]]]"),
        format!("[[{dtype:?},[2]],[[[0],false]]]"),
        format!("[[{dtype:?},[2]],[[[0],0]]]"),
        format!("[[{dtype:?},[2]],[[[2],1]]]"),
        format!("[[{dtype:?},[2]],[[[1],1],[[0],1]]]"),
        format!("[[{dtype:?},[2]],[[[0],1],[[0],2]]]"),
        format!("[[{dtype:?},[2]],{{\"entries\":[]}}]"),
        format!("[[{dtype:?},false],[]]"),
        format!("[[{dtype:?},[2]]"),
        format!("[[{dtype:?},[2]],[1]]"),
        format!("[[{dtype:?},[2]],[1,2,3]]"),
        format!("[[{dtype:?},[2]],[1,\"bad\"]]"),
        format!("[[{dtype:?},[2]],[1,2],3]"),
    ]
    .into_iter()
    .map(|input| (input, None));

    for (input, expected) in entries.chain(payloads) {
        let encoded = futures::stream::iter([Ok::<_, std::io::Error>(input.into_bytes().into())]);
        let error = destream_json::try_decode::<_, _, Tensor<Txn>>(txn.clone(), encoded)
            .await
            .unwrap_err();
        if let Some(expected) = expected {
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
});

test!(dense_and_sparse_stream_errors_drop_source_leases, async {
    for sparse in [false, true] {
        let (_, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![2, 128].into(), sparse, 128).await;
        tensor.fill(&txn, 1.0.into()).await.unwrap();
        tensor.commit(txn.id()).await.unwrap();
        tensor.finalize(&txn.id()).await.unwrap();
        let next = txn.at(11);
        // Unpolled and partially consumed streams must each own their leases.
        let mut stream = if sparse {
            tensor
                .sparse_elements(next.id())
                .await
                .unwrap()
                .map_ok(|(_, n)| n)
                .boxed()
        } else {
            tensor.elements(next.id()).await.unwrap()
        };

        assert!(
            tensor
                .write_value(&next, &[0, 0], 2.0.into())
                .await
                .is_err()
        );

        assert!(stream.try_next().await.unwrap().is_some());
        drop(stream);
        tensor
            .write_value(&next, &[0, 0], 2.0.into())
            .await
            .unwrap();

        tensor.rollback(&next.id()).await.unwrap();
        {
            let Tensor::F64(value) = &tensor else {
                unreachable!()
            };
            super::storage::tests::corrupt_block(value.base().unwrap(), txn.at(12).id(), &[0, 0])
                .await;
            let mut stream = if sparse {
                tensor
                    .sparse_elements(txn.at(12).id())
                    .await
                    .unwrap()
                    .map_ok(|(_, n)| n)
                    .boxed()
            } else {
                tensor.elements(txn.at(12).id()).await.unwrap()
            };

            assert!(stream.try_next().await.is_err());
            assert!(stream.try_next().await.unwrap().is_none());
            // Reads no longer reserve the block after failure, even while the
            // exhausted stream remains in scope. A write detects corruption,
            // rather than a leaked reservation conflict.
            let error = tensor
                .write_value(&txn.at(12), &[0, 0], 3.0.into())
                .await
                .unwrap_err();

            assert!(error.requires_recovery());
        }
    }
});

test!(tensor_facade_read_write_value_roundtrip, async {
    let (_root, txn) = setup().await;
    let tensor = Tensor::dense_f32(&txn, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .await
        .expect("tensor");

    let value = tensor
        .read_value(txn.id(), &[1, 0])
        .await
        .expect("read value");

    let value: f64 = value.cast_into();
    assert_eq!(value, 3.0);

    tensor
        .write_value(&txn, &[0, 1], number_general::Number::from(9.5_f64))
        .await
        .expect("write value");

    assert_eq!(values(&tensor, txn.id()).await, vec![1.0, 9.5, 3.0, 4.0]);
});

test!(tensor_facade_cast_roundtrip, async {
    let (_root, txn) = setup().await;
    let tensor = Tensor::dense_f32(&txn, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .await
        .expect("tensor");

    let as_f64 = tensor
        .clone()
        .cast(txn.id(), NumberType::Float(FloatType::F64))
        .await
        .expect("cast f64");

    assert_eq!(as_f64.number_type(), NumberType::Float(FloatType::F64));
    assert_eq!(values(&as_f64, txn.id()).await, vec![1.0, 2.0, 3.0, 4.0]);

    let as_u64 = tensor
        .cast(txn.id(), NumberType::UInt(UIntType::U64))
        .await
        .expect("cast u64");

    assert_eq!(as_u64.number_type(), NumberType::UInt(UIntType::U64));
    assert_eq!(values(&as_u64, txn.id()).await, vec![1.0, 2.0, 3.0, 4.0]);
});

test!(tensor_facade_reduce_and_reduce_axes, async {
    let (_root, txn) = setup().await;
    let tensor = Tensor::dense_f32(&txn, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .await
        .expect("tensor");

    let TensorReduceResult::Scalar(sum) = tensor.sum(txn.id(), None, false).await.unwrap() else {
        panic!("scalar sum");
    };

    let sum: f64 = sum.cast_into();
    assert_eq!(sum, 10.0);

    let reduced = tensor
        .sum(txn.id(), Some(vec![1]), false)
        .await
        .expect("reduce axis 1");

    let TensorReduceResult::Tensor(reduced) = reduced else {
        panic!("expected tensor reduction output");
    };

    assert_eq!(reduced.shape(), &[2]);
    assert_eq!(values(&reduced, txn.id()).await, vec![3.0, 7.0]);
});

test!(tensor_facade_broadcast_roundtrip, async {
    let (_root, txn) = setup().await;
    let tensor = Tensor::dense_f32(&txn, vec![2, 1], vec![1.0, 2.0])
        .await
        .expect("tensor");

    let broadcast = tensor.broadcast(vec![2, 3].into()).expect("broadcast");

    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_eq!(
        values(&broadcast, txn.id()).await,
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
});

test!(tensor_facade_matmul_roundtrip, async {
    let (_root, txn) = setup().await;
    let left = Tensor::dense_f32(&txn, vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .await
        .expect("left");

    let right = Tensor::dense_f32(&txn, vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .await
        .expect("right");

    let product = left.matmul(txn.id(), &right).await.expect("matmul");
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(
        values(&product, txn.id()).await,
        vec![58.0, 64.0, 139.0, 154.0]
    );
});

test!(tensor_facade_slice_roundtrip, async {
    let (_root, txn) = setup().await;
    let tensor = Tensor::dense_u64(&txn, vec![2, 3], vec![1, 2, 3, 4, 5, 6])
        .await
        .expect("tensor");

    let range: Range = vec![AxisRange::In(0, 2, 1), AxisRange::In(1, 3, 1)]
        .into_iter()
        .collect();

    let sliced = tensor.slice(range).expect("slice");
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(values(&sliced, txn.id()).await, vec![2.0, 3.0, 5.0, 6.0]);
});

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn tensor_metadata_boundaries_reject_unrepresentable_results() {
    Box::pin(async {
        use futures::FutureExt;
        use tc_value::Value;

        let (_, txn) = setup().await;
        let context = txn.context().await.unwrap();
        let held = context.write().await;
        for shape in [vec![], vec![0], vec![1; MAX_TENSOR_METADATA_RANK + 1]] {
            assert!(
                Tensor::dense_f64(&txn, shape, vec![1.])
                    .now_or_never()
                    .expect("invalid metadata rejects before storage admission")
                    .is_err()
            );
        }

        drop(held);
        assert!(context.read().await.is_empty());
        let at_limit = Tensor::dense_f64(&txn, vec![1; MAX_TENSOR_METADATA_RANK], vec![2.])
            .await
            .unwrap();

        assert_eq!(
            at_limit.schema().unwrap().shape().len(),
            MAX_TENSOR_METADATA_RANK
        );

        assert!(
            at_limit
                .clone()
                .reshape(vec![1; MAX_TENSOR_METADATA_RANK + 1].into())
                .is_err()
        );

        assert!(
            at_limit
                .clone()
                .broadcast(vec![1; MAX_TENSOR_METADATA_RANK + 1].into())
                .is_err()
        );

        assert!(at_limit.clone().unsqueeze(vec![0].into()).is_err());
        let schema = at_limit.schema().unwrap();
        let wire_schema = Value::from(schema.clone());
        let json = destream_json::encode(&wire_schema)
            .unwrap()
            .map_err(std::io::Error::other);

        let from_json: TensorSchema = destream_json::try_decode((), json).await.unwrap();
        let from_tbon: TensorSchema =
            tbon::de::try_decode((), tbon::en::encode(&wire_schema).unwrap())
                .await
                .unwrap();

        assert_eq!(from_json, schema);
        assert_eq!(from_tbon, schema);

        for sparse in [false, true] {
            let (tensor, _) = create(&txn, vec![2, 2].into(), sparse, 2).await;
            tensor.write_value(&txn, &[1, 1], 3.0.into()).await.unwrap();
            let computed = tensor.abs(txn.id()).await.unwrap();
            let scalar: Range = vec![AxisRange::At(0), AxisRange::At(1)].into();
            let empty: Range = vec![AxisRange::In(0, 0, 1), AxisRange::In(0, 2, 1)].into();
            for source in [tensor.clone(), computed.clone()] {
                assert!(source.clone().slice(scalar.clone()).is_err());
                assert!(source.clone().slice(empty.clone()).is_err());
                let accepted = source.transpose(None).unwrap().flip(0).unwrap();
                assert_eq!(accepted.schema().unwrap().shape(), &[2, 2]);
                assert_eq!(
                    accepted.read_value(txn.id(), &[0, 1]).await.unwrap(),
                    3.0.into()
                );
            }

            drop(computed);
            tensor.write_value(&txn, &[0, 0], 4.0.into()).await.unwrap();
            assert_eq!(
                tensor.read_value(txn.id(), &[1, 1]).await.unwrap(),
                3.0.into()
            );

            let TensorReduceResult::Scalar(sum) = tensor.sum(txn.id(), None, false).await.unwrap()
            else {
                panic!("full reduction remains scalar")
            };

            assert_eq!(sum, 7.0.into());
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restoration_normalizes_scalar_sparse_layouts() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let store = crate::persistence::storage::Store::create(
            txn.subcontext_unique().context().await.unwrap(),
        )
        .await
        .unwrap();

        let native = fensor::Tensor::<PersistentFile, f64>::from_sparse_elements(
            store.base().clone(),
            fensor::TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 3].into())
                .unwrap(),
            Layout::Sparse { axis: Some(1) },
            4,
            futures::stream::iter([Ok::<_, fensor::Error>((vec![1, 2], 7.0))]),
        )
        .await
        .unwrap();

        let source = Tensor::<Txn>::F64(TensorAccess::from_base(
            super::storage::TensorBase::from_native(native, store)
                .await
                .unwrap(),
        ));

        let (target, _) = create(&txn, vec![2, 3].into(), true, 2).await;
        assert_eq!(source.schema().unwrap(), target.schema().unwrap());
        target.restore(&txn, &source).await.unwrap();
        assert_eq!(
            values(&target, txn.id()).await,
            vec![0., 0., 0., 0., 0., 7.]
        );

        Collection::from(target.clone())
            .restore(&txn, &Collection::from(source))
            .await
            .unwrap();

        assert_eq!(
            target.read_value(txn.id(), &[1, 2]).await.unwrap(),
            7.0.into()
        );

        target.rollback(&txn.id()).await.unwrap();
    })
    .await;
}
