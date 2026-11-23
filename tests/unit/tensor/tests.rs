use std::time::Duration;

use fensor::TensorSource;
use futures::TryStreamExt;
use number_general::{FloatType, Number, UIntType};
use safecast::CastInto;
pub(super) use support::{Txn, create, setup};
use support::{id, values};

use super::*;
use crate::test::tensor as support;
use crate::{Collection, PersistentFile};
use tc_ir::{IntoView, Transact, Transaction};

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
        ];
        for view in views {
            let view = view.clone();
            assert!(!view.is_persistent());
            assert!(dispatch!(&view.data, value => value.owner().is_err()));
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
            assert!(view.restore_from(&txn, &tensor).await.is_err());
            assert!(view.commit(txn.id()).await.is_err());
            assert!(view.rollback(&txn.id()).await.is_err());
            assert!(view.finalize(&txn.id()).await.is_err());
            assert!(view.sync().await.is_err());
            assert!(view.sync_all().await.is_err());
            assert_eq!(values(&view, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
        }
        assert_eq!(values(&tensor, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
        // Explicit gather selections remain read-only even when their indices
        // are unique. Repeated indices must not acquire last-writer semantics.
        for indices in [vec![0, 1], vec![1, 0], vec![0, 0]] {
            let selection: Range = vec![AxisRange::Of(indices), AxisRange::In(0, 2, 1)].into();
            assert!(
                tensor
                    .fill_range(&txn, selection.clone(), 9.0.into())
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("Tensor geometry is read-only")
            );
            assert!(
                tensor
                    .write_tensor_range(&txn, selection, tensor.clone())
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("Tensor assignment requires writable geometry and matching shapes")
            );
            assert_eq!(values(&tensor, txn.id()).await, [0.0, 3.0, 0.0, 0.0]);
        }
        let copied = tensor
            .clone()
            .into_read_only()
            .materialize(&txn, txn.subcontext_unique().context().await.unwrap())
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
    let (tensor, _) = create(&txn, vec![2, 4].into(), true, 2).await;
    tensor.write_value(&txn, &[0, 0], 4.0.into()).await.unwrap();
    tensor.write_value(&txn, &[1, 3], 7.0.into()).await.unwrap();
    tensor.commit(txn.id()).await.unwrap();
    tensor
        .write_value(&txn.at(12), &[0, 0], 0.0.into())
        .await
        .unwrap();
    tensor.commit(id(12)).await.unwrap();
    assert_eq!(
        values(&tensor, id(11)).await,
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    );
    assert_eq!(
        values(&tensor, id(13)).await,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    );
    let Data::F64(typed) = &tensor.data else {
        unreachable!()
    };
    let owner = typed.owner().unwrap();
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

test!(
    restart_loads_only_canonical_and_caller_replay_restores_commits,
    async {
        let (_, txn) = setup().await;
        let (tensor, dir) = create(&txn, vec![4].into(), true, 2).await;
        let schema = tensor.schema().unwrap();
        tensor.write_value(&txn, &[0], 3.0.into()).await.unwrap();
        tensor.commit(id(10)).await.unwrap();
        tensor.finalize(&id(10)).await.unwrap();
        tensor
            .write_value(&txn.at(11), &[3], 9.0.into())
            .await
            .unwrap();
        tensor.commit(id(11)).await.unwrap();
        tensor.sync_all().await.unwrap();
        let path = dir.read().await.path().to_owned();
        let cache =
            freqfs::Cache::<PersistentFile>::new(128 * 1024, None, 0, Duration::from_secs(3));
        let reopened = Tensor::<Txn>::load(cache.load(path).unwrap(), schema)
            .await
            .unwrap();
        assert_eq!(values(&reopened, id(10)).await, [3.0, 0.0, 0.0, 0.0]);
        reopened
            .write_value(&txn.at(11), &[3], 9.0.into())
            .await
            .unwrap();
        reopened.commit(id(11)).await.unwrap();
        assert_eq!(values(&reopened, id(12)).await, [3.0, 0.0, 0.0, 9.0]);
    }
);

test!(copy_restore_and_hash_ignore_physical_layout, async {
    let (_, txn) = setup().await;
    let (left, _) = create(&txn, vec![2, 3].into(), true, 2).await;
    let (right, _) = create(&txn, vec![2, 3].into(), true, 4).await;
    left.write_value(&txn, &[1, 2], 7.0.into()).await.unwrap();
    right.restore_from(&txn, &left).await.unwrap();
    let left_collection = Collection::Tensor(left.clone());
    let right_collection = Collection::Tensor(right.clone());
    assert_eq!(
        left_collection.hash(txn.id()).await.unwrap(),
        right_collection.hash(txn.id()).await.unwrap()
    );
    let copy = left
        .materialize(&txn, txn.subcontext_unique().context().await.unwrap())
        .await
        .unwrap();
    let full_copy = left
        .copy_values(&txn, txn.subcontext_unique().context().await.unwrap())
        .await
        .unwrap();
    left.fill(&txn, 0.0.into()).await.unwrap();
    assert_eq!(
        values(&copy, txn.id()).await,
        [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    );
    assert_eq!(
        values(&full_copy, txn.id()).await,
        values(&copy, txn.id()).await
    );
    right.rollback(&txn.id()).await.unwrap();
    assert_eq!(values(&right, id(11)).await, [0.0; 6]);
});

test!(
    copy_retains_sparse_axis_hints_and_uses_schema_default,
    async {
        let (_, txn) = setup().await;
        let layout = Layout::Sparse { axis: Some(0) };
        let native = fensor::Tensor::<PersistentFile, f64>::from_sparse_elements(
            txn.subcontext_unique().context().await.unwrap(),
            fensor::TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 3].into())
                .unwrap(),
            layout,
            3,
            futures::stream::iter([Ok::<_, fensor::Error>((vec![1, 2], 2.5))]),
        )
        .await
        .unwrap();
        let tensor = Tensor::<Txn> {
            data: Data::F64(Typed::stored(native)),
        };
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
            let copy = source
                .materialize(&txn, txn.subcontext_unique().context().await.unwrap())
                .await
                .unwrap();
            assert_eq!(copy.layout(), expected_layout);
            assert_eq!(
                values(&copy, txn.id()).await,
                values(&source, txn.id()).await
            );
            let dir = txn.subcontext_unique().context().await.unwrap();
            let full_copy = source.copy_values(&txn, dir.clone()).await.unwrap();
            assert_eq!(full_copy.layout(), expected_layout);
            assert_eq!(
                values(&full_copy, txn.id()).await,
                values(&source, txn.id()).await
            );
            let schema = full_copy.schema().unwrap();
            drop(full_copy);
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
        let unsupported = view
            .materialize(&txn, txn.subcontext_unique().context().await.unwrap())
            .await;
        let error = unsupported.err().unwrap();
        assert_eq!(error.code(), tc_error::ErrorKind::BadRequest);
        assert!(error.to_string().contains("copy"));
        let full_copy = view
            .copy_values(&txn, txn.subcontext_unique().context().await.unwrap())
            .await
            .unwrap();
        assert_eq!(
            values(&full_copy, txn.id()).await,
            values(&view, txn.id()).await
        );
        drop(view);
        source
            .write_value(&txn, &[1, 4096], 3.0.into())
            .await
            .unwrap();
        assert_eq!(
            full_copy.read_value(txn.id(), &[4096, 1]).await.unwrap(),
            7.0.into()
        );
    }
);

test!(dense_and_sparse_codecs_preserve_semantic_schema, async {
    use tc_value::Value;

    let (_root, txn) = Txn::owned(id(10), 256 * 1024, 0);
    for axis in [None, Some(0), Some(1)] {
        let schema =
            TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 3].into(), axis).unwrap();
        let wire = Value::from(schema.clone());
        let Value::Tuple(fields) = &wire else {
            panic!("schema must be a tuple")
        };
        assert_eq!(fields.len(), if axis.is_some() { 3 } else { 2 });
        if let Some(axis) = axis {
            assert_eq!(fields[2], Value::from(axis as u64));
        }
        assert_eq!(TensorSchema::try_from(wire).unwrap(), schema);
        let tensor = Tensor::<Txn>::create(
            txn.subcontext_unique().context().await.unwrap(),
            schema.clone(),
            2,
        )
        .await
        .unwrap();
        tensor.write_value(&txn, &[1, 2], 7.0.into()).await.unwrap();
        for binary in [false, true] {
            let view = tensor.clone().into_view(txn.clone()).await.unwrap();
            let decoded: Tensor<Txn> = if binary {
                tbon::de::try_decode(txn.clone(), tbon::en::encode(view).unwrap())
                    .await
                    .unwrap()
            } else {
                let encoded = destream_json::encode(view)
                    .unwrap()
                    .map_err(std::io::Error::other);
                destream_json::try_decode(txn.clone(), encoded)
                    .await
                    .unwrap()
            };
            assert_eq!(decoded.schema().unwrap(), schema);
            let copied_dir = txn.subcontext_unique().context().await.unwrap();
            let copied = decoded.materialize(&txn, copied_dir.clone()).await.unwrap();
            assert_eq!(
                values(&copied, txn.id()).await,
                values(&decoded, txn.id()).await
            );
            copied.sync_all().await.unwrap();
            drop(copied);
            let reopened = Tensor::<Txn>::load(copied_dir, schema.clone())
                .await
                .unwrap();
            assert_eq!(
                values(&reopened, txn.id()).await,
                values(&decoded, txn.id()).await
            );
            assert_eq!(
                values(&decoded, txn.id()).await,
                [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
            );
        }
    }
});

test!(finalize_and_sync_share_no_cross_domain_guards, async {
    let (_, txn) = setup().await;
    let (tensor, _) = create(&txn, vec![8].into(), true, 2).await;
    tensor.fill(&txn, 3.0.into()).await.unwrap();
    tensor.commit(txn.id()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        let cutoff = id(10);
        let (finalized, synced) = tokio::join!(tensor.finalize(&cutoff), tensor.sync_all());
        finalized.unwrap();
        synced.unwrap();
    })
    .await
    .unwrap();
    assert_eq!(values(&tensor, id(11)).await, [3.0; 8]);
});

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

test!(failed_staged_assignment_preserves_pending_state, async {
    let (_, txn) = setup().await;
    let limit = fensor::MAX_BATCH_ELEMENTS;
    let size = limit + 3;
    let (tensor, _) = create(&txn, vec![size as u64].into(), false, 256).await;
    tensor.write_value(&txn, &[0], 7.0.into()).await.unwrap();
    let Data::F64(value) = &tensor.data else {
        unreachable!()
    };
    let mut expected = vec![0.; size];
    expected[0] = 7.;

    for (name, count, source_error, message) in [
        ("early source failure", 1, true, "supplied stream failed"),
        (
            "source failure after batch",
            limit,
            true,
            "supplied stream failed",
        ),
        (
            "too few",
            size - 1,
            false,
            "too few Tensor assignment values",
        ),
        (
            "too many",
            size + 1,
            false,
            "too many Tensor assignment values",
        ),
    ] {
        let input = (0..count)
            .map(|_| Ok(1.0))
            .chain(source_error.then(|| Err(TCError::bad_request("supplied stream failed"))));
        let error = Typed::write_elements(
            &fensor::TensorView::new(value.owner().unwrap().clone()),
            &txn,
            futures::stream::iter(input),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains(message), "{name}: {error}");
        assert_eq!(values(&tensor, txn.id()).await, expected, "{name}");
    }
    tensor.commit(txn.id()).await.unwrap();
    assert_eq!(values(&tensor, id(11)).await, expected);

    let next = txn.at(12);
    Typed::write_elements(
        &fensor::TensorView::new(value.owner().unwrap().clone()),
        &next,
        futures::stream::iter((0..size).map(|_| Ok(2.0))),
    )
    .await
    .unwrap();
    tensor.commit(next.id()).await.unwrap();
    assert_eq!(values(&tensor, id(13)).await, vec![2.; size]);
});

test!(
    failed_materialization_retains_delta_and_reports_error,
    async {
        for sparse in [false, true] {
            let (_, txn) = setup().await;
            let (tensor, dir) = create(&txn, vec![4].into(), sparse, 2).await;
            tensor.write_value(&txn, &[0], 1.0.into()).await.unwrap();
            tensor.write_value(&txn, &[2], 1.0.into()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
            tensor.finalize(&txn.id()).await.unwrap();
            let next = txn.at(11);
            tensor.write_value(&next, &[0], 3.0.into()).await.unwrap();
            tensor.write_value(&next, &[2], 7.0.into()).await.unwrap();
            tensor.commit(next.id()).await.unwrap();
            let mut retained = tensor.elements(id(12)).await.unwrap();
            assert_eq!(retained.try_next().await.unwrap(), Some(3.0.into()));
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };
            let bad_block = value
                .owner()
                .unwrap()
                .geometry()
                .block_position(&[2])
                .unwrap()
                .0;
            if sparse {
                let values = dir.read().await.get_dir("values").unwrap().clone();
                let primary = values.read().await.get_dir("primary").unwrap().clone();
                let root = primary
                    .read()
                    .await
                    .get_file("00000000-0000-0000-0000-000000000000")
                    .unwrap()
                    .clone();
                let mut node = root.write::<fensor::SparseNode<f64>>().await.unwrap();
                let b_tree::Node::Leaf(rows) = &mut *node else {
                    panic!("two canonical rows fit in one leaf")
                };
                let row = rows
                    .iter_mut()
                    .find(|row| row[0] == fensor::SparseCell::Key(bad_block))
                    .unwrap();
                row[1] = fensor::SparseCell::Key(99);
            } else {
                let blocks = dir.read().await.get_dir("blocks").cloned().unwrap();
                blocks.write().await.delete("1").await;
            }
            assert_eq!(
                tensor.finalize(&next.id()).await.unwrap_err().code(),
                tc_error::ErrorKind::Internal
            );
            let Data::F64(value) = &tensor.data else {
                unreachable!()
            };
            let native = {
                let state = value.owner().unwrap().core.state.read().unwrap();
                assert_eq!(state.finalized, Some(txn.id()));
                assert!(state.committed.contains_key(&next.id()));
                state.persistent.clone()
            };
            assert_eq!(
                native.read_logical_block(0).await.unwrap()[0],
                3.0,
                "first block was materialized"
            );
            // Failed canonical application requires caller-owned disposal, not retry.
            drop(retained);
            drop(native);
            drop(tensor);
        }
    }
);

test!(
    strict_reopen_requires_correct_payloads_and_metadata,
    async {
        let (_, txn) = setup().await;
        for corruption in 0..4 {
            let (tensor, dir) = create(&txn, vec![4].into(), corruption == 3, 2).await;
            let schema = tensor.schema().unwrap();
            let blocks = dir.read().await.get_dir("blocks").cloned().unwrap();
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
                    dir.write().await.delete("values").await;
                }
            }
            assert!(Tensor::<Txn>::load(dir.clone(), schema).await.is_err());
            if corruption == 0 {
                assert!(!blocks.read().await.contains("metadata"));
            } else if corruption == 3 {
                assert!(!dir.read().await.contains("values"));
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

test!(computed_self_assignment_consumes_its_input_leases, async {
    let (_, txn) = setup().await;
    let tensor = Tensor::dense_f64(&txn, vec![2], vec![1.0, 2.0])
        .await
        .unwrap();
    tensor.write_value(&txn, &[0], 3.0.into()).await.unwrap();
    let expression = tensor.add_scalar(txn.id(), 1.0.into()).await.unwrap();
    tensor.write_tensor(&txn, expression).await.unwrap();
    assert_eq!(values(&tensor, txn.id()).await, [4.0, 3.0]);
    let expression = tensor.add_scalar(txn.id(), 1.0.into()).await.unwrap();
    let retained = expression.clone();
    assert!(tensor.write_tensor(&txn, expression).await.is_err());
    assert_eq!(values(&retained, txn.id()).await, [5.0, 4.0]);
    drop(retained);
    assert_eq!(values(&tensor, txn.id()).await, [4.0, 3.0]);
});

test!(
    stream_errors_release_permits_without_requiring_drop,
    async {
        let (_, txn) = setup().await;
        let (tensor, dir) = create(&txn, vec![4].into(), false, 2).await;
        let mut values = tensor.elements(txn.id()).await.unwrap();
        let blocks = dir.read().await.get_dir("blocks").cloned().unwrap();
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
            tensor.commit(id(11)).await.unwrap();
            tensor.commit(id(12)).await.unwrap();
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
        let Data::F64(value) = &tensor.data else {
            unreachable!()
        };
        fensor::TensorWrite::write_value(
            &value.owner().unwrap().canonical().unwrap(),
            &[0, 0],
            2.0,
        )
        .await
        .unwrap();
    }
    {
        let Data::F64(value) = &tensor.data else {
            unreachable!()
        };
        fensor::TensorWrite::write_value(
            &value.owner().unwrap().canonical().unwrap(),
            &[0, 2],
            4.0,
        )
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

test!(streamed_codec_rejects_invalid_payloads, async {
    use tc_value::Value;
    let (_, txn) = setup().await;
    let dense = Value::from(
        TensorSchema::new(NumberType::Float(FloatType::F64), vec![2].into(), None).unwrap(),
    );
    let sparse = Value::from(
        TensorSchema::new(NumberType::Float(FloatType::F64), vec![2].into(), Some(0)).unwrap(),
    );
    let dtype = tc_value::number_type_path(&NumberType::Float(FloatType::F64)).to_string();
    for axis in [
        Value::Number((-1i64).into()),
        Value::from(2u64),
        Value::Number(0.5f64.into()),
        Value::String("sparse".into()),
        Value::None,
    ] {
        let schema = Value::Tuple(vec![
            Value::String(dtype.clone()),
            Value::Tuple(vec![Value::from(2u64), Value::from(3u64)]),
            axis,
        ]);
        assert!(TensorSchema::try_from(schema.clone()).is_err());
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
            "invalid Tensor dtype",
        ),
        (
            "invalid sparse axis",
            format!("{dtype:?},[1],1"),
            "sparse axis out of bounds",
        ),
        (
            "extra schema field",
            format!("{dtype:?},[1],0,0"),
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
    for (name, entry, expected) in [
        (
            "missing coordinate",
            "[]",
            "expected sequence element is missing",
        ),
        (
            "missing value",
            "[[0]]",
            "expected sequence element is missing",
        ),
        (
            "short coordinate",
            "[[],1]",
            "sparse coordinate rank mismatch",
        ),
        ("long coordinate", "[[0,0],1]", "too many Tensor dimensions"),
        (
            "extra entry field",
            "[[0],1,0]",
            "unexpected sparse entry field",
        ),
    ] {
        let input = format!("[[{dtype:?},[2],0],[{entry}]]");
        let encoded = futures::stream::iter([Ok::<_, std::io::Error>(input.into_bytes().into())]);
        let error = match destream_json::try_decode::<_, _, Tensor<Txn>>(txn.clone(), encoded).await
        {
            Ok(_) => panic!("{name}: accepted invalid sparse entry"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected), "{name}: {error}");
    }
    for input in [
        format!("[[{dtype:?},[2],0],[[0,1]]]"),
        format!("[[{dtype:?},[2],0],[[[0],false]]]"),
        format!("[[{dtype:?},[2],0],[[[0],1,{{]]]"),
        format!("[[{dtype:?},false],[]]"),
        format!("[[{dtype:?},[2]]"),
    ] {
        let encoded = futures::stream::iter([Ok::<_, std::io::Error>(input.into_bytes().into())]);
        assert!(
            destream_json::try_decode::<_, _, Tensor<Txn>>(txn.clone(), encoded)
                .await
                .is_err()
        );
    }
    let entry = |coord: u64, value: f64| {
        Value::Tuple(vec![
            Value::Tuple(vec![Value::Number(coord.into())]),
            Value::Number(value.into()),
        ])
    };
    for payload in [
        vec![dense.clone(), Value::Tuple(vec![Value::Number(1u8.into())])],
        vec![
            dense.clone(),
            Value::Tuple(vec![
                Value::Number(1u8.into()),
                Value::Number(2u8.into()),
                Value::Number(3u8.into()),
            ]),
        ],
        vec![
            dense.clone(),
            Value::Tuple(vec![Value::Number(1u8.into()), Value::String("bad".into())]),
        ],
        vec![
            dense,
            Value::Tuple(vec![Value::Number(1u8.into()), Value::Number(2u8.into())]),
            Value::Number(3u8.into()),
        ],
        vec![sparse.clone(), Value::Tuple(vec![entry(0, 0.)])],
        vec![sparse.clone(), Value::Tuple(vec![entry(2, 1.)])],
        vec![
            sparse.clone(),
            Value::Tuple(vec![entry(1, 1.), entry(0, 1.)]),
        ],
        vec![sparse, Value::Tuple(vec![entry(0, 1.), entry(0, 2.)])],
    ] {
        let encoded = destream_json::encode(Value::Tuple(payload))
            .unwrap()
            .map_err(std::io::Error::other);
        assert!(
            destream_json::try_decode::<_, _, Tensor<Txn>>(txn.clone(), encoded)
                .await
                .is_err()
        );
    }
});

test!(dense_and_sparse_stream_errors_drop_source_leases, async {
    for sparse in [false, true] {
        let (_, txn) = setup().await;
        let (tensor, dir) = create(&txn, vec![2, 128].into(), sparse, 128).await;
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
            if sparse {
                let values = dir.read().await.get_dir("values").unwrap().clone();
                let primary = values.read().await.get_dir("primary").unwrap().clone();
                let files: Vec<_> = {
                    let directory = primary.read().await;
                    directory
                        .iter()
                        .map(|(name, _)| directory.get_file(name).unwrap().clone())
                        .collect()
                };
                let mut corrupted = 0;
                // Values belong to leaves; compact routing separators contain no
                // payload copy which could hide this corruption.
                for file in files {
                    let mut node = file.write::<fensor::SparseNode<f64>>().await.unwrap();
                    let b_tree::Node::Leaf(rows) = &mut *node else {
                        continue;
                    };
                    for row in rows {
                        if row[0] == fensor::SparseCell::Key(0) {
                            row[1] = fensor::SparseCell::Key(99);
                            corrupted += 1;
                        }
                    }
                }
                assert_eq!(corrupted, 1);
            } else {
                let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
                blocks.write().await.delete("0").await;
            }
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
            let expected = if sparse {
                "invalid sparse payload"
            } else {
                "missing"
            };
            assert!(error.to_string().contains(expected), "{error}");
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
        use fensor::{TensorTransform, TensorView};
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

            // Exercise schema extraction defensively on the formerly accepted native scalar view.
            let Data::F64(Typed::Owner(owner)) = &tensor.data else {
                panic!("f64 owner")
            };

            let native = TensorView::new(owner.as_ref().clone())
                .slice(scalar)
                .unwrap();

            let unsupported = Tensor {
                data: Data::F64(Typed::View(Box::new(native))),
            };

            assert!(unsupported.schema().is_err());
            assert!(unsupported.into_view(txn.clone()).await.is_err());
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
        let native = fensor::Tensor::<PersistentFile, f64>::from_sparse_elements(
            txn.subcontext_unique().context().await.unwrap(),
            fensor::TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 3].into())
                .unwrap(),
            Layout::Sparse { axis: Some(1) },
            4,
            futures::stream::iter([Ok::<_, fensor::Error>((vec![1, 2], 7.0))]),
        )
        .await
        .unwrap();

        let source = Tensor::<Txn> {
            data: Data::F64(Typed::stored(native)),
        };

        let (target, _) = create(&txn, vec![2, 3].into(), true, 2).await;
        assert_eq!(source.schema().unwrap(), target.schema().unwrap());
        target.restore_from(&txn, &source).await.unwrap();
        assert_eq!(
            values(&target, txn.id()).await,
            vec![0., 0., 0., 0., 0., 7.]
        );

        Collection::from(target.clone())
            .restore_from(&txn, &Collection::from(source))
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
