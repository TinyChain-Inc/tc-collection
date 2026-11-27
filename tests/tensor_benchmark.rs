//! Opt-in, production-library measurements. Run through fensor/benchmarks/compare.py
//! with `--record-prefix COLLECTION --suite-env COLLECTION_BENCH_SUITE`.
use std::time::{Duration, Instant};

use futures::{FutureExt, TryStreamExt};
use number_general::Number;
use tc_collection::tensor::{AxisRange, Range, Tensor, TensorSchema};
use tc_collection::{PersistentFile, Restore, StorageContext};
use tc_ir::{IntoView, Transact, Transaction};

#[path = "common/storage.rs"]
mod storage;
#[path = "common/tensor.rs"]
mod support;

use support::{Txn, create, id, values};

#[path = "benchmark/block_reads.rs"]
mod block_reads;
#[path = "benchmark/capacity_policy.rs"]
mod capacity_policy;
#[path = "benchmark/layout.rs"]
mod tensor_layout;

async fn setup() -> (std::sync::Arc<support::TempRoot>, Txn) {
    match std::env::var("COLLECTION_BENCH_CACHE_BYTES") {
        Ok(capacity) => {
            let capacity = capacity
                .parse()
                .expect("COLLECTION_BENCH_CACHE_BYTES must be a byte count");

            support::setup_with_capacity(capacity).await
        }
        Err(std::env::VarError::NotPresent) => support::setup().await,
        Err(error) => panic!("invalid benchmark cache capacity: {error}"),
    }
}

fn record(workload: &str, operation: &str, temperature: &str, elapsed: Duration) {
    println!(
        "COLLECTION,{workload},{operation},{temperature},elapsed,ns,{}",
        elapsed.as_nanos()
    );
}

fn layout(sparse: bool) -> &'static str {
    if sparse { "sparse" } else { "dense" }
}

async fn finalization() {
    for sparse in [false, true] {
        let (_root, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![128].into(), sparse, 1).await;
        let workload = format!("finalization/{}", layout(sparse));
        let start = Instant::now();
        for i in 0..128 {
            tensor.write_value(&txn, &[i], 1.0.into()).await.unwrap();
        }

        record(&workload, "point_updates", "mixed", start.elapsed());
        tensor.commit(txn.id()).await.unwrap();
        let start = Instant::now();
        tensor.finalize(&txn.id()).boxed().await.unwrap();
        record(&workload, "finalize_sync", "mixed", start.elapsed());
        assert!(values(&tensor, id(11)).await.iter().all(|v| *v == 1.));
    }
}

async fn mutation() {
    for sparse in [false, true] {
        for transformed in [false, true] {
            let (_root, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![16, 64].into(), sparse, 64).await;
            // Retain the historical record name and equivalent constant fill,
            // delivering the selection to its owner instead of a writable view.
            let selection: Range = vec![AxisRange::In(0, 16, 1), AxisRange::In(0, 64, 1)].into();
            let mode = if transformed {
                "transformed_fill"
            } else {
                "fill"
            };

            let workload = format!("mutation/{}/{mode}", layout(sparse));
            let start = Instant::now();
            for version in 10..14 {
                let txn = txn.at(version);
                if transformed {
                    tensor
                        .fill_range(&txn, selection.clone(), 2.0.into())
                        .boxed()
                        .await
                        .unwrap();
                } else {
                    tensor.fill(&txn, 2.0.into()).boxed().await.unwrap();
                }

                tensor.commit(txn.id()).await.unwrap();
            }

            record(&workload, "write_commit", "mixed", start.elapsed());
            let start = Instant::now();
            assert!(values(&tensor, id(14)).await.iter().all(|v| *v == 2.));
            record(&workload, "scan", "warm", start.elapsed());
            if sparse {
                let start = Instant::now();
                assert_eq!(sparse_count(&tensor, 14).await, 1024);
                record(&workload, "sparse_scan", "warm", start.elapsed());
            }

            let start = Instant::now();
            tensor.finalize(&id(13)).boxed().await.unwrap();
            record(&workload, "finalize_sync", "mixed", start.elapsed());
        }
    }
}

// A zero fill replaces existing pending data and exercises the explicit zeros
// that suppress canonical ones through finalization.
async fn zero_replacements() {
    for sparse in [false, true] {
        let (_root, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![32, 129].into(), sparse, 64).await;
        tensor.fill(&txn, 1.0.into()).boxed().await.unwrap();
        tensor.commit(txn.id()).await.unwrap();
        tensor.finalize(&txn.id()).boxed().await.unwrap();
        let next = txn.at(11);
        // Keep an existing pending delta for the full fill to replace.
        tensor.fill(&next, 2.0.into()).boxed().await.unwrap();
        let workload = format!("zero_replacements/{}", layout(sparse));
        let start = Instant::now();
        tensor
            .fill_range(
                &next,
                vec![AxisRange::In(0, 32, 1), AxisRange::In(0, 129, 1)].into(),
                0.0.into(),
            )
            .boxed()
            .await
            .unwrap();

        record(&workload, "zero_patch", "mixed", start.elapsed());
        tensor
            .write_value(&next, &[0, 0], 3.0.into())
            .await
            .unwrap();

        tensor
            .write_value(&next, &[31, 128], 4.0.into())
            .await
            .unwrap();

        tensor.commit(next.id()).await.unwrap();
        let start = Instant::now();
        let observed = values(&tensor, id(12)).await;
        record(&workload, "scan", "warm", start.elapsed());
        assert_eq!(observed.len(), 32 * 129);
        assert_eq!(observed[0], 3.0);
        assert_eq!(observed[observed.len() - 1], 4.0);
        assert!(observed[1..observed.len() - 1].iter().all(|v| *v == 0.0));
        let start = Instant::now();
        tensor.finalize(&next.id()).boxed().await.unwrap();
        record(&workload, "finalize", "mixed", start.elapsed());
        let start = Instant::now();
        record(&workload, "sync", "mixed", start.elapsed());
        assert_eq!(values(&tensor, id(12)).await, observed);
    }
}

async fn assignment() {
    for sparse in [false, true] {
        let (_root, txn) = setup().await;
        let source = Tensor::dense_f64(&txn, vec![32, 32], (0..1024).map(f64::from).collect())
            .boxed()
            .await
            .unwrap();

        let (target, _) = create(&txn, vec![32, 32].into(), sparse, 64).await;
        let workload = format!("assignment/{}", layout(sparse));
        let start = Instant::now();
        target
            .write_tensor(&txn, source.clone().transpose(None).unwrap())
            .boxed()
            .await
            .unwrap();

        record(&workload, "transpose", "mixed", start.elapsed());
        let start = Instant::now();
        target.write_tensor(&txn, source).boxed().await.unwrap();
        record(&workload, "replace_pending", "mixed", start.elapsed());
        assert_eq!(
            values(&target, txn.id()).await,
            (0..1024).map(f64::from).collect::<Vec<_>>()
        );

        target.commit(txn.id()).await.unwrap();
        let start = Instant::now();
        target.finalize(&txn.id()).boxed().await.unwrap();
        record(&workload, "finalize_sync", "mixed", start.elapsed());
    }
}

async fn construction() {
    let (_root, txn) = setup().await;
    let start = Instant::now();
    let tensor = Tensor::dense_f64(&txn, vec![64, 128], vec![1.; 8192])
        .boxed()
        .await
        .unwrap();

    record("construction/dense", "literal", "mixed", start.elapsed());
    let start = Instant::now();
    for _ in 0..16 {
        assert_eq!(values(&tensor, txn.id()).await.len(), 8192);
    }

    record("construction/dense", "stream", "warm", start.elapsed());
    for sparse in [false, true] {
        let (source, _) = create(&txn, vec![64, 128].into(), sparse, 4096).await;
        source.fill(&txn, 1.0.into()).boxed().await.unwrap();
        let encoded = destream_json::encode(source.into_view(txn.clone()).await.unwrap())
            .unwrap()
            .map_err(std::io::Error::other);

        let start = Instant::now();
        let decoded: Tensor<Txn> = destream_json::try_decode(txn.clone(), encoded)
            .boxed()
            .await
            .unwrap();

        record(
            &format!("construction/{}", layout(sparse)),
            "decode",
            "mixed",
            start.elapsed(),
        );

        assert_eq!(values(&decoded, txn.id()).await, vec![1.; 8192]);
    }
}

async fn restoration() {
    for sparse in [false, true] {
        let (_root, txn) = setup().await;
        let (source, _) = create(&txn, vec![32, 65].into(), sparse, 256).await;
        source.fill(&txn, 2.0.into()).boxed().await.unwrap();
        let (target, _) = create(&txn, vec![32, 65].into(), sparse, 64).await;
        let start = Instant::now();
        target.restore(&txn, &source).boxed().await.unwrap();
        record(
            &format!("restoration/{}", layout(sparse)),
            "restore",
            "mixed",
            start.elapsed(),
        );

        assert!(values(&target, txn.id()).await.iter().all(|v| *v == 2.));
    }

    const WIDTH: u64 = 4096;
    const SOURCE_CAPACITY: usize = 256;
    for rows in [32, 128] {
        for target_capacity in [SOURCE_CAPACITY, 64] {
            let (_root, txn) = setup().await;
            let shape = [rows, WIDTH];
            let schema = TensorSchema::new(
                number_general::FloatType::F64.into(),
                shape.as_slice().into(),
                Some(0),
            )
            .unwrap();
            let source = Tensor::create(
                txn.subcontext_unique().context().await.unwrap(),
                schema.clone(),
                SOURCE_CAPACITY,
            )
            .boxed()
            .await
            .unwrap();
            let target = Tensor::create(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
                target_capacity,
            )
            .boxed()
            .await
            .unwrap();
            let source_coords = [[0, 0], [rows - 1, WIDTH - 1]];
            let target_coords = [[0, WIDTH / 4], [rows - 1, WIDTH / 2]];
            for (tensor, coords) in [(&source, source_coords), (&target, target_coords)] {
                for coord in coords {
                    tensor.write_value(&txn, &coord, 2.0.into()).await.unwrap();
                }

                tensor.commit(txn.id()).await.unwrap();
                tensor.finalize(&txn.id()).boxed().await.unwrap();
            }

            let reader = txn.at(11);
            let workload =
                format!("restoration/sparse/{rows}x{WIDTH}/{SOURCE_CAPACITY}-to-{target_capacity}");
            let start = Instant::now();
            target.restore(&reader, &source).boxed().await.unwrap();
            record(&workload, "restore", "mixed", start.elapsed());
            for coord in source_coords {
                assert_eq!(
                    target.read_value(reader.id(), &coord).await.unwrap(),
                    2.0.into()
                );
            }
            for coord in target_coords {
                assert_eq!(
                    target.read_value(reader.id(), &coord).await.unwrap(),
                    0.0.into()
                );
            }
        }
    }
}

async fn sparse_count(tensor: &Tensor<Txn>, transaction: u16) -> usize {
    tensor
        .sparse_elements(id(transaction))
        .await
        .unwrap()
        .try_fold(0usize, |n, _| futures::future::ready(Ok(n + 1)))
        .await
        .unwrap()
}

async fn overlay() {
    // Small selections use bounded logical evaluation; larger ones exercise
    // occupied-index traversal across batches with the same block geometry.
    let cases = if std::env::var_os("COLLECTION_BENCH_SMOKE").is_some() {
        [(256, 1), (4097, 1)]
    } else {
        [(256, 20), (8193, 3)]
    };

    for (size, repetitions) in cases {
        for depth in [0, 1, 4, 16] {
            let (_root, txn) = setup().await;
            let (tensor, _) = create(&txn, vec![size].into(), true, 8).await;
            tensor.fill(&txn, 1.0.into()).boxed().await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
            tensor.finalize(&txn.id()).boxed().await.unwrap();
            for n in 0..depth {
                let transaction = txn.at(11 + n);
                for offset in 0..8 {
                    tensor
                        .write_value(
                            &transaction,
                            &[u64::from(n) * 8 + offset],
                            if offset % 2 == 0 {
                                0.0.into()
                            } else {
                                2.0.into()
                            },
                        )
                        .await
                        .unwrap();
                }

                tensor.commit(transaction.id()).await.unwrap();
            }

            let start = Instant::now();
            for _ in 0..repetitions {
                assert_eq!(
                    sparse_count(&tensor, 40).await,
                    size as usize - usize::from(depth) * 4
                );
            }

            record(
                &format!("overlay/{size}/{depth}"),
                "sparse_scan",
                "warm",
                start.elapsed(),
            );
        }
    }
}

async fn check_scan(tensor: &Tensor<Txn>) {
    let mut values = tensor.elements(id(14)).await.unwrap();
    let mut count = 0;
    while let Some(value) = values.try_next().await.unwrap() {
        assert_eq!(value, Number::from(2.0));
        count += 1;
    }

    assert_eq!(count, 1024);
}

async fn reads() {
    for sparse in [false, true] {
        let (_root, txn) = setup().await;
        let (tensor, _) = create(&txn, vec![16, 64].into(), sparse, 64).await;
        let workload = format!("reads/{}", layout(sparse));
        let start = Instant::now();
        for version in 10..14 {
            let txn = txn.at(version);
            tensor.fill(&txn, 2.0.into()).boxed().await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
        }

        record(&workload, "write_commit", "mixed", start.elapsed());
        let start = Instant::now();
        check_scan(&tensor).await;
        record(&workload, "scan", "first", start.elapsed());
        let start = Instant::now();
        for _ in 0..200 {
            check_scan(&tensor).await;
        }

        record(&workload, "scan", "warm", start.elapsed());
    }
}

async fn integrity() {
    use tc_collection::btree::{BTree, StorageConfig};
    use tc_collection::table::{Column, PersistentTable, Range, TableSchema};
    use tc_value::{NumberType, Value, ValueType};

    for depth in [1, 16] {
        let (_root, txn) = setup().await;
        let tree = BTree::<Txn>::create(
            txn.subcontext_unique().context().await.unwrap(),
            tc_collection::btree::BTreeSchema::try_from(vec![
                tc_collection::btree::BTreeColumnSchema {
                    name: "key".into(),
                    dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
                    max_size: None,
                },
            ])
            .unwrap(),
        )
        .await
        .unwrap();

        let schema = TableSchema::new(
            vec![Column {
                name: "id".parse().unwrap(),
                dtype: ValueType::Number(NumberType::Number),
            }],
            Vec::new(),
            Vec::new(),
            StorageConfig::default(),
        )
        .unwrap();

        let table = PersistentTable::<Txn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            schema,
        )
        .await
        .unwrap();

        let (tensor, _) = create(&txn, vec![1024].into(), false, 64).await;
        for version in 0..depth {
            let txn = txn.at(10 + version);
            for key in (version..256).step_by(depth as usize) {
                tree.insert_row(&txn, vec![Value::from(u64::from(key))])
                    .await
                    .unwrap();

                table
                    .upsert_row(&txn, vec![Value::from(u64::from(key))], Vec::new())
                    .await
                    .unwrap();
            }

            tensor.fill(&txn, 2.0.into()).boxed().await.unwrap();
            tree.commit(txn.id()).await.unwrap();
            table.commit(txn.id()).await.unwrap();
            tensor.commit(txn.id()).await.unwrap();
        }

        let expression = tensor.add(id(40), &tensor).await.unwrap();
        // Warm every path before timing; all histories use the same resident budget.
        for pass in 0..2 {
            for kind in ["btree", "table", "tensor", "shared_tensor"] {
                let start = Instant::now();
                for _ in 0..20 {
                    match kind {
                        "btree" => assert_eq!(tree.count(id(40)).await.unwrap(), 256),
                        "table" => {
                            let mut rows = table
                                .rows(id(40), Range::default(), Vec::new(), false)
                                .await
                                .unwrap();

                            let mut count = 0;
                            while rows.try_next().await.unwrap().is_some() {
                                count += 1;
                            }

                            assert_eq!(count, 256);
                        }
                        _ => {
                            let source = if kind == "tensor" {
                                &tensor
                            } else {
                                &expression
                            };

                            let expected = if kind == "tensor" { 2.0 } else { 4.0 };
                            let mut stream = source.elements(id(40)).await.unwrap();
                            let mut count = 0;
                            while let Some(value) = stream.try_next().await.unwrap() {
                                assert_eq!(value, Number::from(expected));
                                count += 1;
                            }

                            assert_eq!(count, 1024);
                        }
                    }
                }

                if pass == 1 {
                    record(
                        &format!("integrity/{kind}/{depth}"),
                        "scan",
                        "warm",
                        start.elapsed(),
                    );
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "isolated release measurements, not a timing gate"]
async fn benchmark() {
    println!(
        "Tokio workers: {}",
        tokio::runtime::Handle::current().metrics().num_workers()
    );

    println!(
        "cache capacity: {} bytes",
        std::env::var("COLLECTION_BENCH_CACHE_BYTES")
            .unwrap_or_else(|_| support::DEFAULT_CACHE_BYTES.to_string())
    );

    let suite = std::env::var("COLLECTION_BENCH_SUITE").unwrap_or_else(|_| "all".into());
    let mut selected = false;
    for name in [
        "finalization",
        "mutation",
        "assignment",
        "construction",
        "restoration",
        "overlay",
        "reads",
        "integrity",
        "zero_replacements",
        "block_reads",
        "capacity_policy",
        "layout",
    ] {
        if suite != "all" && suite != name {
            continue;
        }

        selected = true;
        match name {
            "finalization" => finalization().boxed().await,
            "mutation" => mutation().boxed().await,
            "assignment" => assignment().boxed().await,
            "construction" => construction().boxed().await,
            "restoration" => restoration().boxed().await,
            "overlay" => overlay().boxed().await,
            "reads" => reads().boxed().await,
            "integrity" => integrity().boxed().await,
            "zero_replacements" => zero_replacements().boxed().await,
            "block_reads" => block_reads::run().boxed().await,
            "capacity_policy" => capacity_policy::run().boxed().await,
            "layout" => tensor_layout::run().boxed().await,
            _ => unreachable!(),
        }
    }

    assert!(selected, "unknown COLLECTION_BENCH_SUITE: {suite}");
}
