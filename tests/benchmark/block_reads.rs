//! Resident block-size controls with real transactional histories. Native traffic
//! and retained node bytes are not observed by this production-library adapter.

use std::time::Instant;

use futures::{FutureExt, TryStreamExt};
use number_general::{FloatType, Number, NumberType};
use tc_collection::tensor::AxisRange;
use tc_ir::{Transact, Transaction};

use super::{StorageContext, Tensor, TensorSchema, Txn, id, record, setup};

async fn scan(tensor: &Tensor<Txn>, expected: &[f64]) {
    let mut values = tensor.elements(id(40)).await.unwrap();
    for &expected in expected {
        assert_eq!(
            values.try_next().await.unwrap(),
            Some(Number::from(expected))
        );
    }

    assert!(values.try_next().await.unwrap().is_none());
}

async fn occupied(tensor: &Tensor<Txn>, expected: &[f64], columns: usize) {
    let mut values = tensor.sparse_elements(id(40)).await.unwrap();
    let mut count = 0;
    while let Some((coord, value)) = values.try_next().await.unwrap() {
        assert_eq!(coord.len(), 2);
        let offset = coord[0] as usize * columns + coord[1] as usize;
        assert_ne!(expected[offset], 0.0);
        assert_eq!(value, Number::from(expected[offset]));
        count += 1;
    }

    assert_eq!(
        count,
        expected.iter().filter(|&&value| value != 0.0).count()
    );
}

async fn singleton(tensor: &Tensor<Txn>, expected: &[f64], columns: usize) {
    for (row, column) in [(0, 0), (1, 1024), (2, 2048), (3, 4095)] {
        assert_eq!(
            tensor.read_value(id(40), &[row, column]).await.unwrap(),
            Number::from(expected[row as usize * columns + column as usize])
        );
    }
}

pub(super) async fn run() {
    let repetitions = if std::env::var_os("COLLECTION_BENCH_SMOKE").is_some() {
        1
    } else {
        4
    };

    println!("block_reads: resident timings; native traffic and retained bytes are unmeasured");

    for (axis, capacity) in [(1, 1), (0, 64), (0, 512), (0, 4096)] {
        for full in [false, true] {
            // Scalar blocks have identical isolated/full occupancy.
            if axis == 1 && !full {
                continue;
            }

            for depth in [0, 1, 16] {
                let (_root, txn) = setup().await;
                let shape = [4u64, 4096];
                let columns = shape[1] as usize;
                let mut expected = (0..shape[0] as usize * columns)
                    .map(|offset| {
                        if full || offset % capacity == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();

                let schema = TensorSchema::new(
                    NumberType::Float(FloatType::F64),
                    shape.as_slice().into(),
                    Some(axis),
                )
                .unwrap();

                let dir = txn.subcontext_unique().context().await.unwrap();
                let tensor = Tensor::<Txn>::create(dir, schema, capacity)
                    .boxed()
                    .await
                    .unwrap();

                if full {
                    tensor.fill(&txn, 1.0.into()).boxed().await.unwrap();
                } else {
                    for offset in (0..expected.len()).step_by(capacity) {
                        tensor
                            .write_value(
                                &txn,
                                &[(offset / columns) as u64, (offset % columns) as u64],
                                1.0.into(),
                            )
                            .await
                            .unwrap();
                    }
                }

                tensor.commit(txn.id()).await.unwrap();
                tensor.finalize(&txn.id()).await.unwrap();
                let blocks = expected.len() / capacity;
                for version in 0..depth {
                    let next = txn.at(11 + version);
                    let offset = usize::from(version) % blocks * capacity;
                    let value = f64::from(version) + 2.0;
                    tensor
                        .write_value(
                            &next,
                            &[(offset / columns) as u64, (offset % columns) as u64],
                            value.into(),
                        )
                        .await
                        .unwrap();

                    tensor.commit(next.id()).await.unwrap();
                    expected[offset] = value;
                }

                let selected = (0..256)
                    .map(|i| (i * 17 % columns) as u64)
                    .collect::<Vec<_>>();

                let scattered_expected = (0..shape[0] as usize)
                    .flat_map(|row| {
                        selected
                            .iter()
                            .map(move |&column| row * columns + column as usize)
                    })
                    .map(|offset| expected[offset])
                    .collect::<Vec<_>>();

                let scattered = tensor
                    .clone()
                    .slice(vec![AxisRange::In(0, shape[0], 1), AxisRange::Of(selected)].into())
                    .unwrap();

                let occupancy = if full { "full" } else { "isolated" };
                let workload =
                    format!("block_reads/axis{axis}/cap{capacity}/{occupancy}/depth{depth}");

                for (metric, count) in [
                    ("elements", expected.len()),
                    ("block_elements", capacity),
                    ("blocks", blocks),
                    (
                        "nonzero_elements",
                        expected.iter().filter(|&&value| value != 0.0).count(),
                    ),
                    ("history", usize::from(depth)),
                    ("repetitions", repetitions),
                    ("singleton_reads_per_repetition", 4),
                ] {
                    println!("COLLECTION,{workload},geometry,resident,{metric},count,{count}");
                }

                // Warm independently and consume streams completely before timing
                // another operation; all fixture construction stays outside timing.
                scan(&tensor, &expected).await;
                scan(&scattered, &scattered_expected).await;
                occupied(&tensor, &expected, columns).await;
                singleton(&tensor, &expected, columns).await;

                let start = Instant::now();
                for _ in 0..repetitions {
                    scan(&tensor, &expected).await;
                }

                record(&workload, "sequential", "warm", start.elapsed());

                let start = Instant::now();
                for _ in 0..repetitions {
                    scan(&scattered, &scattered_expected).await;
                }

                record(&workload, "scattered", "warm", start.elapsed());

                let start = Instant::now();
                for _ in 0..repetitions {
                    occupied(&tensor, &expected, columns).await;
                }

                record(&workload, "occupied", "warm", start.elapsed());

                let start = Instant::now();
                for _ in 0..repetitions {
                    singleton(&tensor, &expected, columns).await;
                }

                record(&workload, "singleton", "warm", start.elapsed());
            }
        }
    }
}
