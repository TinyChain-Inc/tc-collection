//! Resident transformed reads with fixed storage and transactional histories.

use std::time::Instant;

use fensor::TensorSource;
use futures::{FutureExt, TryStreamExt, stream};
use number_general::{FloatType, Number, NumberType};
use safecast::CastFrom;
use tc_collection::tensor::TensorReduceResult;
use tc_ir::{Transact, Transaction};

use super::{PersistentFile, StorageContext, Tensor, TensorSchema, Txn, id, record, setup};

async fn consume(source: &Tensor<Txn>, expected: f64, count: usize, mode: &str) {
    if mode == "sum" {
        let TensorReduceResult::Scalar(value) = source.sum(id(40), None, false).await.unwrap()
        else {
            panic!("terminal sum");
        };

        assert_eq!(value, Number::from(expected));
    } else {
        let mut elements = source.elements(id(40)).await.unwrap();
        let mut received = 0;
        let mut sum = 0.0;
        while let Some(value) = elements.try_next().await.unwrap() {
            sum += f64::cast_from(value);
            received += 1;
        }

        assert_eq!(received, count);
        assert_eq!(sum, expected);
    }
}

pub(super) async fn run() {
    let repetitions = if std::env::var_os("COLLECTION_BENCH_SMOKE").is_some() {
        1
    } else {
        4
    };

    for capacity in [64, 512, 4096] {
        for full in [false, true] {
            for depth in [0u16, 1, 16] {
                let (_root, txn) = setup().await;
                let shape = [4u64, 4097];
                let columns = shape[1] as usize;
                let mut values = (0..shape[0] as usize * columns)
                    .map(|i| if full || i % 257 == 0 { 1f64 } else { 0.0 })
                    .collect::<Vec<_>>();

                let schema = TensorSchema::new(
                    NumberType::Float(FloatType::F64),
                    shape.as_slice().into(),
                    Some(0),
                )
                .unwrap();

                let dir = txn.subcontext_unique().context().await.unwrap();
                let native = fensor::Tensor::<PersistentFile, f64>::from_sparse_elements(
                    dir.clone(),
                    fensor::TensorSchema::new(schema.dtype(), shape.as_slice().into()).unwrap(),
                    schema.layout(),
                    capacity,
                    stream::iter(values.iter().enumerate().filter(|(_, v)| **v != 0.0).map(
                        |(i, &v)| {
                            Ok::<_, fensor::Error>((
                                vec![(i / columns) as u64, (i % columns) as u64],
                                v,
                            ))
                        },
                    )),
                )
                .boxed()
                .await
                .unwrap();

                let geometry = native.storage_geometry();
                native.sync_all().await.unwrap();
                drop(native);
                let tensor = Tensor::<Txn>::load(dir, schema).boxed().await.unwrap();
                for version in 0..depth {
                    let next = txn.at(11 + version);
                    let offset = usize::from(version) * 257;
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
                    values[offset] = value;
                }

                let expected = values.iter().sum();
                let occupancy = if full { "full" } else { "isolated" };
                let base = format!("layout/cap{capacity}/{occupancy}/depth{depth}");
                for (metric, value) in [
                    ("block_elements", geometry.block_capacity()),
                    ("blocks", geometry.block_count() as usize),
                    ("elements", values.len()),
                    ("history", usize::from(depth)),
                    ("repetitions", repetitions),
                ] {
                    println!("COLLECTION,{base},geometry,resident,{metric},count,{value}");
                }

                let owned = tensor.abs(id(40)).await.unwrap().transpose(None).unwrap();
                let mut nested = tensor.clone();
                for _ in 0..4 {
                    nested = nested.transpose(None).unwrap().abs(id(40)).await.unwrap();
                }

                for (kind, source) in [("owned", &owned), ("nested", &nested)] {
                    let workload = format!("{base}/{kind}");
                    for mode in ["row", "sum"] {
                        consume(source, expected, values.len(), mode).await;
                        let start = Instant::now();
                        for _ in 0..repetitions {
                            consume(source, expected, values.len(), mode).await;
                        }

                        record(&workload, mode, "warm", start.elapsed());
                    }
                }
            }
        }
    }
}
