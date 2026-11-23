//! Capacity controls retain the same logical values and update coordinates across
//! layouts. Payload-byte records exclude index and container overhead.

use std::mem::size_of;
use std::time::Instant;

use fensor::{TensorElement, TensorFileEntry, TensorSource};
use futures::{FutureExt, TryStreamExt, stream};
use number_general::Number;
use tc_collection::tensor::AxisRange;
use tc_ir::{Transact, Transaction};

use super::{PersistentFile, StorageContext, Tensor, TensorSchema, Txn, id, record, setup};

async fn scan<T: Copy + Into<Number>>(tensor: &Tensor<Txn>, expected: &[T], version: u16) {
    let mut values = tensor.elements(id(version)).await.unwrap();
    for &expected in expected {
        assert_eq!(values.try_next().await.unwrap(), Some(expected.into()));
    }

    assert!(values.try_next().await.unwrap().is_none());
}

async fn singleton<T: Copy + Into<Number>>(tensor: &Tensor<Txn>, expected: &[T]) {
    for (row, column) in [(0, 0), (1, 1024), (2, 2048), (3, 4095)] {
        assert_eq!(
            tensor.read_value(id(40), &[row, column]).await.unwrap(),
            expected[row as usize * 4096 + column as usize].into()
        );
    }
}

async fn typed<T>(dtype: &str, repetitions: usize)
where
    T: TensorElement + From<u8>,
    PersistentFile: TensorFileEntry<T>,
{
    for (axis, capacity) in [(1, 1), (0, 64), (0, 512), (0, 4096)] {
        for pattern in ["isolated", "clustered", "full"] {
            for depth in [0, 16] {
                let (_root, txn) = setup().await;
                let shape = [4u64, 4096];
                let columns = shape[1] as usize;
                let mut expected = (0..shape[0] as usize * columns)
                    .map(|offset| {
                        let column = offset % columns;
                        let nonzero = match pattern {
                            "isolated" => column.is_multiple_of(1024),
                            "clustered" => column < 512,
                            "full" => true,
                            _ => unreachable!(),
                        };

                        T::from(u8::from(nonzero))
                    })
                    .collect::<Vec<_>>();

                let schema =
                    TensorSchema::new(T::dtype(), shape.as_slice().into(), Some(axis)).unwrap();

                let dir = txn.subcontext_unique().context().await.unwrap();
                let entries = stream::iter(
                    expected
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| **value != T::from(0))
                        .map(|(offset, &value)| {
                            Ok::<_, fensor::Error>((
                                vec![(offset / columns) as u64, (offset % columns) as u64],
                                value,
                            ))
                        }),
                );

                let native = fensor::Tensor::<PersistentFile, T>::from_sparse_elements(
                    dir.clone(),
                    fensor::TensorSchema::new(schema.dtype(), shape.as_slice().into()).unwrap(),
                    schema.layout(),
                    capacity,
                    entries,
                )
                .boxed()
                .await
                .unwrap();

                let geometry = native.storage_geometry();
                assert_eq!(geometry.block_capacity(), capacity);
                native.sync_all().await.unwrap();
                drop(native);

                let canonical_nonzero = expected.iter().filter(|&&v| v != T::from(0)).count();
                let occupied_blocks = expected
                    .chunks(capacity)
                    .filter(|block| block.iter().any(|&v| v != T::from(0)))
                    .count();

                let tensor = Tensor::<Txn>::load(dir, schema).boxed().await.unwrap();
                for version in 0..depth {
                    let next = txn.at(11 + version);
                    let row = usize::from(version) % shape[0] as usize;
                    let column = usize::from(version) / shape[0] as usize * 1024;
                    let value = T::from(version as u8 + 2);
                    tensor
                        .write_value(&next, &[row as u64, column as u64], value.into())
                        .await
                        .unwrap();

                    tensor.commit(next.id()).await.unwrap();
                    expected[row * columns + column] = value;
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

                let workload = format!(
                    "capacity_policy/{dtype}/axis{axis}/cap{capacity}/{pattern}/depth{depth}"
                );

                for (metric, count) in [
                    ("elements", expected.len()),
                    ("element_bytes", size_of::<T>()),
                    ("block_elements", geometry.block_capacity()),
                    ("block_bytes", geometry.block_capacity() * size_of::<T>()),
                    ("blocks", geometry.block_count() as usize),
                    ("canonical_occupied_blocks", occupied_blocks),
                    ("canonical_nonzero_elements", canonical_nonzero),
                    (
                        "canonical_payload_bytes",
                        occupied_blocks * geometry.block_capacity() * size_of::<T>(),
                    ),
                    (
                        "visible_nonzero_elements",
                        expected.iter().filter(|&&v| v != T::from(0)).count(),
                    ),
                    (
                        "visible_occupied_blocks",
                        expected
                            .chunks(capacity)
                            .filter(|block| block.iter().any(|&v| v != T::from(0)))
                            .count(),
                    ),
                    ("history", usize::from(depth)),
                    ("repetitions", repetitions),
                    ("singleton_reads_per_repetition", 4),
                    ("point_writes", 4),
                ] {
                    println!("COLLECTION,{workload},geometry,resident,{metric},count,{count}");
                }

                scan(&tensor, &expected, 40).await;
                scan(&scattered, &scattered_expected, 40).await;
                singleton(&tensor, &expected).await;

                let start = Instant::now();
                for _ in 0..repetitions {
                    scan(&tensor, &expected, 40).await;
                }

                record(&workload, "sequential", "warm", start.elapsed());

                let start = Instant::now();
                for _ in 0..repetitions {
                    scan(&scattered, &scattered_expected, 40).await;
                }

                record(&workload, "scattered", "warm", start.elapsed());

                let start = Instant::now();
                for _ in 0..repetitions {
                    singleton(&tensor, &expected).await;
                }

                record(&workload, "singleton", "warm", start.elapsed());

                // The same four logical writes follow every read case. This keeps
                // mutation and finalization comparisons independent of block IDs.
                let writes = [(0, 0), (1, 511), (2, 1024), (3, 4095)];
                let start = Instant::now();
                for (i, &(row, column)) in writes.iter().enumerate() {
                    let next = txn.at(41 + i as u16);
                    tensor
                        .write_value(&next, &[row, column], T::from(100 + i as u8).into())
                        .await
                        .unwrap();

                    tensor.commit(next.id()).await.unwrap();
                }

                record(&workload, "write_commit", "mixed", start.elapsed());
                for (i, &(row, column)) in writes.iter().enumerate() {
                    expected[row as usize * columns + column as usize] = T::from(100 + i as u8);
                }

                let start = Instant::now();
                tensor.finalize(&id(44)).boxed().await.unwrap();
                tensor.sync_all().await.unwrap();
                record(&workload, "finalize_sync", "mixed", start.elapsed());
                scan(&tensor, &expected, 50).await;
            }
        }
    }
}

pub(super) async fn run() {
    let repetitions = if std::env::var_os("COLLECTION_BENCH_SMOKE").is_some() {
        1
    } else {
        4
    };

    typed::<u8>("u8", repetitions).boxed().await;
    typed::<f64>("f64", repetitions).boxed().await;
}
