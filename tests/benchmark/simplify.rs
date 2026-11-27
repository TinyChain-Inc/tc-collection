//! Owner-level measurements; elapsed times describe costs and never select policy.
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use collate::Collate;
use futures::{TryStreamExt, stream};
use tc_collection::Restore;
use tc_value::{Value, ValueCollator};

use super::*;

#[derive(Clone)]
struct Counted<C> {
    inner: C,
    comparisons: Arc<AtomicUsize>,
}

impl<C: PartialEq> PartialEq for Counted<C> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<C: Eq> Eq for Counted<C> {}

impl<C: Collate> Collate for Counted<C> {
    type Value = C::Value;

    fn cmp(&self, left: &C::Value, right: &C::Value) -> std::cmp::Ordering {
        self.comparisons.fetch_add(1, Ordering::Relaxed);
        self.inner.cmp(left, right)
    }
}

fn comparisons(workload: &str, operation: &str, count: &AtomicUsize) {
    println!(
        "PERSISTENCE,{workload},{operation},warm,comparisons,count,{}",
        count.swap(0, Ordering::Relaxed)
    );
}

pub(super) async fn run() {
    primary_reads().await;
    restoration().await;
    tied_keys().await;
    unions().await;
    tensor_history().await;
}

async fn primary_reads() {
    for indexes in [0, 1, 8] {
        let (_root, txn) = TestTxn::owned(
            storage_id(1),
            RESIDENT_CACHE_BYTES,
            MATERIALIZED_TENSOR_BYTES,
        );

        let column = |name: &str| Column {
            name: name.parse().unwrap(),
            dtype: ValueType::Number(NumberType::Number),
        };

        let schema = TableSchema::new(
            vec![column("key")],
            vec![column("value")],
            (0..indexes)
                .map(|index| (format!("index_{index}"), vec!["value".parse().unwrap()]))
                .collect(),
            StorageConfig::default(),
        )
        .unwrap();

        let native = b_table::TableLock::create(
            schema,
            ValueCollator::default(),
            txn.context().await.unwrap(),
        )
        .await
        .unwrap();

        native
            .write()
            .await
            .upsert(vec![1u64.into()], vec![2u64.into()])
            .await
            .unwrap();

        let workload = format!("evaluation/primary/{indexes}");
        let start = Instant::now();
        for _ in 0..4096 {
            assert!(native.get_row(&[1u64.into()]).await.unwrap().is_some());
        }

        record(&workload, "get_row_4096", start.elapsed());
        let start = Instant::now();
        for _ in 0..4096 {
            assert!(native.contains(&[1u64.into()]).await.unwrap());
        }

        record(&workload, "contains_4096", start.elapsed());
    }
}

async fn restoration() {
    for table in [false, true] {
        for rows in [256u64, 4096] {
            for depth in [0, 16] {
                let (_root, txn) = TestTxn::owned(
                    storage_id(1),
                    RESIDENT_CACHE_BYTES,
                    MATERIALIZED_TENSOR_BYTES,
                );

                let target = create(&txn, table).await;
                let source = create(&txn, table).await;
                for key in 0..rows {
                    insert(&target, &txn, key).await;
                    if key % 2 == 0 {
                        insert(&source, &txn, key).await;
                    }
                }

                for value in [&target, &source] {
                    value.commit(txn.id()).await.unwrap();
                    value.finalize(&txn.id()).await.unwrap();
                }

                for version in 0..depth {
                    let writer = txn.at(version + 2);
                    insert(&target, &writer, rows + u64::from(version)).await;
                    target.commit(writer.id()).await.unwrap();
                }

                let reader = txn.at(100);
                let workload = format!(
                    "evaluation/restore/{}/{rows}/{depth}",
                    if table { "table" } else { "btree" }
                );

                let start = Instant::now();
                target.restore(&reader, &source).await.unwrap();
                record(&workload, "restore", start.elapsed());
                assert_eq!(
                    target.hash(reader.id()).await.unwrap(),
                    source.hash(reader.id()).await.unwrap()
                );

                target.rollback(&reader.id()).await.unwrap();
            }
        }
    }
}

async fn tied_keys() {
    for size in [64, 256, 1024] {
        let (_root, txn) = TestTxn::owned(
            storage_id(1),
            RESIDENT_CACHE_BYTES,
            MATERIALIZED_TENSOR_BYTES,
        );

        let count = Arc::new(AtomicUsize::new(0));
        let native = b_tree::BTreeLock::create(
            tc_collection::btree::BTreeSchema::try_from(vec![
                tc_collection::btree::BTreeColumnSchema {
                    name: "key".into(),
                    dtype: tc_value::ValueType::Number(tc_value::NumberType::Number),
                    max_size: None,
                },
            ])
            .unwrap(),
            Counted {
                inner: ValueCollator::default(),
                comparisons: count.clone(),
            },
            txn.context().await.unwrap(),
        )
        .await
        .unwrap();

        // Every value has infinite magnitude but a distinct exact identity.
        let key = |index| {
            Value::from(number_general::Number::from(number_general::Complex::from(
                [index as f64, f64::INFINITY],
            )))
        };

        native
            .write()
            .await
            .insert_sorted(stream::iter((0..size).map(|index| Ok(vec![key(index)]))))
            .await
            .unwrap();

        count.store(0, Ordering::Relaxed);
        let workload = format!("evaluation/ties/{size}");
        let start = Instant::now();
        for _ in 0..16 {
            assert!(
                native
                    .read()
                    .await
                    .get(&[key(size - 1)])
                    .await
                    .unwrap()
                    .is_some()
            );
        }

        record(&workload, "get_16", start.elapsed());
        comparisons(&workload, "get_16", &count);
        let start = Instant::now();
        native.validate().await.unwrap();
        record(&workload, "validate", start.elapsed());
        comparisons(&workload, "validate", &count);
    }
}

async fn unions() {
    for inputs in [1, 16, 256, 1024] {
        let count = Arc::new(AtomicUsize::new(0));
        let collator = Counted {
            inner: collate::Collator::<usize>::default(),
            comparisons: count.clone(),
        };

        let streams = (0..inputs)
            .map(|input| {
                stream::iter(
                    (0..32).map(move |index| Ok::<_, std::io::Error>(index * inputs + input)),
                )
            })
            .collect();

        let workload = format!("evaluation/union/{inputs}");
        let start = Instant::now();
        let mut merged = Box::pin(collate::try_union(collator, streams));
        let mut expected = 0;
        while let Some(key) = merged.try_next().await.unwrap() {
            assert_eq!(key, expected);
            expected += 1;
        }

        assert_eq!(expected, inputs * 32);
        record(&workload, "union", start.elapsed());
        comparisons(&workload, "union", &count);
    }
}

async fn tensor_history() {
    for depth in [0, 16, 128, 512] {
        let (_root, txn) = TestTxn::owned(
            storage_id(1),
            RESIDENT_CACHE_BYTES,
            MATERIALIZED_TENSOR_BYTES,
        );

        let schema = TensorSchema::new(
            NumberType::Float(number_general::FloatType::F64),
            vec![1024].into(),
            Some(0),
        )
        .unwrap();

        let tensor = Tensor::<TestTxn>::create(txn.context().await.unwrap(), schema, 1)
            .await
            .unwrap();

        for version in 0..depth {
            let writer = txn.at(version + 2);
            tensor
                .write_value(&writer, &[u64::from(version)], 1.0.into())
                .await
                .unwrap();

            tensor.commit(writer.id()).await.unwrap();
        }

        let reader = txn.at(depth + 3);
        let start = Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let mut entries = tensor.sparse_elements(reader.id()).await.unwrap();
            let mut count = 0;
            while entries.try_next().await.unwrap().is_some() {
                count += 1;
            }

            assert_eq!(count, depth);
        })
        .await;

        let workload = format!("evaluation/tensor_history/{depth}");
        record(&workload, "occupied_scan", start.elapsed());
        println!(
            "PERSISTENCE,{workload},occupied_scan,warm,timed_out,count,{}",
            usize::from(result.is_err())
        );
    }
}
