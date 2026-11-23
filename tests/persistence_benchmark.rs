//! Opt-in measurements over the production collection implementation.
use std::time::{Duration, Instant};

use futures::{StreamExt, TryStreamExt};

use tc_collection::btree::{BTreeColumnSchema, StorageConfig};
use tc_collection::collection::CollectionSchema;
use tc_collection::table::{Column, Table, TableSchema};
use tc_collection::tensor::{Tensor, TensorSchema};
use tc_collection::{Collection, PersistentFile, StorageContext};
use tc_ir::{Transact, Transaction};
use tc_value::{NumberType, ValueType};

#[path = "common/storage.rs"]
mod storage;
use storage::TestTxn;

#[path = "benchmark/simplify.rs"]
mod simplify;

fn record(workload: &str, operation: &str, elapsed: Duration) {
    println!(
        "PERSISTENCE,{workload},{operation},warm,elapsed,ns,{}",
        elapsed.as_nanos()
    );
}

fn selected(name: &str) -> bool {
    std::env::var("PERSISTENCE_BENCH_SUITE").map_or(true, |suite| suite == "all" || suite == name)
}

async fn create(txn: &TestTxn, table: bool) -> Collection<TestTxn> {
    let dtype = ValueType::Number(NumberType::Number);
    let schema = if table {
        CollectionSchema::Table(
            TableSchema::new(
                vec![Column {
                    name: "key".parse().unwrap(),
                    dtype: dtype.clone(),
                }],
                vec![Column {
                    name: "value".parse().unwrap(),
                    dtype,
                }],
                vec![("by_value".parse().unwrap(), vec!["value".parse().unwrap()])],
                StorageConfig::default(),
            )
            .unwrap(),
        )
    } else {
        CollectionSchema::BTree(vec![BTreeColumnSchema {
            name: "key".into(),
            dtype,
            max_size: None,
        }])
    };

    Collection::create(txn.subcontext_unique().context().await.unwrap(), schema)
        .await
        .unwrap()
}

async fn insert(collection: &Collection<TestTxn>, txn: &TestTxn, key: u64) {
    match collection {
        Collection::BTree(tree) => tree
            .persistent()
            .unwrap()
            .insert_row(txn, vec![key.into()])
            .await
            .unwrap(),
        Collection::Table(table) => {
            let Table::File(table) = table.as_ref() else {
                unreachable!()
            };

            table
                .upsert_row(txn, vec![key.into()], vec![key.into()])
                .await
                .unwrap();
        }
        Collection::Tensor(_) => unreachable!(),
    }
}

async fn contains(collection: &Collection<TestTxn>, txn: &TestTxn, key: u64) -> bool {
    match collection {
        Collection::BTree(tree) => tree.contains_row(txn.id(), &[key.into()]).await.unwrap(),
        Collection::Table(table) => {
            let Table::File(table) = table.as_ref() else {
                unreachable!()
            };

            table
                .read_row(txn.id(), &[key.into()])
                .await
                .unwrap()
                .is_some()
        }
        Collection::Tensor(_) => unreachable!(),
    }
}

async fn count(collection: &Collection<TestTxn>, txn: &TestTxn) -> usize {
    match collection {
        Collection::BTree(tree) => tree
            .keys(txn.id())
            .await
            .unwrap()
            .try_fold(0, |count, _| async move { Ok(count + 1) })
            .await
            .unwrap(),
        Collection::Table(table) => table
            .row_stream(txn.id())
            .await
            .unwrap()
            .try_fold(0, |count, _| async move { Ok(count + 1) })
            .await
            .unwrap(),
        Collection::Tensor(_) => unreachable!(),
    }
}

async fn history() {
    for table in [false, true] {
        for versions in [0, 1, 16, 128, 1024, 4096] {
            let (_root, txn) = TestTxn::owned(storage_id(1), 64 * 1024 * 1024, 64 * 1024 * 1024);
            let collection = create(&txn, table).await;
            for key in 0..32 {
                insert(&collection, &txn, key).await;
            }

            collection.commit(txn.id()).await.unwrap();
            collection.finalize(&txn.id()).await.unwrap();
            let writes = Instant::now();
            for version in 0..versions {
                let writer = txn.at(version + 2);
                insert(&collection, &writer, u64::from(version) + 32).await;
                collection.commit(writer.id()).await.unwrap();
            }

            let reader = txn.at(versions + 3);
            let workload = format!(
                "history/{}/{versions}",
                if table { "table" } else { "btree" }
            );

            record(&workload, "retained_writes", writes.elapsed());
            for (operation, key, expected) in [
                ("point_newest", u64::from(versions) + 31, true),
                ("point_absent", u64::MAX, false),
            ] {
                let start = Instant::now();
                for _ in 0..16 {
                    assert_eq!(contains(&collection, &reader, key).await, expected);
                }

                record(&workload, operation, start.elapsed());
            }

            let start = Instant::now();
            let result =
                tokio::time::timeout(Duration::from_secs(30), count(&collection, &reader)).await;

            println!(
                "PERSISTENCE,{workload},scan,warm,timed_out,count,{}",
                usize::from(result.is_err())
            );

            if let Ok(actual) = result {
                assert_eq!(actual, usize::from(versions) + 32);
            }

            record(&workload, "scan", start.elapsed());
            if let Collection::Table(table) = &collection {
                let Table::File(table) = table.as_ref() else {
                    unreachable!()
                };

                let start = Instant::now();
                let result = tokio::time::timeout(Duration::from_secs(30), async {
                    let mut rows = table
                        .rows(
                            reader.id(),
                            Default::default(),
                            vec!["value".parse().unwrap()],
                            false,
                        )
                        .await
                        .unwrap();

                    let mut actual = 0;
                    while rows.try_next().await.unwrap().is_some() {
                        actual += 1;
                    }

                    actual
                })
                .await;

                println!(
                    "PERSISTENCE,{workload},secondary_scan,warm,timed_out,count,{}",
                    usize::from(result.is_err())
                );

                if let Ok(actual) = result {
                    assert_eq!(actual, usize::from(versions) + 32);
                }

                record(&workload, "secondary_scan", start.elapsed());
            }
        }
    }
}

async fn load() {
    for table in [false, true] {
        let (_work, txn) = TestTxn::owned(storage_id(1), 64 * 1024 * 1024, 64 * 1024 * 1024);
        let root = storage::TempRoot::new("cold-data");
        let (schema, released) = {
            let cache = freqfs::Cache::<PersistentFile>::new(
                64 * 1024 * 1024,
                None,
                0,
                Duration::from_secs(3),
            );

            let native = cache.clone().load(root.to_path_buf()).unwrap();
            let source = create(&txn, table).await;
            for key in 0..4096 {
                insert(&source, &txn, key).await;
            }

            source.commit(txn.id()).await.unwrap();
            let stored = source.materialize(&txn.at(2), native).await.unwrap();
            stored.sync_all().await.unwrap();
            (stored.schema().unwrap(), std::sync::Arc::downgrade(&cache))
        };

        tokio::time::timeout(Duration::from_secs(3), async {
            while released.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let start = Instant::now();
        let cache =
            freqfs::Cache::<PersistentFile>::new(64 * 1024 * 1024, None, 0, Duration::from_secs(3));

        let loaded = Collection::<TestTxn>::load(cache.load(root.to_path_buf()).unwrap(), schema)
            .await
            .unwrap();

        let workload = format!("load/{}", if table { "table" } else { "btree" });
        record(&workload, "cold_load", start.elapsed());
        let start = Instant::now();
        insert(&loaded, &txn.at(3), 4096).await;
        record(&workload, "mutate_loaded", start.elapsed());
        assert_eq!(count(&loaded, &txn.at(3)).await, 4097);
    }
}

async fn slow_reader() {
    for table in [false, true] {
        for partial in [false, true] {
            let (_root, txn) = TestTxn::owned(storage_id(1), 64 * 1024 * 1024, 64 * 1024 * 1024);
            let collection = create(&txn, table).await;
            for key in 0..32 {
                insert(&collection, &txn, key).await;
            }

            collection.commit(txn.id()).await.unwrap();
            collection.finalize(&txn.id()).await.unwrap();
            insert(&collection, &txn.at(2), 32).await;
            collection.commit(txn.at(2).id()).await.unwrap();
            let mut rows: futures::stream::BoxStream<'static, tc_error::TCResult<()>> =
                match &collection {
                    Collection::BTree(tree) => tree
                        .keys(txn.at(3).id())
                        .await
                        .unwrap()
                        .map_ok(|_| ())
                        .boxed(),
                    Collection::Table(table) => table
                        .row_stream(txn.at(3).id())
                        .await
                        .unwrap()
                        .map_ok(|_| ())
                        .map_err(Into::into)
                        .boxed(),
                    Collection::Tensor(_) => unreachable!(),
                };

            if partial {
                rows.try_next().await.unwrap().unwrap();
            }

            let cutoff = txn.at(2).id();
            let finalizing = collection.finalize(&cutoff);
            tokio::pin!(finalizing);
            let start = Instant::now();
            let blocked = match futures::poll!(&mut finalizing) {
                std::task::Poll::Pending => {
                    drop(rows);
                    tokio::time::timeout(Duration::from_secs(3), finalizing)
                        .await
                        .unwrap()
                        .unwrap();

                    true
                }
                std::task::Poll::Ready(result) => {
                    result.unwrap();
                    drop(rows);
                    false
                }
            };

            println!(
                "PERSISTENCE,slow_reader/{}/{partial},release_finalize,warm,blocked,count,{}",
                if table { "table" } else { "btree" },
                usize::from(blocked)
            );

            record(
                &format!(
                    "slow_reader/{}/{partial}",
                    if table { "table" } else { "btree" }
                ),
                "release_finalize",
                start.elapsed(),
            );
        }
    }
}

fn storage_id(n: u16) -> tc_ir::TxnId {
    tc_ir::TxnId::from_parts(tc_ir::NetworkTime::from_nanos(1), n)
}

async fn capture() {
    for table in [false, true] {
        let (_root, txn) = TestTxn::owned(storage_id(1), 64 * 1024 * 1024, 64 * 1024 * 1024);
        let collection = create(&txn, table).await;
        for key in 0..4096 {
            insert(&collection, &txn, key).await;
        }

        collection.commit(txn.id()).await.unwrap();
        collection.finalize(&txn.id()).await.unwrap();
        let reader = txn.at(2);
        measure_capture(&collection, &reader, if table { "table" } else { "btree" }).await;
    }

    for sparse in [false, true] {
        for exponent in [14, 18, 22] {
            let size = 1_u64 << exponent;
            let (_root, txn) = TestTxn::owned(storage_id(1), 64 * 1024 * 1024, 64 * 1024 * 1024);
            let schema = TensorSchema::new(
                NumberType::Float(number_general::FloatType::F64),
                vec![size].into(),
                sparse.then_some(0),
            )
            .unwrap();

            let tensor = Tensor::<TestTxn>::create(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
                fensor::MAX_BLOCK_CAPACITY,
            )
            .await
            .unwrap();

            if sparse {
                for index in 0..16 {
                    tensor
                        .write_value(&txn, &[index * (size / 16)], 1.0.into())
                        .await
                        .unwrap();
                }
            } else {
                tensor.fill(&txn, 1.0.into()).await.unwrap();
            }

            tensor.commit(txn.id()).await.unwrap();
            tensor.finalize(&txn.id()).await.unwrap();
            let collection = Collection::Tensor(tensor);
            measure_capture(
                &collection,
                &txn.at(2),
                &format!("tensor/{}/{size}", if sparse { "sparse" } else { "dense" }),
            )
            .await;
        }
    }
}

async fn measure_capture(source: &Collection<TestTxn>, txn: &TestTxn, workload: &str) {
    let destination = txn.subcontext_unique().context().await.unwrap();
    let start = Instant::now();
    let captured = source.materialize(txn, destination).await.unwrap();
    record(workload, "materialize", start.elapsed());
    let start = Instant::now();
    let hash = captured.hash(txn.id()).await.unwrap();
    record(workload, "hash", start.elapsed());
    assert_eq!(hash, source.hash(txn.id()).await.unwrap());
}

#[test]
#[ignore = "opt-in paired performance measurements"]
fn benchmark() {
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(Box::pin(async {
            if selected("history") {
                history().await;
            }

            if selected("capture") {
                capture().await;
            }

            if selected("load") {
                load().await;
            }

            if selected("slow_reader") {
                slow_reader().await;
            }

            if selected("evaluation") {
                simplify::run().await;
            }
        }));
}
