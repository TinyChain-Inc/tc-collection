use futures::TryStreamExt;
use tc_error::TCResult;
use tc_ir::{Transaction, TxnId};

use super::{Delta, PersistentTable, Range, TableFile, TableSchema, upsert};
use crate::btree::StorageConfig;
use crate::table::Column;
use crate::test::tensor::{Txn, setup};
use crate::{CopyFrom, Restore, StorageContext};

impl<Txn: StorageContext> PersistentTable<Txn> {
    pub(in crate::table) fn finalized(&self) -> TCResult<Option<TxnId>> {
        Ok(self.owner.read_state().finalized)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn selected_delta_baseline_and_scan_setup_survive_finalized_workspace_cleanup() {
    use futures::FutureExt;

    Box::pin(async {
        let (_, txn) = setup().await;
        let schema = TableSchema::new(
            vec![Column {
                name: "id".parse().unwrap(),
                dtype: ValueType::Number(tc_value::NumberType::Number),
            }],
            vec![Column {
                name: "value".parse().unwrap(),
                dtype: ValueType::String,
            }],
            vec![],
            StorageConfig::default(),
        )
        .unwrap();

        let table = PersistentTable::<Txn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            schema,
        )
        .await
        .unwrap();

        let writes = txn.subcontext("writes");
        table
            .upsert_row(&writes, vec![1u64.into()], vec!["original".into()])
            .await
            .unwrap();

        table.commit(txn.id()).await.unwrap();
        let later = txn.at(11);
        let snapshot = table.owner.visible_snapshot(later.id()).unwrap();
        let mut point = table.owner.point_read(later.id()).unwrap();
        let (entered, mut observed) = tokio::sync::oneshot::channel();
        let (resume, paused) = tokio::sync::oneshot::channel();
        let mut pause = Some((entered, paused));
        let mut read = Box::pin(tokio::task::unconstrained(point.resolve(|delta| {
            let pause = pause.take();
            async move {
                if let Some((entered, paused)) = pause {
                    entered.send(()).unwrap();
                    paused.await.unwrap();
                }

                delta.lookup(&[1u64.into()]).await
            }
        })));

        assert!(read.as_mut().now_or_never().is_none());
        observed.try_recv().unwrap();
        table.finalize(txn.id()).await.unwrap();
        writes
            .context()
            .await
            .unwrap()
            .write()
            .await
            .truncate_and_sync()
            .await
            .unwrap();

        resume.send(()).unwrap();
        assert!(read.await.unwrap().is_none());
        assert_eq!(
            point
                .canonical()
                .unwrap()
                .get_row(&[1u64.into()])
                .await
                .unwrap()
                .unwrap()[1],
            Value::from("original")
        );

        let rows: Vec<_> = table
            .visible_rows(snapshot, b_table::Range::default(), &[], false)
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][1], Value::from("original"));
        table
            .upsert_row(&later, vec![1u64.into()], vec!["updated".into()])
            .await
            .unwrap();

        let pending = table
            .owner
            .read_state()
            .pending
            .get(&later.id())
            .unwrap()
            .clone();

        assert_eq!(
            pending
                .deletes
                .get_row(&[1u64.into()])
                .await
                .unwrap()
                .unwrap()[1],
            Value::from("original")
        );
    })
    .await;
}

use tc_value::{Value, ValueType};

#[tokio::test]
async fn before_after_deltas_preserve_indexes_ties_and_repeated_changes() {
    use number_general::{Complex, Number};

    use crate::Collection;
    use crate::table::Table;

    let (_, txn) = setup().await;
    let schema = TableSchema::new(
        vec![Column {
            name: "id".parse().unwrap(),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        }],
        vec![Column {
            name: "value".parse().unwrap(),
            dtype: ValueType::String,
        }],
        vec![("by_value".into(), vec!["value".parse().unwrap()])],
        StorageConfig {
            order: 4,
            ..StorageConfig::default()
        },
    )
    .unwrap();

    let table =
        PersistentTable::<Txn>::try_new(txn.subcontext_unique().context().await.unwrap(), schema)
            .await
            .unwrap();

    let a = Value::from(Number::from(Complex::from([4f64, 3.])));
    let b = Value::from(Number::from(Complex::from([3f64, 4.])));
    for key in [a.clone(), b.clone()] {
        table
            .upsert_row(&txn, vec![key], vec!["old".into()])
            .await
            .unwrap();
    }

    table.commit(txn.id()).await.unwrap();
    table.finalize(txn.id()).await.unwrap();
    let writer = txn.at(11);
    table
        .upsert_row(&writer, vec![a.clone()], vec!["new".into()])
        .await
        .unwrap();

    table
        .upsert_row(&writer, vec![a.clone()], vec!["newer".into()])
        .await
        .unwrap();

    table.delete_row(&writer, vec![a.clone()]).await.unwrap();
    table
        .insert_row(&writer, vec![a.clone()], vec!["final".into()])
        .await
        .unwrap();

    let fresh = Value::from(9u64);
    table
        .upsert_row(&writer, vec![fresh.clone()], vec!["temporary".into()])
        .await
        .unwrap();

    table.delete_row(&writer, vec![fresh]).await.unwrap();
    table
        .update(
            &writer,
            Range::default(),
            std::iter::once(("value".parse().unwrap(), Value::from("bulk"))).collect(),
        )
        .await
        .unwrap();

    let order = vec!["value".parse().unwrap()];
    for reverse in [false, true] {
        let rows = table
            .rows(writer.id(), Range::default(), order.clone(), reverse)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let expected = if reverse {
            vec![a.clone(), b.clone()]
        } else {
            vec![b.clone(), a.clone()]
        };

        assert_eq!(
            rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            expected
        );

        assert!(rows.iter().all(|row| row[1] == Value::from("bulk")));
    }

    for value in ["old", "new", "newer", "final", "temporary"] {
        let range = std::iter::once((
            "value".parse().unwrap(),
            b_table::ColumnRange::Eq(Value::from(value)),
        ))
        .collect::<std::collections::HashMap<_, _>>()
        .into();

        assert!(
            table
                .rows(writer.id(), range, order.clone(), false)
                .await
                .unwrap()
                .try_next()
                .await
                .unwrap()
                .is_none()
        );
    }

    let delta = table
        .owner
        .read_state()
        .pending
        .get(&writer.id())
        .unwrap()
        .clone();

    assert_eq!(
        delta
            .deletes
            .get_row(std::slice::from_ref(&a))
            .await
            .unwrap()
            .unwrap()[1],
        Value::from("old")
    );

    let collection = Collection::from(table.clone());
    let hash = collection.hash(writer.id()).await.unwrap();
    let copy = crate::table::PersistentTable::copy_from(
        &writer,
        writer.subcontext_unique().context().await.unwrap(),
        &Table::from(table.clone()),
    )
    .await
    .unwrap();

    assert_eq!(
        Collection::from(copy).hash(writer.id()).await.unwrap(),
        hash
    );

    table.commit(writer.id()).await.unwrap();
    table.finalize(writer.id()).await.unwrap();
    assert_eq!(collection.hash(txn.at(12).id()).await.unwrap(), hash);
    let native = table.owner.read_state().persistent.clone();
    native.validate().await.unwrap();
    table
        .upsert_row(&txn.at(12), vec![Value::from(f64::NAN)], vec!["nan".into()])
        .await
        .unwrap();

    assert!(
        table
            .read_row(txn.at(12).id(), &[Value::from(f64::NAN)])
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn interrupted_table_mutation_is_discarded_by_rollback() {
    Box::pin(async {
        for cancel in [false, true] {
            let (_, txn) = setup().await;
            let column = |name: &str| Column {
                name: name.parse().unwrap(),
                dtype: ValueType::Number(tc_value::NumberType::Number),
            };

            let schema = TableSchema::new(
                vec![column("id")],
                vec![column("value")],
                vec![("by_value".into(), vec!["value".parse().unwrap()])],
                StorageConfig::default(),
            )
            .unwrap();

            let table = PersistentTable::<Txn>::try_new(
                txn.subcontext_unique().context().await.unwrap(),
                schema,
            )
            .await
            .unwrap();

            let workspace = txn.subcontext_unique().context().await.unwrap();
            let delta = Delta::create(table.schema.clone(), workspace.clone())
                .await
                .unwrap();

            table
                .owner
                .write_state()
                .pending
                .insert(txn.id(), delta.clone());

            // The original row must survive a failed or cancelled replacement.
            delta
                .add_to_deletes(vec![1u64.into()], vec![2u64.into()])
                .await
                .unwrap();

            let snapshot = table.owner.visible_snapshot(txn.id()).unwrap();
            if cancel {
                let blocked = delta.inserts.write().await;
                let mut update =
                    Box::pin(table.upsert_row(&txn, vec![1u64.into()], vec![3u64.into()]));

                assert!(futures::poll!(&mut update).is_pending());

                drop(update);
                drop(blocked);
            } else {
                upsert(&delta.inserts, vec![1u64.into()], vec![2u64.into()])
                    .await
                    .unwrap();

                let inserts = workspace.read().await.get_dir("inserts").unwrap().clone();
                let auxiliary = inserts.read().await.get_dir("by_value").unwrap().clone();
                let root = auxiliary
                    .read()
                    .await
                    .get_file("00000000-0000-0000-0000-000000000000")
                    .unwrap()
                    .clone();

                *root.write::<b_table::Node<Value>>().await.unwrap() =
                    b_tree::Node::Leaf(Vec::new());

                assert!(
                    table
                        .upsert_row(&txn, vec![1u64.into()], vec![3u64.into()])
                        .await
                        .is_err()
                );

                assert!(delta.already_deleted(&[1u64.into()]).await.unwrap());
            }

            drop(snapshot); // Do not reuse results from the interrupted transaction.
            assert!(delta.inserts.try_write().is_ok());
            table.rollback(txn.id()).unwrap();
            assert!(
                table
                    .read_row(txn.id(), &[1u64.into()])
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn bulk_source_error_after_rows_requires_rollback() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let column = |name: &str| Column {
            name: name.parse().unwrap(),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        };

        let schema = TableSchema::new(
            vec![column("id")],
            vec![column("value")],
            Vec::new(),
            StorageConfig {
                block_size: 4096,
                order: 4,
            },
        )
        .unwrap();

        let dir = txn.subcontext_unique().context().await.unwrap();
        let table = PersistentTable::<Txn>::try_new(dir.clone(), schema)
            .await
            .unwrap();

        let canonical = table.owner.read_state().persistent.clone();
        for key in 0..8u64 {
            upsert(&canonical, vec![key.into()], vec![0u64.into()])
                .await
                .unwrap();
        }

        let primary = dir.read().await.get_dir("primary").unwrap().clone();
        let root = primary
            .read()
            .await
            .get_file("00000000-0000-0000-0000-000000000000")
            .unwrap()
            .clone();

        let missing = match &*root.read::<b_table::Node<Value>>().await.unwrap() {
            b_tree::Node::Index(_, children) => children.last().unwrap().to_string(),
            _ => panic!("multiple source leaves"),
        };

        assert!(primary.write().await.delete(&missing).await);
        let mut values = tc_ir::Map::new();
        values.insert("value".parse().unwrap(), 9u64.into());
        assert!(table.update(&txn, Range::default(), values).await.is_err());
        assert!(
            table.owner.state.read().unwrap().pending.is_empty(),
            "source failure must leave pending membership unchanged"
        );

        let root = txn.context().await.unwrap();
        let workspaces: Vec<_> = root
            .read()
            .await
            .iter()
            .filter_map(|(_, entry)| entry.as_dir().cloned())
            .collect();

        let mut staged_rows = 0;
        for workspace in workspaces {
            let inserts = workspace.read().await.get_dir("inserts").cloned();
            if let Some(inserts) = inserts {
                let staged = TableFile::load(
                    table.schema.clone(),
                    crate::persistence::value_order(),
                    inserts,
                )
                .unwrap();

                staged_rows += staged.read().await.count(Range::default()).await.unwrap();
            }
        }

        assert!(
            staged_rows > 0,
            "earlier rows remain staged in delegated storage"
        );

        table.rollback(txn.id()).unwrap();
    })
    .await;
}

#[tokio::test]
async fn equal_norm_keys_preserve_identity_and_range_selection() {
    use collate::Collate;
    use number_general::{Complex, Number};
    use tc_value::ValueCollator;

    let (_, txn) = setup().await;
    let schema = TableSchema::new(
        vec![Column {
            name: "id".parse().unwrap(),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        }],
        vec![Column {
            name: "value".parse().unwrap(),
            dtype: ValueType::String,
        }],
        vec![],
        StorageConfig::default(),
    )
    .unwrap();

    let table =
        PersistentTable::<Txn>::try_new(txn.subcontext_unique().context().await.unwrap(), schema)
            .await
            .unwrap();

    let native = table.owner.read_state().persistent.clone();
    {
        let _guard = native.write().await;
        let invalid = native.get_row(&[]);
        let error = tokio::time::timeout(std::time::Duration::from_secs(1), invalid)
            .await
            .unwrap()
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    let left = Value::from(Number::from(Complex::from([3_f64, 4_f64])));
    let right = Value::from(Number::from(Complex::from([4_f64, 3_f64])));
    assert_ne!(left, right);
    assert_eq!(
        ValueCollator::default().cmp(&left, &right),
        std::cmp::Ordering::Less
    );

    table
        .upsert_row(&txn, vec![left.clone()], vec!["left".into()])
        .await
        .unwrap();

    assert!(
        table
            .read_row(txn.id(), std::slice::from_ref(&right))
            .await
            .unwrap()
            .is_none()
    );

    table
        .upsert_row(&txn, vec![right.clone()], vec!["right".into()])
        .await
        .unwrap();

    for (key, value) in [(left.clone(), "left"), (right.clone(), "right")] {
        let row = table
            .read_row(txn.id(), std::slice::from_ref(&key))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(row.as_slice(), &[key, Value::from(value)]);
    }

    for key in [&left, &right] {
        let range: Range<_, _> = [("id".parse().unwrap(), b_table::ColumnRange::Eq(key.clone()))]
            .into_iter()
            .collect();

        let selected = table
            .rows(txn.id(), range, Vec::new(), false)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(selected.len(), 1);
        assert_eq!(&selected[0][0], key);
    }

    table.commit(txn.id()).await.unwrap();
    table.finalize(txn.id()).await.unwrap();
    let later = txn.at(11);
    table
        .upsert_row(&later, vec![left.clone()], vec!["new-left".into()])
        .await
        .unwrap();

    for reverse in [false, true] {
        let rows = table
            .rows(later.id(), Range::default(), Vec::new(), reverse)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter()
                .any(|row| row.as_slice() == [left.clone(), Value::from("new-left")])
        );

        assert!(
            rows.iter()
                .any(|row| row.as_slice() == [right.clone(), Value::from("right")])
        );
    }

    table.delete_row(&later, vec![right.clone()]).await.unwrap();
    let rows = table
        .rows(later.id(), Range::default(), Vec::new(), false)
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].as_slice(), &[left, Value::from("new-left")]);
    table.commit(later.id()).await.unwrap();
    table.finalize(later.id()).await.unwrap();
    assert!(
        table
            .read_row(txn.at(12).id(), &[right])
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn deep_visible_versions_use_normal_worker_stacks() {
    use tc_ir::{NetworkTime, TxnId};

    for phase in ["drop", "poll"] {
        if !crate::test::depth::child(
            concat!(
                module_path!(),
                "::deep_visible_versions_use_normal_worker_stacks"
            )
            .strip_prefix("tc_collection::")
            .unwrap(),
            phase,
        )
        .await
        {
            continue;
        }

        tokio::spawn(async move {
            for occupied in [false, true] {
                let root = crate::test::TempRoot::new("overlay-depth");
                // Two streams per retained version; admission must not mask stack depth.
                let cache = freqfs::Cache::<crate::PersistentFile>::new(
                    64 * 1024 * 1024,
                    Some(4096),
                    0,
                    std::time::Duration::from_secs(3),
                );

                let txn = Txn::new(
                    TxnId::from_parts(NetworkTime::from_nanos(1), 10),
                    cache.load(root.to_path_buf()).unwrap(),
                );

                let schema = TableSchema::new(
                    vec![Column {
                        name: "id".parse().unwrap(),
                        dtype: ValueType::Number(tc_value::NumberType::Number),
                    }],
                    vec![Column {
                        name: "value".parse().unwrap(),
                        dtype: ValueType::String,
                    }],
                    vec![],
                    StorageConfig::default(),
                )
                .unwrap();

                let collection = PersistentTable::<Txn>::try_new(
                    txn.subcontext_unique().context().await.unwrap(),
                    schema,
                )
                .await
                .unwrap();

                let delta = collection.pending_delta_for_txn(&txn).await.unwrap();
                if occupied {
                    delta
                        .upsert(vec![1_u64.into()], vec!["retained".into()])
                        .await
                        .unwrap();
                }

                {
                    let mut state = collection.owner.write_state();
                    state.pending.clear();
                    // Reuse native fixtures to measure version depth rather than directory setup.
                    for nonce in 0..1024 {
                        state.committed.insert(
                            TxnId::from_parts(NetworkTime::from_nanos(1), nonce),
                            Some(delta.clone()),
                        );
                    }
                }

                let read_id = TxnId::from_parts(NetworkTime::from_nanos(1), 4096);
                eprintln!("{phase}: constructing visible stream");
                let mut stream = collection
                    .rows(read_id, Range::default(), vec![], false)
                    .await
                    .unwrap();

                eprintln!("{phase}: stream constructed");
                if phase == "poll" {
                    if occupied {
                        assert_eq!(
                            stream.try_next().await.unwrap().unwrap().as_slice(),
                            &[1_u64.into(), Value::from("retained")]
                        );
                    }

                    assert!(stream.try_next().await.unwrap().is_none());
                }

                drop(stream);
                eprintln!("{phase}: stream released");
            }
        })
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn point_lookup_skips_shadowed_corruption() {
    Box::pin(async {
        let (_root, txn) = setup().await;
        let column = |name: &str| Column {
            name: name.parse().unwrap(),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        };

        let schema = TableSchema::new(
            vec![column("id")],
            vec![column("value")],
            vec![],
            StorageConfig::default(),
        )
        .unwrap();

        let dir = txn.subcontext_unique().context().await.unwrap();
        let table = PersistentTable::<Txn>::try_new(dir.clone(), schema)
            .await
            .unwrap();

        let canonical = table.owner.read_state().persistent.clone();
        upsert(&canonical, vec![0u64.into()], vec![0u64.into()])
            .await
            .unwrap();

        assert!(
            table
                .read_row(txn.id(), &[0u64.into()])
                .await
                .unwrap()
                .is_some()
        );

        assert!(
            table
                .read_row(txn.id(), &[9u64.into()])
                .await
                .unwrap()
                .is_none()
        );

        table.delete_row(&txn, vec![0u64.into()]).await.unwrap();
        table
            .insert_row(&txn, vec![1u64.into()], vec![1u64.into()])
            .await
            .unwrap();

        table.commit(txn.id()).await.unwrap();
        let primary = dir.read().await.get_dir("primary").unwrap().clone();
        assert!(
            primary
                .write()
                .await
                .delete("00000000-0000-0000-0000-000000000000")
                .await
        );

        let later = txn.at(11);
        // Mutation lookup must also stop at the newer committed deletion.
        table
            .insert_row(&later, vec![0u64.into()], vec![2u64.into()])
            .await
            .unwrap();

        assert_eq!(
            table
                .read_row(later.id(), &[0u64.into()])
                .await
                .unwrap()
                .unwrap()
                .as_slice(),
            &[Value::from(0u64), Value::from(2u64)]
        );

        assert!(
            table
                .read_row(later.id(), &[1u64.into()])
                .await
                .unwrap()
                .is_some()
        );

        assert!(
            table
                .read_row(later.id(), &[2u64.into()])
                .await
                .unwrap_err()
                .requires_recovery()
        );

        table.delete_row(&later, vec![1u64.into()]).await.unwrap();
        assert!(
            table
                .read_row(later.id(), &[1u64.into()])
                .await
                .unwrap()
                .is_none()
        );

        table.rollback(later.id()).unwrap();
        assert!(
            table
                .read_row(later.id(), &[0u64.into()])
                .await
                .unwrap()
                .is_none()
        );

        assert!(
            table
                .read_row(later.id(), &[1u64.into()])
                .await
                .unwrap()
                .is_some()
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn canonical_count_yields_cooperatively() {
    use futures::FutureExt;

    Box::pin(async {
        let (_, txn) = setup().await;
        let schema = TableSchema::new(
            vec![Column {
                name: "id".parse().unwrap(),
                dtype: ValueType::Number(tc_value::NumberType::Number),
            }],
            vec![Column {
                name: "value".parse().unwrap(),
                dtype: ValueType::String,
            }],
            vec![],
            StorageConfig {
                order: 512,
                block_size: 4096,
            },
        )
        .unwrap();

        let table = PersistentTable::<Txn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            schema,
        )
        .await
        .unwrap();

        let native = table.owner.read_state().persistent.clone();
        {
            let mut write = native.write().await;
            for key in 0..256_u64 {
                write
                    .upsert(vec![key.into()], vec!["ready".into()])
                    .await
                    .unwrap();
            }
        }

        // One resident leaf keeps native I/O from supplying the scheduling yield.
        tokio::task::yield_now().await;
        let count = async { native.read().await.count(Range::default()).await };
        assert_eq!(
            count
                .now_or_never()
                .expect("resident native control is ready")
                .unwrap(),
            256
        );

        tokio::task::yield_now().await;
        let mut count = Box::pin(table.count(txn.id()));
        assert!(futures::poll!(&mut count).is_pending());
        assert_eq!(count.await.unwrap(), 256);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restoration_self_and_conflicting_sources_are_bounded() {
    use futures::FutureExt;

    Box::pin(async {
        let (_, txn) = setup().await;
        let make = |dir| async move {
            let schema = TableSchema::new(
                vec![Column {
                    name: "id".parse().unwrap(),
                    dtype: ValueType::Number(tc_value::NumberType::Number),
                }],
                vec![Column {
                    name: "label".parse().unwrap(),
                    dtype: ValueType::String,
                }],
                vec![("by_label".into(), vec!["label".parse().unwrap()])],
                StorageConfig::default(),
            )
            .unwrap();

            PersistentTable::<Txn>::try_new(dir, schema).await.unwrap()
        };

        let seed = |store: PersistentTable<Txn>, txn: Txn, key: u64| async move {
            store
                .upsert_row(&txn, vec![key.into()], vec!["row".into()])
                .await
                .unwrap();
        };

        let has = |store: PersistentTable<Txn>, txn: Txn, key: u64| async move {
            store.contains_row(txn.id(), &[key.into()]).await.unwrap()
        };

        for reverse in [false, true] {
            let left = make(txn.subcontext_unique().context().await.unwrap()).await;
            let right = make(txn.subcontext_unique().context().await.unwrap()).await;
            seed(left.clone(), txn.clone(), 1).await;
            seed(right.clone(), txn.clone(), 2).await;
            let (target, source) = if reverse {
                (&right, &left)
            } else {
                (&left, &right)
            };

            let key = if reverse { 2 } else { 1 };

            let held = target
                .rows(txn.id(), Range::default(), vec![], false)
                .await
                .unwrap();

            target
                .restore(&txn, &target.clone())
                .now_or_never()
                .expect("self-restore does not wait")
                .unwrap();

            drop(held);
            assert!(has(target.clone(), txn.clone(), key).await);

            let context = txn.context().await.unwrap();
            let context_guard = context.write().await;
            let mut restore = Box::pin(target.restore(&txn, source));
            assert!(
                tokio::task::unconstrained(restore.as_mut())
                    .now_or_never()
                    .is_none()
            );

            assert!(
                source
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .is_err(),
                "source snapshot is admitted before staging"
            );

            let reverse_result = source
                .restore(&txn, target)
                .now_or_never()
                .expect("reciprocal restore rejects without waiting");

            assert!(reverse_result.is_err());
            drop(restore);
            drop(context_guard);
            assert!(
                has(target.clone(), txn.clone(), key).await,
                "cancelled staging preserves pending data"
            );

            drop(
                target
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );

            drop(
                source
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );

            target.rollback(txn.id()).unwrap();
            source.rollback(txn.id()).unwrap();
        }

        let dir = txn.subcontext_unique().context().await.unwrap();
        let target = make(dir.clone()).await;
        let source = make(txn.subcontext_unique().context().await.unwrap()).await;
        seed(target.clone(), txn.at(9), 1).await;
        target.commit(txn.at(9).id()).await.unwrap();
        target.finalize(txn.at(9).id()).await.unwrap();
        seed(target.clone(), txn.clone(), 3).await;
        seed(source.clone(), txn.clone(), 2).await;
        target.restore(&txn, &source).await.unwrap();
        assert!(has(target.clone(), txn.clone(), 2).await);
        assert!(!has(target.clone(), txn.clone(), 3).await);
        target.rollback(txn.id()).unwrap();
        assert!(has(target.clone(), txn.clone(), 1).await);
        assert!(!has(target.clone(), txn.clone(), 2).await);
        source.commit(txn.id()).await.unwrap();
        let later = txn.at(11);
        target.restore(&later, &source).await.unwrap();
        target.commit(later.id()).await.unwrap();
        assert!(target.restore(&later, &target).await.is_err());
        target.finalize(later.id()).await.unwrap();
        assert!(target.restore(&later, &target).await.is_err());
        target.sync_all().await.unwrap();
        let path = dir.read().await.path().to_owned();
        drop(target);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            256 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let reopened =
            PersistentTable::<Txn>::load(cache.load(path).unwrap(), source.schema().clone())
                .await
                .unwrap();

        assert!(has(reopened, txn.at(12), 2).await);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn restoration_source_failure_preserves_pending_data() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let make = |dir| async move {
            PersistentTable::<Txn>::try_new(
                dir,
                TableSchema::new(
                    vec![Column {
                        name: "id".parse().unwrap(),
                        dtype: ValueType::Number(tc_value::NumberType::Number),
                    }],
                    vec![Column {
                        name: "value".parse().unwrap(),
                        dtype: ValueType::String,
                    }],
                    vec![],
                    StorageConfig::default(),
                )
                .unwrap(),
            )
            .await
            .unwrap()
        };

        let target = make(txn.subcontext_unique().context().await.unwrap()).await;
        let source_dir = txn.subcontext_unique().context().await.unwrap();
        let source = make(source_dir.clone()).await;
        target
            .upsert_row(&txn, vec![3_u64.into()], vec!["original".into()])
            .await
            .unwrap();

        let native = source_dir.read().await.get_dir("primary").unwrap().clone();
        let root = native
            .read()
            .await
            .get_file("00000000-0000-0000-0000-000000000000")
            .unwrap()
            .clone();

        *root
            .write::<b_table::Node<tc_value::Value>>()
            .await
            .unwrap() = b_tree::Node::Leaf(vec![Default::default()]);

        let error = target.restore(&txn, &source).await.unwrap_err();
        assert!(error.requires_recovery());
        assert!(
            target
                .read_row(txn.id(), &[3_u64.into()])
                .await
                .unwrap()
                .unwrap()[1]
                == tc_value::Value::from("original")
        );

        drop(root.try_write::<b_table::Node<tc_value::Value>>().unwrap());
        drop(
            source
                .owner
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        drop(
            target
                .owner
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        let selection = source.limit(1).select(vec!["id".parse().unwrap()]).unwrap();
        assert!(
            selection
                .count(txn.id())
                .await
                .unwrap_err()
                .requires_recovery()
        );

        assert!(
            selection
                .is_empty(txn.id())
                .await
                .unwrap_err()
                .requires_recovery()
        );

        assert!(
            selection
                .for_each_row_in_order(txn.id(), |_| panic!(
                    "corrupt source must not reach callback"
                ))
                .await
                .unwrap_err()
                .requires_recovery()
        );

        target.rollback(txn.id()).unwrap();
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn copying_preserves_equal_norm_keys_and_rejects_ambiguity() {
    use futures::FutureExt;
    use number_general::{Complex, Number};

    use crate::Collection;
    use crate::table::Table;

    Box::pin(async {
        let (_, txn) = setup().await;
        let column = |name: &str| Column {
            name: name.parse().unwrap(),
            dtype: ValueType::Number(tc_value::NumberType::Number),
        };

        let schema = TableSchema::new(
            vec![column("id")],
            vec![column("first"), column("second")],
            vec![],
            StorageConfig {
                order: 4,
                ..StorageConfig::default()
            },
        )
        .unwrap();

        let source = PersistentTable::<Txn>::try_new(
            txn.subcontext_unique().context().await.unwrap(),
            schema.clone(),
        )
        .await
        .unwrap();

        let keys: Vec<_> = [[3., 4.], [4., 3.], [0., 5.], [-3., 4.], [-4., 3.]]
            .into_iter()
            .map(|parts| Value::from(Number::from(Complex::from(parts))))
            .collect();

        for (i, key) in keys.iter().enumerate() {
            source
                .upsert_row(
                    &txn,
                    vec![key.clone()],
                    vec![(i as u64).into(), ((keys.len() - i) as u64).into()],
                )
                .await
                .unwrap();
        }

        let original = Collection::from(source.clone())
            .hash(txn.id())
            .await
            .unwrap();

        let view = Table::from(source.clone());
        let out = txn.subcontext_unique().context().await.unwrap();
        let copy = crate::table::PersistentTable::copy_from(&txn, out.clone(), &view)
            .await
            .unwrap();

        assert_eq!(copy.count(txn.id()).await.unwrap(), keys.len() as u64);
        assert_eq!(copy.schema(), source.schema());
        assert_eq!(
            Collection::from(copy.clone()).hash(txn.id()).await.unwrap(),
            original
        );

        for key in &keys {
            assert!(
                copy.read_row(txn.id(), std::slice::from_ref(key))
                    .await
                    .unwrap()
                    .is_some()
            );
        }

        // Distinct primary keys determine order before projected value columns.
        let reordered = source
            .select(&[
                "id".parse().unwrap(),
                "second".parse().unwrap(),
                "first".parse().unwrap(),
            ])
            .unwrap();

        let projected = crate::table::PersistentTable::copy_from(
            &txn,
            txn.subcontext_unique().context().await.unwrap(),
            &Table::from(reordered),
        )
        .await
        .unwrap();

        assert_eq!(
            projected
                .read_row(txn.id(), &keys[..1])
                .await
                .unwrap()
                .unwrap()
                .as_slice(),
            &[keys[0].clone(), (keys.len() as u64).into(), 0_u64.into()]
        );

        let reversed = source.slice(Range::default(), &[], true);

        let error = crate::table::PersistentTable::copy_from(
            &txn,
            txn.subcontext_unique().context().await.unwrap(),
            &Table::from(reversed),
        )
        .await
        .unwrap_err();

        assert!(!error.requires_recovery());
        assert!(error.to_string().contains("canonical key order"));
        assert_eq!(
            Collection::from(source.clone())
                .hash(txn.id())
                .await
                .unwrap(),
            original
        );

        let delta = source
            .owner
            .read_state()
            .pending
            .get(&txn.id())
            .unwrap()
            .clone();

        let held = delta.inserts.write().await;
        let cancelled = txn.subcontext_unique().context().await.unwrap();
        let mut copying = Box::pin(crate::table::PersistentTable::copy_from(
            &txn, cancelled, &view,
        ));

        assert!(
            tokio::task::unconstrained(copying.as_mut())
                .now_or_never()
                .is_none()
        );

        assert!(
            source
                .owner
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .is_err()
        );

        drop(copying);
        drop(held);
        drop(
            source
                .owner
                .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                .unwrap(),
        );

        source
            .upsert_row(
                &txn,
                vec![keys[0].clone()],
                vec![0_u64.into(), 100_u64.into()],
            )
            .await
            .unwrap();

        assert_eq!(
            Collection::from(copy.clone()).hash(txn.id()).await.unwrap(),
            original
        );

        copy.sync_all().await.unwrap();
        let path = out.read().await.path().to_owned();
        drop(copy);
        drop(out);
        let cache = freqfs::Cache::<crate::PersistentFile>::new(
            256 * 1024,
            None,
            0,
            std::time::Duration::from_secs(3),
        );

        let reopened = PersistentTable::<Txn>::load(cache.load(path).unwrap(), schema)
            .await
            .unwrap();

        assert_eq!(
            Collection::from(reopened).hash(txn.id()).await.unwrap(),
            original
        );

        for duplicate in [true, false] {
            let dir = txn.subcontext_unique().context().await.unwrap();
            let schema = TableSchema::new(
                vec![column("id")],
                vec![column("value")],
                vec![],
                StorageConfig::default(),
            )
            .unwrap();

            let source = PersistentTable::<Txn>::try_new(dir.clone(), schema)
                .await
                .unwrap();

            let primary = dir.read().await.get_dir("primary").unwrap().clone();
            let root = primary
                .read()
                .await
                .get_file("00000000-0000-0000-0000-000000000000")
                .unwrap()
                .clone();

            let rows = if duplicate {
                vec![
                    vec![keys[0].clone(), 1_u64.into()],
                    vec![keys[0].clone(), 3_u64.into()],
                ]
            } else {
                vec![Vec::new()]
            };

            *root.write::<b_table::Node<Value>>().await.unwrap() = b_tree::Node::Leaf(rows.clone());
            let error = crate::table::PersistentTable::copy_from(
                &txn,
                txn.subcontext_unique().context().await.unwrap(),
                &Table::from(source.clone()),
            )
            .await
            .unwrap_err();

            assert!(error.requires_recovery());
            if duplicate {
                assert!(error.to_string().contains("ambiguous Table primary key"));
            }

            let guard = root.try_write::<b_table::Node<Value>>().unwrap();
            let b_tree::Node::Leaf(actual) = &*guard else {
                panic!("leaf fixture")
            };

            assert_eq!(actual, &rows);
            drop(guard);
            drop(
                source
                    .owner
                    .try_acquire_write_permit(txn.id(), txn_lock::set::Range::All)
                    .unwrap(),
            );
        }
    })
    .await;
}
