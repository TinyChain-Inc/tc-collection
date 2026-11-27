async fn decode_entry(entry: &super::PersistentFile) -> super::PersistentFile {
    tbon::de::try_decode((), tbon::en::encode(entry).unwrap())
        .await
        .unwrap()
}

use super::*;

#[tokio::test]
async fn decoded_native_blocks_do_not_grow_retained_footprints() {
    use get_size::GetSize;

    for count in [1, 3, 5, 9, 17] {
        for entry in [
            PersistentFile::U64(vec![7; count]),
            PersistentFile::Node(b_tree::Node::Leaf(vec![
                vec![tc_value::Value::from("text")];
                count
            ])),
        ] {
            let decoded = decode_entry(&entry).await;
            assert!(decoded.get_size() <= entry.get_size());
        }
    }
}

#[tokio::test]
async fn generic_node_bounds_follow_rows_without_tensor_limits() {
    use tc_value::Value;

    for (width, count) in [
        (0, 0),
        (1, 1),
        (2, 2),
        (4, 4),
        (5, 5),
        (fensor::MAX_BLOCK_CAPACITY + 1, 1),
    ] {
        let row = vec![Value::from(vec![Value::from("nested"), Value::from(3u64)]); width];
        let rows = vec![row; count];
        decode_entry(&PersistentFile::Node(b_tree::Node::Leaf(rows.clone()))).await;
        if count != 0 {
            let node = b_tree::Node::Index(rows, vec![Default::default(); count + 1]);
            decode_entry(&PersistentFile::Node(node)).await;
        }
    }
}

#[tokio::test]
async fn buffered_payload_saves_flush_exact_bytes_and_report_errors() {
    use futures::TryStreamExt;

    let path = std::env::temp_dir().join(format!(
        "tc-payload-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    for len in [1, fensor::MAX_BLOCK_CAPACITY] {
        let entry = PersistentFile::F64(vec![1.; len]);
        decode_entry(&entry).await;
        let expected = tbon::en::encode(&entry)
            .unwrap()
            .try_fold(Vec::new(), |mut bytes, chunk| async move {
                bytes.extend_from_slice(&chunk);
                Ok(bytes)
            })
            .await
            .unwrap();

        let mut file = tokio::fs::File::create(&path).await.unwrap();
        assert_eq!(entry.save(&mut file).await.unwrap(), expected.len() as u64);
        assert_eq!(tokio::fs::read(&path).await.unwrap(), expected);
    }

    let mut readonly = tokio::fs::File::open(&path).await.unwrap();
    assert!(
        PersistentFile::F64(vec![1.])
            .save(&mut readonly)
            .await
            .is_err()
    );

    drop(readonly);
    tokio::fs::remove_file(path).await.unwrap();
}

#[tokio::test]
async fn tensor_files_preserve_dtype_tags_and_bound_decoding() {
    let extra = (21u8, (vec![4u64], true, None::<u64>, 1u64, 0u64));
    assert!(
        tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(extra).unwrap())
            .await
            .is_err()
    );

    let metadata = (21u8, (vec![4u64], true, None::<u64>, 1u64));
    let decoded: PersistentFile = tbon::de::try_decode((), tbon::en::encode(metadata).unwrap())
        .await
        .unwrap();

    assert!(AsType::<fensor::TensorMetadata<f64>>::as_type(&decoded).is_some());
    decode_entry(&decoded).await;

    async fn assert_tag(tag: u8, entry: PersistentFile) {
        let (actual, _): (u8, de::IgnoredAny) =
            tbon::de::try_decode((), tbon::en::encode(&entry).unwrap())
                .await
                .unwrap();

        assert_eq!(actual, tag);
        assert_eq!(
            std::mem::discriminant(&decode_entry(&entry).await),
            std::mem::discriminant(&entry)
        );
    }

    // Expected tags are the persisted wire contract, independent of the catalog.
    macro_rules! check_dtype {
        ($dense:ident, $metadata:ident, $sparse:ident; $tag:literal, $meta_tag:literal, $sparse_tag:literal) => {
            assert_tag($tag, PersistentFile::$dense(vec![Default::default()])).await;
            assert_tag(
                $meta_tag,
                PersistentFile::$metadata(
                    fensor::TensorMetadata::new(vec![1].into(), fensor::Layout::Dense, 1).unwrap(),
                ),
            )
            .await;

            assert_tag(
                $sparse_tag,
                PersistentFile::$sparse(
                    b_tree::Node::Leaf(vec![vec![
                        fensor::SparseCell::Key(0),
                        fensor::SparseCell::Payload(vec![Default::default()]),
                    ]])
                    .into(),
                ),
            )
            .await;
        };
    }

    assert_tag(0, PersistentFile::Node(b_tree::Node::Leaf(vec![]))).await;
    assert_tag(1, PersistentFile::TensorIndex(b_tree::Node::Leaf(vec![]))).await;
    check_dtype!(U8, MetadataU8, SparseU8; 2, 3, 42);
    check_dtype!(U16, MetadataU16, SparseU16; 4, 5, 43);
    check_dtype!(U32, MetadataU32, SparseU32; 6, 7, 44);
    check_dtype!(U64, MetadataU64, SparseU64; 8, 9, 45);
    check_dtype!(I8, MetadataI8, SparseI8; 10, 11, 46);
    check_dtype!(I16, MetadataI16, SparseI16; 12, 13, 47);
    check_dtype!(I32, MetadataI32, SparseI32; 14, 15, 48);
    check_dtype!(I64, MetadataI64, SparseI64; 16, 17, 49);
    check_dtype!(F32, MetadataF32, SparseF32; 18, 19, 50);
    check_dtype!(F64, MetadataF64, SparseF64; 20, 21, 51);
    #[cfg(feature = "complex")]
    {
        check_dtype!(C32, MetadataC32, SparseC32; 22, 23, 52);
        check_dtype!(C64, MetadataC64, SparseC64; 24, 25, 53);
    }

    let entry = PersistentFile::F64(vec![1.0, 2.0]);
    let decoded = decode_entry(&entry).await;
    assert!(AsType::<Vec<f64>>::as_type(&decoded).is_some());
    assert!(AsType::<Vec<u64>>::as_type(&decoded).is_none());
    let sparse = PersistentFile::from(b_tree::Node::Leaf(vec![vec![
        fensor::SparseCell::Key(0),
        fensor::SparseCell::Payload(vec![3f64]),
    ]]));

    let decoded = decode_entry(&sparse).await;
    assert!(AsType::<crate::TensorSparseNode<f64>>::as_type(&decoded).is_some());
    assert!(AsType::<crate::TensorSparseNode<f32>>::as_type(&decoded).is_none());
    let row = vec![1];
    for leaf in [false, true] {
        let node = if leaf {
            b_tree::Node::Leaf(vec![row.clone()])
        } else {
            b_tree::Node::Index(vec![row.clone()], vec![Default::default(); 2])
        };

        let entry = PersistentFile::TensorIndex(node);
        let decoded = decode_entry(&entry).await;
        let PersistentFile::TensorIndex(node) = decoded else {
            panic!("wrong payload tag")
        };

        assert_eq!(node.is_leaf(), leaf);
        let rows = match node {
            b_tree::Node::Leaf(rows) | b_tree::Node::Index(rows, _) => rows,
        };

        assert_eq!(rows.as_slice(), std::slice::from_ref(&row));
        assert_eq!(rows[0].capacity(), rows[0].len());
    }

    for rows in [
        vec![vec![]],
        vec![vec![1, 2]],
        vec![vec![1, 2, 3]],
        vec![vec![1, 2, 3, 4]],
        vec![vec![1]; fensor::MAX_BLOCK_CAPACITY + 1],
    ] {
        let node = PersistentFile::TensorIndex(b_tree::Node::Leaf(rows));
        assert!(
            tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(&node).unwrap())
                .await
                .is_err()
        );
    }

    for children in [0, 1, 3] {
        let node = PersistentFile::TensorIndex(b_tree::Node::Index(
            vec![vec![1]],
            vec![Default::default(); children],
        ));

        assert!(
            tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(&node).unwrap(),)
                .await
                .is_err()
        );
    }

    let node = PersistentFile::TensorIndex(b_tree::Node::Index(
        vec![vec![1]],
        vec![Default::default(); fensor::MAX_BLOCK_CAPACITY + 1],
    ));

    assert!(
        tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(&node).unwrap())
            .await
            .is_err()
    );

    let oversized = (20u8, vec![0.0f64; fensor::MAX_BLOCK_CAPACITY + 1]);
    assert!(
        tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(oversized).unwrap())
            .await
            .is_err()
    );

    let malformed = (21u8, (vec![4u64], false, Option::<u64>::None, vec![0u64]));
    assert!(
        tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(malformed).unwrap())
            .await
            .is_err()
    );

    let oversized_rank = (
        21u8,
        (
            vec![1u64; crate::tensor::MAX_TENSOR_METADATA_RANK + 1],
            false,
            Option::<u64>::None,
            vec![1u64],
        ),
    );

    assert!(
        tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(oversized_rank).unwrap())
            .await
            .is_err()
    );

    // Even a structurally valid separator-only node must use a current tag.
    for tag in (30u8..=41).chain([255u8]) {
        let index = b_tree::Node::Index(vec![(0u64,)], vec![Default::default(); 2]);
        assert!(
            tbon::de::try_decode::<_, _, PersistentFile>(
                (),
                tbon::en::encode((tag, index)).unwrap()
            )
            .await
            .is_err()
        );
    }
}
