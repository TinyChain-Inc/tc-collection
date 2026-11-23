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
async fn sparse_nodes_retain_bits_and_reject_unbounded_or_wrong_payloads() {
    use fensor::SparseCell::{Key, Payload};
    use futures::{TryStreamExt, stream};

    let bits = [0u64, 1 << 63, 0x7ff8000000000123, f64::INFINITY.to_bits()];
    for len in [1, 3, fensor::MAX_BLOCK_CAPACITY] {
        let values: Vec<_> = (0..len)
            .map(|i| f64::from_bits(bits[i % bits.len()]))
            .collect();

        let entry = PersistentFile::SparseF64(b_table::Node::Leaf(vec![vec![
            Key(1),
            Payload(values.clone()),
        ]]));

        let decoded = decode_entry(&entry).await;
        assert!(AsType::<fensor::SparseNode<f32>>::as_type(&decoded).is_none());
        let PersistentFile::SparseF64(b_table::Node::Leaf(rows)) = decoded else {
            panic!("wrong payload tag")
        };

        let Payload(actual) = &rows[0][1] else {
            panic!("wrong cell kind")
        };

        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    for row in [
        vec![Key(2)],
        vec![Key(2), Payload(vec![f64::from_bits(bits[2]), -0.])],
    ] {
        let node = b_table::Node::Index(vec![row.clone()], vec![Default::default(); 2]);
        let decoded = decode_entry(&PersistentFile::SparseF64(node)).await;
        let PersistentFile::SparseF64(b_table::Node::Index(rows, children)) = decoded else {
            panic!("wrong compact node kind")
        };

        assert_eq!(rows, vec![row]);
        assert_eq!(children.len(), 2);
        if let Some(Payload(values)) = rows[0].get(1) {
            assert_eq!(values[0].to_bits(), bits[2]);
            assert_eq!(values[1].to_bits(), (-0.0f64).to_bits());
        }
    }

    let maximum = PersistentFile::SparseF64(b_table::Node::Leaf(
        (0..4)
            .map(|id| vec![Key(id), Payload(vec![1.; fensor::MAX_BLOCK_CAPACITY])])
            .collect(),
    ));

    decode_entry(&maximum).await;
    #[cfg(feature = "complex")]
    {
        let value = fensor::complex::Complex64::new(f64::from_bits(bits[2]), -0.);
        let entry = PersistentFile::SparseC64(b_table::Node::Leaf(vec![vec![
            Key(1),
            Payload(vec![value]),
        ]]));

        let decoded = decode_entry(&entry).await;
        let PersistentFile::SparseC64(b_table::Node::Leaf(rows)) = decoded else {
            panic!("wrong complex tag")
        };

        assert_eq!(rows[0][1], Payload(vec![value]));
    }

    struct Raw(Vec<u8>);

    impl<'en> en::IntoStream<'en> for Raw {
        fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
            encoder.encode_array_u8(stream::iter([self.0]))
        }
    }

    async fn rejects<'en, V: en::IntoStream<'en> + 'en>(value: V) {
        let chunks = tbon::en::encode(value)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert!(
            tbon::de::try_decode::<_, _, PersistentFile>(
                (),
                stream::iter(chunks.clone().into_iter().map(Ok::<_, std::io::Error>))
            )
            .await
            .is_err()
        );
    }

    let b_table::Node::Index(_, children) =
        b_table::Node::<u64>::Index(Vec::new(), vec![Default::default(); 3])
    else {
        unreachable!()
    };

    for count in [0, 1, 3] {
        rejects((39u8, (false, vec![(0u64,)], children[..count].to_vec()))).await;
    }

    rejects((39u8, (true, vec![(0u64,)], Vec::<u64>::new()))).await;
    rejects((
        39u8,
        (false, vec![Vec::<u64>::new()], children[..2].to_vec()),
    ))
    .await;

    rejects((39u8, (false, vec![("bad block",)], children[..2].to_vec()))).await;
    rejects((
        39u8,
        (
            false,
            vec![(0u64, Raw(vec![0; 8]), 1u64)],
            children[..2].to_vec(),
        ),
    ))
    .await;

    // Old copied-minimum nodes had one complete row for every child.
    rejects((
        39u8,
        (false, vec![(0u64, Raw(vec![0; 8]))], children[..1].to_vec()),
    ))
    .await;

    for bytes in [0, 1, 7, 9, (fensor::MAX_BLOCK_CAPACITY + 1) * 8] {
        rejects((
            39u8,
            (true, vec![(0u64, Raw(vec![0; bytes]))], Vec::<u64>::new()),
        ))
        .await;
    }

    for (rows, values) in [(1025, 1), (5, fensor::MAX_BLOCK_CAPACITY)] {
        let rows: Vec<_> = (0..rows)
            .map(|id| (id as u64, Raw(vec![0; values * 8])))
            .collect();

        rejects((39u8, (true, rows, Vec::<u64>::new()))).await;
    }

    rejects((
        39u8,
        (true, vec![(0u64, 0u64, Raw(vec![0; 8]))], Vec::<u64>::new()),
    ))
    .await;

    rejects((39u8, (true, vec![(0u64, Raw(vec![0; 8]))], vec![0u64]))).await;
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
}

#[tokio::test]
async fn tensor_files_preserve_dtype_tags_and_bound_decoding() {
    let entry = PersistentFile::F64(vec![1.0, 2.0]);
    let decoded = decode_entry(&entry).await;
    assert!(AsType::<Vec<f64>>::as_type(&decoded).is_some());
    assert!(AsType::<Vec<u64>>::as_type(&decoded).is_none());
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

    // An unknown tag must not cause a decoder to reinterpret its payload.
    let unknown = (255u8, vec![0.0f64]);
    assert!(
        tbon::de::try_decode::<_, _, PersistentFile>((), tbon::en::encode(unknown).unwrap())
            .await
            .is_err()
    );
}
