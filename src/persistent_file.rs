//! Tagged native collection files. Domain values retain format-neutral codecs.

use destream::{de, en};
use freqfs::{FileLoad, FileSave};
use get_size::GetSize;
use safecast::AsType;
use tc_value::Value;

mod sparse_codec;

pub type CollectionNode = b_tree::Node<Vec<Vec<Value>>>;

#[cfg(feature = "complex")]
pub trait ComplexFile:
    fensor::TensorFileEntry<fensor::complex::Complex32>
    + fensor::TensorFileEntry<fensor::complex::Complex64>
{
}

#[cfg(feature = "complex")]
impl<T> ComplexFile for T where
    T: fensor::TensorFileEntry<fensor::complex::Complex32>
        + fensor::TensorFileEntry<fensor::complex::Complex64>
{
}

#[cfg(not(feature = "complex"))]
pub trait ComplexFile {}

#[cfg(not(feature = "complex"))]
impl<T> ComplexFile for T {}

/// Caller-owned native file composition; no erased storage handles.
pub trait CollectionFile:
    Clone
    + FileLoad
    + FileSave
    + GetSize
    + AsType<b_table::Node<u64>>
    + From<b_table::Node<u64>>
    + AsType<CollectionNode>
    + From<CollectionNode>
    + ComplexFile
    + fensor::TensorFileEntry<u8>
    + fensor::TensorFileEntry<u16>
    + fensor::TensorFileEntry<u32>
    + fensor::TensorFileEntry<u64>
    + fensor::TensorFileEntry<i8>
    + fensor::TensorFileEntry<i16>
    + fensor::TensorFileEntry<i32>
    + fensor::TensorFileEntry<i64>
    + fensor::TensorFileEntry<f32>
    + fensor::TensorFileEntry<f64>
    + Send
    + Sync
    + 'static
{
}

impl<F> CollectionFile for F where
    F: Clone
        + FileLoad
        + FileSave
        + GetSize
        + AsType<b_table::Node<u64>>
        + From<b_table::Node<u64>>
        + AsType<CollectionNode>
        + From<CollectionNode>
        + ComplexFile
        + fensor::TensorFileEntry<u8>
        + fensor::TensorFileEntry<u16>
        + fensor::TensorFileEntry<u32>
        + fensor::TensorFileEntry<u64>
        + fensor::TensorFileEntry<i8>
        + fensor::TensorFileEntry<i16>
        + fensor::TensorFileEntry<i32>
        + fensor::TensorFileEntry<i64>
        + fensor::TensorFileEntry<f32>
        + fensor::TensorFileEntry<f64>
        + Send
        + Sync
        + 'static
{
}

/// Collection-only TBON adapter; tags preserve payload typing across reloads.
#[derive(Clone)]
pub enum PersistentFile {
    SparseU8(fensor::SparseNode<u8>),
    SparseU16(fensor::SparseNode<u16>),
    SparseU32(fensor::SparseNode<u32>),
    SparseU64(fensor::SparseNode<u64>),
    SparseI8(fensor::SparseNode<i8>),
    SparseI16(fensor::SparseNode<i16>),
    SparseI32(fensor::SparseNode<i32>),
    SparseI64(fensor::SparseNode<i64>),
    SparseF32(fensor::SparseNode<f32>),
    SparseF64(fensor::SparseNode<f64>),
    #[cfg(feature = "complex")]
    SparseC32(fensor::SparseNode<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    SparseC64(fensor::SparseNode<fensor::complex::Complex64>),
    Node(CollectionNode),
    TensorIndex(b_table::Node<u64>),
    U8(Vec<u8>),
    MetadataU8(fensor::TensorMetadata<u8>),
    U16(Vec<u16>),
    MetadataU16(fensor::TensorMetadata<u16>),
    U32(Vec<u32>),
    MetadataU32(fensor::TensorMetadata<u32>),
    U64(Vec<u64>),
    MetadataU64(fensor::TensorMetadata<u64>),
    I8(Vec<i8>),
    MetadataI8(fensor::TensorMetadata<i8>),
    I16(Vec<i16>),
    MetadataI16(fensor::TensorMetadata<i16>),
    I32(Vec<i32>),
    MetadataI32(fensor::TensorMetadata<i32>),
    I64(Vec<i64>),
    MetadataI64(fensor::TensorMetadata<i64>),
    F32(Vec<f32>),
    MetadataF32(fensor::TensorMetadata<f32>),
    F64(Vec<f64>),
    MetadataF64(fensor::TensorMetadata<f64>),
    #[cfg(feature = "complex")]
    C32(Vec<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    MetadataC32(fensor::TensorMetadata<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    C64(Vec<fensor::complex::Complex64>),
    #[cfg(feature = "complex")]
    MetadataC64(fensor::TensorMetadata<fensor::complex::Complex64>),
}

safecast::as_type!(PersistentFile, Node, CollectionNode);
safecast::as_type!(PersistentFile, TensorIndex, b_table::Node<u64>);
safecast::as_type!(PersistentFile, U8, Vec<u8>);
safecast::as_type!(PersistentFile, MetadataU8, fensor::TensorMetadata<u8>);
safecast::as_type!(PersistentFile, U16, Vec<u16>);
safecast::as_type!(PersistentFile, MetadataU16, fensor::TensorMetadata<u16>);
safecast::as_type!(PersistentFile, U32, Vec<u32>);
safecast::as_type!(PersistentFile, MetadataU32, fensor::TensorMetadata<u32>);
safecast::as_type!(PersistentFile, U64, Vec<u64>);
safecast::as_type!(PersistentFile, MetadataU64, fensor::TensorMetadata<u64>);
safecast::as_type!(PersistentFile, I8, Vec<i8>);
safecast::as_type!(PersistentFile, MetadataI8, fensor::TensorMetadata<i8>);
safecast::as_type!(PersistentFile, I16, Vec<i16>);
safecast::as_type!(PersistentFile, MetadataI16, fensor::TensorMetadata<i16>);
safecast::as_type!(PersistentFile, I32, Vec<i32>);
safecast::as_type!(PersistentFile, MetadataI32, fensor::TensorMetadata<i32>);
safecast::as_type!(PersistentFile, I64, Vec<i64>);
safecast::as_type!(PersistentFile, MetadataI64, fensor::TensorMetadata<i64>);
safecast::as_type!(PersistentFile, F32, Vec<f32>);
safecast::as_type!(PersistentFile, MetadataF32, fensor::TensorMetadata<f32>);
safecast::as_type!(PersistentFile, F64, Vec<f64>);
safecast::as_type!(PersistentFile, MetadataF64, fensor::TensorMetadata<f64>);
#[cfg(feature = "complex")]
safecast::as_type!(PersistentFile, C32, Vec<fensor::complex::Complex32>);
#[cfg(feature = "complex")]
safecast::as_type!(
    PersistentFile,
    MetadataC32,
    fensor::TensorMetadata<fensor::complex::Complex32>
);
#[cfg(feature = "complex")]
safecast::as_type!(PersistentFile, C64, Vec<fensor::complex::Complex64>);
#[cfg(feature = "complex")]
safecast::as_type!(
    PersistentFile,
    MetadataC64,
    fensor::TensorMetadata<fensor::complex::Complex64>
);

// Only payload dispatch is generated; codecs retain structural validation.
macro_rules! sparse_payload_dispatch {
    ($d:tt; $( $(#[$attr:meta])* $variant:ident : $dtype:ty = $tag:literal ),+ $(,)?) => {
        macro_rules! encode_sparse {
            ($d value:expr, $d encoder:expr; $d($d rest:tt)*) => {
                match $d value {
                    $( $(#[$attr])* PersistentFile::$variant(node) => en::IntoStream::into_stream(($tag as u8, sparse_codec::Encoded(node)), $d encoder), )+
                    $d($d rest)*
                }
            };
        }
        macro_rules! decode_sparse {
            ($d tag:expr, $d seq:ident; $d($d rest:tt)*) => {
                match $d tag {
                    $( $(#[$attr])* $tag => PersistentFile::$variant($d seq.expect_next::<sparse_codec::Decoded<$dtype>>(()).await?.0), )+
                    $d($d rest)*
                }
            };
        }
        macro_rules! size_sparse {
            ($d value:expr; $d($d rest:tt)*) => {
                match $d value {
                    $( $(#[$attr])* PersistentFile::$variant(node) => node.get_heap_size(), )+
                    $d($d rest)*
                }
            };
        }
    };
}
sparse_payload_dispatch!($;
    SparseU8:u8=30, SparseU16:u16=31, SparseU32:u32=32, SparseU64:u64=33,
    SparseI8:i8=34, SparseI16:i16=35, SparseI32:i32=36, SparseI64:i64=37,
    SparseF32:f32=38, SparseF64:f64=39,
    #[cfg(feature="complex")] SparseC32:fensor::complex::Complex32=40,
    #[cfg(feature="complex")] SparseC64:fensor::complex::Complex64=41,
);

impl<'en> en::ToStream<'en> for PersistentFile {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encode_sparse!(self, encoder;
            Self::Node(value) => en::IntoStream::into_stream((0u8, value), encoder),
            Self::TensorIndex(value) => en::IntoStream::into_stream((1u8, value), encoder),
            Self::U8(value) => en::IntoStream::into_stream((2u8, value), encoder),
            Self::MetadataU8(value) => en::IntoStream::into_stream((3u8, value), encoder),
            Self::U16(value) => en::IntoStream::into_stream((4u8, value), encoder),
            Self::MetadataU16(value) => en::IntoStream::into_stream((5u8, value), encoder),
            Self::U32(value) => en::IntoStream::into_stream((6u8, value), encoder),
            Self::MetadataU32(value) => en::IntoStream::into_stream((7u8, value), encoder),
            Self::U64(value) => en::IntoStream::into_stream((8u8, value), encoder),
            Self::MetadataU64(value) => en::IntoStream::into_stream((9u8, value), encoder),
            Self::I8(value) => en::IntoStream::into_stream((10u8, value), encoder),
            Self::MetadataI8(value) => en::IntoStream::into_stream((11u8, value), encoder),
            Self::I16(value) => en::IntoStream::into_stream((12u8, value), encoder),
            Self::MetadataI16(value) => en::IntoStream::into_stream((13u8, value), encoder),
            Self::I32(value) => en::IntoStream::into_stream((14u8, value), encoder),
            Self::MetadataI32(value) => en::IntoStream::into_stream((15u8, value), encoder),
            Self::I64(value) => en::IntoStream::into_stream((16u8, value), encoder),
            Self::MetadataI64(value) => en::IntoStream::into_stream((17u8, value), encoder),
            Self::F32(value) => en::IntoStream::into_stream((18u8, value), encoder),
            Self::MetadataF32(value) => en::IntoStream::into_stream((19u8, value), encoder),
            Self::F64(value) => en::IntoStream::into_stream((20u8, value), encoder),
            Self::MetadataF64(value) => en::IntoStream::into_stream((21u8, value), encoder),
            #[cfg(feature = "complex")]
            Self::C32(value) => en::IntoStream::into_stream(
                (22u8, value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>()),
                encoder,
            ),
            #[cfg(feature = "complex")]
            Self::MetadataC32(value) => en::IntoStream::into_stream((23u8, value), encoder),
            #[cfg(feature = "complex")]
            Self::C64(value) => en::IntoStream::into_stream(
                (24u8, value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>()),
                encoder,
            ),
            #[cfg(feature = "complex")]
            Self::MetadataC64(value) => en::IntoStream::into_stream((25u8, value), encoder),
        )
    }
}

/// Reject oversized payloads while decoding, before an untrusted length allocates.
struct BlockValues<T>(Vec<T>);

impl<T: de::FromStream<Context = ()> + Send> de::FromStream for BlockValues<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);

        impl<T: de::FromStream<Context = ()> + Send> de::Visitor for Visitor<T> {
            type Value = BlockValues<T>;

            fn expecting() -> &'static str {
                "one bounded tensor block"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element(()).await? {
                    if values.len() == fensor::MAX_BLOCK_CAPACITY {
                        return Err(de::Error::custom("tensor block exceeds storage capacity"));
                    }

                    values.push(value);
                }

                Ok(BlockValues(values.into_boxed_slice().into_vec()))
            }
        }

        decoder.decode_seq(Visitor(std::marker::PhantomData)).await
    }
}

// Tensor touched-block keys contain one logical block ID.
// Validate the row while decoding instead of trusting an encoded sequence length.
struct IndexRow(Vec<u64>);

impl de::FromStream for IndexRow {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor;

        impl de::Visitor for Visitor {
            type Value = IndexRow;

            fn expecting() -> &'static str {
                "a Tensor touched-block key"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let id = seq.expect_next::<u64>(()).await?;
                if seq.next_element::<u64>(()).await?.is_some() {
                    return Err(de::Error::custom("invalid Tensor index key width"));
                }

                Ok(IndexRow(vec![id]))
            }
        }

        decoder.decode_seq(Visitor).await
    }
}

// Tensor indexes have bounded rows and native node fanout. The
// decoder never trusts the wire's declared sequence lengths for allocation.
struct IndexNode(b_table::Node<u64>);

impl de::FromStream for IndexNode {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor;

        impl de::Visitor for Visitor {
            type Value = IndexNode;

            fn expecting() -> &'static str {
                "a bounded native Tensor index node"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let leaf = seq.expect_next::<bool>(()).await?;
                let rows = seq.expect_next::<BlockValues<IndexRow>>(()).await?.0;
                let rows: Vec<_> = rows.into_iter().map(|row| row.0).collect();
                let node = if leaf {
                    b_tree::Node::Leaf(rows)
                } else {
                    let children = seq.expect_next::<BlockValues<_>>(()).await?.0;
                    if rows.is_empty() || children.len() != rows.len() + 1 {
                        return Err(de::Error::custom("invalid Tensor index separator count"));
                    }

                    b_tree::Node::Index(rows, children)
                };

                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected Tensor index field"));
                }

                Ok(IndexNode(node))
            }
        }

        decoder.decode_seq(Visitor).await
    }
}

impl de::FromStream for PersistentFile {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor;

        impl de::Visitor for Visitor {
            type Value = PersistentFile;

            fn expecting() -> &'static str {
                "a tagged native collection file"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let value = decode_sparse!(seq.expect_next::<u8>(()).await?, seq;
                    0 => PersistentFile::Node(seq.expect_next(()).await?),
                    1 => PersistentFile::TensorIndex(seq.expect_next::<IndexNode>(()).await?.0),
                    2 => PersistentFile::U8(seq.expect_next::<BlockValues<u8>>(()).await?.0),
                    3 => PersistentFile::MetadataU8(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    4 => PersistentFile::U16(seq.expect_next::<BlockValues<u16>>(()).await?.0),
                    5 => PersistentFile::MetadataU16(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    6 => PersistentFile::U32(seq.expect_next::<BlockValues<u32>>(()).await?.0),
                    7 => PersistentFile::MetadataU32(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    8 => PersistentFile::U64(seq.expect_next::<BlockValues<u64>>(()).await?.0),
                    9 => PersistentFile::MetadataU64(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    10 => PersistentFile::I8(seq.expect_next::<BlockValues<i8>>(()).await?.0),
                    11 => PersistentFile::MetadataI8(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    12 => PersistentFile::I16(seq.expect_next::<BlockValues<i16>>(()).await?.0),
                    13 => PersistentFile::MetadataI16(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    14 => PersistentFile::I32(seq.expect_next::<BlockValues<i32>>(()).await?.0),
                    15 => PersistentFile::MetadataI32(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    16 => PersistentFile::I64(seq.expect_next::<BlockValues<i64>>(()).await?.0),
                    17 => PersistentFile::MetadataI64(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    18 => PersistentFile::F32(seq.expect_next::<BlockValues<f32>>(()).await?.0),
                    19 => PersistentFile::MetadataF32(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    20 => PersistentFile::F64(seq.expect_next::<BlockValues<f64>>(()).await?.0),
                    21 => PersistentFile::MetadataF64(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    #[cfg(feature = "complex")]
                    22 => PersistentFile::C32(
                        seq.expect_next::<BlockValues<(f32, f32)>>(())
                            .await?
                            .0
                            .into_iter()
                            .map(|(re, im)| fensor::complex::Complex32::new(re, im))
                            .collect(),
                    ),
                    #[cfg(feature = "complex")]
                    23 => PersistentFile::MetadataC32(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    #[cfg(feature = "complex")]
                    24 => PersistentFile::C64(
                        seq.expect_next::<BlockValues<(f64, f64)>>(())
                            .await?
                            .0
                            .into_iter()
                            .map(|(re, im)| fensor::complex::Complex64::new(re, im))
                            .collect(),
                    ),
                    #[cfg(feature = "complex")]
                    25 => PersistentFile::MetadataC64(seq.expect_next::<fensor::TensorMetadata<_>>(crate::tensor::MAX_TENSOR_METADATA_RANK).await?),
                    tag => {
                        return Err(de::Error::custom(format!(
                            "unknown collection file tag {tag}"
                        )));
                    }
                );

                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected collection file field"));
                }

                Ok(value)
            }
        }

        decoder.decode_seq(Visitor).await
    }
}

impl FileLoad for PersistentFile {
    async fn load(
        _: &std::path::Path,
        file: tokio::fs::File,
        _: std::fs::Metadata,
    ) -> std::io::Result<Self> {
        tbon::de::read_from((), file).await.map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
        })
    }
}

impl FileSave for PersistentFile {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        use futures::TryStreamExt;
        use tokio::io::AsyncWriteExt;

        let mut file = tokio::io::BufWriter::new(file);
        let mut stream = tbon::en::encode(self).map_err(std::io::Error::other)?;
        let mut size = 0;
        while let Some(chunk) = stream.try_next().await.map_err(std::io::Error::other)? {
            file.write_all(&chunk).await?;
            size += chunk.len() as u64;
        }

        file.flush().await?;
        Ok(size)
    }
}

impl GetSize for PersistentFile {
    fn get_heap_size(&self) -> usize {
        size_sparse!(self;
            Self::Node(node) => node.get_heap_size(),
            Self::TensorIndex(node) => node.get_heap_size(),
            Self::U8(value) => value.get_heap_size(),
            Self::MetadataU8(value) => value.get_heap_size(),
            Self::U16(value) => value.get_heap_size(),
            Self::MetadataU16(value) => value.get_heap_size(),
            Self::U32(value) => value.get_heap_size(),
            Self::MetadataU32(value) => value.get_heap_size(),
            Self::U64(value) => value.get_heap_size(),
            Self::MetadataU64(value) => value.get_heap_size(),
            Self::I8(value) => value.get_heap_size(),
            Self::MetadataI8(value) => value.get_heap_size(),
            Self::I16(value) => value.get_heap_size(),
            Self::MetadataI16(value) => value.get_heap_size(),
            Self::I32(value) => value.get_heap_size(),
            Self::MetadataI32(value) => value.get_heap_size(),
            Self::I64(value) => value.get_heap_size(),
            Self::MetadataI64(value) => value.get_heap_size(),
            Self::F32(value) => value.get_heap_size(),
            Self::MetadataF32(value) => value.get_heap_size(),
            Self::F64(value) => value.get_heap_size(),
            Self::MetadataF64(value) => value.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::C32(value) => value.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::MetadataC32(value) => value.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::C64(value) => value.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::MetadataC64(value) => value.get_heap_size(),
        )
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! delegate_collection_payload {
    ($file:ty, $variant:ident, $payload:ty) => {
        impl From<$payload> for $file {
            fn from(value: $payload) -> Self {
                Self::$variant($crate::PersistentFile::from(value))
            }
        }

        impl safecast::AsType<$payload> for $file {
            fn as_type(&self) -> Option<&$payload> {
                match self {
                    Self::$variant(file) => safecast::AsType::as_type(file),
                    _ => None,
                }
            }

            fn as_type_mut(&mut self) -> Option<&mut $payload> {
                match self {
                    Self::$variant(file) => safecast::AsType::as_type_mut(file),
                    _ => None,
                }
            }

            fn into_type(self) -> Option<$payload> {
                match self {
                    Self::$variant(file) => safecast::AsType::into_type(file),
                    _ => None,
                }
            }
        }
    };
}

// Feature selection occurs here, not in a downstream macro expansion.
macro_rules! tensor_file_delegation {
    ($d:tt; $($complex:ident),* $(,)?) => {
        #[macro_export]
        macro_rules! delegate_tensor_files {
            ($d file:ty, $d variant:ident) => {
                $d crate::delegate_collection_payload!($d file, $d variant, $d crate::TensorIndexNode);
                $d crate::delegate_tensor_files!(@types $d file, $d variant;
                    u8,u16,u32,u64,i8,i16,i32,i64,f32,f64 $(,$d crate::$complex)*);
            };
            (@types $d file:ty, $d variant:ident; $d($d dtype:ty),+) => {
                $d(
                    $d crate::delegate_collection_payload!($d file, $d variant, Vec<$d dtype>);
                    $d crate::delegate_collection_payload!($d file, $d variant, $d crate::TensorMetadata<$d dtype>);
                    $d crate::delegate_collection_payload!($d file, $d variant, $d crate::TensorSparseNode<$d dtype>);
                )+
            };
        }
    };
}

#[cfg(not(feature = "complex"))]
tensor_file_delegation!($;);
#[cfg(feature = "complex")]
tensor_file_delegation!($; TensorComplex32, TensorComplex64);

#[cfg(test)]
#[path = "../tests/unit/persistent_file/tests.rs"]
mod tests;

safecast::as_type!(PersistentFile, SparseU8, fensor::SparseNode<u8>);
safecast::as_type!(PersistentFile, SparseU16, fensor::SparseNode<u16>);
safecast::as_type!(PersistentFile, SparseU32, fensor::SparseNode<u32>);
safecast::as_type!(PersistentFile, SparseU64, fensor::SparseNode<u64>);
safecast::as_type!(PersistentFile, SparseI8, fensor::SparseNode<i8>);
safecast::as_type!(PersistentFile, SparseI16, fensor::SparseNode<i16>);
safecast::as_type!(PersistentFile, SparseI32, fensor::SparseNode<i32>);
safecast::as_type!(PersistentFile, SparseI64, fensor::SparseNode<i64>);
safecast::as_type!(PersistentFile, SparseF32, fensor::SparseNode<f32>);
safecast::as_type!(PersistentFile, SparseF64, fensor::SparseNode<f64>);
#[cfg(feature = "complex")]
safecast::as_type!(
    PersistentFile,
    SparseC32,
    fensor::SparseNode<fensor::complex::Complex32>
);
#[cfg(feature = "complex")]
safecast::as_type!(
    PersistentFile,
    SparseC64,
    fensor::SparseNode<fensor::complex::Complex64>
);
