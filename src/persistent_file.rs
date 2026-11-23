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

// One catalog owns native payload variants, conversions, and stable wire tags.
// Validation remains in the bounded decoders below and in sparse_codec.
macro_rules! persistent_files {
    (
        nodes [$($node:ident($node_type:ty) = $node_tag:literal => |$seq:ident| $decode:expr;)*]
        real [$($dense:ident, $metadata:ident, $sparse:ident: $dtype:ty =
            ($dense_tag:literal, $metadata_tag:literal, $sparse_tag:literal);)*]
        complex [$($cdense:ident, $cmetadata:ident, $csparse:ident: $ctype:ty =
            ($cdense_tag:literal, $cmetadata_tag:literal, $csparse_tag:literal);)*]
    ) => {
        /// Collection-only TBON adapter; tags preserve payload typing across reloads.
        #[derive(Clone)]
        pub enum PersistentFile {
            $($sparse(fensor::SparseNode<$dtype>),)*
            $(#[cfg(feature = "complex")] $csparse(fensor::SparseNode<$ctype>),)*
            $($node($node_type),)*
            $($dense(Vec<$dtype>), $metadata(fensor::TensorMetadata<$dtype>),)*
            $(
                #[cfg(feature = "complex")]
                $cdense(Vec<$ctype>),
                #[cfg(feature = "complex")]
                $cmetadata(fensor::TensorMetadata<$ctype>),
            )*
        }

        $(safecast::as_type!(PersistentFile, $node, $node_type);)*
        $(
            safecast::as_type!(PersistentFile, $dense, Vec<$dtype>);
            safecast::as_type!(PersistentFile, $metadata, fensor::TensorMetadata<$dtype>);
            safecast::as_type!(PersistentFile, $sparse, fensor::SparseNode<$dtype>);
        )*
        $(
            #[cfg(feature = "complex")]
            safecast::as_type!(PersistentFile, $cdense, Vec<$ctype>);
            #[cfg(feature = "complex")]
            safecast::as_type!(PersistentFile, $cmetadata, fensor::TensorMetadata<$ctype>);
            #[cfg(feature = "complex")]
            safecast::as_type!(PersistentFile, $csparse, fensor::SparseNode<$ctype>);
        )*

        impl<'en> en::ToStream<'en> for PersistentFile {
            fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
                match self {
                    $(Self::$node(value) => en::IntoStream::into_stream(($node_tag as u8, value), encoder),)*
                    $(
                        Self::$dense(value) => en::IntoStream::into_stream(($dense_tag as u8, value), encoder),
                        Self::$metadata(value) => en::IntoStream::into_stream(($metadata_tag as u8, value), encoder),
                        Self::$sparse(value) => en::IntoStream::into_stream(
                            ($sparse_tag as u8, sparse_codec::Encoded(value)), encoder,
                        ),
                    )*
                    $(
                        #[cfg(feature = "complex")]
                        Self::$cdense(value) => en::IntoStream::into_stream(
                            ($cdense_tag as u8, value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>()),
                            encoder,
                        ),
                        #[cfg(feature = "complex")]
                        Self::$cmetadata(value) => en::IntoStream::into_stream(
                            ($cmetadata_tag as u8, value), encoder,
                        ),
                        #[cfg(feature = "complex")]
                        Self::$csparse(value) => en::IntoStream::into_stream(
                            ($csparse_tag as u8, sparse_codec::Encoded(value)), encoder,
                        ),
                    )*
                }
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
                        let value = match seq.expect_next::<u8>(()).await? {
                            $($node_tag => {
                                let $seq = &mut seq;
                                PersistentFile::$node($decode)
                            },)*
                            $(
                                $dense_tag => PersistentFile::$dense(seq.expect_next::<BlockValues<$dtype>>(()).await?.0),
                                $metadata_tag => PersistentFile::$metadata(
                                    seq.expect_next(crate::tensor::MAX_TENSOR_METADATA_RANK).await?,
                                ),
                                $sparse_tag => PersistentFile::$sparse(seq.expect_next::<sparse_codec::Decoded<$dtype>>(()).await?.0),
                            )*
                            $(
                                #[cfg(feature = "complex")]
                                $cdense_tag => PersistentFile::$cdense(
                                    seq.expect_next::<BlockValues<(_, _)>>(()).await?.0
                                        .into_iter().map(|(re, im)| <$ctype>::new(re, im)).collect(),
                                ),
                                #[cfg(feature = "complex")]
                                $cmetadata_tag => PersistentFile::$cmetadata(
                                    seq.expect_next(crate::tensor::MAX_TENSOR_METADATA_RANK).await?,
                                ),
                                #[cfg(feature = "complex")]
                                $csparse_tag => PersistentFile::$csparse(seq.expect_next::<sparse_codec::Decoded<$ctype>>(()).await?.0),
                            )*
                            tag => return Err(de::Error::custom(format!("unknown collection file tag {tag}"))),
                        };

                        if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                            return Err(de::Error::custom("unexpected collection file field"));
                        }

                        Ok(value)
                    }
                }

                decoder.decode_seq(Visitor).await
            }
        }

        impl GetSize for PersistentFile {
            fn get_heap_size(&self) -> usize {
                match self {
                    $(Self::$node(value) => value.get_heap_size(),)*
                    $(
                        Self::$dense(value) => value.get_heap_size(),
                        Self::$metadata(value) => value.get_heap_size(),
                        Self::$sparse(value) => value.get_heap_size(),
                    )*
                    $(
                        #[cfg(feature = "complex")]
                        Self::$cdense(value) => value.get_heap_size(),
                        #[cfg(feature = "complex")]
                        Self::$cmetadata(value) => value.get_heap_size(),
                        #[cfg(feature = "complex")]
                        Self::$csparse(value) => value.get_heap_size(),
                    )*
                }
            }
        }
    };
}

persistent_files! {
    nodes [
        Node(CollectionNode) = 0 => |seq| seq.expect_next(()).await?;
        TensorIndex(b_table::Node<u64>) = 1 => |seq| seq.expect_next::<IndexNode>(()).await?.0;
    ]
    real [
        U8, MetadataU8, SparseU8: u8 = (2, 3, 30);
        U16, MetadataU16, SparseU16: u16 = (4, 5, 31);
        U32, MetadataU32, SparseU32: u32 = (6, 7, 32);
        U64, MetadataU64, SparseU64: u64 = (8, 9, 33);
        I8, MetadataI8, SparseI8: i8 = (10, 11, 34);
        I16, MetadataI16, SparseI16: i16 = (12, 13, 35);
        I32, MetadataI32, SparseI32: i32 = (14, 15, 36);
        I64, MetadataI64, SparseI64: i64 = (16, 17, 37);
        F32, MetadataF32, SparseF32: f32 = (18, 19, 38);
        F64, MetadataF64, SparseF64: f64 = (20, 21, 39);
    ]
    complex [
        C32, MetadataC32, SparseC32: fensor::complex::Complex32 = (22, 23, 40);
        C64, MetadataC64, SparseC64: fensor::complex::Complex64 = (24, 25, 41);
    ]
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
