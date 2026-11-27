//! Tagged native collection files. Domain values retain format-neutral codecs.

use destream::{de, en};
use freqfs::{FileLoad, FileSave};
use get_size::GetSize;
use safecast::AsType;
use tc_value::Value;

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
    + AsType<crate::CollectionMetadata>
    + From<crate::CollectionMetadata>
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
        + AsType<crate::CollectionMetadata>
        + From<crate::CollectionMetadata>
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
// Local decoders bound adapter payloads; fensor validates sparse rows.
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
                            ($sparse_tag as u8, value), encoder,
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
                            ($csparse_tag as u8, value), encoder,
                        ),
                    )*
                }
            }
        }

        impl de::FromStream for PersistentFile {
            type Context = ();

            async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
                let mut fields = decoder.open_container(de::Kind::Seq, None).await?;
                if fields.slot() == de::Slot::End {
                    return Err(de::Error::custom("missing collection file tag"));
                }

                let tag = <u8 as de::FromStream>::from_stream((), decoder).await?;
                decoder.finish_child(&mut fields).await?;
                if fields.slot() == de::Slot::End {
                    return Err(de::Error::custom("missing collection file payload"));
                }

                let value = match tag {
                    $($node_tag => {
                        let $seq = &mut *decoder;
                        PersistentFile::$node($decode)
                    },)*
                    $(
                        $dense_tag => PersistentFile::$dense(
                            decode_block_values::<$dtype, _>(decoder).await?,
                        ),
                        $metadata_tag => PersistentFile::$metadata(
                            <fensor::TensorMetadata<$dtype> as de::FromStream>::from_stream(crate::tensor::MAX_TENSOR_METADATA_RANK, decoder).await?,
                        ),
                        $sparse_tag => PersistentFile::$sparse(
                            <fensor::SparseNode<$dtype> as de::FromStream>::from_stream((), decoder).await?,
                        ),
                    )*
                    $(
                        #[cfg(feature = "complex")]
                        $cdense_tag => PersistentFile::$cdense(
                            decode_block_values::<(_, _), _>(decoder).await?
                                .into_iter().map(|(re, im)| <$ctype>::new(re, im)).collect(),
                        ),
                        #[cfg(feature = "complex")]
                        $cmetadata_tag => PersistentFile::$cmetadata(
                            <fensor::TensorMetadata<$ctype> as de::FromStream>::from_stream(crate::tensor::MAX_TENSOR_METADATA_RANK, decoder).await?,
                        ),
                        #[cfg(feature = "complex")]
                        $csparse_tag => PersistentFile::$csparse(
                            <fensor::SparseNode<$ctype> as de::FromStream>::from_stream((), decoder).await?,
                        ),
                    )*
                    tag => return Err(de::Error::custom(format!("unknown collection file tag {tag}"))),
                };

                decoder.finish_child(&mut fields).await?;
                if fields.slot() != de::Slot::End {
                    <de::IgnoredAny as de::FromStream>::from_stream((), decoder).await?;
                    return Err(de::Error::custom("unexpected collection file field"));
                }

                Ok(value)
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
        Metadata(crate::CollectionMetadata) = 26 => |decoder| <crate::CollectionMetadata as de::FromStream>::from_stream((), decoder).await?;
        Node(CollectionNode) = 0 => |decoder| <CollectionNode as de::FromStream>::from_stream((), decoder).await?;
        TensorIndex(b_table::Node<u64>) = 1 => |decoder| b_tree::Node::decode_bounded::<TensorIndexRows, _>((), decoder, fensor::MAX_BLOCK_CAPACITY).await?;
    ]
    real [
        // Tags 30 through 41 identified the retired packed-byte sparse nodes.
        U8, MetadataU8, SparseU8: u8 = (2, 3, 42);
        U16, MetadataU16, SparseU16: u16 = (4, 5, 43);
        U32, MetadataU32, SparseU32: u32 = (6, 7, 44);
        U64, MetadataU64, SparseU64: u64 = (8, 9, 45);
        I8, MetadataI8, SparseI8: i8 = (10, 11, 46);
        I16, MetadataI16, SparseI16: i16 = (12, 13, 47);
        I32, MetadataI32, SparseI32: i32 = (14, 15, 48);
        I64, MetadataI64, SparseI64: i64 = (16, 17, 49);
        F32, MetadataF32, SparseF32: f32 = (18, 19, 50);
        F64, MetadataF64, SparseF64: f64 = (20, 21, 51);
    ]
    complex [
        C32, MetadataC32, SparseC32: fensor::complex::Complex32 = (22, 23, 52);
        C64, MetadataC64, SparseC64: fensor::complex::Complex64 = (24, 25, 53);
    ]
}

/// Reject oversized payloads while decoding, before an untrusted length allocates.
async fn decode_block_values<T: de::FromStream<Context = ()> + Send, D: de::Decoder>(
    decoder: &mut D,
) -> Result<Vec<T>, D::Error> {
    let mut rows = decoder.open_container(de::Kind::Seq, None).await?;
    let mut values = Vec::new();
    while rows.slot() != de::Slot::End {
        if values.len() == fensor::MAX_BLOCK_CAPACITY {
            return Err(de::Error::custom("tensor block exceeds storage capacity"));
        }

        let value = T::from_stream((), decoder).await?;
        decoder.finish_child(&mut rows).await?;
        values.push(value);
    }

    Ok(values.into_boxed_slice().into_vec())
}

struct TensorIndexRows(Vec<Vec<u64>>);

impl From<TensorIndexRows> for Vec<Vec<u64>> {
    fn from(rows: TensorIndexRows) -> Self {
        rows.0
    }
}

impl de::FromStream for TensorIndexRows {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let rows = decode_block_values::<[u64; 1], _>(decoder).await?;
        Ok(Self(rows.into_iter().map(Vec::from).collect()))
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
macro_rules! collection_file_delegation {
    ($d:tt; $($complex:ident),* $(,)?) => {
        #[macro_export]
        macro_rules! delegate_collection_files {
            ($d file:ty, $d variant:ident) => {
                $d crate::delegate_collection_payload!($d file, $d variant, $d crate::CollectionNode);
                $d crate::delegate_collection_payload!($d file, $d variant, $d crate::CollectionMetadata);
                $d crate::delegate_collection_payload!($d file, $d variant, $d crate::TensorIndexNode);
                $d crate::delegate_collection_files!(@types $d file, $d variant;
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
collection_file_delegation!($;);

#[cfg(feature = "complex")]
collection_file_delegation!($; TensorComplex32, TensorComplex64);

#[cfg(test)]
#[path = "../tests/unit/persistent_file/tests.rs"]
mod tests;

// BTree operates on the native node. Its persisted owner supplies the codec,
// while these conversions move or borrow that same payload without copying it.
impl<T: fensor::TensorElement> From<b_tree::Node<Vec<Vec<fensor::SparseCell<T>>>>>
    for PersistentFile
where
    Self: From<fensor::SparseNode<T>>,
{
    fn from(node: b_tree::Node<Vec<Vec<fensor::SparseCell<T>>>>) -> Self {
        Self::from(fensor::SparseNode::from(node))
    }
}

impl<T: fensor::TensorElement> AsType<b_tree::Node<Vec<Vec<fensor::SparseCell<T>>>>>
    for PersistentFile
where
    Self: AsType<fensor::SparseNode<T>>,
{
    fn as_type(&self) -> Option<&b_tree::Node<Vec<Vec<fensor::SparseCell<T>>>>> {
        <Self as AsType<fensor::SparseNode<T>>>::as_type(self).map(|node| &node.0)
    }

    fn as_type_mut(&mut self) -> Option<&mut b_tree::Node<Vec<Vec<fensor::SparseCell<T>>>>> {
        <Self as AsType<fensor::SparseNode<T>>>::as_type_mut(self).map(|node| &mut node.0)
    }

    fn into_type(self) -> Option<b_tree::Node<Vec<Vec<fensor::SparseCell<T>>>>> {
        <Self as AsType<fensor::SparseNode<T>>>::into_type(self).map(|node| node.0)
    }
}
