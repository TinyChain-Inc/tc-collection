#![forbid(unsafe_code)]
#![deny(clippy::needless_question_mark)]

mod class;
mod decode;
mod encode;
mod persistence;
mod persistent_file;
mod stream;
mod txn;
mod view;

pub mod btree;
pub mod collection;
pub mod route;
pub mod table;
pub mod tensor;

pub use fensor::TensorMetadata;
#[cfg(feature = "complex")]
pub use fensor::complex::{Complex32 as TensorComplex32, Complex64 as TensorComplex64};

pub use class::{BTreeType, CollectionType, TableType, TensorType};
pub use collection::Collection;
pub use decode::decode_collection;
pub use persistent_file::{CollectionFile, CollectionNode, PersistentFile};
pub use route::CollectionState;
pub use table::Table;
pub use txn::{CopyFrom, Restore, StorageContext};
pub use view::CollectionView;

pub type TensorSparseNode<T> = fensor::SparseNode<T>;
pub type TensorIndexNode = b_table::Node<u64>;

#[cfg(test)]
#[path = "../tests/common/unit.rs"]
mod test;
