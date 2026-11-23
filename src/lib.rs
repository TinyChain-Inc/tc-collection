#![forbid(unsafe_code)]
#![deny(clippy::needless_question_mark)]

mod persistence;
mod persistent_file;
pub use persistent_file::{CollectionFile, CollectionNode, PersistentFile};

mod stream;
mod txn;
pub use txn::StorageContext;

pub mod btree;
pub mod collection;
pub mod table;
pub mod tensor;

pub use collection::Collection;
pub use table::Table;
mod class;
pub use class::{BTreeType, CollectionType, TableType, TensorType};
mod decode;
pub use decode::decode_collection;
mod encode;
mod view;
pub use view::CollectionView;

pub mod route;
pub use fensor::TensorMetadata;

pub use route::CollectionState;
pub type TensorSparseNode<T> = fensor::SparseNode<T>;

pub type TensorIndexNode = b_table::Node<u64>;

#[cfg(feature = "complex")]
pub use fensor::complex::{Complex32 as TensorComplex32, Complex64 as TensorComplex64};

#[cfg(test)]
#[path = "../tests/common/unit.rs"]
mod test;
