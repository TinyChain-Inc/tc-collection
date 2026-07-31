#![deny(clippy::needless_question_mark)]

pub use fensor;

pub mod btree;
pub mod collection;
pub mod table;

pub use collection::Collection;
pub use table::Table;
