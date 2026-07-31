#![deny(clippy::needless_question_mark)]

pub use fensor;

pub mod btree;
pub mod collection;
pub mod table;

pub use collection::{Collection, CollectionRoute, CollectionRouter};
pub use table::{TableResponse, TableRoute, TableRouter, TableStatic, TableStaticRoute};
