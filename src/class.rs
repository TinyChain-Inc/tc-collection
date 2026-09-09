use pathlink::{Label, PathBuf, PathSegment, label, path_label};
use tc_value::class::{Class, NativeClass};

const BTREE_PATH: pathlink::PathLabel = path_label(&["state", "collection", "btree"]);
const TABLE_PATH: pathlink::PathLabel = path_label(&["state", "collection", "table"]);
const TENSOR_PATH: pathlink::PathLabel = path_label(&["state", "collection", "tensor"]);

const STATE: Label = label("state");
const COLLECTION: Label = label("collection");

macro_rules! collection_class {
    ($name:ident, $path:ident, $label:literal) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl Class for $name {}

        impl NativeClass for $name {
            fn from_path(path: &[PathSegment]) -> Option<Self> {
                (path.len() == $path.len()
                    && path
                        .iter()
                        .zip($path[..].iter())
                        .all(|(segment, expected)| segment.as_str() == *expected))
                .then_some(Self)
            }

            fn path(&self) -> PathBuf {
                PathBuf::new()
                    .append(STATE)
                    .append(COLLECTION)
                    .append(label($label))
            }
        }
    };
}

collection_class!(BTreeType, BTREE_PATH, "btree");
collection_class!(TableType, TABLE_PATH, "table");
collection_class!(TensorType, TENSOR_PATH, "tensor");

/// TinyChain collection classes, owned by the collection crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
    Table(TableType),
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        BTreeType::from_path(path)
            .map(Self::BTree)
            .or_else(|| TableType::from_path(path).map(Self::Table))
            .or_else(|| TensorType::from_path(path).map(Self::Tensor))
    }

    fn path(&self) -> PathBuf {
        match self {
            Self::BTree(class) => class.path(),
            Self::Table(class) => class.path(),
            Self::Tensor(class) => class.path(),
        }
    }
}

impl From<BTreeType> for CollectionType {
    fn from(class: BTreeType) -> Self {
        Self::BTree(class)
    }
}

impl From<TableType> for CollectionType {
    fn from(class: TableType) -> Self {
        Self::Table(class)
    }
}

impl From<TensorType> for CollectionType {
    fn from(class: TensorType) -> Self {
        Self::Tensor(class)
    }
}
