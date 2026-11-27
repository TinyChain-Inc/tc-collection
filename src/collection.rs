use async_hash::{Digest, Hash, Sha256, hash_try_stream};
use safecast::{CastFrom, TryCastFrom};
use tc_ir::{Sha256Hash, Transact, TxnId};
use tc_value::Value;
use tc_value::class::NativeClass;

use crate::btree::{BTree, BTreeColumnSchema, BTreeSchema};
use crate::table::{PersistentTable, Table};
use crate::tensor::Tensor;

fn hash_collection(descriptor: (pathlink::PathBuf, Value), contents: Sha256Hash) -> Sha256Hash {
    let mut digest = Sha256::new();
    digest.update(Hash::<Sha256>::hash(descriptor));
    digest.update(contents);
    digest.finalize()
}

/// Schema required to strictly reopen a native persistent collection.
#[derive(Clone, Debug)]
pub enum CollectionSchema {
    BTree(BTreeSchema),
    Table(crate::table::TableSchema),
    Tensor(crate::tensor::TensorSchema),
}

impl From<CollectionSchema> for (pathlink::PathBuf, Value) {
    fn from(schema: CollectionSchema) -> Self {
        match schema {
            CollectionSchema::BTree(columns) => (
                crate::BTreeType.path(),
                Value::Tuple(columns.columns().iter().cloned().map(Value::from).collect()),
            ),
            CollectionSchema::Table(schema) => (crate::TableType.path(), Value::cast_from(schema)),
            CollectionSchema::Tensor(schema) => (crate::TensorType.path(), schema.into()),
        }
    }
}

impl TryFrom<(pathlink::PathBuf, Value)> for CollectionSchema {
    type Error = tc_error::TCError;

    fn try_from((path, schema): (pathlink::PathBuf, Value)) -> Result<Self, Self::Error> {
        let class = crate::CollectionType::from_path(&path)
            .ok_or_else(|| tc_error::TCError::bad_request("invalid collection class"))?;

        match class {
            crate::CollectionType::BTree(_) => {
                let Value::Tuple(columns) = schema else {
                    return Err(tc_error::TCError::bad_request(
                        "expected BTree schema columns",
                    ));
                };

                let columns: Vec<_> = columns
                    .into_iter()
                    .map(|column| {
                        BTreeColumnSchema::try_cast_from(column, |_| {
                            tc_error::TCError::bad_request("invalid BTree schema column")
                        })
                    })
                    .collect::<Result<_, _>>()?;

                BTreeSchema::try_from(columns).map(Self::BTree)
            }
            crate::CollectionType::Table(_) => {
                crate::table::TableSchema::try_from(schema).map(Self::Table)
            }
            crate::CollectionType::Tensor(_) => {
                crate::tensor::TensorSchema::try_from(schema).map(Self::Tensor)
            }
        }
    }
}

#[derive(Debug)]
pub enum Collection<Txn: crate::StorageContext> {
    BTree(Box<BTree<Txn>>),
    Table(Box<Table<Txn>>),
    Tensor(Tensor<Txn>),
}

impl<Txn: crate::StorageContext> Clone for Collection<Txn> {
    fn clone(&self) -> Self {
        match self {
            Self::BTree(btree) => Self::BTree(Box::new((**btree).clone())),
            Self::Table(table) => Self::Table(Box::new((**table).clone())),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
        }
    }
}

impl<Txn: crate::StorageContext> From<Tensor<Txn>> for Collection<Txn> {
    fn from(tensor: Tensor<Txn>) -> Self {
        Self::Tensor(tensor)
    }
}

impl<Txn: crate::StorageContext> From<PersistentTable<Txn>> for Collection<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self::Table(Box::new(table.into()))
    }
}

impl<Txn: crate::StorageContext> TryCastFrom<Collection<Txn>> for Tensor<Txn> {
    fn can_cast_from(collection: &Collection<Txn>) -> bool {
        matches!(collection, Collection::Tensor(_))
    }

    fn opt_cast_from(collection: Collection<Txn>) -> Option<Self> {
        match collection {
            Collection::Tensor(tensor) => Some(tensor),
            Collection::BTree(_) | Collection::Table(_) => None,
        }
    }
}

impl<Txn: crate::StorageContext> From<Table<Txn>> for Collection<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self::Table(Box::new(table))
    }
}

impl<Txn: crate::StorageContext> Collection<Txn> {
    pub fn schema(&self) -> tc_error::TCResult<CollectionSchema> {
        match self {
            Self::BTree(view) => Ok(CollectionSchema::BTree(view.schema().clone())),
            Self::Table(table) => Ok(CollectionSchema::Table(table.schema().clone())),
            Self::Tensor(tensor) => tensor.schema().map(CollectionSchema::Tensor),
        }
    }

    /// Create an unpublished persistent collection in caller-delegated empty storage.
    pub async fn create(
        dir: freqfs::DirLock<Txn::File>,
        schema: CollectionSchema,
    ) -> tc_error::TCResult<Self> {
        match schema {
            CollectionSchema::BTree(schema) => {
                Ok(Self::BTree(Box::new(BTree::create(dir, schema).await?)))
            }
            CollectionSchema::Table(schema) => {
                Ok(PersistentTable::try_new(dir, schema).await?.into())
            }
            CollectionSchema::Tensor(schema) => {
                Tensor::create(dir, schema, fensor::MAX_BLOCK_CAPACITY)
                    .await
                    .map(Self::Tensor)
            }
        }
    }

    /// Strictly load native storage using its recorded semantic schema.
    pub async fn load(
        dir: freqfs::DirLock<Txn::File>,
        schema: CollectionSchema,
    ) -> tc_error::TCResult<Self> {
        match schema {
            CollectionSchema::BTree(schema) => Ok(Self::BTree(Box::new(
                Box::pin(BTree::load(dir, schema)).await?,
            ))),
            CollectionSchema::Tensor(schema) => {
                Box::pin(Tensor::load(dir, schema)).await.map(Self::Tensor)
            }
            CollectionSchema::Table(schema) => Box::pin(PersistentTable::load(dir, schema))
                .await
                .map(Self::from)
                .map_err(Into::into),
        }
    }

    /// Hash the class, logical schema, and contents without encoding.
    pub async fn hash(&self, txn_id: TxnId) -> tc_error::TCResult<Sha256Hash> {
        if let Self::Tensor(tensor) = self
            && matches!(tensor.layout(), crate::tensor::Layout::Sparse { .. })
        {
            let descriptor = (
                crate::TensorType.path(),
                tensor.schema()?.sparse_hash_schema(),
            );
            let elements = tensor.sparse_elements(txn_id).await?;
            let contents = hash_try_stream::<Sha256, _, _, _>(elements).await?;
            return Ok(hash_collection(descriptor, contents));
        }

        self.hash_native_capture(txn_id).await
    }

    /// Hash a native capture with its reopening descriptor for local integrity checks.
    pub async fn hash_native_capture(&self, txn_id: TxnId) -> tc_error::TCResult<Sha256Hash> {
        let descriptor = self.schema()?.into();
        let hash = match self {
            Self::BTree(view) => Box::pin(view.hash_contents(txn_id)).await?,
            Self::Table(table) => Box::pin(table.hash_contents(txn_id)).await?,
            Self::Tensor(tensor) => Box::pin(tensor.hash_contents(txn_id)).await?,
        };

        Ok(hash_collection(descriptor, hash))
    }

    /// Complete accepted recovery work on an exclusively owned, unpublished value.
    /// A durable decision skips the callback; otherwise it runs once before commit.
    pub async fn recover<Replay, Replayed>(
        &self,
        id: TxnId,
        replay: Replay,
    ) -> tc_error::TCResult<()>
    where
        Replay: FnOnce() -> Replayed + Send,
        Replayed: std::future::Future<Output = tc_error::TCResult<()>> + Send,
    {
        match self {
            Self::BTree(owner) => owner.recover(id, replay).await,
            Self::Table(owner) => owner.recover(id, replay).await,
            Self::Tensor(owner) => owner.recover(id, replay).await,
        }
    }

    /// Whether this value is a full persistent collection owner.
    pub fn is_persistent(&self) -> bool {
        match self {
            Self::BTree(view) => view.is_persistent(),
            Self::Table(table) => table.is_persistent(),
            Self::Tensor(tensor) => tensor.is_persistent(),
        }
    }

    pub fn as_btree(&self) -> Option<&BTree<Txn>> {
        match self {
            Self::BTree(btree) => btree.is_persistent().then_some(btree),
            _ => None,
        }
    }

    pub fn as_table(&self) -> Option<&Table<Txn>> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }
}

impl<Txn: crate::StorageContext> crate::Restore for Collection<Txn> {
    type Txn = Txn;

    /// Stage a same-kind, same-schema native replacement in the caller's transaction.
    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        match (self, source) {
            (Self::BTree(target), Self::BTree(source)) => target.restore(txn, source).await,
            (Self::Table(target), Self::Table(source)) => target.restore(txn, source).await,
            (Self::Tensor(target), Self::Tensor(source)) => target.restore(txn, source).await,
            _ => Err(tc_error::TCError::bad_request(
                "unsupported collection restoration",
            )),
        }
    }
}

impl<Txn: crate::StorageContext> crate::CopyFrom<&Collection<Txn>> for Collection<Txn> {
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
        source: &Collection<Txn>,
    ) -> tc_error::TCResult<Self> {
        match source {
            Self::BTree(view) => BTree::copy_from(txn, dir, view.as_ref())
                .await
                .map(|view| Self::BTree(Box::new(view))),
            Self::Table(table) => PersistentTable::copy_from(txn, dir, table.as_ref())
                .await
                .map(Self::from),
            Self::Tensor(tensor) => Tensor::copy_from(txn, dir, tensor).await.map(Self::Tensor),
        }
    }
}

impl<Txn: crate::StorageContext> Transact for Collection<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(btree) => Transact::commit(btree.as_ref(), txn_id).await,
            Self::Table(table) => Transact::commit(table.as_ref(), txn_id).await,
            Self::Tensor(tensor) => Transact::commit(tensor, txn_id).await,
        }
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match self {
                Self::BTree(btree) => Transact::rollback(btree.as_ref(), &txn_id).await,
                Self::Table(table) => Transact::rollback(table.as_ref(), &txn_id).await,
                Self::Tensor(tensor) => Transact::rollback(tensor, &txn_id).await,
            }
        }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match self {
                Self::BTree(btree) => Transact::finalize(btree.as_ref(), &txn_id).await,
                Self::Table(table) => Transact::finalize(table.as_ref(), &txn_id).await,
                Self::Tensor(tensor) => Transact::finalize(tensor, &txn_id).await,
            }
        }
    }
}
