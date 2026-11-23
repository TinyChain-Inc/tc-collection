use std::ops::Bound;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use safecast::{CastFrom, TryCastFrom};
use tc_ir::{IntoView, Sha256Hash, Transact, TxnId};
use tc_value::Value;
use tc_value::class::NativeClass;

use crate::btree::{BTree, BTreeColumnSchema, BTreeSlice};
use crate::table::{PersistentTable, Table};
use crate::tensor::Tensor;

/// Schema required to strictly reopen a native persistent collection.
#[derive(Clone, Debug)]
pub enum CollectionSchema {
    BTree(Vec<BTreeColumnSchema>),
    Table(crate::table::TableSchema),
    Tensor(crate::tensor::TensorSchema),
}

impl From<CollectionSchema> for (pathlink::PathBuf, Value) {
    fn from(schema: CollectionSchema) -> Self {
        match schema {
            CollectionSchema::BTree(columns) => (
                crate::BTreeType.path(),
                Value::Tuple(columns.into_iter().map(Value::from).collect()),
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

                if columns.is_empty() {
                    return Err(tc_error::TCError::bad_request(
                        "BTree schema must contain columns",
                    ));
                }

                Ok(Self::BTree(columns))
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

/// A mutable owner or a read-only selection of its transactional contents.
/// Selecting the entire owner does not transfer mutation or lifecycle authority.
/// Schema metadata is immutable after checked construction.
///
/// The former public variants cannot be used to forge a schema/owner pair:
///
/// ```compile_fail
/// use tc_collection::{StorageContext, collection::BTreeView};
/// fn change_schema<T: StorageContext>(view: &mut BTreeView<T>) {
///     view.schema[0].name = "different".into();
/// }
/// ```
///
/// ```compile_fail
/// use tc_collection::{StorageContext, btree::BTree, collection::BTreeView};
/// fn forge<T: StorageContext>(tree: BTree<T>) {
///     let _ = BTreeView::File(Vec::new(), tree);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct BTreeView<Txn: crate::StorageContext> {
    schema: Arc<Vec<BTreeColumnSchema>>,
    source: BTreeSource<Txn>,
}

#[derive(Clone, Debug)]
enum BTreeSource<Txn: crate::StorageContext> {
    File(BTree<Txn>),
    Slice(Box<BTreeSlice<Txn>>),
}

impl<Txn: crate::StorageContext> BTreeView<Txn> {
    /// Pair semantic columns with a matching typed owner without native I/O.
    /// Untyped raw trees remain usable directly, but cannot back a typed view.
    /// Callers must relinquish native aliases before handing an owner to Chain.
    pub fn new(schema: Vec<BTreeColumnSchema>, btree: BTree<Txn>) -> tc_error::TCResult<Self> {
        btree.validate_columns(&schema)?;
        Ok(Self {
            schema: Arc::new(schema),
            source: BTreeSource::File(btree),
        })
    }

    pub fn schema(&self) -> &[BTreeColumnSchema] {
        self.schema.as_slice()
    }

    pub(crate) fn is_persistent(&self) -> bool {
        matches!(&self.source, BTreeSource::File(_))
    }

    /// Return the mutable owner only when this value owns persistent storage.
    pub fn persistent(&self) -> tc_error::TCResult<&BTree<Txn>> {
        match &self.source {
            BTreeSource::File(btree) => Ok(btree),
            BTreeSource::Slice(_) => Err(tc_error::TCError::bad_request(
                "expected a persistent BTree owner",
            )),
        }
    }

    /// Synchronize canonical storage of a persistent owner.
    pub async fn sync(&self) -> tc_error::TCResult<()> {
        self.persistent()?.sync().await.map_err(Into::into)
    }

    /// Make canonical storage durable without publishing pending versions.
    pub async fn sync_all(&self) -> tc_error::TCResult<()> {
        self.persistent()?.sync_all().await.map_err(Into::into)
    }

    /// Select a read-only view, including when the selected range is unbounded.
    pub fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        let slice = match &self.source {
            BTreeSource::File(btree) => btree.slice(bounds, reverse),
            BTreeSource::Slice(slice) => slice.slice(bounds, reverse),
        };

        Self {
            schema: self.schema.clone(),
            source: BTreeSource::Slice(Box::new(slice)),
        }
    }

    pub async fn keys(&self, txn_id: TxnId) -> tc_error::TCResult<crate::btree::Keys> {
        match &self.source {
            BTreeSource::File(btree) => {
                btree
                    .keys(txn_id, (Bound::Unbounded, Bound::Unbounded), false)
                    .await
            }
            BTreeSource::Slice(slice) => slice.keys(txn_id).await,
        }
    }

    async fn hash_contents(&self, id: TxnId) -> tc_error::TCResult<Sha256Hash> {
        async_hash::hash_try_stream::<async_hash::Sha256, _, _, _>(Box::pin(self.keys(id)).await?)
            .await
    }

    pub async fn contains_row(&self, txn_id: TxnId, key: &[Value]) -> tc_error::TCResult<bool> {
        match &self.source {
            BTreeSource::File(btree) => btree.contains_row(txn_id, key).await,
            BTreeSource::Slice(slice) => slice.contains_row(txn_id, key).await,
        }
    }

    pub async fn count(&self, txn_id: TxnId) -> tc_error::TCResult<u64> {
        match &self.source {
            BTreeSource::File(btree) => btree.count(txn_id).await,
            BTreeSource::Slice(slice) => slice.count(txn_id).await,
        }
    }

    pub async fn is_empty(&self, txn_id: TxnId) -> tc_error::TCResult<bool> {
        match &self.source {
            BTreeSource::File(btree) => btree.is_empty(txn_id).await,
            BTreeSource::Slice(slice) => slice.is_empty(txn_id).await,
        }
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        match &self.source {
            BTreeSource::File(btree) => btree.finalized_key_stream().await,
            BTreeSource::Slice(slice) => slice.finalized_key_stream().await,
        }
    }
}

impl<Txn: crate::StorageContext> IntoView for BTreeView<Txn> {
    type Txn = Txn;

    type View = crate::view::BTreeView;

    async fn into_view(self, txn: Txn) -> tc_error::TCResult<Self::View> {
        let keys = self
            .keys(txn.id())
            .await
            .map_err(|error| txn.storage_error(error))?;

        let keys = keys.map_stream(|stream| {
            stream
                .map(move |row| row.map_err(|error| txn.storage_error(error)))
                .boxed()
        });

        Ok(crate::view::BTreeView {
            schema: Arc::unwrap_or_clone(self.schema),
            keys,
        })
    }
}

#[derive(Debug)]
pub enum Collection<Txn: crate::StorageContext> {
    BTree(Box<BTreeView<Txn>>),
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
            Self::BTree(view) => Ok(CollectionSchema::BTree(view.schema().to_vec())),
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
            CollectionSchema::BTree(columns) => {
                let schema = crate::btree::BTreeSchema::try_from(columns.as_slice())?;

                let btree = BTree::try_with_schema(dir, schema).await?;
                Ok(Self::BTree(Box::new(BTreeView::new(columns, btree)?)))
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
            CollectionSchema::BTree(columns) => {
                let schema = crate::btree::BTreeSchema::try_from(columns.as_slice())?;

                Ok(Self::BTree(Box::new(BTreeView::new(
                    columns,
                    // Keep native loader state separate from composed State and
                    // Chain recovery futures on ordinary worker stacks.
                    Box::pin(BTree::load(dir, schema)).await?,
                )?)))
            }
            CollectionSchema::Tensor(schema) => {
                Box::pin(Tensor::load(dir, schema)).await.map(Self::Tensor)
            }
            CollectionSchema::Table(schema) => Box::pin(PersistentTable::load(dir, schema))
                .await
                .map(Self::from)
                .map_err(Into::into),
        }
    }

    /// Hash the class and semantic schema, then ordered native contents, without encoding.
    pub async fn hash(&self, txn_id: TxnId) -> tc_error::TCResult<Sha256Hash> {
        use async_hash::{Digest, Hash, Sha256};

        let (class, schema): (pathlink::PathBuf, Value) = self.schema()?.into();
        let schema_hash = Hash::<Sha256>::hash((class, schema));

        let hash = match self {
            Self::BTree(view) => Box::pin(view.hash_contents(txn_id)).await?,
            Self::Table(table) => Box::pin(table.hash_contents(txn_id)).await?,
            Self::Tensor(tensor) => Box::pin(tensor.hash_contents(txn_id)).await?,
        };

        let mut digest = Sha256::new();
        digest.update(schema_hash);
        digest.update(hash);
        Ok(digest.finalize())
    }

    /// Whether this value is a full persistent collection owner.
    pub fn is_persistent(&self) -> bool {
        match self {
            Self::BTree(view) => view.is_persistent(),
            Self::Table(table) => table.is_persistent(),
            Self::Tensor(tensor) => tensor.is_persistent(),
        }
    }

    /// Synchronize canonical storage without publishing pending transaction versions.
    pub async fn sync(&self) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(view) => view.sync().await,
            Self::Table(table) => table.sync().await,
            Self::Tensor(tensor) => tensor.sync().await,
        }
    }

    /// Explicitly make canonical storage durable without publishing pending versions.
    pub async fn sync_all(&self) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(view) => view.sync_all().await,
            Self::Table(table) => table.sync_all().await,
            Self::Tensor(tensor) => tensor.sync_all().await,
        }
    }

    pub fn as_btree(&self) -> Option<&BTree<Txn>> {
        match self {
            Self::BTree(btree) => btree.persistent().ok(),
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

impl<Txn: crate::StorageContext> crate::Restore for BTreeView<Txn> {
    type Txn = Txn;

    async fn restore(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let target = self.persistent()?;
        let original = source.persistent()?;
        if self.schema() != source.schema() {
            return Err(tc_error::TCError::bad_request(
                "BTree restoration schema mismatch",
            ));
        }

        target.restore(txn, original).await
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

impl<Txn: crate::StorageContext> crate::CopyFrom<&BTreeView<Txn>> for BTreeView<Txn> {
    type Txn = Txn;

    async fn copy_from(
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
        source: &BTreeView<Txn>,
    ) -> tc_error::TCResult<Self> {
        if matches!(&source.source, BTreeSource::Slice(slice) if slice.is_reversed()) {
            return Err(tc_error::TCError::bad_request(
                "cannot materialize a reversed BTree view in canonical key order",
            ));
        }

        let schema = crate::btree::BTreeSchema::try_from(source.schema())?;

        let target = BTree::try_with_schema(dir, schema).await?;
        let mut keys = source.keys(txn.id()).await?;
        while let Some(mut key) = keys.try_next().await? {
            target
                .load_literal_row(if source.schema().len() == 1 {
                    key.remove(0)
                } else {
                    Value::Tuple(key)
                })
                .await?;
        }

        target.sync().await?;
        Ok(Self {
            schema: source.schema.clone(),
            source: BTreeSource::File(target),
        })
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
            Self::BTree(view) => BTreeView::copy_from(txn, dir, view.as_ref())
                .await
                .map(|view| Self::BTree(Box::new(view))),
            Self::Table(table) => PersistentTable::copy_from(txn, dir, table.as_ref())
                .await
                .map(Self::from),
            Self::Tensor(tensor) => Tensor::copy_from(txn, dir, tensor).await.map(Self::Tensor),
        }
    }
}

impl<Txn: crate::StorageContext> Transact for BTreeView<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        Transact::commit(self.persistent()?, txn_id).await
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { Transact::rollback(self.persistent()?, &txn_id).await }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move { Transact::finalize(self.persistent()?, &txn_id).await }
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
