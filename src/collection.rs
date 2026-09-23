use std::cmp::Ordering;
use std::ops::Bound;

use collate::Collate;
use futures::TryStreamExt;
use safecast::{CastFrom, TryCastFrom};
use tc_ir::{Sha256Hash, Transact, TxnId};
use tc_value::class::NativeClass;
use tc_value::{Value, ValueCollator};

use crate::btree::{BTree, BTreeColumnSchema};
use crate::table::{PersistentTable, Table};
use crate::tensor::Tensor;

/// Schema required to strictly reopen a native persistent collection.
#[derive(Clone, Debug)]
pub enum CollectionSchema {
    BTree(Vec<BTreeColumnSchema>),
    Table(crate::table::TableSchema),
}

impl From<CollectionSchema> for (pathlink::PathBuf, Value) {
    fn from(schema: CollectionSchema) -> Self {
        match schema {
            CollectionSchema::BTree(columns) => (
                crate::BTreeType.path(),
                Value::Tuple(columns.into_iter().map(Value::from).collect()),
            ),
            CollectionSchema::Table(schema) => (crate::TableType.path(), Value::cast_from(schema)),
        }
    }
}

impl TryCastFrom<(pathlink::PathBuf, Value)> for CollectionSchema {
    fn can_cast_from(value: &(pathlink::PathBuf, Value)) -> bool {
        Self::opt_cast_from(value.clone()).is_some()
    }

    fn opt_cast_from((path, schema): (pathlink::PathBuf, Value)) -> Option<Self> {
        match crate::CollectionType::from_path(&path)? {
            crate::CollectionType::BTree(_) => {
                let Value::Tuple(columns) = schema else {
                    return None;
                };
                let columns: Vec<_> = columns
                    .into_iter()
                    .map(BTreeColumnSchema::opt_cast_from)
                    .collect::<Option<_>>()?;
                crate::btree::BTreeSchema::can_cast_from(&columns).then_some(Self::BTree(columns))
            }
            crate::CollectionType::Table(_) => {
                crate::table::TableSchema::opt_cast_from(schema).map(Self::Table)
            }
            crate::CollectionType::Tensor(_) => None,
        }
    }
}

#[derive(Debug)]
pub struct BTreeView<Txn: crate::StorageContext> {
    pub schema: Vec<BTreeColumnSchema>,
    pub btree: BTree<Txn>,
    pub bounds: (Bound<Value>, Bound<Value>),
    pub reverse: bool,
}

impl<Txn: crate::StorageContext> Clone for BTreeView<Txn> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            btree: self.btree.clone(),
            bounds: self.bounds.clone(),
            reverse: self.reverse,
        }
    }
}

impl<Txn: crate::StorageContext> BTreeView<Txn> {
    /// Copy the visible keys into unpublished caller-delegated storage.
    pub async fn copy_into(
        &self,
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
    ) -> tc_error::TCResult<Self> {
        let schema = crate::btree::BTreeSchema::try_cast_from(self.schema.clone(), |_| {
            tc_error::TCError::bad_request("invalid BTree schema")
        })?;
        let target = BTree::try_with_schema(dir, schema)?;

        let mut keys = self
            .btree
            .keys(txn.id(), self.bounds.clone(), self.reverse)
            .await?;
        while let Some(mut key) = keys.try_next().await? {
            target
                .load_literal_row(if self.schema.len() == 1 {
                    key.remove(0)
                } else {
                    Value::Tuple(key)
                })
                .await?;
        }

        target.sync().await?;
        Ok(Self::new(self.schema.clone(), target))
    }

    pub fn new(schema: Vec<BTreeColumnSchema>, btree: BTree<Txn>) -> Self {
        Self {
            schema,
            btree,
            bounds: (Bound::Unbounded, Bound::Unbounded),
            reverse: false,
        }
    }

    pub fn slice(&self, bounds: (Bound<Value>, Bound<Value>), reverse: bool) -> Self {
        Self {
            schema: self.schema.clone(),
            btree: self.btree.clone(),
            bounds: (
                max_lower_bound(self.bounds.0.clone(), bounds.0),
                min_upper_bound(self.bounds.1.clone(), bounds.1),
            ),
            reverse,
        }
    }

    pub async fn finalized_key_stream(&self) -> std::io::Result<b_tree::Keys<Value>> {
        self.btree
            .finalized_key_stream_in(self.bounds.clone(), self.reverse)
            .await
    }
}

#[derive(Debug)]
pub enum Collection<Txn: crate::StorageContext> {
    BTree(Box<BTreeView<Txn>>),
    Table(Box<Table<Txn>>),
    Tensor(Tensor),
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

impl<Txn: crate::StorageContext> From<PersistentTable<Txn>> for Collection<Txn> {
    fn from(table: PersistentTable<Txn>) -> Self {
        Self::Table(Box::new(table.into()))
    }
}

fn max_lower_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => Bound::Included(max_value(left, right)),
        (Bound::Included(left), Bound::Excluded(right)) => {
            if compare_values(&left, &right) == Ordering::Greater {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if compare_values(&left, &right) == Ordering::Less {
                Bound::Included(right)
            } else {
                Bound::Excluded(left)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => Bound::Excluded(max_value(left, right)),
    }
}

fn min_upper_bound(left: Bound<Value>, right: Bound<Value>) -> Bound<Value> {
    match (left, right) {
        (Bound::Unbounded, bound) | (bound, Bound::Unbounded) => bound,
        (Bound::Included(left), Bound::Included(right)) => Bound::Included(min_value(left, right)),
        (Bound::Included(left), Bound::Excluded(right)) => {
            if compare_values(&left, &right) == Ordering::Less {
                Bound::Included(left)
            } else {
                Bound::Excluded(right)
            }
        }
        (Bound::Excluded(left), Bound::Included(right)) => {
            if compare_values(&left, &right) != Ordering::Greater {
                Bound::Excluded(left)
            } else {
                Bound::Included(right)
            }
        }
        (Bound::Excluded(left), Bound::Excluded(right)) => Bound::Excluded(min_value(left, right)),
    }
}

fn compare_values(left: &Value, right: &Value) -> Ordering {
    ValueCollator::default().cmp(left, right)
}

fn max_value(left: Value, right: Value) -> Value {
    if compare_values(&left, &right) == Ordering::Less {
        right
    } else {
        left
    }
}

fn min_value(left: Value, right: Value) -> Value {
    if compare_values(&left, &right) == Ordering::Greater {
        right
    } else {
        left
    }
}

impl<Txn: crate::StorageContext> TryCastFrom<Collection<Txn>> for Tensor {
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
            Self::BTree(view) => Ok(CollectionSchema::BTree(view.schema.clone())),
            Self::Table(table) => Ok(CollectionSchema::Table(table.schema().clone())),
            Self::Tensor(_) => Err(tc_error::TCError::bad_request(
                "persistent Tensor values are not supported",
            )),
        }
    }

    /// Copy a consistent value into unpublished caller-owned storage.
    pub async fn copy_into(
        &self,
        txn: &Txn,
        dir: freqfs::DirLock<Txn::File>,
    ) -> tc_error::TCResult<Self> {
        match self {
            Self::BTree(view) => view
                .copy_into(txn, dir)
                .await
                .map(|v| Self::BTree(Box::new(v))),
            Self::Table(table) => table.copy_into(txn, dir).await.map(Self::from),
            Self::Tensor(_) => Err(tc_error::TCError::bad_request(
                "persistent Tensor values are not supported",
            )),
        }
    }

    /// Strictly load native storage using its recorded semantic schema.
    pub async fn load(
        dir: freqfs::DirLock<Txn::File>,
        schema: CollectionSchema,
    ) -> tc_error::TCResult<Self> {
        match schema {
            CollectionSchema::BTree(columns) => {
                let schema = crate::btree::BTreeSchema::try_cast_from(columns.clone(), |_| {
                    tc_error::TCError::bad_request("invalid BTree schema")
                })?;

                Ok(Self::BTree(Box::new(BTreeView::new(
                    columns,
                    BTree::load(dir, schema).await?,
                ))))
            }
            CollectionSchema::Table(schema) => PersistentTable::load(dir, schema)
                .await
                .map(Self::from)
                .map_err(Into::into),
        }
    }

    /// Hash the class and semantic schema, then ordered native contents, without encoding.
    pub async fn hash(&self, txn_id: TxnId) -> tc_error::TCResult<Sha256Hash> {
        use async_hash::{Digest, Hash, Sha256, hash_try_stream};

        let (class, schema): (pathlink::PathBuf, Value) = self.schema()?.into();
        let schema_hash = Hash::<Sha256>::hash((class, schema));

        let hash = match self {
            Self::BTree(view) => {
                hash_try_stream::<Sha256, _, _, _>(
                    view.btree
                        .keys(txn_id, view.bounds.clone(), view.reverse)
                        .await?,
                )
                .await?
            }
            Self::Table(table) => {
                hash_try_stream::<Sha256, _, _, _>(
                    table.row_stream(txn_id).await?.map_ok(|row| row.into_vec()),
                )
                .await?
            }
            Self::Tensor(_) => {
                return Err(tc_error::TCError::bad_request(
                    "persistent Tensor values are not supported",
                ));
            }
        };

        let mut digest = Sha256::new();
        digest.update(schema_hash);
        digest.update(hash);
        Ok(digest.finalize())
    }

    /// Whether this value is a full persistent owner rather than a view or Tensor.
    pub fn is_persistent(&self) -> bool {
        match self {
            Self::BTree(view) => {
                view.bounds == (Bound::Unbounded, Bound::Unbounded) && !view.reverse
            }
            Self::Table(table) => table.is_persistent(),
            Self::Tensor(_) => false,
        }
    }

    /// Stage a same-kind, same-schema native replacement in the caller's transaction.
    pub async fn restore_from(&self, txn: &Txn, source: &Self) -> tc_error::TCResult<()> {
        let schema: (pathlink::PathBuf, Value) = self.schema()?.into();
        let source_schema: (pathlink::PathBuf, Value) = source.schema()?.into();
        if !self.is_persistent() || !source.is_persistent() || schema != source_schema {
            return Err(tc_error::TCError::bad_request(
                "restoration requires matching persistent collections",
            ));
        }
        match (self, source) {
            (Self::BTree(target), Self::BTree(source)) => {
                target.btree.restore_from(txn, &source.btree).await
            }
            (Self::Table(target), Self::Table(source)) => {
                target
                    .persistent()?
                    .restore_from(txn, source.persistent()?)
                    .await
            }
            _ => Err(tc_error::TCError::bad_request(
                "unsupported collection restoration",
            )),
        }
    }

    /// Synchronize canonical storage without publishing pending transaction versions.
    pub async fn sync(&self) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(view) => view.btree.sync().await.map_err(Into::into),
            Self::Table(table) => table.sync().await,
            Self::Tensor(_) => Err(tc_error::TCError::bad_request(
                "persistent Tensor subjects are not supported",
            )),
        }
    }

    /// Explicitly make canonical storage durable without publishing pending versions.
    pub async fn sync_all(&self) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(view) => view.btree.sync_all().await.map_err(Into::into),
            Self::Table(table) => table.sync_all().await,
            Self::Tensor(_) => Err(tc_error::TCError::bad_request(
                "persistent Tensor subjects are not supported",
            )),
        }
    }
    pub fn as_btree(&self) -> Option<&BTree<Txn>> {
        match self {
            Self::BTree(btree) => Some(&btree.btree),
            _ => None,
        }
    }

    pub fn into_btree(self) -> BTree<Txn> {
        match self {
            Self::BTree(btree) => btree.btree,
            _ => panic!("Collection is not a BTree"),
        }
    }

    pub fn as_table(&self) -> Option<&Table<Txn>> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }

    pub fn into_table(self) -> Table<Txn> {
        match self {
            Self::Table(table) => *table,
            _ => panic!("Collection is not a Table"),
        }
    }
}

impl<Txn: crate::StorageContext> Transact for Collection<Txn> {
    async fn commit(&self, txn_id: TxnId) -> tc_error::TCResult<()> {
        match self {
            Self::BTree(btree) => Transact::commit(&btree.btree, txn_id).await?,
            Self::Table(table) => {
                if let Table::File(t) = table.as_ref() {
                    Transact::commit(t, txn_id).await?;
                }
            }
            Self::Tensor(_) => {}
        }

        Ok(())
    }

    fn rollback(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::rollback(&btree.btree, &txn_id).await?,
                Self::Table(table) => {
                    if let Table::File(t) = table.as_ref() {
                        Transact::rollback(t, &txn_id).await?;
                    }
                }
                Self::Tensor(_) => {}
            }

            Ok(())
        }
    }

    fn finalize(
        &self,
        txn_id: &TxnId,
    ) -> impl std::future::Future<Output = tc_error::TCResult<()>> + Send {
        let txn_id = *txn_id;
        async move {
            match &self {
                Self::BTree(btree) => Transact::finalize(&btree.btree, &txn_id).await?,
                Self::Table(table) => {
                    if let Table::File(t) = table.as_ref() {
                        Transact::finalize(t, &txn_id).await?;
                    }
                }
                Self::Tensor(_) => {}
            }

            Ok(())
        }
    }
}
