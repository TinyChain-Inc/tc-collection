//! Caller-owned Tensor storage shared by unit regressions and production benchmarks.
use std::sync::Arc;

use freqfs::DirLock;
use futures::{FutureExt, TryStreamExt};
use number_general::FloatType;
use safecast::CastInto;

pub use super::storage::{TempRoot, TestTxn as Txn};
use super::{PersistentFile, StorageContext, Tensor, TensorSchema};
use tc_ir::{NetworkTime, TxnId};
use tc_value::NumberType;

pub fn id(n: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), n)
}

pub async fn setup() -> (Arc<TempRoot>, Txn) {
    setup_with_capacity(256 * 1024).await
}

pub async fn setup_with_capacity(capacity: usize) -> (Arc<TempRoot>, Txn) {
    Txn::owned(id(10), capacity, 64 * 1024)
}

pub async fn create(
    txn: &Txn,
    shape: fensor::Shape,
    sparse: bool,
    capacity: usize,
) -> (Tensor<Txn>, DirLock<PersistentFile>) {
    let dir = txn.subcontext_unique().context().await.unwrap();
    let axis = sparse.then_some(shape.len() - 1);
    let schema = TensorSchema::new(NumberType::Float(FloatType::F64), shape, axis).unwrap();
    (
        Tensor::create(dir.clone(), schema, capacity)
            .boxed()
            .await
            .unwrap(),
        dir,
    )
}

pub async fn values(tensor: &Tensor<Txn>, id: TxnId) -> Vec<f64> {
    tensor
        .elements(id)
        .await
        .unwrap()
        .map_ok(|n| -> f64 { n.cast_into() })
        .try_collect()
        .await
        .unwrap()
}
