//! Caller-owned Tensor storage shared by unit regressions and production benchmarks.
use std::sync::Arc;

use freqfs::DirLock;
use futures::{FutureExt, TryStreamExt};
use number_general::FloatType;
use safecast::CastInto;
use tc_ir::{NetworkTime, TxnId};
use tc_value::NumberType;

pub use super::storage::{TempRoot, TestTxn as Txn};
use super::{PersistentFile, StorageContext, Tensor, TensorSchema};

// A small resident target shared by regressions and benchmarks: 32,768 f64
// values before metadata. Larger workloads can exercise eviction without large
// fixture memory. This is a reproducible baseline, not a measured optimum.
pub const DEFAULT_CACHE_BYTES: usize = 256 * 1024;

// Limit explicit Tensor materialization to 8,192 f64 values before metadata;
// streaming larger logical tensors remains supported and is tested separately.
const MATERIALIZED_BYTES: usize = 64 * 1024;

pub fn id(n: u16) -> TxnId {
    TxnId::from_parts(NetworkTime::from_nanos(1), n)
}

pub async fn setup() -> (Arc<TempRoot>, Txn) {
    setup_with_capacity(DEFAULT_CACHE_BYTES).await
}

pub async fn setup_with_capacity(capacity: usize) -> (Arc<TempRoot>, Txn) {
    Txn::owned(id(10), capacity, MATERIALIZED_BYTES)
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
