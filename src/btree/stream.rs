use tc_error::TCResult;
use tc_value::Value;

/// A transaction-consistent, permit-bound stream of BTree keys.
pub type Keys = crate::stream::GuardedStream<TCResult<Vec<Value>>, crate::stream::ReadPermit>;
