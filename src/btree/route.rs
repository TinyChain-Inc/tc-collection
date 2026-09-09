use std::ops::Bound;

use pathlink::PathSegment;
use safecast::CastInto;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Map, Transaction};
use tc_value::Value;

use crate::Collection;
use crate::collection::BTreeView;
use crate::route::CollectionState;

type BTreeBounds = (Bound<Value>, Bound<Value>);
struct BTreeOp<S: CollectionState> {
    view: BTreeView<S::Txn>,
    state: std::marker::PhantomData<fn() -> S>,
}

struct DeleteRow<S: CollectionState>(BTreeOp<S>);
struct Contains<S: CollectionState>(BTreeOp<S>);
struct Count<S: CollectionState>(BTreeOp<S>);
struct IsEmpty<S: CollectionState>(BTreeOp<S>);
struct Slice<S: CollectionState>(BTreeOp<S>);
struct Insert<S: CollectionState>(BTreeOp<S>);
struct Delete<S: CollectionState>(BTreeOp<S>);

fn operation<S: CollectionState>(view: &BTreeView<S::Txn>) -> BTreeOp<S> {
    BTreeOp {
        view: view.clone(),
        state: std::marker::PhantomData,
    }
}

impl<S, Txn> tc_ir::Route<S> for BTreeView<Txn>
where
    S: CollectionState<Txn = Txn>,
    Txn: crate::StorageContext,
{
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn tc_ir::Handler<'a, S> + 'a>> {
        let handler: Box<dyn tc_ir::Handler<'a, S> + 'a> = match path {
            [] => Box::new(DeleteRow(operation::<S>(self))),
            [segment] => match segment.as_str() {
                "contains" => Box::new(Contains(operation::<S>(self))),
                "count" => Box::new(Count(operation::<S>(self))),
                "is_empty" => Box::new(IsEmpty(operation::<S>(self))),
                "slice" => Box::new(Slice(operation::<S>(self))),
                "insert" => Box::new(Insert(operation::<S>(self))),
                "delete" => Box::new(Delete(operation::<S>(self))),
                _ => return None,
            },
            _ => return None,
        };
        Some(handler)
    }
}

impl<Txn: crate::StorageContext> BTreeView<Txn> {
    fn slice_from_key<S: CollectionState>(&self, key: S) -> TCResult<Self> {
        if key.is_none() {
            Ok(self.clone())
        } else {
            let (bounds, reverse) = slice_bounds_from_state(key)?;
            Ok(self.slice(bounds, reverse))
        }
    }
}

fn row_from_state<S: CollectionState>(state: S, context: &str) -> TCResult<Vec<Value>> {
    state
        .into_tuple()?
        .into_iter()
        .map(|value| value.into_value())
        .collect::<TCResult<Vec<_>>>()
        .map_err(|_| TCError::bad_request(format!("expected {context} values")))
}

fn slice_bounds_from_state<S: CollectionState>(key: S) -> TCResult<(BTreeBounds, bool)> {
    let map = key.into_map()?;
    let start = bound_from_map(&map, "start", Bound::Included)?;
    let end = bound_from_map(&map, "end", Bound::Excluded)?;
    let reverse_id: Id = "reverse".parse().expect("fixed reverse id");
    let reverse = map
        .get(&reverse_id)
        .cloned()
        .map(|value| value.into_value())
        .transpose()?
        .map(|value| match value {
            Value::Number(number) => Ok(number.cast_into()),
            other => Err(TCError::bad_request(format!(
                "expected BTree reverse boolean, found {other:?}"
            ))),
        })
        .transpose()?
        .unwrap_or(false);
    Ok(((start, end), reverse))
}

fn bound_from_map<S: CollectionState>(
    map: &Map<S>,
    name: &str,
    bound: fn(Value) -> Bound<Value>,
) -> TCResult<Bound<Value>> {
    let id: Id = name.parse().expect("fixed bound id");
    match map.get(&id).cloned() {
        Some(value) if !value.is_none() => Ok(bound(value.into_value()?)),
        _ => Ok(Bound::Unbounded),
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for DeleteRow<S> {
    fn delete<'txn>(self: Box<Self>) -> Option<tc_ir::DeleteHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                self.0
                    .view
                    .btree
                    .delete_row(txn, row_from_state(S::from(key), "BTree row")?)
                    .await
                    .map_err(|err| TCError::bad_request(err.to_string()))
            })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for Contains<S> {
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                Ok(S::from(Value::from(
                    self.0
                        .view
                        .btree
                        .contains_row(txn.id(), &row_from_state(S::from(key), "BTree row")?)
                        .await,
                )))
            })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for Count<S> {
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let view = self.0.view.slice_from_key(S::from(key))?;
                Ok(S::from(Value::from(
                    view.btree
                        .slice(view.bounds.clone(), view.reverse)
                        .count(txn.id())
                        .await,
                )))
            })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for IsEmpty<S> {
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let view = self.0.view.slice_from_key(S::from(key))?;
                Ok(S::from(Value::from(
                    view.btree
                        .slice(view.bounds.clone(), view.reverse)
                        .is_empty(txn.id())
                        .await,
                )))
            })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for Slice<S> {
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, key| {
            Box::pin(async move {
                let (bounds, reverse) = slice_bounds_from_state(S::from(key))?;
                Ok(S::from(Collection::BTree(Box::new(
                    self.0.view.slice(bounds, reverse),
                ))))
            })
        }))
    }
}

fn row_param<S: CollectionState>(params: &Map<S>) -> TCResult<Vec<Value>> {
    let row_id: Id = "row".parse().expect("valid row parameter");
    row_from_state(
        params
            .get(&row_id)
            .cloned()
            .ok_or_else(|| TCError::bad_request("missing BTree row parameter"))?,
        "BTree row",
    )
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for Insert<S> {
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                self.0
                    .view
                    .btree
                    .insert_row(txn, row_param(&params)?)
                    .await
                    .map_err(|err| TCError::bad_request(err.to_string()))?;
                Ok(S::from(Value::None))
            })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for Delete<S> {
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                self.0
                    .view
                    .btree
                    .delete_row(txn, row_param(&params)?)
                    .await
                    .map_err(|err| TCError::bad_request(err.to_string()))?;
                Ok(S::from(Value::None))
            })
        }))
    }
}
