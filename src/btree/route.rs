use std::ops::Bound;

use pathlink::PathSegment;
use safecast::CastInto;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Map, Scalar};
use tc_value::Value;

use crate::Collection;
use crate::collection::BTreeView;
use crate::route::CollectionState;

type BTreeBounds = (Bound<Value>, Bound<Value>);
#[derive(Clone)]
pub struct BTreeRoutes<S: CollectionState> {
    view: BTreeView<S::Txn>,
    state: std::marker::PhantomData<fn() -> S>,
}

impl<S: CollectionState> BTreeRoutes<S> {
    pub fn new(view: BTreeView<S::Txn>) -> Self {
        Self {
            view,
            state: std::marker::PhantomData,
        }
    }
}

pub struct BTreeRoute<S: CollectionState> {
    view: BTreeView<S::Txn>,
    path: Vec<PathSegment>,
    state: std::marker::PhantomData<fn() -> S>,
}

impl<S: CollectionState> tc_ir::Route<S> for BTreeRoutes<S> {
    type Handler = BTreeRoute<S>;

    fn route(&self, path: &[PathSegment]) -> Option<Self::Handler> {
        let known = path.is_empty()
            || matches!(path, [segment] if matches!(segment.as_str(), "contains" | "count" | "is_empty" | "slice" | "insert" | "delete"));
        known.then(|| BTreeRoute {
            view: self.view.clone(),
            path: path.to_vec(),
            state: std::marker::PhantomData,
        })
    }
}
impl<S: CollectionState> BTreeRoute<S> {
    async fn get(&self, txn: &S::Txn, request: Scalar) -> TCResult<S> {
        self.view
            .get(&self.path, S::from_scalar(request), txn)
            .await?
            .ok_or_else(|| TCError::method_not_allowed(tc_ir::Method::Get, "BTree"))
    }

    async fn post(&self, txn: &S::Txn, request: Map<S>) -> TCResult<S> {
        self.view
            .post(&self.path, request, txn)
            .await?
            .ok_or_else(|| TCError::method_not_allowed(tc_ir::Method::Post, "BTree"))
    }

    async fn delete(&self, txn: &S::Txn, request: Scalar) -> TCResult<()> {
        let _ = self
            .view
            .delete(&self.path, S::from_scalar(request), txn)
            .await?
            .ok_or_else(|| TCError::method_not_allowed(tc_ir::Method::Delete, "BTree"))?;
        Ok(())
    }
}

impl<Txn: crate::StorageContext> BTreeView<Txn> {
    async fn get<S: CollectionState<Txn = Txn>>(
        &self,
        path: &[PathSegment],
        key: S,
        txn: &Txn,
    ) -> TCResult<Option<S>> {
        if path.len() != 1 {
            return Ok(None);
        }
        let state = match path[0].as_str() {
            "contains" => S::from_value(Value::from(
                self.btree
                    .contains_row(txn.id(), &row_from_state(key, "BTree row")?)
                    .await,
            )),
            "count" => {
                let view = self.slice_from_key(key)?;
                S::from_value(Value::from(
                    view.btree
                        .slice(view.bounds.clone(), view.reverse)
                        .count(txn.id())
                        .await,
                ))
            }
            "is_empty" => {
                let view = self.slice_from_key(key)?;
                S::from_value(Value::from(
                    view.btree
                        .slice(view.bounds.clone(), view.reverse)
                        .is_empty(txn.id())
                        .await,
                ))
            }
            "slice" => {
                let (bounds, reverse) = slice_bounds_from_state(key)?;
                S::from_collection(Collection::BTree(Box::new(self.slice(bounds, reverse))))
            }
            _ => return Ok(None),
        };
        Ok(Some(state))
    }

    async fn post<S: CollectionState<Txn = Txn>>(
        &self,
        path: &[PathSegment],
        params: Map<S>,
        txn: &Txn,
    ) -> TCResult<Option<S>> {
        if path.len() != 1 {
            return Ok(None);
        }
        let row_id: Id = "row".parse().expect("valid row parameter");
        let row = row_from_state(
            params
                .get(&row_id)
                .cloned()
                .ok_or_else(|| TCError::bad_request("missing BTree row parameter"))?,
            "BTree row",
        )?;
        match path[0].as_str() {
            "insert" => self.btree.insert_row(txn, row).await,
            "delete" => self.btree.delete_row(txn, row).await,
            _ => return Ok(None),
        }
        .map_err(|err| TCError::bad_request(err.to_string()))?;
        Ok(Some(S::none()))
    }

    async fn delete<S: CollectionState<Txn = Txn>>(
        &self,
        path: &[PathSegment],
        key: S,
        txn: &Txn,
    ) -> TCResult<Option<S>> {
        if !path.is_empty() {
            return Ok(None);
        }
        self.btree
            .delete_row(txn, row_from_state(key, "BTree row")?)
            .await
            .map_err(|err| TCError::bad_request(err.to_string()))?;
        Ok(Some(S::none()))
    }

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

impl<S: CollectionState> tc_ir::Handler<S> for BTreeRoute<S> {
    async fn get(&self, txn: &S::Txn, key: Scalar) -> TCResult<S> {
        BTreeRoute::get(self, txn, key).await
    }

    async fn post(&self, txn: &S::Txn, params: Map<S>) -> TCResult<S> {
        BTreeRoute::post(self, txn, params).await
    }

    async fn delete(&self, txn: &S::Txn, key: Scalar) -> TCResult<()> {
        BTreeRoute::delete(self, txn, key).await
    }
}
