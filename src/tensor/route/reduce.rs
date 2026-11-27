//! Reduce Tensor terminal handlers.

use number_general::Number;
use safecast::CastInto;
use tc_error::{TCError, TCResult};
use tc_ir::Map;
use tc_value::Value;

use crate::{CollectionState, StorageContext};
use super::super::{Tensor, TensorReduceResult};
use super::{axes, no_parameters, optional, parameter, shape, state};

fn reduction<S: CollectionState>(value: TensorReduceResult<S::Txn>) -> S {
    match value {
        TensorReduceResult::Scalar(number) => S::from(Value::Number(number)),
        TensorReduceResult::Tensor(tensor) => state(tensor),
    }
}

fn keepdims<S: CollectionState>(params: &mut Map<S>) -> TCResult<bool> {
    match optional(params, "keepdims")
        .map(CollectionState::into_value)
        .transpose()?
    {
        None => Ok(false),
        Some(Value::Number(number)) => Ok(number.cast_into()),
        _ => Err(TCError::bad_request("expected keepdims boolean")),
    }
}

macro_rules! reduce_handler {
    ($($handler:ident => $method:ident),* $(,)?) => {
        $(pub(super) struct $handler<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

        impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
            for $handler<'a, Txn>
        {
            fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
            where
                'txn: 'a,
            {
                Some(Box::new(move |txn, mut params| {
                    Box::pin(async move {
                        let axes = optional(&mut params, "axes").map(axes).transpose()?;
                        let keepdims = keepdims(&mut params)?;
                        no_parameters(&params)?;
                        Ok(reduction(self.0.$method(txn.id(), axes, keepdims).await?))
                    })
                }))
            }
        })*
    };
}

reduce_handler! {
    Sum => sum,
    Product => product,
    Min => min,
    Max => max,
    Mean => mean,
    Norm => norm,
    Std => std,
}

pub(super) struct All<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for All<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                no_parameters(&params)?;
                Ok(S::from(Value::Number(Number::from(
                    self.0.all(txn.id()).await?,
                ))))
            })
        }))
    }
}

pub(super) struct Any<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Any<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                no_parameters(&params)?;
                Ok(S::from(Value::Number(Number::from(
                    self.0.any(txn.id()).await?,
                ))))
            })
        }))
    }
}

pub(super) struct BroadcastReduce<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for BroadcastReduce<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, mut params| {
            Box::pin(async move {
                let target = shape(parameter(&mut params, "target_shape")?)?;
                no_parameters(&params)?;
                Ok(state(self.0.broadcast_reduce(txn.id(), target).await?))
            })
        }))
    }
}
