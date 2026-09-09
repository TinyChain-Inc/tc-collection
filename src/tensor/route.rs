use number_general::{FloatType, Number, UIntType};
use pathlink::{PathBuf, PathSegment};
use safecast::CastInto;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, Map, Scalar};
use tc_value::{NumberType, Value, number_type_from_path, number_type_path};

use crate::Collection;
use crate::route::CollectionState;
use crate::tensor::{
    AxisRange, Range, Tensor, TensorReduceResult, batched_matmul, broadcast_add,
    broadcast_reduce_sum, tensor_op_result, tensor_transpose,
};

type GetTensorFn<S> = fn(&Tensor, S) -> TCResult<S>;
type PostTensorFn<S> = fn(&Tensor, Map<S>) -> TCResult<S>;

struct GetTensor<'a, S> {
    tensor: &'a Tensor,
    operation: GetTensorFn<S>,
    state: std::marker::PhantomData<fn() -> S>,
}

struct PostTensor<'a, S> {
    tensor: &'a Tensor,
    operation: PostTensorFn<S>,
    state: std::marker::PhantomData<fn() -> S>,
}

struct Transpose<'a, S> {
    tensor: &'a Tensor,
    state: std::marker::PhantomData<fn() -> S>,
}

impl<S: CollectionState> tc_ir::Route<S> for Tensor {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn tc_ir::Handler<'a, S> + 'a>> {
        let handler: Box<dyn tc_ir::Handler<'a, S> + 'a> = match path {
            [] => Box::new(GetTensor {
                tensor: self,
                operation: |tensor, key| {
                    tensor_state(
                        tensor
                            .clone()
                            .slice(tensor_range_from_state(key, tensor.shape())?),
                    )
                },
                state: std::marker::PhantomData,
            }),
            [segment] => match segment.as_str() {
                "broadcast" => get_tensor(self, |tensor, key| {
                    tensor_state(tensor.clone().broadcast(shape_from_state(key)?))
                }),
                "cast" => get_tensor(self, |tensor, key| {
                    tensor_state(tensor.clone().cast(tensor_dtype_from_state(key)?))
                }),
                "expand_dims" => get_tensor(self, |tensor, key| {
                    tensor_state(tensor.clone().expand_dims(optional_shape_from_state(key)?))
                }),
                "reshape" => get_tensor(self, |tensor, key| {
                    tensor_state(tensor.clone().reshape(shape_from_state(key)?))
                }),
                "transpose" => Box::new(Transpose {
                    tensor: self,
                    state: std::marker::PhantomData,
                }),
                "dtype" => post_tensor(self, |tensor, _| {
                    Ok(S::from(Value::String(
                        number_type_path(&tensor.number_type()).to_string(),
                    )))
                }),
                "ndim" => post_tensor(self, |tensor, _| {
                    Ok(S::from(Value::Number(Number::from(
                        tensor.shape().len() as u64
                    ))))
                }),
                "shape" => post_tensor(self, |tensor, _| {
                    Ok(S::from(Scalar::Tuple(
                        tensor
                            .shape()
                            .iter()
                            .map(|dim| Scalar::Value(Value::Number(Number::from(*dim as u64))))
                            .collect(),
                    )))
                }),
                "size" => post_tensor(self, |tensor, _| {
                    Ok(S::from(Value::Number(Number::from(tensor.size() as u64))))
                }),
                "all" => post_tensor(self, |tensor, _| tensor_truthy_state(tensor, true)),
                "any" => post_tensor(self, |tensor, _| tensor_truthy_state(tensor, false)),
                "cond" => post_tensor(self, |tensor, params| {
                    tensor_state(Tensor::cond(
                        tensor,
                        &tensor_param(&params, "then")?,
                        &tensor_param(&params, "or_else")?,
                    ))
                }),
                "max" => post_tensor(self, |tensor, params| tensor_reduce(tensor, params, "max")),
                "min" => post_tensor(self, |tensor, params| tensor_reduce(tensor, params, "min")),
                "mean" => post_tensor(self, |tensor, params| tensor_reduce(tensor, params, "mean")),
                "norm" => post_tensor(self, |tensor, params| tensor_reduce(tensor, params, "norm")),
                "product" => post_tensor(self, |tensor, params| {
                    tensor_reduce(tensor, params, "product")
                }),
                "std" => post_tensor(self, |tensor, params| tensor_reduce(tensor, params, "std")),
                "sum" => post_tensor(self, |tensor, params| tensor_reduce(tensor, params, "sum")),
                "broadcast_reduce" => post_tensor(self, |tensor, params| {
                    Ok(S::from(Collection::Tensor(tensor_op_result(
                        broadcast_reduce_sum(tensor, &shape_param(&params, "target_shape")?),
                    )?)))
                }),
                "matmul" => post_tensor(self, |tensor, params| {
                    Ok(S::from(Collection::Tensor(tensor_op_result(
                        batched_matmul(tensor, &tensor_param(&params, "r")?),
                    )?)))
                }),
                "add" => post_tensor(self, |tensor, params| {
                    Ok(S::from(Collection::Tensor(tensor_op_result(
                        broadcast_add(tensor, &tensor_param(&params, "r")?),
                    )?)))
                }),
                "sub" => post_tensor(self, |tensor, params| tensor_binary(tensor, params, "sub")),
                "mul" => post_tensor(self, |tensor, params| tensor_binary(tensor, params, "mul")),
                "div" => post_tensor(self, |tensor, params| tensor_binary(tensor, params, "div")),
                "and" => post_tensor(self, |tensor, params| tensor_binary(tensor, params, "and")),
                "or" => post_tensor(self, |tensor, params| tensor_binary(tensor, params, "or")),
                "xor" => post_tensor(self, |tensor, params| tensor_binary(tensor, params, "xor")),
                "not" => post_tensor(self, |tensor, _| tensor_state(tensor.unary_not())),
                _ => return None,
            },
            _ => return None,
        };
        Some(handler)
    }
}

fn get_tensor<'a, S: CollectionState>(
    tensor: &'a Tensor,
    operation: GetTensorFn<S>,
) -> Box<dyn tc_ir::Handler<'a, S> + 'a> {
    Box::new(GetTensor {
        tensor,
        operation,
        state: std::marker::PhantomData,
    })
}

fn post_tensor<'a, S: CollectionState>(
    tensor: &'a Tensor,
    operation: PostTensorFn<S>,
) -> Box<dyn tc_ir::Handler<'a, S> + 'a> {
    Box::new(PostTensor {
        tensor,
        operation,
        state: std::marker::PhantomData,
    })
}

fn tensor_state<S: CollectionState>(tensor: Result<Tensor, impl std::fmt::Display>) -> TCResult<S> {
    tensor
        .map(Collection::Tensor)
        .map(S::from)
        .map_err(TCError::bad_request)
}

fn tensor_transpose_get<S: CollectionState>(tensor: &Tensor, key: S) -> TCResult<S> {
    tensor_transpose(tensor, &shape_from_state(key)?)
        .map(Collection::Tensor)
        .map(S::from)
        .map_err(TCError::bad_request)
}

fn tensor_transpose_post<S: CollectionState>(tensor: &Tensor, params: Map<S>) -> TCResult<S> {
    Ok(S::from(Collection::Tensor(tensor_op_result(
        tensor_transpose(tensor, &shape_param(&params, "perm")?),
    )?)))
}

fn tensor_reduce<S: CollectionState>(tensor: &Tensor, params: Map<S>, op: &str) -> TCResult<S> {
    match tensor
        .reduce_axes(
            op,
            optional_axes_param(&params)?,
            bool_param(&params, "keepdims")?,
        )
        .map_err(TCError::bad_request)?
    {
        TensorReduceResult::Scalar(number) => Ok(S::from(Value::Number(number))),
        TensorReduceResult::Tensor(tensor) => Ok(S::from(Collection::Tensor(tensor))),
    }
}

fn tensor_binary<S: CollectionState>(tensor: &Tensor, params: Map<S>, op: &str) -> TCResult<S> {
    tensor
        .binary_op(&tensor_param(&params, "r")?, op)
        .map(Collection::Tensor)
        .map(S::from)
        .map_err(TCError::bad_request)
}

pub(crate) fn tensor_literal<S: CollectionState>(key: S, value: S) -> TCResult<Tensor> {
    let key = key.into_tuple()?;
    if key.len() != 2 {
        return Err(TCError::bad_request(
            "tensor literal key must be [dtype, shape]",
        ));
    }
    let dtype = tensor_dtype_from_state(key[0].clone())?;
    let shape = shape_from_state(key[1].clone())?;
    let values = numbers_from_state(value, "tensor literal values")?;
    match dtype {
        NumberType::Float(FloatType::F32) => {
            Tensor::dense_f32(shape, values.into_iter().map(CastInto::cast_into).collect())
                .map_err(TCError::bad_request)
        }
        NumberType::Float(FloatType::F64) => {
            Tensor::dense_f64(shape, values.into_iter().map(CastInto::cast_into).collect())
                .map_err(TCError::bad_request)
        }
        NumberType::UInt(UIntType::U64) => {
            Tensor::dense_u64(shape, values.into_iter().map(CastInto::cast_into).collect())
                .map_err(TCError::bad_request)
        }
        dtype => Err(TCError::bad_request(format!(
            "unsupported tensor literal dtype {dtype}"
        ))),
    }
}

fn tensor_param<S: CollectionState>(params: &Map<S>, name: &str) -> TCResult<Tensor> {
    required_param(params, name)?.into_tensor()
}
fn shape_param<S: CollectionState>(params: &Map<S>, name: &str) -> TCResult<Vec<usize>> {
    shape_from_state(required_param(params, name)?)
}
fn optional_axes_param<S: CollectionState>(params: &Map<S>) -> TCResult<Option<Vec<usize>>> {
    if let Some(axes) = optional_param(params, "axes")? {
        return optional_axes_from_state(axes);
    }
    optional_param(params, "axis")?
        .map(optional_axes_from_state)
        .transpose()
        .map(|axes| axes.flatten())
}
fn bool_param<S: CollectionState>(params: &Map<S>, name: &str) -> TCResult<bool> {
    optional_param(params, name)?
        .map(|value| value.into_value())
        .transpose()?
        .map(|value| match value {
            Value::Number(number) => Ok(number.cast_into()),
            other => Err(TCError::bad_request(format!(
                "expected tensor {name} boolean, found {other:?}"
            ))),
        })
        .transpose()
        .map(|value| value.unwrap_or(false))
}
fn required_param<S: CollectionState>(params: &Map<S>, name: &str) -> TCResult<S> {
    optional_param(params, name)?
        .ok_or_else(|| TCError::bad_request(format!("missing tensor parameter {name}")))
}
fn optional_param<S: CollectionState>(params: &Map<S>, name: &str) -> TCResult<Option<S>> {
    let id: Id = name
        .parse()
        .map_err(|err| TCError::internal(format!("invalid tensor parameter {name}: {err}")))?;
    Ok(params.get(&id).cloned())
}
fn numbers_from_state<S: CollectionState>(state: S, context: &str) -> TCResult<Vec<Number>> {
    state
        .into_tuple()?
        .into_iter()
        .map(|state| match state.into_value()? {
            Value::Number(number) => Ok(number),
            other => Err(TCError::bad_request(format!(
                "expected {context} numbers, found {other:?}"
            ))),
        })
        .collect()
}
fn shape_from_state<S: CollectionState>(state: S) -> TCResult<Vec<usize>> {
    numbers_from_state(state, "tensor shape")?
        .into_iter()
        .map(|number| number_to_usize(number, "tensor shape dimension"))
        .collect()
}
fn optional_shape_from_state<S: CollectionState>(state: S) -> TCResult<Option<Vec<usize>>> {
    if state.is_none() {
        Ok(None)
    } else {
        shape_from_state(state).map(Some)
    }
}
fn optional_axes_from_state<S: CollectionState>(state: S) -> TCResult<Option<Vec<usize>>> {
    if state.is_none() {
        return Ok(None);
    };
    match state.clone().into_value() {
        Ok(Value::Number(number)) => Ok(Some(vec![number_to_usize(
            number,
            "tensor reduction axis",
        )?])),
        _ => shape_from_state(state).map(Some),
    }
}
fn tensor_dtype_from_state<S: CollectionState>(state: S) -> TCResult<NumberType> {
    let raw = match state.into_value()? {
        Value::String(dtype) => dtype,
        Value::Link(link) => link.to_string(),
        other => {
            return Err(TCError::bad_request(format!(
                "expected tensor dtype string or link, found {other:?}"
            )));
        }
    };
    parse_tensor_number_type(&raw)
        .ok_or_else(|| TCError::bad_request(format!("unsupported tensor dtype {raw}")))
}
fn parse_tensor_number_type(raw: &str) -> Option<NumberType> {
    match raw {
        "f32" => Some(NumberType::Float(FloatType::F32)),
        "f64" => Some(NumberType::Float(FloatType::F64)),
        "u64" => Some(NumberType::UInt(UIntType::U64)),
        _ => raw
            .parse::<PathBuf>()
            .ok()
            .and_then(|path| number_type_from_path(path.as_ref())),
    }
}
fn tensor_range_from_state<S: CollectionState>(bounds: S, shape: &[usize]) -> TCResult<Range> {
    let bounds = bounds.into_tuple()?;
    if bounds.len() != shape.len() {
        return Err(TCError::bad_request(format!(
            "tensor slice bounds rank {} does not match tensor rank {}",
            bounds.len(),
            shape.len()
        )));
    };
    bounds
        .into_iter()
        .zip(shape.iter().copied())
        .enumerate()
        .map(|(axis, (bound, dim))| tensor_axis_range_from_state(bound, axis, dim))
        .collect()
}
fn tensor_axis_range_from_state<S: CollectionState>(
    state: S,
    axis: usize,
    dim: usize,
) -> TCResult<AxisRange> {
    if let Ok(Value::Number(number)) = state.clone().into_value() {
        let index = number_to_usize(number, "tensor slice index")?;
        if index >= dim {
            return Err(TCError::bad_request(format!(
                "tensor slice index {index} is out of bounds for axis {axis} with dim {dim}"
            )));
        };
        return Ok(AxisRange::At(index));
    }
    let parts = state.into_tuple()?;
    if parts.is_empty() || parts.len() > 3 {
        return Err(TCError::bad_request(
            "tensor slice range must have 1 to 3 components",
        ));
    };
    let start = state_usize(parts[0].clone(), "tensor slice start")?;
    let stop = parts
        .get(1)
        .cloned()
        .map(|value| state_usize(value, "tensor slice stop"))
        .transpose()?
        .unwrap_or(dim);
    let step = parts
        .get(2)
        .cloned()
        .map(|value| state_usize(value, "tensor slice step"))
        .transpose()?
        .unwrap_or(1);
    if step == 0 || start > stop || stop > dim {
        return Err(TCError::bad_request(format!(
            "invalid tensor slice range for axis {axis}"
        )));
    };
    Ok(AxisRange::In(start, stop, step))
}
fn state_usize<S: CollectionState>(state: S, context: &str) -> TCResult<usize> {
    match state.into_value()? {
        Value::Number(number) => number_to_usize(number, context),
        other => Err(TCError::bad_request(format!(
            "expected {context} number, found {other:?}"
        ))),
    }
}
fn number_to_usize(number: Number, context: &str) -> TCResult<usize> {
    let value: i64 = number.cast_into();
    if value < 0 {
        return Err(TCError::bad_request(format!(
            "expected {context} to be non-negative"
        )));
    };
    Ok(value as usize)
}
fn tensor_truthy_state<S: CollectionState>(tensor: &Tensor, all: bool) -> TCResult<S> {
    let values = tensor.values_f64().map_err(TCError::bad_request)?;
    Ok(S::from(Value::Number(Number::Bool(
        (if all {
            values.iter().all(|value| *value != 0.0)
        } else {
            values.iter().any(|value| *value != 0.0)
        })
        .into(),
    ))))
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for GetTensor<'a, S> {
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, key| {
            Box::pin(async move { (self.operation)(self.tensor, S::from(key)) })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for PostTensor<'a, S> {
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, params| {
            Box::pin(async move { (self.operation)(self.tensor, params) })
        }))
    }
}

impl<'a, S: CollectionState> tc_ir::Handler<'a, S> for Transpose<'a, S> {
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, key| {
            Box::pin(async move { tensor_transpose_get(self.tensor, S::from(key)) })
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, params| {
            Box::pin(async move { tensor_transpose_post(self.tensor, params) })
        }))
    }
}
