//! Select each Tensor route once; operation families own terminal handlers.

use number_general::Number;
use pathlink::{PathBuf, PathSegment};
use safecast::CastInto;
use tc_error::{TCError, TCResult};
use tc_ir::Map;
use tc_value::{NumberType, Value, number_type_from_path, number_type_path};

use super::{AxisRange, Data, Range, Shape as TensorShape, Tensor, TensorSchema, Typed};
use crate::{Collection, CollectionState, StorageContext};

// Generate concrete borrowing handlers; argument policy stays at this boundary.
macro_rules! unary_handler {
    ($($(#[$attr:meta])* $handler:ident => $method:ident),* $(,)?) => {
        $($(#[$attr])*
        pub(super) struct $handler<'a, Txn: $crate::StorageContext>(
            pub(super) &'a $crate::tensor::Tensor<Txn>,
        );

        $(#[$attr])*
        impl<'a, S, Txn> tc_ir::Handler<'a, S> for $handler<'a, Txn>
        where
            S: $crate::CollectionState<Txn = Txn>,
            Txn: $crate::StorageContext,
        {
            fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
            where
                'txn: 'a,
            {
                Some(Box::new(move |txn, params| {
                    Box::pin(async move {
                        super::no_parameters(&params)?;
                        Ok(super::state(self.0.$method(txn.id()).await?))
                    })
                }))
            }
        })*
    };
}

mod binary;
mod reduce;
mod transform;
mod unary;

use binary::{Add, And, Cond, Div, Eq, Ge, Gt, Le, Log, Lt, Mul, Ne, Or, Pow, Rem, Sub, Xor};
use reduce::{All, Any, BroadcastReduce, Max, Mean, Min, Norm, Product, Std, Sum};
use transform::{
    AsDense, Broadcast, Cast, Copy, CopyValues, Diag, ExpandDims, Flip, Matmul, Mt, Reshape,
    Squeeze, Transpose,
};
#[cfg(feature = "complex")]
use transform::{Fft, Fft2, Ifft, Ifft2, Mh};
use unary::{
    Abs, Acos, Asin, Atan, Cos, Cosh, Exp, IsInf, IsNan, Ln, Not, Round, Sin, Sinh, Tan, Tanh,
};
#[cfg(feature = "complex")]
use unary::{Angle, Conj, Im, Re};

struct Root<'a, Txn: StorageContext>(&'a Tensor<Txn>);
struct Dtype<'a, Txn: StorageContext>(&'a Tensor<Txn>);
struct Shape<'a, Txn: StorageContext>(&'a Tensor<Txn>);
struct Size<'a, Txn: StorageContext>(&'a Tensor<Txn>);
struct Ndim<'a, Txn: StorageContext>(&'a Tensor<Txn>);

impl<S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Route<S> for Tensor<Txn> {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn tc_ir::Handler<'a, S> + 'a>> {
        match path {
            [] => Some(Box::new(Root(self))),
            [segment] => match segment.as_str() {
                "add" => Some(Box::new(Add(self))),
                "sub" => Some(Box::new(Sub(self))),
                "mul" => Some(Box::new(Mul(self))),
                "div" => Some(Box::new(Div(self))),
                "pow" => Some(Box::new(Pow(self))),
                "rem" => Some(Box::new(Rem(self))),
                "log" => Some(Box::new(Log(self))),
                "eq" => Some(Box::new(Eq(self))),
                "ne" => Some(Box::new(Ne(self))),
                "gt" => Some(Box::new(Gt(self))),
                "ge" => Some(Box::new(Ge(self))),
                "lt" => Some(Box::new(Lt(self))),
                "le" => Some(Box::new(Le(self))),
                "and" => Some(Box::new(And(self))),
                "or" => Some(Box::new(Or(self))),
                "xor" => Some(Box::new(Xor(self))),
                "matmul" => Some(Box::new(Matmul(self))),
                "not" => Some(Box::new(Not(self))),
                "abs" => Some(Box::new(Abs(self))),
                "exp" => Some(Box::new(Exp(self))),
                "ln" => Some(Box::new(Ln(self))),
                "round" => Some(Box::new(Round(self))),
                "sin" => Some(Box::new(Sin(self))),
                "asin" => Some(Box::new(Asin(self))),
                "sinh" => Some(Box::new(Sinh(self))),
                "cos" => Some(Box::new(Cos(self))),
                "acos" => Some(Box::new(Acos(self))),
                "cosh" => Some(Box::new(Cosh(self))),
                "tan" => Some(Box::new(Tan(self))),
                "atan" => Some(Box::new(Atan(self))),
                "tanh" => Some(Box::new(Tanh(self))),
                "is_nan" => Some(Box::new(IsNan(self))),
                "is_inf" => Some(Box::new(IsInf(self))),
                "mt" => Some(Box::new(Mt(self))),
                "diag" => Some(Box::new(Diag(self))),
                #[cfg(feature = "complex")]
                "conj" => Some(Box::new(Conj(self))),
                #[cfg(feature = "complex")]
                "re" => Some(Box::new(Re(self))),
                #[cfg(feature = "complex")]
                "im" => Some(Box::new(Im(self))),
                #[cfg(feature = "complex")]
                "angle" => Some(Box::new(Angle(self))),
                #[cfg(feature = "complex")]
                "mh" => Some(Box::new(Mh(self))),
                #[cfg(feature = "complex")]
                "fft2" => Some(Box::new(Fft2(self))),
                #[cfg(feature = "complex")]
                "ifft2" => Some(Box::new(Ifft2(self))),
                #[cfg(feature = "complex")]
                "fft" => Some(Box::new(Fft(self))),
                #[cfg(feature = "complex")]
                "ifft" => Some(Box::new(Ifft(self))),
                "sum" => Some(Box::new(Sum(self))),
                "product" => Some(Box::new(Product(self))),
                "min" => Some(Box::new(Min(self))),
                "max" => Some(Box::new(Max(self))),
                "mean" => Some(Box::new(Mean(self))),
                "norm" => Some(Box::new(Norm(self))),
                "std" => Some(Box::new(Std(self))),
                "dtype" => Some(Box::new(Dtype(self))),
                "shape" => Some(Box::new(Shape(self))),
                "size" => Some(Box::new(Size(self))),
                "ndim" => Some(Box::new(Ndim(self))),
                "reshape" => Some(Box::new(Reshape(self))),
                "broadcast" => Some(Box::new(Broadcast(self))),
                "expand_dims" => Some(Box::new(ExpandDims(self))),
                "squeeze" => Some(Box::new(Squeeze(self))),
                "flip" => Some(Box::new(Flip(self))),
                "cast" => Some(Box::new(Cast(self))),
                "transpose" => Some(Box::new(Transpose(self))),
                "all" => Some(Box::new(All(self))),
                "any" => Some(Box::new(Any(self))),
                "cond" => Some(Box::new(Cond(self))),
                "broadcast_reduce" => Some(Box::new(BroadcastReduce(self))),
                "copy" => Some(Box::new(Copy(self))),
                "copy_values" => Some(Box::new(CopyValues(self))),
                "as_dense" => Some(Box::new(AsDense(self))),
                _ => None,
            },
            _ => None,
        }
    }
}

fn state<S: CollectionState>(tensor: Tensor<S::Txn>) -> S {
    S::from(Collection::Tensor(tensor))
}

fn parameter<S: CollectionState>(params: &mut Map<S>, name: &str) -> TCResult<S> {
    params
        .remove(name)
        .ok_or_else(|| TCError::bad_request(format!("missing Tensor parameter {name}")))
}

fn optional<S: CollectionState>(params: &mut Map<S>, name: &str) -> Option<S> {
    params.remove(name).filter(|value| !value.is_none())
}

fn no_parameters<S>(params: &Map<S>) -> TCResult<()> {
    if let Some(name) = params.keys().next() {
        return Err(TCError::bad_request(format!(
            "unexpected Tensor parameter {name}"
        )));
    }

    Ok(())
}

fn shape<S: CollectionState>(state: S) -> TCResult<TensorShape> {
    state
        .into_tuple()?
        .into_iter()
        .map(|value| match value.into_value()? {
            Value::Number(number) => super::schema::dimension(number),
            _ => Err(TCError::bad_request("expected a Tensor dimension")),
        })
        .collect()
}

fn axes<S: CollectionState>(state: S) -> TCResult<Vec<usize>> {
    if let Some(number) = state.as_number() {
        return Ok(vec![
            usize::try_from(super::schema::dimension(number)?).map_err(TCError::bad_request)?,
        ]);
    }

    shape(state)?
        .into_iter()
        .map(|axis| usize::try_from(axis).map_err(TCError::bad_request))
        .collect()
}

fn dtype<S: CollectionState>(state: S) -> TCResult<NumberType> {
    let raw = match state.into_value()? {
        Value::String(raw) => raw,
        Value::Link(link) => link.to_string(),
        _ => return Err(TCError::bad_request("expected Tensor dtype path")),
    };

    let dtype = match raw.as_str() {
        "u8" => Some(NumberType::UInt(number_general::UIntType::U8)),
        "u16" => Some(NumberType::UInt(number_general::UIntType::U16)),
        "u32" => Some(NumberType::UInt(number_general::UIntType::U32)),
        "u64" => Some(NumberType::UInt(number_general::UIntType::U64)),
        "i8" => Some(NumberType::Int(number_general::IntType::I8)),
        "i16" => Some(NumberType::Int(number_general::IntType::I16)),
        "i32" => Some(NumberType::Int(number_general::IntType::I32)),
        "i64" => Some(NumberType::Int(number_general::IntType::I64)),
        "f32" => Some(NumberType::Float(number_general::FloatType::F32)),
        "f64" => Some(NumberType::Float(number_general::FloatType::F64)),
        "c32" => Some(NumberType::Complex(number_general::ComplexType::C32)),
        "c64" => Some(NumberType::Complex(number_general::ComplexType::C64)),
        _ => raw
            .parse::<PathBuf>()
            .ok()
            .and_then(|path| number_type_from_path(&path)),
    };

    dtype.ok_or_else(|| TCError::bad_request("unsupported Tensor dtype"))
}

fn range<S: CollectionState>(state: S, shape: &[u64]) -> TCResult<Range> {
    if state.is_none() {
        return Ok(shape.iter().map(|&dim| AxisRange::In(0, dim, 1)).collect());
    }

    let values = state.into_tuple()?;
    if values.len() != shape.len() {
        return Err(TCError::bad_request("Tensor selection rank mismatch"));
    }

    values
        .into_iter()
        .zip(shape)
        .map(|(value, &dim)| {
            if value.is_none() {
                return Ok(AxisRange::In(0, dim, 1));
            }

            if let Some(number) = value.as_number() {
                let coord = super::schema::dimension(number)?;
                if coord >= dim {
                    return Err(TCError::bad_request("Tensor coordinate out of bounds"));
                }

                return Ok(AxisRange::At(coord));
            }

            let fields = value.into_tuple()?;
            if fields.is_empty() || fields.len() > 3 {
                return Err(TCError::bad_request("expected [start, stop, step]"));
            }

            let mut numbers = fields.into_iter().map(|value| match value.into_value()? {
                Value::Number(number) => super::schema::dimension(number),
                _ => Err(TCError::bad_request("expected unsigned range component")),
            });

            let start = numbers.next().expect("nonempty")?;
            let stop = numbers.next().transpose()?.unwrap_or(dim);
            let step = numbers.next().transpose()?.unwrap_or(1);
            if step == 0 || start > stop || stop > dim {
                return Err(TCError::bad_request("invalid Tensor range"));
            }

            Ok(AxisRange::In(start, stop, step))
        })
        .collect()
}

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Root<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let key = S::from(key);
                if key.is_none() {
                    return Ok(state(self.0.clone().into_read_only()));
                }

                let range = range(key, self.0.shape())?;
                if range.iter().all(|axis| matches!(axis, AxisRange::At(_))) {
                    let coord: Vec<u64> = range
                        .iter()
                        .map(|axis| match axis {
                            AxisRange::At(coord) => *coord,
                            _ => unreachable!(),
                        })
                        .collect();

                    return Ok(S::from(Value::Number(
                        self.0.read_value(txn.id(), &coord).await?,
                    )));
                }

                Ok(state(self.0.clone().slice(range)?))
            })
        }))
    }

    fn put<'txn>(self: Box<Self>) -> Option<tc_ir::PutHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            Box::pin(async move {
                let range = range(S::from(key), self.0.shape())?;
                if let Some(number) = value.as_number() {
                    if range.iter().all(|axis| matches!(axis, AxisRange::At(_))) {
                        let coord: Vec<u64> = range
                            .iter()
                            .map(|axis| match axis {
                                AxisRange::At(coord) => *coord,
                                _ => unreachable!(),
                            })
                            .collect();

                        return self.0.write_value(txn, &coord, number).await;
                    }

                    return self.0.fill_range(txn, range, number).await;
                }

                self.0
                    .write_tensor_range(txn, range, value.into_tensor()?)
                    .await
            })
        }))
    }

    fn delete<'txn>(self: Box<Self>) -> Option<tc_ir::DeleteHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let range = range(S::from(key), self.0.shape())?;
                if range.iter().all(|axis| matches!(axis, AxisRange::At(_))) {
                    let coord: Vec<u64> = range
                        .iter()
                        .map(|axis| match axis {
                            AxisRange::At(coord) => *coord,
                            _ => unreachable!(),
                        })
                        .collect();

                    self.0.write_value(txn, &coord, Number::from(0u8)).await
                } else {
                    self.0.fill_range(txn, range, Number::from(0u8)).await
                }
            })
        }))
    }
}

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Dtype<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                let _ = (txn, params);
                Ok(S::from(Value::String(
                    number_type_path(&self.0.number_type()).to_string(),
                )))
            })
        }))
    }
}

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Shape<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                let _ = (txn, params);
                Ok(S::from(Value::Tuple(
                    self.0.shape().iter().copied().map(Value::from).collect(),
                )))
            })
        }))
    }
}

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Size<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                let _ = (txn, params);
                Ok(S::from(Value::from(self.0.size()?)))
            })
        }))
    }
}

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Ndim<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                let _ = (txn, params);
                Ok(S::from(Value::from(self.0.shape().len() as u64)))
            })
        }))
    }
}

pub(crate) async fn tensor_literal<S: CollectionState>(
    txn: &S::Txn,
    key: S,
    values: S,
) -> TCResult<Tensor<S::Txn>> {
    let mut fields = key.into_tuple()?;
    if !(2..=3).contains(&fields.len()) {
        return Err(TCError::bad_request(
            "Tensor literal key must be [dtype, shape, optional sparse axis]",
        ));
    }

    let sparse_axis = match (fields.len() == 3).then(|| fields.pop().expect("literal arity")) {
        None => None,
        Some(value) => match value.into_value()? {
            Value::Number(axis) => Some(
                usize::try_from(super::schema::dimension(axis)?).map_err(TCError::bad_request)?,
            ),
            _ => return Err(TCError::bad_request("expected a numeric sparse axis")),
        },
    };

    let mut fields = fields.into_iter();
    let schema = TensorSchema::new(
        dtype(fields.next().expect("literal arity"))?,
        shape(fields.next().expect("literal arity"))?,
        sparse_axis,
    )?;

    let sparse = schema.is_sparse();
    let values = values.into_tuple()?;
    if !sparse && values.len() as u64 != schema.native().shape().iter().product::<u64>() {
        return Err(TCError::bad_request(
            "Tensor literal value count differs from shape",
        ));
    }

    let capacity = fensor::MAX_BLOCK_CAPACITY;
    let dir = txn.subcontext_unique().context().await?;
    let data = dtype_factory!(schema.dtype(), T => {
        if sparse {
            let entries = futures::stream::iter(values.into_iter().map(|value| {
                let fields = value.into_tuple()?;
                if fields.len() != 2 { return Err(TCError::bad_request("expected sparse [coordinate, value]")); }
                let mut fields = fields.into_iter();
                let coord = shape(fields.next().expect("sparse entry arity"))?;
                let Value::Number(number) = fields.next().expect("sparse entry arity").into_value()? else {
                    return Err(TCError::bad_request("expected numeric Tensor value"));
                };
                if number == Number::from(0u8) { return Err(TCError::bad_request("sparse values must be nonzero")); }
                Ok((coord.into_vec(), number.cast_into()))
            }));
            fensor::Tensor::<_, T>::from_sparse_elements(dir, schema.native().clone(), schema.layout(), capacity, entries).await?
        } else {
            let values = futures::stream::iter(values.into_iter().map(|value| {
                let Value::Number(number) = value.into_value()? else {
                    return Err(TCError::bad_request("expected numeric Tensor value"));
                };
                Ok::<T, TCError>(number.cast_into())
            }));
            fensor::Tensor::<_, T>::from_values(dir, schema.native().clone(), capacity, values).await?
        }
    }, dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {dtype}"))));

    Ok(Tensor { data })
}
