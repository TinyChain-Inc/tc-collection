use fensor::{
    TensorAbs, TensorBoolean, TensorBooleanScalar, TensorCast, TensorCompare, TensorCompareScalar,
    TensorMatMul, TensorMath, TensorMathScalar, TensorMatrixUnary, TensorNumeric, TensorReduce,
    TensorReduceAll, TensorReduceBoolean, TensorStatistics, TensorTrig, TensorUnary,
    TensorUnaryBoolean, TensorWhere, broadcast_shape,
};
#[cfg(feature = "complex")]
use fensor::{TensorComplex, TensorFourier, TensorMatrixUnaryComplex};
use number_general::Number;
use safecast::CastInto;

use super::{Data, Shape, Tensor, TensorReduceResult, Typed};
use crate::StorageContext;
use tc_error::{TCError, TCResult};
use tc_ir::TxnId;
use tc_value::NumberType;

fn reduction_axes(shape: &[u64], axes: Option<Vec<usize>>) -> TCResult<fensor::Axes> {
    let axes = axes
        .map(Into::into)
        .unwrap_or_else(|| (0..shape.len()).collect());

    fensor::reduction_axes(shape.len(), axes).map_err(TCError::bad_request)
}

macro_rules! native_binary {
    ($id:expr, $left:expr, $right:expr, $method:ident, $family:ident => $output:ident, $error:expr) => {
        dtype_rows!($family => mapped_types($output => native_binary(@cases $id, $left, $right, $method, $error)))
    };
    (@cases $id:expr, $left:expr, $right:expr, $method:ident, $error:expr;
     [$($(#[$attr:meta])* $input:ident => $output:ident),* $(,)?]) => {
        match (&$left.data, &$right.data) {
            $($(#[$attr])* (Data::$input(left), Data::$input(right)) => {
                let expressions = Typed::expressions([left, right], $id)?;
                let [left, right] = expressions;

                Data::$output(Typed::computed(
                    fensor::TensorExpression::new(left.$method(&right).await?)?,
                    $id,
                ))
            },)*
            _ => return Err($error),
        }
    };
}

macro_rules! native_unary {
    (@call $value:ident, $id:expr, $method:ident, ($($arg:expr),*)) => {
        $value.expression($id).await?.$method($($arg),*).await?
    };
    ($id:expr, $source:expr, $method:ident $args:tt, $family:ident => component $(, [$($error_attr:meta),*] $error:expr)?) => {
        component_types!($family => native_unary(@cases $id, $source, $method $args $(, [$($error_attr),*] $error)?))
    };
    ($id:expr, $source:expr, $method:ident $args:tt, $family:ident => $output:ident $(, [$($error_attr:meta),*] $error:expr)?) => {
        dtype_rows!($family => mapped_types($output => native_unary(@cases $id, $source, $method $args $(, [$($error_attr),*] $error)?)))
    };
    (@cases $id:expr, $source:expr, $method:ident $args:tt $(, [$($error_attr:meta),*] $error:expr)?;
     [$($(#[$attr:meta])* $input:ident => $output:ident),* $(,)?]) => {
        match &$source.data {
            $($(#[$attr])* Data::$input(value) => Data::$output(Typed::computed(
                fensor::TensorExpression::new(native_unary!(@call value, $id, $method, $args))?,
                $id,
            )),)*
            $($(#[$error_attr])* _ => return Err($error),)?
        }
    };
}

macro_rules! native_conditional {
    ($id:expr, $condition:expr, $then:expr, $or_else:expr, $family:ident) => {
        dtype_rows!($family => native_conditional(@cases $id, $condition, $then, $or_else))
    };
    (@cases $id:expr, $condition:expr, $then:expr, $or_else:expr; $($(#[$attr:meta])* $dtype:ident),* $(,)?) => {
        match (&$then.data, &$or_else.data) {
            $($(#[$attr])* (Data::$dtype(left), Data::$dtype(right)) => {
                let condition_source = &$condition;
                let (condition, left, right) = if let Data::$dtype(condition) = &condition_source.data {
                    let [condition, left, right] = Typed::expressions([condition, left, right], $id)?;
                    (fensor::TensorExpression::new(condition.ne_scalar(Default::default()).await?)?, left, right)
                } else {
                    let condition = dispatch!(&$condition.data, value => fensor::TensorExpression::new(
                        value.try_expression($id)?.ne_scalar(Default::default()).await?
                    )?);
                    let [left, right] = Typed::expressions([left, right], $id)?;
                    (condition, left, right)
                };

                Data::$dtype(Typed::computed(
                    fensor::TensorExpression::new(condition.cond(&left, &right).await?)?,
                    $id,
                ))
            },)*
            _ => return Err(TCError::bad_request("conditional branches require matching Tensor dtypes")),
        }
    };
}

macro_rules! native_terminal {
    ($id:expr, $source:expr, $method:ident, $family:ident $(, [$($error_attr:meta),*] $error:expr)?) => {
        dtype_rows!($family => native_terminal(@cases $id, $source, $method $(, [$($error_attr),*] $error)?))
    };
    (@cases $id:expr, $source:expr, $method:ident $(, [$($error_attr:meta),*] $error:expr)?; $($(#[$attr:meta])* $dtype:ident),* $(,)?) => {
        match &$source.data {
            $($(#[$attr])* Data::$dtype(value) => {
                let result = value.expression($id).await?.$method().await?;
                Ok(TensorReduceResult::Scalar(Number::from(result)))
            },)*
            $($(#[$error_attr])* _ => Err($error),)?
        }
    };
}

macro_rules! binary_methods {
    ($($method:ident($($dispatch:tt)*));* $(;)?) => {$(
        pub async fn $method(&self, id: TxnId, right: &Self) -> TCResult<Self> {
            let (left, right) = self.broadcast_pair(right)?;
            let data = native_binary!(id, left, right, $method, $($dispatch)*, TCError::bad_request(
                concat!(stringify!($method), " requires matching supported Tensor dtypes"),
            ));

            Ok(Self { data })
        }
    )*};
}

macro_rules! scalar_methods {
    ($($method:ident($($dispatch:tt)*));* $(;)?) => {$(
        pub async fn $method(&self, id: TxnId, scalar: Number) -> TCResult<Self> {
            let data = native_unary!(id, self, $method(scalar.cast_into()), $($dispatch)*);

            Ok(Self { data })
        }
    )*};
}

macro_rules! unary_methods {
    ($($(#[$attr:meta])* $method:ident($($dispatch:tt)*));* $(;)?) => {$(
        $(#[$attr])*
        pub async fn $method(&self, id: TxnId) -> TCResult<Self> {
            let data = native_unary!(id, self, $method(), $($dispatch)*);

            Ok(Self { data })
        }
    )*};
}

impl<Txn: StorageContext> Tensor<Txn> {
    fn broadcast_pair(&self, right: &Self) -> TCResult<(Self, Self)> {
        let shape = broadcast_shape(self.shape(), right.shape())?;
        let left = self.clone().broadcast(shape.clone())?;
        let right = right.clone().broadcast(shape)?;
        Ok((left, right))
    }

    binary_methods! {
        add(all => same);
        sub(all => same);
        mul(all => same);
        div(all => same);
        pow(all => same);
        rem(real => same);
        log(floating => same);
        eq(all => boolean);
        ne(all => boolean);
        and(all => boolean);
        or(all => boolean);
        xor(all => boolean);
        gt(real => boolean);
        ge(real => boolean);
        lt(real => boolean);
        le(real => boolean);
    }

    pub async fn matmul(&self, id: TxnId, right: &Self) -> TCResult<Self> {
        let (left_shape, right_shape) =
            fensor::matmul_broadcast_shapes(self.shape(), right.shape())?;

        let left = self.clone().broadcast(left_shape)?;
        let right = right.clone().broadcast(right_shape)?;
        let data = native_binary!(id, left, right, matmul, all => same, TCError::bad_request(
            "matmul requires matching supported Tensor dtypes",
        ));

        Ok(Self { data })
    }

    scalar_methods! {
        add_scalar(all => same);
        sub_scalar(all => same);
        mul_scalar(all => same);
        div_scalar(all => same);
        pow_scalar(all => same);
        rem_scalar(real => same, [cfg(feature = "complex")] TCError::unsupported("rem_scalar does not support this dtype"));
        log_scalar(floating => same, [] TCError::unsupported("log_scalar does not support this dtype"));
        eq_scalar(all => boolean);
        ne_scalar(all => boolean);
        and_scalar(all => boolean);
        or_scalar(all => boolean);
        xor_scalar(all => boolean);
        gt_scalar(real => boolean, [cfg(feature = "complex")] TCError::unsupported("gt_scalar does not support this dtype"));
        ge_scalar(real => boolean, [cfg(feature = "complex")] TCError::unsupported("ge_scalar does not support this dtype"));
        lt_scalar(real => boolean, [cfg(feature = "complex")] TCError::unsupported("lt_scalar does not support this dtype"));
        le_scalar(real => boolean, [cfg(feature = "complex")] TCError::unsupported("le_scalar does not support this dtype"));
    }

    unary_methods! {
        not(all => boolean);
        abs(all => component);
        exp(floating => same, [] TCError::unsupported("exp does not support this dtype"));
        ln(floating => same, [] TCError::unsupported("ln does not support this dtype"));
        sin(floating => same, [] TCError::unsupported("sin does not support this dtype"));
        asin(floating => same, [] TCError::unsupported("asin does not support this dtype"));
        sinh(floating => same, [] TCError::unsupported("sinh does not support this dtype"));
        cos(floating => same, [] TCError::unsupported("cos does not support this dtype"));
        acos(floating => same, [] TCError::unsupported("acos does not support this dtype"));
        cosh(floating => same, [] TCError::unsupported("cosh does not support this dtype"));
        tan(floating => same, [] TCError::unsupported("tan does not support this dtype"));
        atan(floating => same, [] TCError::unsupported("atan does not support this dtype"));
        tanh(floating => same, [] TCError::unsupported("tanh does not support this dtype"));
        round(float => same, [] TCError::unsupported("round does not support this dtype"));
        is_nan(floating => boolean, [] TCError::unsupported("is_nan does not support this dtype"));
        is_inf(floating => boolean, [] TCError::unsupported("is_inf does not support this dtype"));
        mt(all => same);
        diag(all => same);
        #[cfg(feature = "complex")]
        conj(complex => same, [] TCError::unsupported("conj does not support this dtype"));
        #[cfg(feature = "complex")]
        re(complex => component, [] TCError::unsupported("re does not support this dtype"));
        #[cfg(feature = "complex")]
        im(complex => component, [] TCError::unsupported("im does not support this dtype"));
        #[cfg(feature = "complex")]
        angle(complex => component, [] TCError::unsupported("angle does not support this dtype"));
        #[cfg(feature = "complex")]
        mh(complex => same, [] TCError::unsupported("mh does not support this dtype"));
        #[cfg(feature = "complex")]
        fft(complex => same, [] TCError::unsupported("fft does not support this dtype"));
        #[cfg(feature = "complex")]
        ifft(complex => same, [] TCError::unsupported("ifft does not support this dtype"));
    }

    #[cfg(feature = "complex")]
    pub async fn fft2(&self, id: TxnId) -> TCResult<Self> {
        Ok(Self {
            data: match &self.data {
                Data::C32(value) => Data::C32(Typed::computed(
                    fensor::TensorExpression::new(
                        fensor::fft::fft2(&value.expression(id).await?).await?,
                    )?,
                    id,
                )),
                Data::C64(value) => Data::C64(Typed::computed(
                    fensor::TensorExpression::new(
                        fensor::fft::fft2(&value.expression(id).await?).await?,
                    )?,
                    id,
                )),
                _ => return Err(TCError::unsupported("fft2 requires complex tensors")),
            },
        })
    }

    #[cfg(feature = "complex")]
    pub async fn ifft2(&self, id: TxnId) -> TCResult<Self> {
        Ok(Self {
            data: match &self.data {
                Data::C32(value) => Data::C32(Typed::computed(
                    fensor::TensorExpression::new(
                        fensor::fft::ifft2(&value.expression(id).await?).await?,
                    )?,
                    id,
                )),
                Data::C64(value) => Data::C64(Typed::computed(
                    fensor::TensorExpression::new(
                        fensor::fft::ifft2(&value.expression(id).await?).await?,
                    )?,
                    id,
                )),
                _ => return Err(TCError::unsupported("ifft2 requires complex tensors")),
            },
        })
    }

    pub async fn cast(&self, id: TxnId, dtype: NumberType) -> TCResult<Self> {
        let data = cast_target!(self, id, dtype);

        Ok(Self { data })
    }

    pub async fn all(&self, id: TxnId) -> TCResult<bool> {
        dispatch!(&self.data, value => {
            let result = value.expression(id).await?.all().await?;
            Ok(result)
        })
    }

    pub async fn any(&self, id: TxnId) -> TCResult<bool> {
        dispatch!(&self.data, value => {
            let result = value.expression(id).await?.any().await?;
            Ok(result)
        })
    }

    pub async fn sum(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, sum_all, all);
        }

        let data = native_unary!(id, self, sum(axes, keepdims), all => same);
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn product(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, product_all, all);
        }

        let data = native_unary!(id, self, product(axes, keepdims), all => same);
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn min(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, min_all, real, [cfg(feature = "complex")] TCError::unsupported("min does not support complex tensors"));
        }

        let data = native_unary!(id, self, min(axes, keepdims), real => same, [cfg(feature = "complex")] TCError::unsupported("min does not support complex tensors"));
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn max(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, max_all, real, [cfg(feature = "complex")] TCError::unsupported("max does not support complex tensors"));
        }

        let data = native_unary!(id, self, max(axes, keepdims), real => same, [cfg(feature = "complex")] TCError::unsupported("max does not support complex tensors"));
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn cond(&self, id: TxnId, then: &Self, or_else: &Self) -> TCResult<Self> {
        let shape = broadcast_shape(
            &broadcast_shape(self.shape(), then.shape())?,
            or_else.shape(),
        )?;

        let condition = self.clone().broadcast(shape.clone())?;
        let then = then.clone().broadcast(shape.clone())?;
        let or_else = or_else.clone().broadcast(shape)?;
        let data = native_conditional!(id, condition, then, or_else, all);

        Ok(Self { data })
    }

    pub async fn mean(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, mean_all, all);
        }

        let data = statistic!(self, id, axes, keepdims, mean, C64);
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn norm(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, norm_all, all);
        }

        let data = statistic!(self, id, axes, keepdims, norm, F64);
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn std(
        &self,
        id: TxnId,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> TCResult<TensorReduceResult<Txn>> {
        let axes = reduction_axes(self.shape(), axes)?;
        if !keepdims && axes.len() == self.shape().len() {
            return native_terminal!(id, self, std_all, all);
        }

        let data = statistic!(self, id, axes, keepdims, std, F64);
        Ok(TensorReduceResult::Tensor(Self { data }))
    }

    pub async fn broadcast_reduce(&self, id: TxnId, target: Shape) -> TCResult<Self> {
        let axes = fensor::broadcast_reduce_axes(self.shape(), &target)?;
        let TensorReduceResult::Tensor(value) = self.sum(id, Some(axes.into_vec()), true).await?
        else {
            unreachable!("keepdims")
        };

        value.reshape(target)
    }
}
