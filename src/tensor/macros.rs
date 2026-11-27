//! Private dtype declarations and mechanically identical Tensor delegation.

// Keep the dtype catalog at the invocation site. The dollar token lets the
// generated helpers retain their ordinary macro parameters without a proc macro.
macro_rules! tensor_dtypes {
    ($d:tt;
     integers [$($iv:ident: $it:ty => $ip:pat, $il:ident;)*]
     floats [$($fv:ident: $ft:ty => $fp:pat, $fl:ident;)*]
     complex [$($(#[$ca:meta])* $cv:ident: $ct:ty => $cp:pat, $cl:ident;)*]
    ) => {
        /// A transactional Tensor or read-only Tensor value, selected by numerical dtype.
        #[derive(Clone)]
        pub enum Tensor<Txn: StorageContext> {
            $($iv(TensorAccess<Txn, $it>),)*
            $($fv(TensorAccess<Txn, $ft>),)*
            $($(#[$ca])* $cv(TensorAccess<Txn, $ct>),)*
        }

        $(
            impl<Txn: StorageContext> From<TensorAccess<Txn, $it>> for Tensor<Txn> {
                fn from(value: TensorAccess<Txn, $it>) -> Self { Self::$iv(value) }
            }
        )*
        $(
            impl<Txn: StorageContext> From<TensorAccess<Txn, $ft>> for Tensor<Txn> {
                fn from(value: TensorAccess<Txn, $ft>) -> Self { Self::$fv(value) }
            }
        )*
        $(
            $(#[$ca])*
            impl<Txn: StorageContext> From<TensorAccess<Txn, $ct>> for Tensor<Txn> {
                fn from(value: TensorAccess<Txn, $ct>) -> Self { Self::$cv(value) }
            }
        )*

        macro_rules! dispatch {
            ($d data:expr, $d value:ident => $d body:expr) => {
                dispatch!(@match direct, $d data, $d value => $d body)
            };
            (@wrap $d data:expr, $d value:ident => $d body:expr) => {
                dispatch!(@match wrap, $d data, $d value => $d body)
            };
            (@result direct, $d body:expr) => { $d body };
            (@result wrap, $d body:expr) => { Tensor::from($d body) };
            (@match $d mode:ident, $d data:expr, $d value:ident => $d body:expr) => {
                match $d data {
                    $(Tensor::$iv($d value) => dispatch!(@result $d mode, $d body),)*
                    $(Tensor::$fv($d value) => dispatch!(@result $d mode, $d body),)*
                    $($(#[$ca])* Tensor::$cv($d value) => dispatch!(@result $d mode, $d body),)*
                }
            };
        }

        macro_rules! dispatch_pair {
            ($d left:expr, $d right:expr, $d target:ident, $d source:ident => $d body:expr, $d error:expr) => {
                match ($d left, $d right) {
                    $((Tensor::$iv($d target), Tensor::$iv($d source)) => $d body,)*
                    $((Tensor::$fv($d target), Tensor::$fv($d source)) => $d body,)*
                    $($(#[$ca])* (Tensor::$cv($d target), Tensor::$cv($d source)) => $d body,)*
                    _ => $d error,
                }
            };
        }

        macro_rules! dtype_factory {
            ($d dtype:expr, $d ty:ident => $d body:expr, $d unsupported:ident => $d error:expr) => {
                match $d dtype {
                    $($ip => { type $d ty = $it; Tensor::from(TensorAccess::<_, $d ty>::from_base($d body)) },)*
                    $($fp => { type $d ty = $ft; Tensor::from(TensorAccess::<_, $d ty>::from_base($d body)) },)*
                    $($(#[$ca])* $cp => { type $d ty = $ct; Tensor::from(TensorAccess::<_, $d ty>::from_base($d body)) },)*
                    $d unsupported => $d error,
                }
            };
        }

        macro_rules! cast_target {
            ($d source:expr, $d id:expr, $d dtype:expr) => {
                match $d dtype {
                    $($ip => cast_target!(@cast $d source, $d id, $it),)*
                    $($fp => cast_target!(@cast $d source, $d id, $ft),)*
                    $($(#[$ca])* $cp => cast_target!(@cast $d source, $d id, $ct),)*
                    dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {}", dtype))),
                }
            };
            (@cast $d source:expr, $d id:expr, $d ty:ty) => {
                Tensor::from(TensorAccess::<Txn, $d ty>::computed(
                    dispatch!(&$d source, value => fensor::TensorExpression::new(
                        TensorCast::<$d ty>::cast(&value.expression($d id).await?).await?
                    )?),
                    $d id,
                ))
            };
        }

        macro_rules! dense_literals {
            () => {
                $(dense_literals!(@method $il, $it);)*
                $(dense_literals!(@method $fl, $ft);)*
                $($(#[$ca])* dense_literals!(@method $cl, $ct);)*
            };
            (@method $d method:ident, $d ty:ty) => {
                pub async fn $d method(txn: &Txn, shape: Vec<u64>, values: Vec<$d ty>) -> TCResult<Self> {
                    Ok(TensorAccess::from_base(TensorBase::dense(txn, shape, values).await?).into())
                }
            };
        }

        // Families describe supported scalar data, never an execution strategy.
        macro_rules! dtype_rows {
            (all => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; $($iv,)* $($fv,)* $($(#[$ca])* $cv,)*)
            };
            (real => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; $($iv,)* $($fv,)*)
            };
            (floating => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; $($fv,)* $($(#[$ca])* $cv,)*)
            };
            (float => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; $($fv,)*)
            };
            (complex => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; $($(#[$ca])* $cv,)*)
            };
        }
    };
}

macro_rules! tensor_transforms {
    ($($method:ident($arg:ident: $ty:ty));* $(;)?) => {$(
        pub fn $method(self, $arg: $ty) -> TCResult<Self> {
            Ok(dispatch!(@wrap self, value => value.$method($arg)?))
        }
    )*};
}
