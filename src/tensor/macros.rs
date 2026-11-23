//! Private dtype declarations and mechanically identical Tensor delegation.

// Keep the dtype catalog at the invocation site. The dollar token lets the
// generated helpers retain their ordinary macro parameters without a proc macro.
macro_rules! tensor_dtypes {
    ($d:tt;
     integers [$($iv:ident: $it:ty => $ip:pat, $il:ident;)*]
     floats [$($fv:ident: $ft:ty => $fp:pat, $fl:ident;)*]
     complex [$($(#[$ca:meta])* $cv:ident: $ct:ty => $cp:pat, $cl:ident => $cr:ident;)*]
    ) => {
        #[derive(Clone)]
        enum Data<Txn: StorageContext> {
            $($iv(Typed<Txn, $it>),)*
            $($fv(Typed<Txn, $ft>),)*
            $($(#[$ca])* $cv(Typed<Txn, $ct>),)*
        }

        macro_rules! dispatch {
            ($d data:expr, $d value:ident => $d body:expr) => {
                dispatch!(@match direct, $d data, $d value => $d body)
            };
            (@wrap $d data:expr, $d value:ident => $d body:expr) => {
                dispatch!(@match wrap, $d data, $d value => $d body)
            };
            (@result direct, $d variant:ident, $d body:expr) => { $d body };
            (@result wrap, $d variant:ident, $d body:expr) => { Data::$d variant($d body) };
            (@match $d mode:ident, $d data:expr, $d value:ident => $d body:expr) => {
                match $d data {
                    $(Data::$iv($d value) => dispatch!(@result $d mode, $iv, $d body),)*
                    $(Data::$fv($d value) => dispatch!(@result $d mode, $fv, $d body),)*
                    $($(#[$ca])* Data::$cv($d value) => dispatch!(@result $d mode, $cv, $d body),)*
                }
            };
        }

        macro_rules! dispatch_pair {
            ($d left:expr, $d right:expr, $d target:ident, $d source:ident => $d body:expr, $d error:expr) => {
                match ($d left, $d right) {
                    $((Data::$iv($d target), Data::$iv($d source)) => $d body,)*
                    $((Data::$fv($d target), Data::$fv($d source)) => $d body,)*
                    $($(#[$ca])* (Data::$cv($d target), Data::$cv($d source)) => $d body,)*
                    _ => $d error,
                }
            };
        }

        macro_rules! dtype_factory {
            ($d dtype:expr, $d ty:ident => $d body:expr, $d unsupported:ident => $d error:expr) => {
                match $d dtype {
                    $($ip => { type $d ty = $it; Data::$iv(Typed::stored($d body)) },)*
                    $($fp => { type $d ty = $ft; Data::$fv(Typed::stored($d body)) },)*
                    $($(#[$ca])* $cp => { type $d ty = $ct; Data::$cv(Typed::stored($d body)) },)*
                    $d unsupported => $d error,
                }
            };
        }

        macro_rules! cast_target {
            ($d source:expr, $d id:expr, $d dtype:expr) => {
                match $d dtype {
                    $($ip => cast_target!(@cast $d source, $d id, $iv, $it),)*
                    $($fp => cast_target!(@cast $d source, $d id, $fv, $ft),)*
                    $($(#[$ca])* $cp => cast_target!(@cast $d source, $d id, $cv, $ct),)*
                    dtype => return Err(TCError::bad_request(format!("unsupported Tensor dtype {}", dtype))),
                }
            };
            (@cast $d source:expr, $d id:expr, $d variant:ident, $d ty:ty) => {
                Data::$d variant(Typed::computed(
                    dispatch!(&$d source.data, value => fensor::TensorExpression::new(
                        TensorCast::<$d ty>::cast(&value.expression($d id).await?).await?
                    )?),
                    $d id,
                ))
            };
        }

        macro_rules! dense_literals {
            () => {
                $(dense_literals!(@method $il, $iv, $it);)*
                $(dense_literals!(@method $fl, $fv, $ft);)*
                $($(#[$ca])* dense_literals!(@method $cl, $cv, $ct);)*
            };
            (@method $d method:ident, $d variant:ident, $d ty:ty) => {
                pub async fn $d method(txn: &Txn, shape: Vec<u64>, values: Vec<$d ty>) -> TCResult<Self> {
                    Ok(Self { data: Data::$d variant(Typed::dense(txn, shape, values).await?) })
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

        macro_rules! statistic {
            ($d tensor:expr, $d id:expr, $d axes:expr, $d keepdims:expr, $d method:ident, $d complex:ident) => {
                native_unary!(@cases $d id, $d tensor, $d method($d axes, $d keepdims); [
                    $($iv => F64,)* $($fv => F64,)* $($(#[$ca])* $cv => $d complex,)*
                ])
            };
        }

        macro_rules! component_types {
            (all => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; [
                    $($iv => $iv,)* $($fv => $fv,)* $($(#[$ca])* $cv => $cr,)*
                ])
            };
            (complex => $d callback:ident($d($d args:tt)*)) => {
                $d callback!($d($d args)*; [$($(#[$ca])* $cv => $cr,)*])
            };
        }
    };
}

// Project an operation's explicit output rule over a selected dtype family.
macro_rules! mapped_types {
    (same => $callback:ident($($args:tt)*); $($(#[$attr:meta])* $input:ident),+ $(,)?) => {
        $callback!($($args)*; [$($(#[$attr])* $input => $input),+])
    };
    (boolean => $callback:ident($($args:tt)*); $($(#[$attr:meta])* $input:ident),+ $(,)?) => {
        $callback!($($args)*; [$($(#[$attr])* $input => U8),+])
    };
}

macro_rules! tensor_transforms {
    ($($method:ident($arg:ident: $ty:ty));* $(;)?) => {$(
        pub fn $method(self, $arg: $ty) -> TCResult<Self> {
            Ok(Self {
                data: dispatch!(@wrap self.data, value => value.$method($arg)?),
            })
        }
    )*};
}
