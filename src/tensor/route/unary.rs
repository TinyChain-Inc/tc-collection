//! Unary Tensor terminal handlers.

unary_handler! {
    Not => not,
    Abs => abs,
    Exp => exp,
    Ln => ln,
    Round => round,
    Sin => sin,
    Asin => asin,
    Sinh => sinh,
    Cos => cos,
    Acos => acos,
    Cosh => cosh,
    Tan => tan,
    Atan => atan,
    Tanh => tanh,
    IsNan => is_nan,
    IsInf => is_inf,
    #[cfg(feature = "complex")]
    Conj => conj,
    #[cfg(feature = "complex")]
    Re => re,
    #[cfg(feature = "complex")]
    Im => im,
    #[cfg(feature = "complex")]
    Angle => angle,
}
