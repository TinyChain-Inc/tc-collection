use std::convert::TryFrom;

use number_general::{FloatType, Number, UIntType};
use pathlink::PathBuf;
use safecast::CastInto;
use tc_value::{NumberType, number_type_from_path};

use super::Tensor;

pub(super) fn tensor_from_parts(
    dtype: NumberType,
    shape: Vec<usize>,
    values: Vec<Number>,
) -> Result<Tensor, String> {
    match dtype {
        NumberType::Float(FloatType::F32) => {
            let values = numbers_to_f32(values)?;
            Tensor::dense_f32(shape, values)
        }
        NumberType::Float(FloatType::F64) => {
            let values = numbers_to_f64(values)?;
            Tensor::dense_f64(shape, values)
        }
        NumberType::UInt(UIntType::U64) => {
            let values = numbers_to_u64(values)?;
            Tensor::dense_u64(shape, values)
        }
        other => Err(format!("unsupported tensor dtype {other}")),
    }
}

pub(super) fn tensor_dtype_from_wire(raw: &str) -> Option<NumberType> {
    let path = raw.parse::<PathBuf>().ok()?;
    number_type_from_path(&path)
}

pub(super) fn coerce_shape(dims: Vec<u64>) -> Result<Vec<usize>, String> {
    dims.into_iter()
        .map(|dim| usize::try_from(dim).map_err(|_| format!("invalid dimension {dim}")))
        .collect()
}

fn numbers_to_f32(values: Vec<Number>) -> Result<Vec<f32>, String> {
    values
        .into_iter()
        .map(|number| {
            if matches!(number, Number::Complex(_)) {
                Err("complex numbers are not supported in tensors".into())
            } else {
                Ok(number.cast_into())
            }
        })
        .collect()
}

fn numbers_to_f64(values: Vec<Number>) -> Result<Vec<f64>, String> {
    values
        .into_iter()
        .map(|number| {
            if matches!(number, Number::Complex(_)) {
                Err("complex numbers are not supported in tensors".into())
            } else {
                Ok(number.cast_into())
            }
        })
        .collect()
}

fn numbers_to_u64(values: Vec<Number>) -> Result<Vec<u64>, String> {
    values
        .into_iter()
        .map(|number| {
            ensure_tensor_u64_component(&number)?;
            Ok(number.cast_into())
        })
        .collect()
}

pub(super) fn ensure_tensor_u64_component(number: &Number) -> Result<(), String> {
    match number {
        Number::Bool(_) | Number::UInt(_) => Ok(()),
        Number::Int(int) => {
            let value = i64::from(*int);
            if value < 0 {
                Err("tensor values must be non-negative".into())
            } else {
                Ok(())
            }
        }
        Number::Float(float) => {
            let value = f64::from(*float);
            if !value.is_finite() {
                Err("tensor value must be finite".into())
            } else if value < 0.0 || value.fract() != 0.0 {
                Err(format!(
                    "expected a non-negative whole number but found {value}"
                ))
            } else {
                Ok(())
            }
        }
        Number::Complex(_) => Err("complex numbers are not supported in tensors".into()),
    }
}
