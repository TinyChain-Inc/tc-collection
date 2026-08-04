use ha_ndarray::{ArrayBuf, Buffer, NDArray, NDArrayRead, NDArrayTransform};
use number_general::{FloatType, Number, UIntType};
use safecast::CastInto;
use tc_value::NumberType;

use super::wire::ensure_tensor_u64_component;
use super::{Range, Tensor, TensorReduceResult};

impl Tensor {
    /// Construct a dense `f32` tensor from a shape and flattened values.
    pub fn dense_f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::F32)
            .map_err(|err| err.to_string())
    }

    /// Construct a dense `f64` tensor from a shape and flattened values.
    pub fn dense_f64(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::F64)
            .map_err(|err| err.to_string())
    }

    /// Construct a dense `u64` tensor from a shape and flattened values.
    pub fn dense_u64(shape: Vec<usize>, values: Vec<u64>) -> Result<Self, String> {
        let shape = shape.into();
        let buffer = Buffer::from(values);
        ArrayBuf::new(buffer, shape)
            .map(Box::new)
            .map(Tensor::U64)
            .map_err(|err| err.to_string())
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(array) => array.shape(),
            Tensor::F64(array) => array.shape(),
            Tensor::U64(array) => array.shape(),
        }
    }

    pub fn number_type(&self) -> NumberType {
        match self {
            Tensor::F32(_) => NumberType::Float(FloatType::F32),
            Tensor::F64(_) => NumberType::Float(FloatType::F64),
            Tensor::U64(_) => NumberType::UInt(UIntType::U64),
        }
    }

    pub fn dtype_tag(&self) -> &'static str {
        match self {
            Tensor::F32(_) => "f32",
            Tensor::F64(_) => "f64",
            Tensor::U64(_) => "u64",
        }
    }

    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn flattened_f32(&self) -> Result<Vec<f32>, String> {
        match self {
            Tensor::F32(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F64(_) => Err("tensor dtype is not f32".to_string()),
            Tensor::U64(_) => Err("tensor dtype is not f32".to_string()),
        }
    }

    pub fn flattened_f64(&self) -> Result<Vec<f64>, String> {
        match self {
            Tensor::F64(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F32(_) => Err("tensor dtype is not f64".to_string()),
            Tensor::U64(_) => Err("tensor dtype is not f64".to_string()),
        }
    }

    pub fn flattened_u64(&self) -> Result<Vec<u64>, String> {
        match self {
            Tensor::U64(array) => Ok(array
                .buffer()
                .map_err(|err| err.to_string())?
                .to_slice()
                .map_err(|err| err.to_string())?
                .into_vec()),
            Tensor::F32(_) => Err("tensor dtype is not u64".to_string()),
            Tensor::F64(_) => Err("tensor dtype is not u64".to_string()),
        }
    }

    pub fn values_f64(&self) -> Result<Vec<f64>, String> {
        match self {
            Tensor::F32(_) => Ok(self.flattened_f32()?.into_iter().map(f64::from).collect()),
            Tensor::F64(_) => self.flattened_f64(),
            Tensor::U64(_) => Ok(self
                .flattened_u64()?
                .into_iter()
                .map(|value| value as f64)
                .collect()),
        }
    }

    pub fn from_f64_like(&self, shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        match self {
            Tensor::F64(_) => Tensor::dense_f64(shape, values),
            Tensor::U64(_)
                if values
                    .iter()
                    .all(|value| *value >= 0.0 && value.fract() == 0.0) =>
            {
                Tensor::dense_u64(
                    shape,
                    values.into_iter().map(|value| value as u64).collect(),
                )
            }
            _ => Tensor::dense_f32(
                shape,
                values.into_iter().map(|value| value as f32).collect(),
            ),
        }
    }

    pub fn cast(self, dtype: NumberType) -> Result<Self, String> {
        let dtype = match dtype {
            NumberType::Float(FloatType::F32) => "f32",
            NumberType::Float(FloatType::F64) => "f64",
            NumberType::UInt(UIntType::U64) => "u64",
            other => return Err(format!("unsupported tensor dtype {other}")),
        };

        if self.dtype_tag() == dtype {
            return Ok(self);
        }

        let shape = self.shape().to_vec();
        let values = self.values_f64()?;

        match dtype {
            "f32" => Tensor::dense_f32(shape, values.into_iter().map(|v| v as f32).collect()),
            "f64" => Tensor::dense_f64(shape, values),
            "u64" => {
                let mut out = Vec::with_capacity(values.len());
                for value in values {
                    if !value.is_finite() {
                        return Err("tensor cast to u64 requires finite values".to_string());
                    }
                    if value < 0.0 || value.fract() != 0.0 {
                        return Err(format!(
                            "tensor cast to u64 requires non-negative whole numbers, found {value}"
                        ));
                    }
                    out.push(value as u64);
                }

                Tensor::dense_u64(shape, out)
            }
            other => Err(format!("unsupported tensor dtype {other}")),
        }
    }

    pub fn read_value(&self, coord: &[u64]) -> Result<Number, String> {
        let offset = coord_offset(self.shape(), coord)?;

        match self {
            Tensor::F32(_) => Ok(Number::from(self.flattened_f32()?[offset])),
            Tensor::F64(_) => Ok(Number::from(self.flattened_f64()?[offset])),
            Tensor::U64(_) => Ok(Number::from(self.flattened_u64()?[offset])),
        }
    }

    pub fn write_value(&mut self, coord: &[u64], value: Number) -> Result<(), String> {
        let shape = self.shape().to_vec();
        let offset = coord_offset(&shape, coord)?;

        let next = match self {
            Tensor::F32(_) => {
                ensure_non_complex(&value)?;
                let mut values = self.flattened_f32()?;
                values[offset] = value.cast_into();
                Tensor::dense_f32(shape, values)?
            }
            Tensor::F64(_) => {
                ensure_non_complex(&value)?;
                let mut values = self.flattened_f64()?;
                values[offset] = value.cast_into();
                Tensor::dense_f64(shape, values)?
            }
            Tensor::U64(_) => {
                ensure_tensor_u64_component(&value)?;
                let mut values = self.flattened_u64()?;
                values[offset] = value.cast_into();
                Tensor::dense_u64(shape, values)?
            }
        };

        *self = next;
        Ok(())
    }

    pub fn reshape(self, shape: Vec<usize>) -> Result<Self, String> {
        if shape.iter().product::<usize>() != self.size() {
            return Err("tensor reshape changes size".to_string());
        }

        self.from_f64_like(shape, self.values_f64()?)
    }

    pub fn expand_dims(self, axes: Option<Vec<usize>>) -> Result<Self, String> {
        let mut shape = self.shape().to_vec();

        if let Some(axes) = axes {
            for axis in axes {
                if axis > shape.len() {
                    return Err("expand_dims axis out of bounds".to_string());
                }

                shape.insert(axis, 1);
            }
        } else {
            shape.push(1);
        }

        self.from_f64_like(shape, self.values_f64()?)
    }

    pub fn broadcast(self, shape: Vec<usize>) -> Result<Self, String> {
        let source_shape = self.shape().to_vec();
        if !can_broadcast_to(&source_shape, &shape) {
            return Err(format!(
                "cannot broadcast tensor shape {:?} into {:?}",
                source_shape, shape
            ));
        }

        let source_values = self.values_f64()?;
        let out_len = shape.iter().product::<usize>();
        let mut out = Vec::with_capacity(out_len);

        for linear_idx in 0..out_len {
            let out_coord = unravel_index(linear_idx, &shape);
            let source_coord = project_broadcast_index(&out_coord, &source_shape)?;
            let source_offset = coord_offset_usize(&source_shape, &source_coord)?;
            out.push(source_values[source_offset]);
        }

        self.from_f64_like(shape, out)
    }

    pub fn transpose(self, permutation: Option<Vec<usize>>) -> Result<Self, String> {
        let shape = self.shape().to_vec();
        let permutation = if let Some(permutation) = permutation {
            if permutation.len() != shape.len() {
                return Err("transpose permutation rank must match tensor rank".to_string());
            }

            let mut seen = vec![false; shape.len()];
            for axis in &permutation {
                if *axis >= shape.len() {
                    return Err("transpose axis out of bounds".to_string());
                }
                if seen[*axis] {
                    return Err("transpose permutation contains duplicate axis".to_string());
                }
                seen[*axis] = true;
            }

            permutation
        } else {
            (0..shape.len()).rev().collect()
        };

        let out_shape: Vec<usize> = permutation.iter().map(|axis| shape[*axis]).collect();
        let out_len = out_shape.iter().product::<usize>();
        let values = self.values_f64()?;
        let mut out = vec![0.0; out_len];

        for (linear_idx, out_value) in out.iter_mut().enumerate() {
            let out_coord = unravel_index(linear_idx, &out_shape);
            let mut in_coord = vec![0usize; shape.len()];
            for (out_axis, in_axis) in permutation.iter().copied().enumerate() {
                in_coord[in_axis] = out_coord[out_axis];
            }

            let in_offset = coord_offset_usize(&shape, &in_coord)?;
            *out_value = values[in_offset];
        }

        self.from_f64_like(out_shape, out)
    }

    pub fn slice(self, range: Range) -> Result<Self, String> {
        match self {
            Tensor::F32(array) => {
                let sliced = array.slice(range.clone()).map_err(|err| err.to_string())?;
                let shape = sliced.shape().to_vec();
                let values = sliced
                    .buffer()
                    .map_err(|err| err.to_string())?
                    .to_slice()
                    .map_err(|err| err.to_string())?
                    .into_vec();

                Tensor::dense_f32(shape, values)
            }
            Tensor::F64(array) => {
                let sliced = array.slice(range.clone()).map_err(|err| err.to_string())?;
                let shape = sliced.shape().to_vec();
                let values = sliced
                    .buffer()
                    .map_err(|err| err.to_string())?
                    .to_slice()
                    .map_err(|err| err.to_string())?
                    .into_vec();

                Tensor::dense_f64(shape, values)
            }
            Tensor::U64(array) => {
                let sliced = array.slice(range).map_err(|err| err.to_string())?;
                let shape = sliced.shape().to_vec();
                let values = sliced
                    .buffer()
                    .map_err(|err| err.to_string())?
                    .to_slice()
                    .map_err(|err| err.to_string())?
                    .into_vec();

                Tensor::dense_u64(shape, values)
            }
        }
    }

    pub fn reduce(&self, op: &str) -> Result<Number, String> {
        let values = self.values_f64()?;
        if values.is_empty() {
            return Err("cannot reduce an empty tensor".to_string());
        }

        let value = match op {
            "max" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "min" => values.iter().copied().fold(f64::INFINITY, f64::min),
            "mean" => values.iter().sum::<f64>() / values.len() as f64,
            "norm" => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
            "product" => values.iter().product::<f64>(),
            "std" => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64)
                    .sqrt()
            }
            "sum" => values.iter().sum::<f64>(),
            other => return Err(format!("unsupported tensor reduction {other}")),
        };

        Ok(Number::from(value))
    }

    pub fn reduce_axes(
        &self,
        op: &str,
        axes: Option<Vec<usize>>,
        keepdims: bool,
    ) -> Result<TensorReduceResult, String> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let values = self.values_f64()?;

        if values.is_empty() {
            return Err("cannot reduce an empty tensor".to_string());
        }

        let axes = axes.unwrap_or_else(|| (0..rank).collect());
        let mut reduce_mask = vec![false; rank];
        for axis in axes {
            if axis >= rank {
                return Err(format!("reduction axis {axis} is out of bounds"));
            }
            reduce_mask[axis] = true;
        }

        let out_shape = if keepdims {
            shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| if reduce_mask[axis] { 1 } else { *dim })
                .collect::<Vec<_>>()
        } else {
            shape
                .iter()
                .enumerate()
                .filter_map(|(axis, dim)| (!reduce_mask[axis]).then_some(*dim))
                .collect::<Vec<_>>()
        };

        let in_strides = strides(&shape);
        let out_strides = strides(&out_shape);
        let out_size = if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().product()
        };

        let mut acc = vec![ReduceAccum::default(); out_size];
        for (flat_idx, value) in values.into_iter().enumerate() {
            let coord = coord_from_offset(flat_idx, &shape, &in_strides);

            let out_coord = if keepdims {
                coord
                    .iter()
                    .enumerate()
                    .map(|(axis, c)| if reduce_mask[axis] { 0 } else { *c })
                    .collect::<Vec<_>>()
            } else {
                coord
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, c)| (!reduce_mask[axis]).then_some(*c))
                    .collect::<Vec<_>>()
            };

            let out_offset = offset_from_coord(&out_coord, &out_strides);
            acc[out_offset].update(value);
        }

        let out_values = acc
            .into_iter()
            .map(|acc| acc.finalize(op))
            .collect::<Result<Vec<_>, _>>()?;

        if out_shape.is_empty() {
            Ok(TensorReduceResult::Scalar(Number::from(out_values[0])))
        } else {
            let tensor = self.from_f64_like(out_shape, out_values)?;
            Ok(TensorReduceResult::Tensor(tensor))
        }
    }

    pub fn binary_op(&self, right: &Self, op: &str) -> Result<Self, String> {
        let left_values = self.values_f64()?;
        let right_values = right.values_f64()?;
        let shape = broadcast_shape(self.shape(), right.shape())?;

        let len = shape.iter().product::<usize>();
        let out = (0..len)
            .map(|idx| {
                let out_coord = unravel_index(idx, &shape);
                let left_coord = project_broadcast_index(&out_coord, self.shape())?;
                let right_coord = project_broadcast_index(&out_coord, right.shape())?;

                let l = left_values[coord_offset_usize(self.shape(), &left_coord)?];
                let r = right_values[coord_offset_usize(right.shape(), &right_coord)?];
                match op {
                    "add" => Ok(l + r),
                    "sub" => Ok(l - r),
                    "mul" => Ok(l * r),
                    "div" => Ok(l / r),
                    "and" => Ok(f64::from(l != 0.0 && r != 0.0)),
                    "or" => Ok(f64::from(l != 0.0 || r != 0.0)),
                    "xor" => Ok(f64::from((l != 0.0) ^ (r != 0.0))),
                    other => Err(format!("unsupported tensor binary op {other}")),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.from_f64_like(shape, out)
    }

    pub fn unary_not(&self) -> Result<Self, String> {
        let values = self
            .values_f64()?
            .into_iter()
            .map(|value| f64::from(value == 0.0))
            .collect();

        self.from_f64_like(self.shape().to_vec(), values)
    }

    pub fn cond(cond: &Self, then_tensor: &Self, else_tensor: &Self) -> Result<Self, String> {
        let cond_values = cond.values_f64()?;
        let then_values = then_tensor.values_f64()?;
        let else_values = else_tensor.values_f64()?;

        if then_values.len() != else_values.len() {
            return Err("tensor cond branches must have equal size".to_string());
        }

        if cond_values.len() != 1 && cond_values.len() != then_values.len() {
            return Err("tensor cond must be scalar or equal size".to_string());
        }

        let out = (0..then_values.len())
            .map(|idx| {
                if cond_values[if cond_values.len() == 1 { 0 } else { idx }] != 0.0 {
                    then_values[idx]
                } else {
                    else_values[idx]
                }
            })
            .collect();

        then_tensor.from_f64_like(then_tensor.shape().to_vec(), out)
    }

    pub fn matmul(&self, right: &Self) -> Result<Self, String> {
        let left_shape = self.shape();
        let right_shape = right.shape();
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err("tensor matmul currently supports 2D tensors".to_string());
        }

        let rows = left_shape[0];
        let inner = left_shape[1];
        if right_shape[0] != inner {
            return Err("tensor matmul inner dimensions do not match".to_string());
        }

        let cols = right_shape[1];
        let left_values = self.values_f64()?;
        let right_values = right.values_f64()?;
        let mut out = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += left_values[row * inner + k] * right_values[k * cols + col];
                }
                out[row * cols + col] = sum;
            }
        }

        self.from_f64_like(vec![rows, cols], out)
    }
}

fn ensure_non_complex(number: &Number) -> Result<(), String> {
    if matches!(number, Number::Complex(_)) {
        Err("complex numbers are not supported in tensors".to_string())
    } else {
        Ok(())
    }
}

fn can_broadcast_to(source: &[usize], target: &[usize]) -> bool {
    if source.len() > target.len() {
        return false;
    }

    source
        .iter()
        .rev()
        .zip(target.iter().rev())
        .all(|(left, right)| *left == 1 || left == right)
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, String> {
    let ndim = usize::max(left.len(), right.len());
    let mut out = vec![1usize; ndim];

    for i in 0..ndim {
        let l = left
            .len()
            .checked_sub(i + 1)
            .and_then(|idx| left.get(idx))
            .copied()
            .unwrap_or(1);
        let r = right
            .len()
            .checked_sub(i + 1)
            .and_then(|idx| right.get(idx))
            .copied()
            .unwrap_or(1);

        out[ndim - i - 1] = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(format!(
                "tensor shapes {:?} and {:?} are not broadcast-compatible",
                left, right
            ));
        };
    }

    Ok(out)
}

fn project_broadcast_index(
    out_coord: &[usize],
    source_shape: &[usize],
) -> Result<Vec<usize>, String> {
    if source_shape.len() > out_coord.len() {
        return Err("source rank exceeds output rank for broadcast".to_string());
    }

    let mut source_coord = vec![0usize; source_shape.len()];
    let offset = out_coord.len() - source_shape.len();

    for (axis, dim) in source_shape.iter().copied().enumerate() {
        source_coord[axis] = if dim == 1 {
            0
        } else {
            out_coord[offset + axis]
        };
    }

    Ok(source_coord)
}

fn unravel_index(mut linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coord = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        coord[axis] = linear_idx % dim;
        linear_idx /= dim;
    }

    coord
}

fn coord_offset_usize(shape: &[usize], coord: &[usize]) -> Result<usize, String> {
    if coord.len() != shape.len() {
        return Err("incorrect number of coordinates".to_string());
    }

    let mut offset = 0usize;
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let value = coord[axis];
        if value >= dim {
            return Err(format!("coordinate at axis {axis} is out of bounds"));
        }
        offset += value * stride;
        stride *= dim;
    }

    Ok(offset)
}

fn coord_offset(shape: &[usize], coord: &[u64]) -> Result<usize, String> {
    if coord.len() != shape.len() {
        return Err("incorrect number of coordinates".to_string());
    }

    let mut offset = 0usize;
    let mut stride = 1usize;

    for (axis, dim) in shape.iter().enumerate().rev() {
        let value = usize::try_from(coord[axis])
            .map_err(|_| format!("coordinate at axis {axis} overflows usize"))?;

        if value >= *dim {
            return Err(format!("coordinate at axis {axis} is out of bounds"));
        }

        offset += value * stride;
        stride *= *dim;
    }

    Ok(offset)
}

fn strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn coord_from_offset(offset: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut remainder = offset;
    let mut coord = vec![0; shape.len()];
    for axis in 0..shape.len() {
        let stride = strides[axis];
        coord[axis] = remainder / stride;
        remainder %= stride;
    }
    coord
}

fn offset_from_coord(coord: &[usize], strides: &[usize]) -> usize {
    coord
        .iter()
        .zip(strides.iter())
        .map(|(value, stride)| value * stride)
        .sum()
}

#[derive(Clone, Copy, Debug, Default)]
struct ReduceAccum {
    initialized: bool,
    count: usize,
    sum: f64,
    sumsq: f64,
    product: f64,
    min: f64,
    max: f64,
}

impl ReduceAccum {
    fn update(&mut self, value: f64) {
        if !self.initialized {
            self.initialized = true;
            self.count = 1;
            self.sum = value;
            self.sumsq = value * value;
            self.product = value;
            self.min = value;
            self.max = value;
            return;
        }

        self.count += 1;
        self.sum += value;
        self.sumsq += value * value;
        self.product *= value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn finalize(self, op: &str) -> Result<f64, String> {
        if !self.initialized {
            return Err("cannot reduce an empty tensor".to_string());
        }

        match op {
            "max" => Ok(self.max),
            "min" => Ok(self.min),
            "mean" => Ok(self.sum / self.count as f64),
            "norm" => Ok(self.sumsq.sqrt()),
            "product" => Ok(self.product),
            "std" => {
                let mean = self.sum / self.count as f64;
                Ok((self.sumsq / self.count as f64 - mean * mean)
                    .max(0.0)
                    .sqrt())
            }
            "sum" => Ok(self.sum),
            other => Err(format!("unsupported tensor reduction {other}")),
        }
    }
}
