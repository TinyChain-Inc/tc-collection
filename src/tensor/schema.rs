use fensor::{Layout, Shape};
use number_general::Number;
use safecast::{CastFrom, TryCastFrom};

use tc_error::{TCError, TCResult};
use tc_value::{NumberType, Value, ValueType};

/// Maximum dimensions admitted by collection wire and filesystem metadata.
pub(crate) const MAX_TENSOR_METADATA_RANK: usize = 4096;

/// Tensor dtype, shape, and optional last sparse axis for wire reconstruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSchema {
    native: fensor::TensorSchema,
    sparse_axis: Option<usize>,
}

impl TensorSchema {
    /// Omit the axis for dense storage; otherwise specify the last sparse dimension.
    /// Shapes require 1–4096 nonzero dimensions; scalar results use `Number`.
    pub fn new(dtype: NumberType, shape: Shape, sparse_axis: Option<usize>) -> TCResult<Self> {
        let layout = sparse_axis.map_or(Layout::Dense, |axis| Layout::Sparse { axis: Some(axis) });
        Self::from_geometry(dtype, shape, layout)
    }

    /// Validate Collection metadata without allocating or inspecting native storage.
    pub(super) fn validate_geometry(shape: &[u64], layout: Layout) -> TCResult<Option<usize>> {
        if shape.len() > MAX_TENSOR_METADATA_RANK {
            return Err(TCError::bad_request(
                "Tensor schema exceeds the collection metadata rank bound",
            ));
        }

        let sparse_axis = match layout {
            Layout::Dense => None,
            Layout::Sparse { axis } => axis.or_else(|| shape.len().checked_sub(1)),
        };

        if sparse_axis.is_some_and(|axis| axis >= shape.len()) {
            return Err(TCError::bad_request("sparse axis out of bounds"));
        }

        if shape.is_empty() || shape.contains(&0) {
            return Err(TCError::bad_request(
                "Collection Tensor requires a nonempty shape with nonzero dimensions",
            ));
        }

        Ok(sparse_axis)
    }

    pub(super) fn from_geometry(dtype: NumberType, shape: Shape, layout: Layout) -> TCResult<Self> {
        let sparse_axis = Self::validate_geometry(&shape, layout)?;
        Ok(Self {
            native: fensor::TensorSchema::new(dtype, shape)?,
            sparse_axis,
        })
    }

    pub(super) fn into_native(self) -> fensor::TensorSchema {
        self.native
    }

    pub fn dtype(&self) -> NumberType {
        self.native.dtype()
    }

    pub fn shape(&self) -> &[u64] {
        self.native.shape()
    }

    pub fn is_sparse(&self) -> bool {
        self.sparse_axis.is_some()
    }

    pub(super) fn native(&self) -> &fensor::TensorSchema {
        &self.native
    }

    pub fn layout(&self) -> Layout {
        match self.sparse_axis {
            // The last axis is the native scalar-sparse default.
            Some(axis) => Layout::Sparse {
                axis: (axis + 1 < self.shape().len()).then_some(axis),
            },
            None => Layout::Dense,
        }
    }
}

impl From<TensorSchema> for Value {
    fn from(schema: TensorSchema) -> Self {
        let mut fields = vec![
            Value::cast_from(ValueType::Number(schema.dtype())),
            Value::Tuple(schema.shape().iter().copied().map(Value::from).collect()),
        ];

        if let Some(axis) = schema.sparse_axis {
            fields.push(Value::from(axis as u64));
        }

        Value::Tuple(fields)
    }
}

impl TryFrom<Value> for TensorSchema {
    type Error = TCError;

    fn try_from(value: Value) -> TCResult<Self> {
        let Value::Tuple(fields) = value else {
            return Err(TCError::bad_request("expected Tensor schema fields"));
        };

        if !(2..=3).contains(&fields.len()) {
            return Err(TCError::bad_request(
                "Tensor schema requires dtype, shape, and optional sparse axis",
            ));
        }

        let mut fields = fields.into_iter();
        let Some(ValueType::Number(dtype)) = fields.next().and_then(ValueType::opt_cast_from)
        else {
            return Err(TCError::bad_request("expected Tensor dtype path"));
        };

        let Some(Value::Tuple(shape)) = fields.next() else {
            return Err(TCError::bad_request("expected Tensor shape"));
        };

        let shape: Shape = shape
            .into_iter()
            .map(|dim| match dim {
                Value::Number(value) => dimension(value),
                _ => Err(TCError::bad_request("expected a Tensor dimension")),
            })
            .collect::<TCResult<_>>()?;

        let sparse_axis = match fields.next() {
            None => None,
            Some(Value::Number(axis)) => {
                Some(usize::try_from(dimension(axis)?).map_err(TCError::bad_request)?)
            }
            _ => return Err(TCError::bad_request("expected a Tensor sparse axis")),
        };

        Self::new(dtype, shape, sparse_axis)
    }
}

pub(super) fn dimension(value: Number) -> TCResult<u64> {
    match value {
        Number::UInt(value) => Ok(u64::from(value)),
        Number::Int(value) => u64::try_from(i64::from(value)).map_err(TCError::bad_request),
        Number::Float(value) => {
            let value = f64::from(value);
            if value.is_finite()
                && value >= 0.0
                && value.fract() == 0.0
                && value < 18_446_744_073_709_551_616.0
            {
                Ok(value as u64)
            } else {
                Err(TCError::bad_request(
                    "expected an unsigned integer dimension",
                ))
            }
        }
        _ => Err(TCError::bad_request(
            "expected an unsigned integer dimension",
        )),
    }
}
