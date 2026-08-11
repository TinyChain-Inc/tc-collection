use super::Tensor;

use super::dtype::TensorOpError;

pub fn tensor_transpose(input: &Tensor, permutation: &[usize]) -> Result<Tensor, TensorOpError> {
    let output_shape = transpose_output_shape(input.shape(), permutation)?;

    let result = input
        .clone()
        .transpose(Some(permutation.to_vec()))
        .map_err(|_| invalid_permutation(input.shape().len(), permutation, "execution failed"))?;

    debug_assert_eq!(result.shape(), output_shape.as_slice());
    Ok(result)
}

pub fn transpose_output_shape(
    input_shape: &[usize],
    permutation: &[usize],
) -> Result<Vec<usize>, TensorOpError> {
    validate_permutation(input_shape.len(), permutation)?;
    Ok(permutation.iter().map(|axis| input_shape[*axis]).collect())
}

fn validate_permutation(rank: usize, permutation: &[usize]) -> Result<(), TensorOpError> {
    if permutation.len() != rank {
        return Err(invalid_permutation(
            rank,
            permutation,
            "length must match input rank",
        ));
    }

    let mut seen = vec![false; rank];
    for axis in permutation {
        if *axis >= rank {
            return Err(invalid_permutation(rank, permutation, "axis out of range"));
        }
        if seen[*axis] {
            return Err(invalid_permutation(
                rank,
                permutation,
                "axis appears more than once",
            ));
        }
        seen[*axis] = true;
    }

    Ok(())
}

fn invalid_permutation(rank: usize, permutation: &[usize], reason: &str) -> TensorOpError {
    TensorOpError::InvalidPermutation {
        rank,
        permutation: permutation.to_vec(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_u64_values_exactly() {
        let values = vec![u64::MAX - 3, u64::MAX - 2, u64::MAX - 1, u64::MAX];
        let tensor = Tensor::dense_u64(vec![2, 2], values.clone()).expect("tensor");

        let transposed = tensor_transpose(&tensor, &[1, 0]).expect("transpose");

        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(
            transposed.flattened_u64().expect("values"),
            vec![values[0], values[2], values[1], values[3]]
        );
    }
}
