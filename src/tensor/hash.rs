use async_hash::{Hash, SequenceHash, Sha256};
use futures::TryStreamExt;
use number_general::{Complex, Float, Number};
use tc_error::TCResult;
use tc_ir::Sha256Hash;

pub(super) async fn blocks<T>(mut blocks: fensor::ValueBlockStream<'_, T>) -> TCResult<Sha256Hash>
where
    T: Copy + Default + PartialEq + Into<Number> + Send,
{
    let mut digest = SequenceHash::<Sha256>::default();
    // A Tensor has one dtype. Real zeros have two signs; complex zeros have four.
    let mut zeros = [None::<Sha256Hash>; 4];
    let mut remaining = 128;
    while let Some(values) = blocks.try_next().await? {
        tokio::task::coop::consume_budget().await;
        for value in values {
            let number = value.into();
            let hash = if value == T::default() {
                let slot = match number {
                    Number::Float(Float::F32(value)) => usize::from(value.is_sign_negative()),
                    Number::Float(Float::F64(value)) => usize::from(value.is_sign_negative()),
                    Number::Complex(Complex::C32(value)) => {
                        usize::from(value.re.is_sign_negative()) * 2
                            + usize::from(value.im.is_sign_negative())
                    }
                    Number::Complex(Complex::C64(value)) => {
                        usize::from(value.re.is_sign_negative()) * 2
                            + usize::from(value.im.is_sign_negative())
                    }
                    _ => 0,
                };

                *zeros[slot].get_or_insert_with(|| Hash::<Sha256>::hash(number))
            } else {
                Hash::<Sha256>::hash(number)
            };

            digest.update_hash(&hash);
            remaining -= 1;
            if remaining == 0 {
                tokio::task::yield_now().await;
                remaining = 128;
            }
        }
    }

    Ok(digest.finish())
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/hash.rs"]
mod tests;
