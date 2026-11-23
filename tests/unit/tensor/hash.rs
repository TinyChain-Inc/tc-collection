use async_hash::{Sha256, hash_try_stream};
use futures::StreamExt;
use number_general::Number;

async fn check<T>(values: Vec<T>)
where
    T: Copy + Default + PartialEq + Into<Number> + Send + 'static,
{
    let expected = hash_try_stream::<Sha256, _, _, _>(futures::stream::iter(
        values
            .iter()
            .copied()
            .map(|value| Ok::<_, fensor::Error>(value.into())),
    ))
    .await
    .unwrap();

    for width in [1, 17, 128, 4096] {
        let batches = values
            .chunks(width)
            .map(|chunk| Ok(chunk.to_vec()))
            .collect::<Vec<_>>();

        let actual = super::blocks(futures::stream::iter(batches).boxed())
            .await
            .unwrap();

        assert_eq!(actual, expected);
    }
}

#[tokio::test]
async fn batches_preserve_element_hashes_and_zero_representations() {
    check(Vec::<u8>::new()).await;
    check(vec![0u8; 4097]).await;
    check(vec![0f32, -0f32, 1., f32::NAN, f32::INFINITY, -0.]).await;
    check(vec![0f64, -0f64, 1., f64::NAN, f64::NEG_INFINITY, -0.]).await;
    check(
        (0..4097)
            .map(|i| if i % 31 == 0 { i as f64 } else { -0. })
            .collect(),
    )
    .await;
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn complex_zero_signs_are_independent() {
    let signs = [(0., 0.), (0., -0.), (-0., 0.), (-0., -0.)];
    check(
        signs
            .iter()
            .map(|&(re, im)| fensor::complex::Complex32::new(re, im))
            .collect(),
    )
    .await;

    check(
        signs
            .iter()
            .map(|&(re, im)| fensor::complex::Complex64::new(f64::from(re), f64::from(im)))
            .collect(),
    )
    .await;
}
