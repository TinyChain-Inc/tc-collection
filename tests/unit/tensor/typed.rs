use tc_error::TCError;
use tc_ir::{Transact, Transaction};

use super::Typed;
use crate::tensor::Data;
use crate::test::tensor::{create, id, setup, values};

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn failed_staged_assignment_preserves_pending_state() {
    Box::pin(async {
        let (_, txn) = setup().await;
        let limit = fensor::MAX_BATCH_ELEMENTS;
        let size = limit + 3;
        let (tensor, _) = create(&txn, vec![size as u64].into(), false, 256).await;
        tensor.write_value(&txn, &[0], 7.0.into()).await.unwrap();
        let Data::F64(value) = &tensor.data else {
            unreachable!()
        };

        let mut expected = vec![0.; size];
        expected[0] = 7.;

        for (name, count, source_error, message) in [
            ("early source failure", 1, true, "supplied stream failed"),
            (
                "source failure after batch",
                limit,
                true,
                "supplied stream failed",
            ),
            (
                "too few",
                size - 1,
                false,
                "too few Tensor assignment values",
            ),
            (
                "too many",
                size + 1,
                false,
                "too many Tensor assignment values",
            ),
        ] {
            let input = (0..count)
                .map(|_| Ok(1.0))
                .chain(source_error.then(|| Err(TCError::bad_request("supplied stream failed"))));

            let error = Typed::write_elements(
                &fensor::TensorView::new(value.owner().unwrap().clone()),
                &txn,
                futures::stream::iter(input),
            )
            .await
            .unwrap_err();

            assert!(error.to_string().contains(message), "{name}: {error}");
            assert_eq!(values(&tensor, txn.id()).await, expected, "{name}");
        }

        tensor.commit(txn.id()).await.unwrap();
        assert_eq!(values(&tensor, id(11)).await, expected);

        let next = txn.at(12);
        Typed::write_elements(
            &fensor::TensorView::new(value.owner().unwrap().clone()),
            &next,
            futures::stream::iter((0..size).map(|_| Ok(2.0))),
        )
        .await
        .unwrap();

        tensor.commit(next.id()).await.unwrap();
        assert_eq!(values(&tensor, id(13)).await, vec![2.; size]);
    })
    .await;
}
