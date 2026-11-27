use futures::TryStreamExt;
use tc_error::TCError;

use super::CollectionView;

impl CollectionView {
    // Check the storage error before the wire encoder projects its diagnostics.
    pub(crate) async fn next_error_for_test(&mut self) -> TCError {
        match self {
            Self::BTree(view) => view.keys.try_next().await.unwrap_err(),
            Self::Table(view) => view.rows.try_next().await.unwrap_err(),
            Self::Tensor(_) => unreachable!("row view fixture"),
        }
    }
}
