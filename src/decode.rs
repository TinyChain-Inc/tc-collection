use destream::de::{self, FromStream};

use crate::btree::BTree;
use crate::table::Table;
use crate::tensor::Tensor;
use crate::{Collection, CollectionType};

/// Decode one collection payload after the universal state visitor identifies its class.
pub async fn decode_collection<Txn, D>(
    class: CollectionType,
    txn: Txn,
    decoder: &mut D,
) -> Result<Collection<Txn>, D::Error>
where
    Txn: crate::StorageContext,
    D: de::Decoder,
{
    match class {
        CollectionType::BTree(_) => BTree::<Txn>::from_stream(txn, decoder)
            .await
            .map(|view| Collection::BTree(Box::new(view))),
        CollectionType::Table(_) => Table::<Txn>::from_stream(txn, decoder)
            .await
            .map(Collection::from),
        CollectionType::Tensor(_) => Tensor::<Txn>::from_stream(txn, decoder)
            .await
            .map(Collection::Tensor),
    }
}
