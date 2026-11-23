use destream::de::{self, FromStream};

use crate::btree::DecodedBTreePayload;
use crate::collection::BTreeView;
use crate::table::DecodedTablePayload;
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
        CollectionType::BTree(_) => {
            let payload = DecodedBTreePayload::<Txn>::from_stream(txn, decoder).await?;
            Ok(Collection::BTree(Box::new(
                BTreeView::new(payload.schema, payload.btree).map_err(de::Error::custom)?,
            )))
        }
        CollectionType::Table(_) => {
            let payload = DecodedTablePayload::<Txn>::from_stream(txn, decoder).await?;
            Ok(Collection::from(payload.table))
        }
        CollectionType::Tensor(_) => Tensor::<Txn>::from_stream(txn, decoder)
            .await
            .map(Collection::Tensor),
    }
}
