use destream::de;

use crate::{
    Collection, CollectionType, btree::DecodedBTreePayload, collection::BTreeView,
    table::DecodedTablePayload, tensor::Tensor,
};

/// Decode one collection payload after the universal state visitor identifies its class.
pub async fn decode_collection<Txn, A>(
    class: CollectionType,
    txn: Txn,
    map: &mut A,
) -> Result<Collection<Txn>, A::Error>
where
    Txn: crate::StorageContext,
    A: de::MapAccess,
{
    match class {
        CollectionType::BTree(_) => {
            let payload = map.next_value::<DecodedBTreePayload<Txn>>(txn).await?;
            Ok(Collection::BTree(Box::new(BTreeView::new(
                payload.schema,
                payload.btree,
            ))))
        }
        CollectionType::Table(_) => {
            let payload = map.next_value::<DecodedTablePayload<Txn>>(txn).await?;
            Ok(Collection::from(payload.table))
        }
        CollectionType::Tensor(_) => map.next_value::<Tensor>(()).await.map(Collection::Tensor),
    }
}
