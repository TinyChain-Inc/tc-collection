# tc-collection

`tc-collection` owns TinyChain BTree, Table, Tensor, and aggregate Collection
values. Each concrete collection implements its native behavior, routing, views,
streaming codec, and local transactional lifecycle.

Collections do not own public names, cross-process routing, transaction-ID
allocation, durable history, or reconciliation. Literal and transaction-local
values receive a delegated transaction and allocation context. Long-lived
public hosting is outside this crate.

Persistent BTree/Table owners provide in-memory transactional visibility and
in-place canonical materialization. `PersistentFile` contains native nodes only.
Commit is not independently crash-durable. Native restoration, strict loading,
and caller-owned recovery are described by the
[transactional collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md).
Recreate development fixtures from earlier layouts.

## Tensor

There is one public Tensor type and route family. Its current backend is
in-memory. A future persistent backend is an internal replacement and must
preserve the same State, routing, codec, and transaction contracts.

## Testing

```bash
cargo test --all-targets --all-features
```

See the [transactional collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md),
[crate invariants](AGENTS.md), and [roadmap](ROADMAP.md).
