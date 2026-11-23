# tc-collection

`tc-collection` owns TinyChain BTree, Table, Tensor, and aggregate Collection
values. Each concrete collection implements its native behavior, routing, views,
streaming codec, and local transactional lifecycle.

Use `CopyFrom::copy_from(txn, dir, source)` to construct a native copy, and
`Restore::restore(&owner, txn, &source)` to stage an existing owner's replacement. The
[collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md) defines destination
ownership and failure handling.

Collections do not own public names, cross-process routing, transaction-ID
allocation, durable history, or reconciliation. Literal and transaction-local
values receive a delegated transaction and allocation context. Long-lived
public hosting is outside this crate.

Persistent BTree, Table, and Tensor owners provide in-memory transactional visibility and
in-place canonical materialization. `PersistentFile` stores tagged native nodes, typed Tensor blocks, and metadata.
Commit is not independently crash-durable. Native restoration, strict loading,
and caller-owned recovery are described by the
[transactional collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md).

`TableSchema::try_from(value)` consumes Value-backed schema fields and returns
parsing or constructor failures as `TCError`.
Schema validation and auxiliary declaration rules are specified by the
[Table contract](TRANSACTIONAL_COLLECTION_CONTRACT.md#table-access).

`CollectionSchema::try_from((class_path, value))` and
`TensorSchema::try_from(value)` likewise return structured construction errors.
Native BTree schema preparation
borrows columns through `BTreeSchema::try_from(columns.as_slice())`; it does not
consume or clone their names and size metadata.

Every application mutation of a persistent mutable collection must pass through
its owning Chain. Returned views grant read access, not mutation authority,
including full reads and identity transforms. Selection-write conveniences
address the original owner with explicit bounds. Trusted native callers handing
an unpublished collection to Chain relinquish mutable aliases and must not reopen
the same storage independently.

`collection::BTreeView::new(columns, owner)` returns `TCResult` and requires
matching typed backing storage. Views have private representation and immutable
schema metadata shared by their clones and slices; use `schema()` and `persistent()`
for inspection. Raw untyped BTree operations remain supported directly.

## Tensor

`Tensor<Txn>` uses fensor for native storage and numerical execution. It supports
all ten real dtypes; enable `complex` for the two complex types. Constructors,
point access, assignment, `CopyFrom`, and terminal consumption are asynchronous
and use a delegated transaction or directory. Geometric views are lazy and
read-only.

Use `Tensor::copy_values` or the `copy_values` route for an explicit copy of every
logical value when indexed copying is unsupported. This includes implicit sparse
zeros and can be expensive. Use `as_dense` for a lazy dense view.

Caller file compositions use
`delegate_tensor_files!(ApplicationBlock, Collection)` alongside `CollectionNode`.
See the contract for [Tensor representation and numerical behavior](TRANSACTIONAL_COLLECTION_CONTRACT.md#tensor-values-and-numerical-execution),
[visibility](TRANSACTIONAL_COLLECTION_CONTRACT.md#visibility-and-ordering),
[locking and resource bounds](TRANSACTIONAL_COLLECTION_CONTRACT.md#locking-and-backpressure),
[persistence and hashing](TRANSACTIONAL_COLLECTION_CONTRACT.md#persistence-and-recovery),
and [failure handling](TRANSACTIONAL_COLLECTION_CONTRACT.md#caller-owned-failure-handling).

## Testing

```bash
cargo test --all-targets --all-features
```

See the [transactional collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md),
[crate invariants](AGENTS.md), and [roadmap](ROADMAP.md).
