# tc-collection

`tc-collection` owns TinyChain BTree, Table, Tensor, and aggregate Collection
values. Each concrete collection implements its native behavior, routing, views,
streaming codec, and local transactional lifecycle.

Collections do not own public names, cross-process routing, transaction-ID
allocation, durable history, or reconciliation. Literal and transaction-local
values receive a delegated transaction and allocation context. Long-lived
public hosting is outside this crate.

Persistent BTree, Table, and Tensor owners provide in-memory transactional visibility and
in-place canonical materialization. `PersistentFile` stores tagged native nodes, typed Tensor blocks, and metadata.
Commit is not independently crash-durable. Native restoration, strict loading,
and caller-owned recovery are described by the
[transactional collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md).

`TableSchema::try_from(value)` is the canonical Value-to-Table-schema conversion,
replacing `TryCastFrom<Value>`. It consumes the input fields and returns parsing
or constructor failures as `TCError`. Auxiliary-index names must be strings;
rendering conversions of numbers and links do not apply to schema names.
Schema validation and auxiliary declaration rules are specified by the
[Table contract](TRANSACTIONAL_COLLECTION_CONTRACT.md#table-access).

`CollectionSchema::try_from((class_path, value))` and
`TensorSchema::try_from(value)` likewise return structured construction errors,
replacing their `TryCastFrom` implementations. Native BTree schema preparation
borrows columns through `BTreeSchema::try_from(columns.as_slice())`; it does not
consume or clone their names and size metadata.

Every application mutation of a persistent mutable collection must pass through
its owning Chain. Returned views grant read access, not mutation authority,
including full reads and identity transforms. Selection-write conveniences
address the original owner with explicit bounds. Trusted native callers handing
an unpublished collection to Chain relinquish mutable aliases and must not reopen
the same storage independently.

`collection::BTreeView::new(columns, owner)` now returns `TCResult` and requires
matching typed backing storage. Views have private representation and immutable
schema metadata shared by their clones and slices; use `schema()` and `persistent()`
for inspection. Raw untyped BTree operations remain supported directly.

## Tensor

`Tensor<Txn>` wraps native fensor storage with transaction-local logical-block
replacements. It supports dense and sparse storage, all ten real dtypes, and
`complex` enables the two native complex types. Construction, point access,
assignment, copying, decoding, and terminal consumption are asynchronous and use
a delegated transaction or directory. Geometric transforms are lazy; computed
expressions retain their input read leases and are read-only.

Native literal constructors check supplied element bytes against the delegated
`materialized_tensor_bytes` allowance after allocation. This is not an aggregate
memory guarantee. Streamed construction and copying use fensor's storage-block
capacity independently; reopening preserves stored geometry.

Tensor routing selects each operation once. Private unary, binary, reduction,
and transformation modules own concrete terminal handlers; codecs, views, and
transactional storage retain their separate responsibilities.

Native storage uses compact row-major chunks of at most 4,096 elements. Dense
values have one region; each sparse-coordinate prefix has an independent dense
suffix region. The last chunk in each region stores only its valid elements.
Transaction deltas use fensor's `SparseBlocks` with the same canonical geometry,
plus touched membership to distinguish explicit zero replacements from absence.
Block capacity controls physical conflict boundaries, which may cross rows.

A read selects its transaction's replacement first, then committed replacements
in reverse transaction order, then canonical storage. Explicit zero replacements
suppress older contents. Point operations reserve one logical storage block;
selections reserve conservative logical-block intervals derived from native mapping metadata. Multi-owner operations acquire
without waiting while holding another owner's permits. Assignment consumes its
source, stages only affected blocks in a separate workspace, and merges the patch after validation;
this supports both geometric and computed self-assignment. Retaining another
alias to a computed input can still cause a structured conflict.

Commit publishes visibility without persistence. Finalize streams covered deltas
into native canonical storage. An interrupted pending mutation requires caller-owned
rollback and disposal of affected results before reuse. A failed canonical
materialization requires discarding affected owners and caller-coordinated recovery. `sync()` writes back canonical blocks and the native sparse
index root; `sync_all()` additionally performs delegated durable synchronization.
Later snapshots hand off materialized deltas to canonical storage before caller
workspace cleanup; sparse cursors resume from their last logical key.
Strict loading requires existing, valid files and never initializes missing state.

Tensor schemas contain dtype, logical shape, and dense/sparse semantics. Hashes
exclude block geometry, cache settings, sparse-axis hints, and physical IDs.
Copying and restoration stream into unpublished delegated storage. Geometric
views are read-only; selected mutations pass explicit ranges to their owner.
Explicit copies own independent writable storage. Arithmetic and supported
backend/dtype combinations follow fensor and ha-ndarray. Implicit sparse zeros
are ordinary values, and reductions include
them. Pointwise operations that would turn the implicit background into nonzero
values return a structured error; `as_dense` explicitly creates a lazy dense view
before such an operation, without copying storage. Native mean, population
standard deviation, and Euclidean norm return f64 for real inputs; complex mean
returns c64, while complex standard deviation and norm return f64. Floating point
recomputation retains the backend's reproducibility limits; exact replay retains
accepted values at the caller boundary.

Dense wire payloads remain `[[dtype_path, shape], values]`. Sparse payloads use
`[[dtype_path, shape, sparse_axis], [[coordinate, value], ...]]` with ordered, unique,
nonzero entries. Encoding streams the computed values and omits numerical zeros
from sparse payloads. The optional zero-based axis identifies the last sparse
dimension; the final axis selects scalar-sparse storage. Omitting it selects
dense storage, as in v1. No layout string or key is encoded. Schema and coordinate
metadata, native payloads, and index
decoding are bounded. Explicit value buffers respect the delegated materialization
limit; streamed logical size is not subject
to that whole-value limit. Native execution and cache admission retain their own
bounds. Eligible sparse streams, encoding, copying, and reductions use bounded
occupied-block traversal; reductions account for intervening implicit zeros.
Supported forward slices and permutations retain indexed planning through owned
expressions. Larger sparse selections with unsupported mappings or ordering fail
with a structured error instead of scanning the entire logical shape. Bounded point
reads and explicit row-major reads remain available; computed matrix/Fourier outputs
retain their own logical evaluation. No operation implicitly sorts or persists
intermediates.

Use `Tensor::copy_values` or the `copy_values` route to explicitly copy every logical
value into independent storage when indexed copying is unsupported. This includes
implicit sparse zeros and can be expensive. Ordinary copying and materialization
retain indexed traversal; they do not retry through full-value copying.

Caller file compositions delegate Tensor payloads through
`delegate_tensor_files!(ApplicationBlock, Collection)` alongside their ordinary
`CollectionNode` support. Payload tags determine dtype on reopen; casts do not.

## Testing

```bash
cargo test --all-targets --all-features
```

See the [transactional collection contract](TRANSACTIONAL_COLLECTION_CONTRACT.md),
[crate invariants](AGENTS.md), and [roadmap](ROADMAP.md).
