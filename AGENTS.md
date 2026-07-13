# tc-collection Agent Notes

`tc-collection` defines shard-aware collection behaviors. Keep routing logic shard-
local and lean, and push cross-host orchestration to client libraries.

## Routing and semantics

- Collections shard data across blocks; hosts execute shard-local handlers. Do not add
  cross-host routing logic here—client runtimes own cluster-aware dispatch.
- Keep v1 collection semantics (batching, auth headers, error envelopes) visible in
  docs and examples so clients can migrate without surprises.
- Express new handlers in terms of the shared IR envelopes; avoid adapter-specific
  request/response shapes so Python and WASM clients can reuse the same contracts.
- Do not materialize collection keysets in memory in production codepaths. In
  particular, avoid building `Vec`/`BTreeSet` snapshots of full BTree keys for
  route responses or serialization; consume key streams incrementally via
  iterators/callbacks and encode/output as you iterate.
- Do not construct `freqfs::Cache` in production collection code. `tc-collection`
  constructors and runtime methods must accept host-loaded `Dir`/`DirLock`
  resources rather than creating caches from filesystem paths.

## Documentation and validation

- Update the crate `README` and `ROADMAP` when adjusting placement rules, shard
  manifests, or health snapshot expectations. Design changes should land with concrete
  migration notes, not fallbacks.
- Add focused tests or examples alongside new routing behaviors to avoid introducing
  special-case pathways.
- Keep `TRANSACTIONAL_COLLECTION_CONTRACT.md` aligned with implementation changes and
  use its checklist as the required gate for BTree/Table/Tensor migration slices.

## Locking and transaction safety

- Treat lock ordering as part of the public correctness contract for transactional
  collections. Use one canonical order across BTree/Table/Tensor paths and keep
  it documented in the module that owns transaction lifecycle orchestration.
- Do not hold a BTree/Table/Tensor read or write guard while calling operations
  that may re-enter locking in another domain (for example `sync`, finalize
  reconciliation, or cross-structure merge). Drop guards first, then perform the
  next phase.
- Keep lock lifetimes as short as possible: mutate under guard, release guard,
  then run persistence or replay steps. Avoid helpers that combine these phases
  unless they prove lock independence.
- New migration slices must include deadlock regression coverage for lock-order
  interactions, especially unresolved-finalize recovery and finalize+sync flows.
- Transactional collection data structures must fail closed on any conflict,
  ambiguity, or non-deterministic replay condition. Return structured errors;
  do not attempt opportunistic merge/repair logic in this crate.
- Treat cross-shard or global consistency as chain-owned policy. `tc-collection`
  guarantees local transactional integrity and deterministic replay only; any
  unresolved global conflict must be surfaced upward for `tc-chain`-level
  reconciliation.
- `tc-collection` does not own write-ahead log policy or canonical replay-log
  durability. Chain variants (for example `SyncChain`, `BlockChain`) provide
  authoritative transaction history; collection structures deterministically
  apply that history and fail closed on ambiguity.
