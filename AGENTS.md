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

## Code style

- Do not wrap `?` expressions in `Ok(...)`. The `?` operator already returns
  `Err` on failure and passes through `Ok` on success, so `Ok(expr?)` is
  redundant — write `expr?` (or `expr` if it's the last expression in the
  function body). This is enforced by `#![deny(clippy::needless_question_mark)]`
  in `src/lib.rs`.

## Generic type naming

Use `State` (not e.g. `Resp` or `T`) as the generic type parameter name
when a handler or trait expects a host-level state type — even though
`tc-collection` does not depend on `tc-state` directly.  This matches
the v1 convention (`State: From<Collection> + From<Value> + From<u64>`)
and keeps the codebase consistent across crates.  The bounds on `State`
express what the handler needs (`From<Collection>`, `From<Value>`,
`From<u64>`, etc.) without naming a concrete type.

## Type conversions

Prefer standard trait impls (`From`, `TryFrom`, `CastFrom`, `TryCastFrom`)
over hand-written `parse_*` / `cast_*` / `to_*` helper functions.  This keeps
conversions idiomatic, composable, and discoverable.

- **Infallible conversions** (e.g. encoding a `TableSchema` as a `Value`):
  implement `CastFrom<TableSchema> for Value` (which provides `CastInto` for
  free) or `From<TableSchema> for Value`.  Do not write a `to_value()` method.
- **Fallible conversions** (e.g. decoding a `Value` into a `TableSchema`):
  implement `TryCastFrom<Value> for TableSchema` (which provides
  `TryCastInto` for free).  Callers then write
  `value.try_cast_into(|v| bad_request!(...))?` instead of calling a
  custom `try_from_value` function.
- **String ↔ enum** (e.g. `ValueType`): use `FromStr` and `Display` rather
  than match-based `parse_*` and `*_to_string` helpers.
- **Composite types** (e.g. `Column`, `(Vec<Id>, bool)`, `HashMap<Id, Value>`):
  implement `TryCastFrom<Value>` so handlers can use `.try_cast_into(...)` at
  every call site, matching the v1 pattern.
- When a conversion trait impl belongs in another crate (orphan rules), add
  it there rather than working around it with a local helper.  Add a comment
  noting the cross-crate dependency.
- Use `safecast::TryCastInto` at call sites (not `TryCastFrom` directly) for
  ergonomics: `request.try_cast_into(|v| bad_request!("..."))?`.

## Trait implementations

- When a type has inherent methods that return `Result` (e.g. `commit`/`rollback`/
  `finalize`) and also implements the `Transact` trait (which returns `()`), the
  `Transact` impl is the **single** place that converts `Result` to `()` via
  `.expect()`. Callers that need the infallible API must go through the trait
  method, not call the inherent method directly with a redundant `.expect()`.
  This keeps error-handling policy in one place and avoids duplicate panic
  messages drifting out of sync.

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
- `finalize` must not acquire a semaphore write permit. Finalize is a lifecycle
  operation that merges already-committed data into canon; it is not a new write.
  Synchronization is provided by the state write lock and
  `semaphore.finalize(drop_past=true)` for cleanup. Acquiring `try_write` would
  incorrectly conflict with future read reservations.
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
