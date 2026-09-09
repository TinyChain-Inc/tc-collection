# tc-collection invariants

The repository-local
[`TRANSACTIONAL_COLLECTION_CONTRACT.md`](TRANSACTIONAL_COLLECTION_CONTRACT.md)
is the canonical behavioral contract. The parent
[workspace invariants](https://github.com/TinyChain-Inc/tcv2/blob/main/AGENTS.md)
are non-normative integration context when this repository is used as a submodule.

- Own BTree, Table, Tensor, and Collection values, local operations, routes,
  views, codecs, and deterministic transaction behavior.
- Implement `Route` directly on each concrete collection and delegate the
  `Collection` enum to it. Select each operation once, retain no unresolved
  path, and do not repeat traversal in a terminal handler. The parent workspace's
  [native-routing contract](https://github.com/TinyChain-Inc/tcv2/blob/main/tc-ir/IR_INTERFACE_GUIDELINES.md#native-routing)
  is non-normative integration context for these shared interfaces.
- Represent each selected operation with a small terminal handler borrowing its
  collection receiver. Do not clone the receiver into every handler or replace
  the terminal types with an operation tag, aggregate method switch, adapter
  handler, cross-collection route enum, or forwarding façade.
- Exchange caller-owned native State through `CollectionState`. Delegate its
  conversions to canonical `From`/`TryCastFrom` implementations instead of
  duplicating universal State matching.
- Keep routes, views, and codecs separate: routes know behavior, views acquire
  consistent streams and guards, and codecs know wire structure.
- Stream large scans and keysets. Do not collect unknown-length keys or rows into
  an intermediate vector, map, or set.
- Receive bootstrap- or transaction-delegated storage/allocation handles. Never
  construct a cache, derive a transaction workspace path, or choose caller storage
  policy. Allocate transaction-local deltas from the delegated `StorageContext`;
  store `BTreeLock` and `TableLock` with `StorageContext::File` directly; do not
  add storage-mode enums, erased store traits, forwarding wrappers, erased
  directory handles, or runtime downcasts.
- Standalone named persistent collections are unsupported. Literal and
  transaction-local collections are owned by their enclosing request. Public
  hosting and durable ownership are outside this crate.
- Callers own durable history, replay, canonical-state selection, and
  reconciliation. Collections own local state transitions and fail closed on
  ambiguity; they do not maintain a WAL, select canonical history, or repair
  replicas.
- `Tensor` is one public collection type. A new backend replaces its private
  storage implementation and must not add another Tensor variant, codec, route,
  or adapter path.
- Every `Transact` method is fallible and idempotent. Preserve one documented
  lock order, release guards before cross-domain operations, and test conflicts,
  wait-until-finalize, restart, and finalize/sync deadlocks.
- Use the shared guard-owning stream for values that retain permits or read
  guards. Cancellation and drop release all owned capacity.
