# tc-collection invariants

The repository-local
[`TRANSACTIONAL_COLLECTION_CONTRACT.md`](TRANSACTIONAL_COLLECTION_CONTRACT.md)
is the canonical behavioral contract. The parent
[workspace invariants](https://github.com/TinyChain-Inc/tcv2/blob/main/AGENTS.md)
are non-normative integration context when this repository is used as a submodule.

- Own BTree, Table, Tensor, and Collection values, local operations, routes,
  views, codecs, and deterministic transaction behavior.
- Implement `Route` directly on each concrete collection and delegate the
  `Collection` enum to it. Do not add route enums, adapter handlers, associated
  handler types, or aggregate method switches.
- Select each operation once in `Route`. A terminal handler must not retain an
  unresolved path, repeat path dispatch, or forward to inherent methods which
  merely duplicate its `Handler` verbs.
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
  do not add storage-mode enums, erased directory handles, or runtime downcasts.
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
