# tc-collection

`tc-collection` is the home for TinyChain's sharded collection services and storage behaviors. It documents how collections slice data across nodes, coordinate transactional updates, and expose deterministic health snapshots without diverging from the v1 semantics that clients rely on.

## Scope

- Define collection manifests (capability masks, shard metadata, telemetry hooks) so runtimes and validators agree on placement and custody requirements.
- Describe storage and replication patterns that pair `txn_lock` ordering with shard-local handlers, keeping graph execution local after ingress deserialization.
- Capture how collection routing interacts with the control plane (`tc-chain`) for placement, attestation, and MetricsChain emission.
- Provide migration notes for v1 collection semantics (batching, authorization headers, error envelopes) as new v2 behaviors land.

## Layout

- `ROADMAP.md` — active design and implementation checklist for sharded collections, placement, and telemetry.
- `TRANSACTIONAL_COLLECTION_CONTRACT.md` — required lifecycle, replay, recovery, lock-order, and parity contract shared by BTree/Table/Tensor.
- `examples/` (planned) — focused walkthroughs of shard manifests, routing tables, and deterministic health snapshots once prototypes stabilize.

## Relationships

- **Runtime host (`tc-server`).** Hosts execute handlers resident in the collection shards and expose shard-aware health snapshots to applications and validators; cross-host routing is orchestrated by client libraries.
- **Control-plane ledger (`tc-chain`).** Records shard manifests, placement decisions, and custody/attestation proofs so clients and validators can trust routing hints.
- **IR traits (`tc-ir`).** Collection handlers should consume and emit the same IR envelopes as other adapters to preserve batching ergonomics and transaction semantics for clients.

## v1 Port Discipline

Transactional collection migration in this crate is a strict v1 parity port, not
an invention track.

- Keep one canonical behavior path per feature; do not add fallback or parallel semantics.
- Preserve v1-visible behavior first (transaction visibility, finalize ordering,
  range/count/scan semantics, error envelopes), then optimize after parity gates are green.
- Keep transaction identity immutable and globally scoped; never remap or alias
  `txn_id` values locally.
- Keep Chain variants (`SyncChain`, `BlockChain`) as the authoritative source of
  transaction history/replay policy. Collection structures deterministically apply
  ordered events and fail closed on ambiguity/conflict.
- Treat local snapshots/materializations as optional performance caches only,
  never as WAL authority.

## Canonical namespaces

Collections always live under the standard TinyChain directories. Use `/state` (with
`/state/chain` for consensus-wrapped collections, `/state/collection` for shard-local
stores, `/state/scalar` plus `/state/scalar/value`, `/state/scalar/tuple`,
`/state/scalar/map` for scalar primitives, and the upcoming `/state/media` path for large
objects) for data, `/service` for stateful APIs that wrap those collections, `/lib` for
stateless helpers, `/class` for shared type definitions, and `/host`/`/healthz` for
telemetry. Do not mint bespoke top-level paths for collection work.

## TaskQueue primitive

Long-running workloads must never hold a transaction owner open beyond three seconds. The
approved pattern is a `While`-based queue service:

- **Structure.** A queue is a publisher-defined service whose **entire runtime is a single
  TinyChain `While` loop**. That loop’s `state` field is the only cross-transaction state:
  each iteration (`cond → body → commit`) finishes inside three seconds, persists the new
  `state`, and exits. The `state` may be as small as a `Value` (for one-job queues) or a
  reference into a collection (e.g., `/state/<namespace>/queues/<name>/tasks`). If you only
  ever have one job, that state can be a single `Value`; the model does not require a
  `Table`, it simply reuses whatever storage shape the publisher chooses. Large binaries
  live under `/state/media/...`; the queue stores references to them. There is no bespoke
  `/data` endpoint.
- **Execution.** Each `While` iteration is a short transaction: evaluate `cond`; if
  `true`, fetch the next payload, run the long job (outside the request), persist the
  updated state, commit, and loop. If the host fails mid-loop, another host resumes
  from the persisted state automatically.
- **Developer usage.** Publishers expose an `enqueue` method on their queue service and
  emit references clients can poll for status/results. Worker services implement the
  complementary `While` loop. No manual `claim`/`ack` is needed; the kernel’s existing
  begin/commit flow handles exclusivity.

Every feature that needs more than three seconds—ML training, analytics, backups—must be
expressed via this queue pattern so temporal locality enforcement stays uniform.
