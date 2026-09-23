# Transactional Collection Contract

This document defines the stable lifecycle shared by transactional BTree,
Table, and Tensor implementations. Concrete algorithms and test names belong in
code; future integration work belongs in the roadmap.

## Ownership

A collection owns local transactional state, concurrent mutation isolation,
conflict ordering, and deterministic visibility. Its transactional data structures
and locking primitives enforce these guarantees without relying on a caller to
serialize mutation handlers. Its storage backend is a non-policy primitive.
The caller owns public naming,
routing, durable ordered history, replay, canonical-state selection,
reconciliation, and resynchronization.

Literal and operation-local collections are scoped to their enclosing
transaction workspace. They do not create an independent resource identity or
transaction coordinator. Standalone named collection hosting is outside this
crate's supported API.

## Lifecycle

Every transactional collection implements the shared fallible `Transact`
lifecycle:

- pending mutation is visible only to its transaction;
- commit makes that version eligible for ordered visibility;
- rollback discards that transaction's pending version;
- `finalize(cutoff)` merges committed versions through the cutoff, discards
  other versions through it, and releases their resources.

Commit, rollback, and finalize are deterministic and idempotent. A stale or
duplicate lifecycle call is a no-op only when the collection can prove the
result. Ambiguous state and malformed storage fail closed.

Callers serialize lifecycle decisions for a collection. Before commit or rollback,
they prevent new work in that transaction, finish or cancel its operations, and
release its streams and guards. Before finalization, they do so for transactions
through the cutoff. Collections do not check this precondition by acquiring new
semaphore reservations; unrelated later transactions may continue.

BTree and Table commit move the pending delta into committed visibility under the
shared state lock, then release reservations. Commit performs no persistence and
is not independently crash-durable. Empty decisions need no files. Failed decisions
retain reservations. Rollback discards pending work.

Finalization applies covered deltas in place to native canonical storage, prunes
covered transaction state, then releases reservations. The caller supplies durable
recovery evidence and coordinates canonical synchronization. Interrupted or failed
materialization must not be retried against uncertain live storage; the caller owns the
recovery-required boundary.

## Visibility and ordering

For transaction `T`, a read observes canonical state, committed versions no
later than `T`, and its own pending version. It never observes another
transaction's pending data.

An earlier overlapping pending mutation must be resolved before a later
conflicting access proceeds. The implementation may wait through its canonical
cancel-safe ordering primitive or return a structured conflict; it may not
guess, silently reorder, or expose partial state. Transaction identities are
never remapped during replay.

Iteration, ranges, projections, limits, updates, and truncation preserve this
same snapshot. Large result and mutation sets remain pull-driven rather than
being collected into unbounded intermediate vectors.

## Locking and backpressure

- Use one lock acquisition order across collection variants.
- Never hold a guard while calling a lifecycle, sync, merge, callback, or I/O
  path which may acquire another lock domain.
- Mutate under the narrowest guard, release it, then publish or await unrelated
  work.
- Successful commit and rollback release their transaction's reservations,
  including duplicate decisions. Failed decisions preserve them.
- Streams retain their read guards and capacity permits until completion or
  drop. Waiting remains cancellable and bounded by the request deadline.

Every buffer, stream, cache, and task spawned by a collection must be bounded
and release its capacity on completion, cancellation, or drop. Collection-local
limits compose beneath any broader admission policy supplied by the caller.

## Persistence and recovery

Creation requires empty delegated storage. Loading strictly requires the existing
root and all schema-required indexes, and validates native tree structure and index
consistency. Loading reconstructs only materialized canonical contents. It does not
recover committed visibility, acceptance receipts, or a cutoff. The caller reconstructs
those from its retained WAL with original-ID capabilities and fresh workspaces.

Committed deltas retain their delegated transaction workspaces until finalization.
Host workspace removal follows successful recursive finalization. Explicit
`sync_all()` makes materialized canonical storage durable; `sync()` is buffered
writeback. Neither persists pending or committed workspace deltas as accepted work.

BTree and Table share one private lifecycle owner containing their transaction
state and existing semaphore. It implements visibility selection and decisions once,
delegating delta application to concrete key/row operations. Concrete collections
retain creation, loading, workspace construction, hashing, copying and restoration.
There is no independent publication owner or acceptance metadata.

Native copying streams a transaction-consistent value into unpublished delegated
storage. Native schemas convert to and from a collection class path and a Value
for strict reopening. Collection identity hashes the class and semantic schema,
hashes ordered native contents, then combines those hashes with SHA-256. It
includes column and index definitions but excludes cache configuration and physical
layout; no serialization is used for hashing. The caller coordinates publication
and retention. Native `restore_from` validates kind and semantic schema, builds
insert/delete deltas in a separate delegated workspace, and installs the pending
replacement only after construction succeeds. Rollback preserves the previous
committed value. Collection routes expose no snapshot restoration endpoint.

Native trees mutate individual blocks and delete obsolete nodes in place. They
provide no immutable root generations or unreachable-version reclamation. Native
copying and loading retain corruption checks. Development fixtures from earlier
layouts must be recreated; no migration reader or fallback is provided.

Local materializations may accelerate access but are not an independent source
of canonical history. Collection code must deterministically apply the ordered
mutations supplied by its caller. It must not add a WAL, replay registry,
majority policy, repair heuristic, or alternate ordering.

Corrupt or inconsistent local state returns a structured error. Repair and
canonical selection are outside this crate; collections do not silently
truncate, skip, or reinterpret records.

## Required evidence

A production collection implementation must cover:

- transaction-local pending visibility and absence of cross-transaction leaks;
- ordered committed visibility;
- commit, rollback, duplicate lifecycle calls, and stale/cutoff finalization;
- overlapping reads and writes, cancellation, and lock-order deadlock
  regressions;
- deterministic range/scan/projection behavior and streaming of large sets;
- resource release on success, no-op, cancellation, and error;
- fail-closed malformed state; and
- deterministic application of caller-supplied ordered mutations.

Variant-specific semantics belong beside the owning implementation and tests.
This contract changes only when the shared ownership, visibility, or lifecycle
model changes.
