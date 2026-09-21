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

BTree and Table commit publish pending deltas as committed in memory. Commit and
rollback release the state lock before releasing transaction reservations, and
failed decisions retain reservations. Neither operation persists canonical state.
Finalization merges committed deltas into canonical storage; the durable owner
must synchronize that storage before retiring its recovery evidence.

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

Creation requires empty delegated storage. Loading requires existing roots and
all schema-required indexes; it never initializes missing state. The caller owns
publication and explicitly synchronizes initial canonical state before relying
on restart loading.

Native copying streams a transaction-consistent value into unpublished delegated
storage. Native schemas convert to and from a collection class path and a Value
for strict reopening. Collection identity hashes the class and semantic schema,
hashes ordered native contents, then combines those hashes with SHA-256. It
includes column and index definitions but excludes cache configuration and physical
layout; no serialization is used for hashing. The caller owns publication, durability,
and retention. Collection routes expose no snapshot restoration endpoint.

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
