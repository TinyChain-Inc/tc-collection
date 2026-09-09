# Transactional Collection Contract

This document defines the stable lifecycle shared by transactional BTree,
Table, and Tensor implementations. Concrete algorithms and test names belong in
code; future integration work belongs in the roadmap.

## Ownership

A collection owns local transactional state and deterministic visibility. Its
storage backend is a non-policy primitive. The caller owns public naming,
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
- Every lifecycle exit releases any semaphore reservation it acquired,
  including errors, duplicates, and stale no-ops.
- Streams retain their read guards and capacity permits until completion or
  drop. Waiting remains cancellable and bounded by the request deadline.

Every buffer, stream, cache, and task spawned by a collection must be bounded
and release its capacity on completion, cancellation, or drop. Collection-local
limits compose beneath any broader admission policy supplied by the caller.

## Persistence and recovery

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
