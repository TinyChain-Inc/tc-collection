# Transactional Collection Contract

This document defines the required standardization contract for all transactional
collection structures in this crate (BTree, Table, Tensor) so they compose with
Chain replay and server transaction/finalize invariants.

## Purpose

- Keep collection structures as deterministic transactional state machines while
   storage engines remain non-transactional primitives.
- Guarantee fail-closed local integrity under an ordered stream of chain events.
- Keep WAL/replay authority in Chain so reconciliation policy is centralized.

## v1 Parity Port Rules

1. This migration is a behavior-preserving v1 port.
2. Do not introduce new transaction semantics until parity gates pass.
3. Prefer deleting transitional/fallback paths over adding compatibility branches.
4. Any intentional divergence from v1 must be explicitly documented in the same
   change with rationale and migration impact.
5. Optimization and refactoring work happens only after parity evidence is green.

## Required interfaces

Every transactional collection structure must implement the same lifecycle surface.

1. Persistence contract
   - Optional local materialization snapshots are allowed for performance only.
   - Snapshots are non-authoritative and must always be rebuildable from Chain history.

2. Transaction lifecycle contract
   - Support pending, committed, rollback, and finalize states.
   - Finalize must be idempotent and deterministic.
   - Collection lifecycles must be applied from chain-selected events, not from
     collection-owned WAL policy.

3. Recovery contract
   - Restart must reconstruct the same visible state by reapplying canonical
     Chain events in order.
   - Collections may not invent alternate replay order or local reconciliation policy.

4. Integrity contract
   - Corrupt or malformed persisted state must fail closed with structured errors.
   - No silent repair or implicit fallback paths.

## Replay and ordering invariants

1. Canonical identity
   - Replay order is dictated by the selected Chain variant's canonical identity
     and deterministic predecessor ordering.
   - Do not remap or alias transaction identifiers during replay.

2. Visibility
   - Reads at transaction T see canonical state plus committed deltas <= T plus pending delta at T.
   - If an earlier transaction has an overlapping pending write, the read at T must
     wait until that earlier transaction is finalized (commit/rollback/finalize frontier)
     before returning a value.
   - No visibility leakage from pending deltas of other transactions.

3. Finalize
   - Finalize advances a monotonic frontier.
   - Re-applying finalize for the same frontier is a no-op.
   - No-op/error lifecycle exits (duplicate commit, stale finalize, outdated/invalid
     rollback/commit paths) must still release any transaction semaphore reservation
     acquired for that lifecycle operation.

## Locking and concurrency invariants

1. Use one canonical lock acquisition order across BTree, Table, and Tensor.
2. Do not hold a guard while invoking sync/finalize/merge paths that may acquire additional locks.
3. Keep lock scopes minimal: mutate under guard, drop guard, then publish/return.
4. Lifecycle operations that acquire semaphore reservations (`commit`, `rollback`,
   `finalize`) must always call semaphore finalize/release on every exit path,
   including stale/no-op/error branches, to prevent reservation leaks and
   read starvation/deadlock.

## Chain compatibility requirements

1. Transaction records must be replayable in canonical order without side tables.
   Transaction records and WAL durability are Chain-owned responsibilities.
2. Restart/replay behavior must match tc-state chain construction assumptions.
3. Finalize behavior must match tc-server root-only finalize and transaction ownership invariants.

## Parity checklist (must be green for promotion)

1. Transaction semantics
   - Commit/rollback/finalize visibility tests pass.
   - Deterministic range/count/scan behavior matches parity matrix expectations.

2. Recovery and reliability
   - Prepared and unresolved-finalize restart tests pass under Chain-driven replay.
   - Corruption/malformed state tests fail closed with structured errors.
   - Duplicate replay/idempotency tests pass.

3. Concurrency
   - Lock-order deadlock regressions pass.
   - Concurrent read/write/finalize contention tests pass.

4. Performance
   - Read/write/range/scan benchmarks meet regression budgets vs v1 baselines.

## BTree v1 parity matrix (current verified coverage)

This matrix captures behavior-preserving parity targets for the transactional
BTree port. Each row must map to executable tests.

1. `pending` visibility is txn-local until the pending transaction is finalized;
   later overlapping reads block rather than observing uncommitted state.
   - Evidence: `pending_is_visible_only_to_its_txn` in `src/btree/tests.rs`.
2. Committed deltas are visible in txn order (`<= txn_id`).
   - Evidence: `committed_is_visible_in_txn_order` in `src/btree/tests.rs`.
3. Writes after finalize are rejected.
   - Evidence: `cannot_write_after_commit_or_finalize` in `src/btree/tests.rs`.
4. Deterministic mutation flow (insert/delete/commit/finalize) is preserved.
   - Evidence: `direct_mutation_flow_from_chain_state` in `src/btree/tests.rs`.
5. Streamed row iteration matches expected materialized visibility.
   - Evidence: `streamed_keys_match_materialized_keys` in `src/btree/tests.rs`.
6. Slice/range view behavior is deterministic.
   - Evidence: `slice_keys_match_range_view` in `src/btree/tests.rs`.
7. Earlier overlapping writes fail closed with conflict.
   - Evidence: `overlapping_write_in_past_txn_fails_closed` in `src/btree/tests.rs`.

## BTree parity gap status

Previously identified parity gaps and their current status. Closed gaps map to
executable tests in `src/btree/tests.rs`.

1. Rollback visibility/regression coverage should be explicit. **Closed.**
   - Evidence: `rollback_unblocks_later_read_and_discards_pending` and
     `repeated_rollback_and_finalize_are_idempotent` in `src/btree/tests.rs`.
2. Finalize idempotency/stale finalize no-op coverage should be explicit. **Closed.**
   - Evidence: `stale_finalize_is_noop` and
     `repeated_rollback_and_finalize_are_idempotent` in `src/btree/tests.rs`.
3. Duplicate commit idempotency coverage should be explicit. **Closed.**
   - Evidence: `duplicate_commit_is_idempotent` in `src/btree/tests.rs`.
4. Chain-ordered event replay coverage should be explicit. **Open.**
   - Current coverage is limited to `direct_mutation_flow_from_chain_state`,
     which drives the deterministic mutation flow directly rather than replaying
     canonical chain events. Explicit replay-order coverage depends on
     `tc-chain` integration and remains to be implemented.
5. Lifecycle no-op/error branches should explicitly assert semaphore-reservation
   release (no blocked later reads after stale/no-op operations). **Closed.**
   - Evidence: `lifecycle_noop_paths_do_not_leak_reservations_soak` in
     `src/btree/tests.rs` (200-iteration soak over duplicate commit, stale
     finalize, and finalize paths asserting that later reads do not starve).

## Remaining BTree parity gaps beyond the original list

The following parity checklist items have no test coverage in this crate yet:

1. Restart recovery for prepared and unresolved-finalize paths under
   Chain-driven replay (depends on `tc-chain`).
2. Corruption/malformed persisted state tests failing closed with structured
   errors.
3. Read/write/range/scan benchmarks with regression budgets vs v1 baselines.
4. Table and Tensor lifecycle parity (see `ROADMAP.md` items 4-6); `Collection`
   currently has only the `BTree` variant.

## Table v1 parity port

The transactional `Table` variant has a dedicated parity/port spec that is the
required gate for every Table implementation slice:

- **[`TABLE_PARITY_PORT.md`](TABLE_PARITY_PORT.md)** — v1 inventory (pinned to
  tinychain commit `17ef342e8f7026e4c4a60d2044de9aeb1b145b91`), the
  `b-table`/`tc-collection` authority boundary, no-materialization rules, the
  canonical-plus-delta merge model, the file/module migration map, the
  route/API matrix, the test matrix, fixture datasets, and explicit blockers.

Table implementation work must land the [§7 test
matrix](TABLE_PARITY_PORT.md#7-test-matrix) cases and satisfy the §9 acceptance
criteria before the Table variant is promotable. No production `Table`
implementation exists yet (`Collection` currently has only the `BTree` variant).

## Migration policy

1. BTree, Table, and Tensor must adopt this contract before declaring migration complete.
2. New route or storage behavior that affects replay/visibility/finalize requires checklist updates.
3. Keep this document synchronized with `ROADMAP.md` exit criteria.
