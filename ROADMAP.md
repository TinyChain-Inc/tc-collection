# tc-collection roadmap

## Active deliverables

1. **Shard manifest schema.**
   - Specify shard descriptors (hash ranges, replica sets, capability masks, telemetry hooks) and how they anchor to control-plane records in `tc-chain`.
   - Define how manifests encode custody tags and attestation requirements so validators can gate shard admission.
   - Document migration pointers back to v1 collection behaviors for batching, auth headers, and error envelopes.

2. **Placement and routing plan.**
   - Outline how runtimes publish deterministic health snapshots (CPU/memory/IO windows) that clients and validators can consume before routing or scheduling work.
   - Describe the routing table structure (client-side and host-side) and how it reuses `txn_lock` coordination for cross-shard operations, keeping cross-host dispatch logic in the client library rather than the Rust host binary.
   - Include failure-handling rules (stale snapshots, shard eviction, resharding) and how they surface through MetricsChain.

3. **Local execution graph ergonomics.**
   - Capture how handler DAGs execute inside a shard once the ingress request is deserialized, avoiding repeated HTTP-style dispatch.
   - Define telemetry hooks that tag resource usage with manifest IDs and shard identifiers so billing and analytics stay aligned with placement.
   - Add examples that keep v1 batching semantics visible while documenting intentional v2 behavior changes.

4. **Transactional BTree/Table orchestration over `b-tree` and `b-table` (v1-parity first).**
   - Port discipline: treat this track as a straightforward, behavior-preserving v1 translation. Avoid introducing new transactional semantics, fallback pathways, or local reconciliation heuristics before parity gates pass.
    - Treat `b-tree` and `b-table` as managed non-transactional storage/index primitives, analogous to the `fensor` boundary: storage engines own persistence/index mechanics while `tc-collection` owns transaction lifecycle semantics.
   - Treat Chain variants as the authoritative transaction-history source. `tc-collection` applies ordered chain events deterministically and does not own WAL/reconciliation policy.
    - Implement canonical+delta lifecycle in `tc-collection` for both BTree and Table: `pending`, `committed`, finalize-merge, and deterministic replay ordering.
    - Keep transaction visibility, isolation, recovery behavior, and finalize policy exclusively in `tc-collection`; do not move transaction ownership into `b-tree`/`b-table`.
    - Require v1-parity as an implementation goal, not only functional equivalence:
       - preserve v1 batching semantics, auth-header behavior, and error-envelope behavior where contractually required.
       - preserve deterministic range/query planning behavior and index-selection semantics.
       - preserve operational behavior for delete/merge/count/scan paths where supported.
    - Require reliability parity gates:
       - restart recovery for prepared and unresolved-finalize paths.
       - fail-closed behavior on corruption or malformed persisted state with structured errors.
    - Require performance parity gates:
       - benchmark against v1 baselines for representative read/write/range/scan workloads.
       - define explicit regression budgets and block promotion if budgets are exceeded.
    - Cross-link dependencies and sequencing:
       - replay and chain-construction contract from `tc-state/ROADMAP.md`.
       - transaction/finalize invariants from `tc-server/ROADMAP.md`.
       - shared lifecycle/replay/reliability checklist in `TRANSACTIONAL_COLLECTION_CONTRACT.md`.
    - Exit criteria:
       - commit/rollback/finalize visibility tests pass for BTree and Table.
       - restart and unresolved-finalize recovery tests pass.
       - transactional contract checklist in `TRANSACTIONAL_COLLECTION_CONTRACT.md` is green with evidence links.
       - v1 parity port rules in `TRANSACTIONAL_COLLECTION_CONTRACT.md` are satisfied and any intentional divergence is documented.
       - parity matrix is documented and green for required v1 behaviors.
       - performance and reliability regression gates pass against v1 baselines.
    - **Operation/stream ownership:** expose BTree and Table route behavior through the universal
      `State` operation and async stream contracts. Collection implementations own validation,
      projection, range scans, and incremental output; neither `tc-server`'s executor nor its
      HTTP/PyO3 codecs may match on BTree or Table. A projected slice must stream only the
      selected data and never materialize the enclosing collection.
    - **Table POST contract:** retain v1's map-shaped selector as a shared State/IR form. The
      Table route must not invent an arbitrary parameter wrapper or depend on a `tc-server`
      conversion; complete the shared representation first, then adapt the collection handler.

5. **fensor extraction and integration plan.**
   - Treat `fensor` as a managed dependency under `deps/fensor`. It replaces the current in-memory `ha-ndarray` storage shim behind the existing `tc-collection::tensor::Tensor` type; it must not introduce a second public Tensor type or route family.
   - Own transaction lifecycle orchestration in `tc-collection` (pending/committed deltas plus commit/rollback/finalize) while treating `fensor` as a non-transactional storage primitive.
   - Track storage/index maturation in `deps/fensor/ROADMAP.md` and wire completion milestones back into collection delivery gates.
   - Define the storage cutover from in-memory tensors to `freqfs`-backed `fensor` tensors inside `tc-collection`, without changes to `tc-state`, `tc-server`, client URIs, or adapter behavior.
   - Enforce `fensor` fail-closed integrity semantics: corruption in metadata/data must be surfaced as errors, with no auto-recovery in `fensor`; operational recovery is handled above the storage layer.

6. **Transactional tensor orchestration over `fensor`.**
   - Implement canonical+delta tensor lifecycle in `tc-collection`: `pending`, `committed`, and finalize-merge.
   - Store both block mutations and sparse-index mutations as deltas with deterministic merge ordering.
   - Keep transaction visibility, isolation, and recovery policy exclusively in `tc-collection`; `fensor` remains non-transactional.
    - This tensor track is parallel to the BTree/Table v1-parity track and uses the same transaction-orchestration boundary (engine non-transactional, lifecycle in `tc-collection`).
   - Chain variants remain authoritative for replay-log durability and canonical ordering; Tensor structures apply chain events and fail closed on ambiguity.
   - Exit criteria: commit/rollback/finalize visibility tests and restart recovery tests pass.
   - Exit criteria: transactional contract checklist in `TRANSACTIONAL_COLLECTION_CONTRACT.md` is green for Tensor.
   - Cutover sequence:
     1. Complete fensor dense metadata/block/view operations and streaming reads.
     2. Adapt the existing Tensor implementation to a storage trait implemented by fensor, without exposing that trait at the public API.
     3. Add transactional pending/committed delta handling around fensor roots in `tc-collection` and prove restart/finalize behavior.
     4. Switch the sole Tensor implementation to fensor storage and delete the in-memory `ha-ndarray` storage shim in the same change.
     5. Retain `ha-ndarray` only where fensor explicitly delegates execution; do not retain it as an alternate Tensor persistence or route implementation.

7. **Canonical Tensor API and routing ownership (complete).**
    - `tc-collection` owns Tensor representation, validation, wire encoding,
      metadata/view operations, and numerical/reduction operations.
    - `tc-state` delegates through the universal State operation surface; it
      does not duplicate Tensor semantics.
    - `tc-server` has no Tensor-specific resolver, codec, or PyO3 branch.
    - Tensor uses the same State operation and stream contracts as BTree and
      Table.


## Deferred explorations

- **Hot resharding cookbook.** Draft a worked example for moving hash ranges between shards without violating ordering guarantees, including the ledger entries and validator attestations required.
- **Edge/overlay routing.** Evaluate whether peer-provided `tc://` hints from the control plane can accelerate shard discovery in disconnected environments.
