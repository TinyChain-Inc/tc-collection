# Contributing to tc-collection

Read this repository's [invariants](AGENTS.md) and
[transactional contract](TRANSACTIONAL_COLLECTION_CONTRACT.md).

## Collection types

- Reuse fensor's `Shape`, `Strides`, `Axes`, and `Range` for owned Tensor
  geometry. These aliases use `SmallVec`; inline capacity is an allocation
  optimization, not a rank limit. In production paths, construct geometry
  directly with `from_slice` or iterator collection instead of allocating a
  temporary `Vec` first.
- Use `Vec` for values, bounded batches, native keys and rows, public coordinate
  payloads, and decoder buffers. Preserve existing boundary APIs and validate
  allocation limits independently of the container type.
- Prefer slices for borrowed data and arrays for fixed-cardinality structures
  where the receiving API permits them. Keep large descriptors and variable-depth
  traversal frames heap-backed: a short collection alone does not justify
  `SmallVec`, whose inline storage enlarges its containing object or future.

## Dispatch and runtime data

Use concrete types and trait methods for operations known at construction. Enums
may represent runtime alternatives (such as dtype or stored/computed values) and
explicit traversal states. Do not replace concrete behavior or borrowing terminal
handlers with operation tags. Keep representation matching near its owner and
delegate behavior to the selected concrete value.

## Bounded Tensor execution

Every working collection needs an identifiable bound; name that bound locally
for non-obvious allocations. Never size implicit working buffers from total
logical Tensor cardinality. Streamed filesystem materialization and caller-supplied
literal buffers are permitted; collecting a Tensor stream into RAM is not implicit
materialization policy.

Use `fensor::MAX_BATCH_ELEMENTS` for execution/update batches, independently of
storage block capacity, metadata rank, and index page limits. Equal numeric values
do not couple their owners. Bounded batches do not provide an aggregate memory
limit: concurrent consumers and expression frames may retain multiple buffers.

## Futures and streams

- Collections own transaction visibility and leases; fensor owns numerical
  evaluation, batching, and delivery order. Do not add collection-side evaluators
  or nested buffering.
- Prefer `async fn`, stream combinators, and `try_unfold`. Handwritten polling
  belongs only to scheduling, cooperative progress, or resource-release owners.
  Loops that skip ready items must not rely on an `await` necessarily yielding.
- Use `pin!` for local consumption. Retain `BoxFuture` and `BoxStream` at owned,
  erased, or documented stack-size boundaries; local pinning cannot replace an
  escaping boxed future. Boxing recursive futures alone does not make them
  stack-safe.
- Keep source handles and permits in the consuming stream. Use `GuardedStream`
  outside lazy transforms, so EOF (including a limit), the first error, and drop
  release the source before its permits. Keep cancellation pull-driven; do not
  spawn ingestion tasks to hide ownership.

## Validation

The [standalone CI guide](ci/README.md) describes pinned dependency setup and the
repository's formatting, feature, library-test, and doctest checks. These checks
run on pull requests and pushes to `main`.

The persistence benchmark's `evaluation` suite measures primary reads with varying
auxiliary-index counts, streamed restoration with canonical and retained baselines,
large native collation-tie groups, union fan-out, and sparse Tensor history depth.
It reports comparisons independently of elapsed time and marks timed-out scans.
Restoration rebuilds indexes row by row; measure this explicit tradeoff separately
from capture/materialization. Process RSS includes allocator and cache retention
and is not an isolated live-payload measurement. Keep matched history fixtures and
frozen executables identical across paired comparisons; evaluation-only cases
have no historical baseline. Record build inputs, feature selection, worker count,
and executable hashes with retained results. Timings never gate correctness.

Production code returns structured errors for invalid input, storage failures,
resource exhaustion, transaction conflicts, and poisoned collection state.
Assertions and `expect` require a locally established internal invariant, not
an assumption that an external operation succeeds. Destructors must avoid
introducing panics during cleanup. Test code may assert and unwrap expected results.

Run affected library targets before review. For Tensor or shared lifecycle changes,
include both default and complex configurations. Complete correctness validation is:

```bash
cargo fmt --check
cargo clippy --lib --tests -- -D warnings
cargo clippy --lib --tests --features complex -- -D warnings
cargo test --lib
cargo test --lib --features complex
cargo test --doc
```

During iteration, select the target as well as the test name. A name filter alone
still compiles unrelated integration targets. For example:

```bash
CARGO_BUILD_JOBS=1 cargo test --lib btree::tests
CARGO_BUILD_JOBS=1 cargo test --lib table::tests
CARGO_BUILD_JOBS=1 cargo test --lib stream::overlay::tests
CARGO_BUILD_JOBS=1 cargo test --lib typed_literal_declarations_preserve_dtype_and_validation --features complex
CARGO_BUILD_JOBS=1 cargo check --features benchmarks --test tensor_benchmark
```

Reuse the shared target and keep compiler settings consistent between sessions.
For bounded validation builds, use these settings throughout the run:

```bash
export CARGO_BUILD_JOBS=1
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_RELEASE_DEBUG=0
```

Start edits with `cargo check --lib`; when running strict Clippy for that same
target, omit the redundant check. Select affected test targets and features once.
Test-name filters reduce execution, but the containing test binary still compiles.
Use the complete owner suite for shared changes or release acceptance. Benchmark
compilation is separate. Moving test files or parameterizing cases does not by
itself establish a build-time improvement.

For repeated local edits, `CARGO_INCREMENTAL=1 cargo check --lib` opts into cached
compiler state at additional disk cost. Preserve that choice across the session;
do not switch configurations repeatedly in the same target.

The opt-in `dev-opt` profile uses basic optimization with development assertions
and overflow checks. It disables LTO and debug symbols, while leaving the
production release profile unchanged:

```bash
cargo test --profile dev-opt --lib table::public::tests::get_limit -- --exact
```

Use this only when optimized iteration is useful. It does not replace actual
release-mode stack, numerical, or pre-release gates. Routine Table-only changes
use the default profile and do not repeat complex or release configurations.


Collection changes require focused streaming, transaction visibility, conflict,
finalization, cancellation, and lock-order tests as applicable. Public hosting,
durable history, and reconciliation are outside this repository.

Coordinate concurrency tests with channels, barriers, or explicitly polled
waiters; do not use sleeps to establish acquisition order. Timeouts bound failed
progress. Cover broad and narrow reservations in both orders through collection
operations: Table scans reserve the whole table, point operations reserve keys,
and Tensor selections reserve logical-block intervals. Keep overlap semantics in
`txn_lock` tests and verify range mapping and permit ownership here.

Timing workloads live in the opt-in `tensor_benchmark` integration target, which
links the production library without unit-test instrumentation. The ordinary
suite retains private structural, corruption, and transaction assertions. Tensor
unit tests and benchmarks share their caller-owned storage fixture and automatic
temporary-directory cleanup.

```bash
cargo test --release --features benchmarks --test tensor_benchmark -- --ignored --nocapture
COLLECTION_BENCH_SUITE=overlay cargo test --release --features benchmarks --test tensor_benchmark -- --ignored --nocapture
COLLECTION_BENCH_SUITE=restoration cargo test --release --features benchmarks --test tensor_benchmark -- --ignored --nocapture
```

`COLLECTION_BENCH_CACHE_BYTES` selects the benchmark fixture's cache budget;
the default is `DEFAULT_CACHE_BYTES` in the shared Tensor fixture (256 KiB).
This small resident target permits eviction in larger workloads while keeping
fixture memory modest; it is a reproducible pressure baseline, not a tuned optimum.
Use the same value for both paired builds and record it with the results.
A 64 MiB run provides a larger-cache comparison; verify its measured cache
behavior before treating it as a fully resident control. The separate Tensor
materialization ceiling is 64 KiB and does not limit streamed logical size.
General storage fixtures use a 16 MiB resident target and a separate 256 MiB
materialization ceiling so ordinary tests do not incidentally test admission.
Boundary tests override these policies. None of these defaults claims an exact
working-set threshold; workload, payload geometry, metadata, and platform affect
the retained footprint.

Available suites are `finalization`, `mutation`, `assignment`, `construction`,
`restoration`, `overlay`, `reads`, `integrity`, `zero_replacements`, and
`block_reads`, and `capacity_policy`, plus `layout`; the default is `all`.
`integrity` consumes resident BTree/Table scans and Tensor streams at 1 and 16
visible deltas, including an expression sharing the same Tensor source twice.
It uses 20 warm repetitions after an identical warmup. The benchmark runtime
retains one Tokio worker; comparison-runner worker settings vary numerical
workers, not this explicit Tokio runtime setting.
`mutation` measures whole-owner fills and selected fills through the public
owner-bound write path. Returned geometric views remain read-only.
The `zero_replacements` suite measures zero-patch merging, mixed-value scans, and
durable finalization for both layouts. Sparse scan records include
value consumption, rather than a private index-only probe. The overlay suite
measures 256-element logical evaluation and 8,193-element indexed traversal at
each transaction depth, with 20 and 3 repetitions respectively.
`COLLECTION_BENCH_SMOKE=1` uses 256 and 4,097 elements with one repetition for that
suite; leave it unset for comparisons. Shapes, storage capacities, and
repetitions are fixed in the fixture.

`block_reads` measures sequential reads, scattered selections, singleton reads,
and occupied-value traversal for shape `[4, 4096]` with 0, 1, and 16 visible deltas.
Sparse axis 1 provides a scalar-block control at capacity 1. Sparse axis 0 retains trailing
payloads at capacities 64, 512, and 4096, with either one nonzero per block or
fully populated blocks. Each case warms its reads before four timed repetitions;
`COLLECTION_BENCH_SMOKE=1` reduces this to one repetition. The suite records
geometry and resident runtimes; this adapter does not measure payload copies,
native traffic, or retained node bytes.

`capacity_policy` keeps logical values and update coordinates identical across
capacities 1, 64, 512, and 4096 for `u8` and `f64`. The shape remains `[4, 4096]`:
isolated nonzeros occur every 1024 columns, clustered values occupy the first
512 columns of each row, and the full case populates every element. Each case
runs with 0 or 16 visible deltas. Scalar capacity 1 uses sparse axis 1; the larger
capacities use axis 0. Sequential, scattered, and four singleton reads warm once
before four timed repetitions. Four fixed-coordinate writes with separate commits
and durable finalization follow each read case. Construction and
post-finalization value verification stay outside timing. Geometry records include
effective block capacity and canonical payload bytes, excluding index/container overhead;
these bytes are not an aggregate retained-memory measurement.

For these block controls, use the same adequately provisioned cache for both
builds:

```bash
COLLECTION_BENCH_CACHE_BYTES=67108864 COLLECTION_BENCH_SUITE=block_reads RAYON_NUM_THREADS=4 CARGO_BUILD_JOBS=1 cargo test --release --features benchmarks --test tensor_benchmark -- --ignored --nocapture
```

The entrypoint explicitly uses one Tokio worker, including when the comparison
runner supplies `TOKIO_WORKER_THREADS`. Record that actual setting separately
from `RAYON_NUM_THREADS`; the one- and four-worker comparisons vary numerical
workers only.

For comparisons, compile the same harness against both source snapshots and
retain both executables. The shared
[`fensor` comparison runner](https://github.com/TinyChain-Inc/fensor/blob/main/benchmarks/compare.py)
is a timing tool. Invoke it with
`--record-prefix COLLECTION --suite-env COLLECTION_BENCH_SUITE --suite all`.
It alternates isolated processes and validates matching records. Keep its output
under the ignored benchmark results directory; do not use timing assertions in
the correctness suite.

## Test ownership

The opt-in `persistence_benchmark` target measures visible-history reads and
scans, native materialization and semantic hashing, cold loads followed by a
mutation, and stream release during finalization:

```bash
PERSISTENCE_BENCH_SUITE=all cargo test --release --features benchmarks --test persistence_benchmark -- --ignored --nocapture
```

Suites are `history`, `capture`, `load`, and `slow_reader`. History scans have a
30-second observation limit and emit a separate `timed_out` outcome; a censored
duration is not a completed scan. Compare frozen executables with identical
fixtures and worker counts, without overlapping compilation. Capture measurements
assert semantic hash equality; timings never determine correctness.

History also reports `retained_writes`, the time spent mutating and committing
the retained versions. Compare this with scan costs: Table's first write captures
the original row, so reducing read amplification can increase write setup cost.

Keep production implementations in `src/`, integration targets in `tests/`,
private unit tests in `tests/unit/`, and shared fixtures in `tests/common/`.
Private tests retain their owning Rust module through `#[cfg(test)]` path
inclusion; moving a file must not widen production visibility or rename tests.

Fixtures must not add fields or public APIs to production types. Put corruption
helpers in private test-support modules and observe execution through narrowly
scoped, test-only calls backed by task-local state. Observations must retain any
watched ownership handle and explicitly scope spawned work. Keep definitions,
fixture adapters, and profiling code outside `src/`; normal builds contain none
of that support code.

Ordinary collection tests share one concrete transaction fixture in
`tests/common/storage.rs`, with explicit cache and materialization budgets.
Create one fixture per workspace and use `with_id` or `at` to share its
subcontext counter across transaction versions. Retain temporary-root owners
until their storage handles and streams are dropped. BTree and Table unit tests
share synchronous root creation and fresh-cache reopening helpers in the existing
unit-test support module; reopening never reuses the previous cache.
Tests run on normal Tokio stacks; native overlay-depth regressions run in isolated
processes so stack failures retain diagnostic logs without terminating the suite.

The `layout` suite compares transformed unary and nested owned reads with canonical,
one-delta, and 16-delta sources. It fixes shape `[4, 4097]`, sparse axis 0, capacities
64/512/4096, isolated or fully populated values, and four warm repetitions. Run it
with the same 64 MiB cache in each paired process. Native `layout` timing and
profiling controls separately observe physical traversal and address-planning work.
