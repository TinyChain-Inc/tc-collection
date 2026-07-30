# Table v1 Parity Port & Migration Spec

This document is the behavior-preserving port contract for the transactional
`Table` variant, derived from an inventory of the v1 implementation. It is the
required gate referenced by `TRANSACTIONAL_COLLECTION_CONTRACT.md` and
`ROADMAP.md` item 4. **This issue (#4) delivers the spec only — no production
`Table` implementation is included.** Implementation slices land in later
issues, each gated by the test matrix in [§7](#7-test-matrix).

> **Pinned v1 revision.** The inventory below is taken from the TinyChain v1
> repository at commit
> [`17ef342e8f7026e4c4a60d2044de9aeb1b145b91`](https://github.com/haydnv/tinychain/tree/17ef342e8f7026e4c4a60d2044de9aeb1b145b91/host/collection/src/table)
> (HEAD of `main` as of this inventory). All file references and line-level
> behavior are pinned to that revision. A later implementation issue must
> re-pin if the upstream v1 tree advances.

## 1. Source of truth

The v1 transactional `Table` lives under two locations in the v1 repo:

| v1 path | Role |
| --- | --- |
| `host/collection/src/table/` | The transactional table engine: schema, file/storage, views, stream, public API. |
| `host/collection/src/public/table.rs` | Collection-level routing that dispatches to `Table` and exposes the `schema` route on `Collection`. |

The `table/` module decomposes into six files:

| v1 file | v1 responsibility |
| --- | --- |
| `table/mod.rs` | `Table` enum (`Limited` / `Selection` / `Slice` / `Table`), `TableType` class, and the read/write/stream/slice/order trait surface (`TableInstance`, `TableOrder`, `TableRead`, `TableSlice`, `TableStream`, `TableUpdate`, `TableWrite`) plus the `Key`/`Values`/`Row`/`Range`/`ColumnRange` type aliases. |
| `table/schema.rs` | `TableSchema` (key columns, value columns, primary `IndexSchema`, named auxiliary indices); implements `b_table::Schema`, value (de)serialization, key/value validation, and `range_from_key`. |
| `table/stream.rs` | `Rows<'a>` (a read-permit-bound row stream) and `TableView<'en>` (`IntoView` encoder). `Rows::limit` / `Rows::select` are lazy row transforms applied while streaming. |
| `table/view.rs` | Lazy view structs: `Limited<T>` (row cap), `Selection<T>` (column projection), and `TableSlice<Txn, FE>` (range + order + reverse). Each re-implements the trait surface by delegating to its source. |
| `table/file.rs` | `TableFile<Txn, FE>`: the transactional engine. Owns the canonical version, committed deltas, pending deltas, the finalize frontier, and a range semaphore. Implements `Transact` (commit/rollback/finalize), `Persist`, `CopyFrom`, `Restore`, and `de::FromStream`. |
| `table/public.rs` | Route handlers and the `Route` impls for `Table` / `TableFile` / `Static`: `create`, `copy_from`, row read/slice/upsert/update/truncate/delete, `columns`, `key_columns`, `key_names`, `contains`, `count`, `limit`, `order`, `select`. |

## 2. Boundary decisions

### 2.1 `b-table` vs. `tc-collection` authority

This port reuses the same transaction-orchestration boundary already established
by the BTree port (`ROADMAP.md` item 4, `AGENTS.md`):

- **`b-table` owns non-transactional persistence and index mechanics.** It
  provides `TableLock` (primary + auxiliary `BTreeLock` indices, locked
  primary-first then auxiliary-in-order), `Table` read/write guards, `Schema` /
  `IndexSchema` traits, `Row`/`Rows`/`Range`/`ColumnRange`, query planning
  (`QueryPlan`), and the `merge` / `delete_all` / `upsert` / `delete_row` /
  `truncate` / `count` / `rows` storage operations. `b-table` has **no**
  transaction, visibility, or finalize semantics.
- **`tc-collection` owns transaction lifecycle.** The v2 `Table` (mirroring the
  v2 `BTree` in `src/btree/file.rs`) owns: the canonical version, the
  committed/pending delta sets, the finalize frontier, the range semaphore,
  transactional visibility, isolation (blocking reads behind earlier
  overlapping pending writes), commit/rollback/finalize ordering, and
  deterministic replay application. Transaction ownership **never** moves into
  `b-table`.

### 2.2 No-materialization rules

Query, view, and stream execution must remain lazy and block-streaming. This is
the v1 invariant carried forward verbatim and reinforced by `AGENTS.md`:

1. No full table, index, range, join, or intermediate result may be collected
   solely to execute an operation. Rows are produced as a `Stream` and consumed
   incrementally; the v1 `Rows<'a>` is a permit-bound `TCBoxTryStream<Row>`.
2. `count` is computed by streaming rows and folding (`try_fold(0, |c, _| c+1)`),
   not by materializing the row set. `is_empty` short-circuits on the first row.
3. `truncate` / `update` stream the in-range rows into a temporary `BTreeLock`
   scratch index, then stream that scratch back into the pending delta — never
   buffering the whole affected set in a `Vec`.
4. View composition (`Limited`, `Selection`, `TableSlice`) is structural: each
   view holds a reference to its source and applies its transform (`take`,
   column projection, range/order) lazily during `rows`. The v2 port must keep
   this — views must not eagerly copy rows.
5. Per `AGENTS.md`, never build `Vec`/`BTreeSet` snapshots of full keysets for
   route responses or serialization; consume key/row streams incrementally and
   encode as you iterate.

### 2.3 Canonical-plus-delta (LSM) model and deterministic merge order

The v1 `TableFile` is a canonical-plus-delta store. The v2 port preserves this
model exactly:

- **Canonical version (`canon`)** — a `b_table::TableLock` holding the
  finalized, merged-on-disk state.
- **Committed deltas** — `OrdHashMap<TxnId, Delta>` where each `Delta` is a pair
  of `b_table::TableLock`s: `inserts` and `deletes`. Ordered by `TxnId`.
- **Pending deltas** — `OrdHashMap<TxnId, Delta>` for in-flight transactions.
- **Finalize frontier** — a monotonic `Option<TxnId>`; everything `<=` the
  frontier has been merged into `canon`.

**Deterministic merge order** (v1 `TableFile::into_rows` / `read` / `finalize`):

1. Read canonical rows first, **before** locking `state`, to avoid deadlock with
   `finalize` (v1 comments: *"read-lock the canonical version BEFORE locking
   self.state"*).
2. Merge committed deltas in **ascending `TxnId` order**, taking
   `deltas.iter().take_while(|(id, _)| *id <= &txn_id)`. For each delta, merge
   inserts (`collate::try_merge`) then diff deletes (`collate::try_diff`).
3. If a pending delta exists at `txn_id`, merge it last (txn-local visibility).
4. Read visibility: pending delta at T is visible only to T; committed deltas
   `<= T` are visible to T; no leakage from other transactions' pending deltas.
5. `finalize(txn_id)` merges all committed deltas `<= txn_id` into `canon` via
   `canon.merge(inserts)` + `canon.delete_all(deletes)`, pops them from
   `committed`, prunes `pending` and the `commits` set, and advances the
   frontier. Re-finalizing `<=` the frontier is a no-op.

The merge is deterministic because (a) deltas are totally ordered by immutable
`TxnId`, (b) inserts/deletes within a delta are applied in the fixed
insert-then-delete order, and (c) `canon` is the single merge target. The v2
port must not invent alternate replay order or local reconciliation
(`AGENTS.md`: fail closed on any non-deterministic replay condition).

## 3. File / module migration map

Every v1 `Table` module and public symbol has a v2 disposition. The v2 layout
mirrors `src/btree/` (`mod.rs` / `file.rs` / `stream.rs` / `codec.rs` /
`tests.rs`), adding the schema and view modules that BTree does not need.

| v1 module / symbol | v2 location | `b-table` / `b-tree` primitive backing it | Disposition |
| --- | --- | --- | --- |
| `table/mod.rs` — `Table` enum, `TableType` | `src/table/mod.rs` | — | Port: `Table` view enum (`Limited`/`Selection`/`Slice`/`File`) and `TableType` class. `Collection::Table` variant added in `src/collection.rs`. |
| `table/mod.rs` — `TableInstance` trait | `src/table/mod.rs` | `b_table::Schema` (`schema()` accessor) | Port: trait retained; `schema()` returns the v2 `TableSchema`. |
| `table/mod.rs` — `TableOrder` trait | `src/table/mod.rs` | `b_table::IndexSchema::columns` (order support check) | Port: `order_by` / `reverse` build a `TableSlice` view. |
| `table/mod.rs` — `TableRead` trait | `src/table/mod.rs` | `b_table::Table::get_row` | Port: `read(txn_id, key)` resolves visibility through the delta stack. |
| `table/mod.rs` — `TableSlice` trait | `src/table/mod.rs` | `b_table::Range` / `IndexSchema::supports` | Port: `slice(range)` builds a `TableSlice`. |
| `table/mod.rs` — `TableStream` trait | `src/table/mod.rs` | `b_table::Table::rows` / `count` / `is_empty` | Port: `count`/`is_empty`/`limit`/`select`/`rows`; streams stay lazy. |
| `table/mod.rs` — `TableUpdate` trait | `src/table/mod.rs` | `b_tree::BTreeLock` scratch + `b_table::Table::upsert`/`delete_row` | Port: `truncate(range)` / `update(range, values)` via temp scratch index. |
| `table/mod.rs` — `TableWrite` trait | `src/table/mod.rs` | `b_table::Table::upsert` / `delete_row` | Port: `delete(key)` / `upsert(key, values)` into the pending delta. |
| `table/mod.rs` — `Key`/`Values`/`Row`/`Range`/`ColumnRange` aliases | `src/table/mod.rs` | `b_table::Row` / `b_table::Range` / `b_table::ColumnRange` / `b_tree::Key` | Re-export from `b-table` / `b-tree` (v1 does this verbatim). |
| `table/schema.rs` — `TableSchema` | `src/table/schema.rs` | `b_table::Schema` + `b_tree::Schema` (`IndexSchema`) | Port: struct with `key`/`values`/`primary`/`indices`; implement `b_table::Schema`, validation, `range_from_key`, value (de)serialization. |
| `table/stream.rs` — `Rows<'a>` | `src/table/stream.rs` | `b_table::Rows` wrapped behind a `txn_lock` read permit | Port: permit-bound row stream; `limit`/`select` are lazy stream transforms. |
| `table/stream.rs` — `TableView<'en>` / `IntoView` | `src/table/stream.rs` (codec) | `destream` encoder | Port: streaming encoder; deferred until IR envelope wiring (see [§8](#8-unresolved-questions--blockers) B1). |
| `table/view.rs` — `Limited<T>` | `src/table/view.rs` | — (pure lazy wrapper) | Port: row-cap view delegating to source `rows().limit(n)`. |
| `table/view.rs` — `Selection<T>` | `src/table/view.rs` | `b_table::Schema::primary` (column lookup) | Port: column-projection view; `Rows::select` projects while streaming. |
| `table/view.rs` — `TableSlice<Txn, FE>` | `src/table/view.rs` | `b_table::Table::rows(range, order, reverse, select)` / `QueryPlan` | Port: range+order+reverse view; validates index support before constructing. |
| `table/file.rs` — `TableFile` (engine) | `src/table/file.rs` | `b_table::TableLock` (canon + per-delta inserts/deletes) | Port: mirror `src/btree/file.rs` — `PersistentTable` wrapping `TableLock`, `Delta { inserts, deletes }`, `State { persistent, committed, pending, finalized, txn_root }`, range `Semaphore`, `Transact` impl. |
| `table/file.rs` — `Delta` | `src/table/file.rs` | `b_table::TableLock` pair | Port: `inserts`/`deletes` `TableLock`s; `merge_into` uses `collate::try_merge`/`try_diff`. |
| `table/file.rs` — `State` | `src/table/file.rs` | — | Port: `committed`/`pending` `BTreeMap<TxnId, Delta>`, `finalized`, `txn_root`. |
| `table/file.rs` — `Transact`/`Persist`/`CopyFrom`/`Restore`/`FromStream` | `src/table/file.rs` + `src/table/codec.rs` | `b_table::TableLock::create`/`load`/`sync`; `freqfs::DirLock` | Port: lifecycle per `Transact` trait (`tc_ir::Transact`, as BTree does). `Restore`/`FromStream`/`CopyFrom` are **deferred** (see [§8](#8-unresolved-questions--blockers) B2/B3). |
| `table/public.rs` — route handlers | `src/table/route.rs` | — | Port: handler structs + `Route` impl (see [§4](#4-route--api-matrix)). Express via shared IR envelopes (`AGENTS.md`). |
| `table/public.rs` — `Static` (`create`, `copy_from`) | `src/table/route.rs` | `b_table::TableLock::create` | Port: `create`. `copy_from` stays `not_implemented` parity (v1 stubs it). |
| `table/public.rs` — `KeyOrRange` / `cast_into_range` | `src/table/route.rs` | `b_table::Range` / `ColumnRange` | Port: selector parsing (All / Key / Range). |
| `public/table.rs` — `Collection`/`CollectionType`/`Static` routing | `src/collection.rs` + `src/table/route.rs` | — | Port: add `Collection::Table` dispatch and the collection-level `schema` route. |

## 4. Route / API matrix

The v1 public surface exposed these routes. Each row gives the HTTP verb, the
v1 handler, and the v2 disposition. Error envelopes and auth behavior are
preserved per `AGENTS.md` ("keep v1 collection semantics — batching, auth
headers, error envelopes — visible") and expressed through the shared IR
envelopes rather than adapter-specific shapes.

| Route | Verb | v1 handler | v2 disposition | Error envelope |
| --- | --- | --- | --- | --- |
| `/state/collection/table` (static) | GET | `CreateHandler` | `create(schema)` → `TableFile::create` | `bad_request` on invalid schema |
| `/state/collection/table/copy_from` | POST | `CopyHandler` | **Parity stub** (`not_implemented!("copy a Table")`) — v1 itself does not implement it | `not_implemented` |
| `<table>` | GET | `TableHandler` (All→self, Range→slice, Key→read row) | port: read/slice/read-row dispatch | `bad_request` on invalid selector |
| `<table>` | PUT | `TableHandler` (All/Range→update, Key→upsert) | port: `update(range, values)` / `upsert(key, values)` | `bad_request` on bad values/cols |
| `<table>` | POST | `TableHandler` (body→range→slice) | port: `slice(range)` | `bad_request` on invalid selection |
| `<table>` | DELETE | `TableHandler` (All→truncate, Key→delete, Range→truncate) | port: `truncate(range)` / `delete(key)` | `bad_request` on invalid selector |
| `<table>/columns` | GET | `SchemaHandler` (`column_schema`) | port: return primary column schema | — |
| `<table>/contains` | GET | `ContainsHandler` (All/Key/Range) | port: `is_empty` / `read` / slice `is_empty` | `bad_request` on invalid selector |
| `<table>/count` | GET | `CountHandler` (All/Key/Range) | port: `count` / read-presence (0/1) / slice `count` | `bad_request` on invalid selector |
| `<table>/key_columns` | GET | `SchemaHandler` (`key_columns`) | port: return key column ids | — |
| `<table>/key_names` | GET | `SchemaHandler` (`key_names`) | port: return key column ids | — |
| `<table>/limit` | GET | `LimitHandler` | port: `limit(n)` → `Limited` view | `bad_request` if limit too large |
| `<table>/order` | GET | `OrderHandler` (`(cols, rev)` or `cols`) | port: `order_by(cols, rev)` → `TableSlice` | `bad_request` on invalid column list |
| `<table>/select` | GET | `SelectHandler` | port: `select(cols)` → `Selection` view | `bad_request` if column absent |
| `Collection::schema` | GET | collection `SchemaHandler` | port: extend `Collection` schema route to `Table` | — |

**Auth behavior:** preserved v1 — authorization is enforced at the host/server
layer (`tc-server`), not inside `tc-collection`. `tc-collection` handlers must
not add bespoke auth; they surface structured `tc_error` envelopes upward
(`AGENTS.md`). **Batching:** v1 handlers accept single-row and range forms via
`KeyOrRange`; the v2 port preserves both selector shapes and reuses the shared
IR envelopes so Python/WASM clients reuse the same contracts.

## 5. Intentional v1 divergences

Per the v1 parity port rules (`TRANSACTIONAL_COLLECTION_CONTRACT.md` §"v1 Parity
Port Rules"), any divergence must be documented with rationale and migration
impact. The following are intentional and bounded:

1. **`txn_lock::Semaphore` replaces the v1 `tc_transact::lock::Semaphore`.**
   The v2 BTree port already uses `txn_lock::semaphore::Semaphore<TxnId,
   Collator<Vec<Value>>, Range<Vec<Value>>>` with `try_write`/`read` and
   reservation release on every exit path. The Table port uses the same
   `txn_lock` semaphore over `b_table::Range<Id, Value>` to share one canonical
   lock-acquisition order across BTree/Table/Tensor (contract §"Locking and
   concurrency invariants"). *Rationale:* single canonical lock order;
   reservation-leak regression coverage already proven for BTree. *Migration
   impact:* none for clients; internal only.
2. **`tc_ir::Transact` replaces `tc_transact::Transact`.** The v2 crate targets
   the `tc-ir`/`tc-error`/`tc-value` stack (see `Cargo.toml`), so lifecycle is
   expressed via `tc_ir::Transact` exactly as `src/btree/file.rs` does. Commit
   is fallible (`Result<(), txn_lock::Error>`) and `rollback`/`finalize` panic
   on invariant violation, matching BTree. *Rationale:* stack alignment.
   *Migration impact:* none for clients.
3. **`copy_from` remains unimplemented.** v1's `CopyHandler` returns
   `not_implemented!("copy a Table")`; the port preserves this stub rather than
   inventing copy semantics. *Rationale:* no new semantics before parity.
   *Migration impact:* identical to v1.
4. **No new transaction semantics.** Visibility, isolation, finalize, and replay
   semantics are byte-for-behavior identical to v1. No opportunistic merge,
   repair, or fallback paths are introduced (`AGENTS.md`).

## 6. Fixture datasets

Small, reusable fixture datasets are checked in under `tests/fixtures/` for use
by later implementation/test issues. Each is deliberately tiny so tests stay
fast and deterministic. See `tests/fixtures/MANIFEST.md` for the schema and
intent of each fixture.

| Fixture | Columns (key in **bold**) | Rows | Purpose |
| --- | --- | --- | --- |
| `simple_pk.csv` | **id**:Int, label:String, score:Int, tag:String | 5 | Single-column primary-key reads, upsert/delete, single-column range, count. |
| `index_and_range.csv` | **id**:Int, ref_id:Int, amount:Float, state:String | 8 | Range slice, `select`/`order`/`limit` views, auxiliary index on `ref_id`. |
| `composite_key.csv` | **part_a**:Int, **part_b**:Int, val1:Float, val2:Float | 12 | Composite primary key, auxiliary-index coverage, streaming/scan perf baseline. |

## 7. Test matrix

One planned automated test per behavior, grouped by the contract's parity
checklists. Test names are provisional targets for `src/table/tests.rs`; the
implementation issue(s) will land them. Every behavior must map to an executable
validation case (acceptance criterion: *"Every required behavior maps to an
executable validation case"*).

### 7.1 Lifecycle
| Behavior | Planned test |
| --- | --- |
| `upsert` inserts/updates a row in the pending delta | `upsert_inserts_and_updates_pending_row` |
| `delete` removes a row into the pending delete delta | `delete_moves_row_to_pending_deletes` |
| `commit` promotes pending → committed without merging canon | `commit_promotes_pending_to_committed` |
| `rollback` discards the pending delta and unblocks later reads | `rollback_discards_pending_and_unblocks` |
| `finalize` merges committed deltas into canon and advances frontier | `finalize_merges_committed_into_canon` |
| duplicate `commit` is idempotent (no-op) | `duplicate_commit_is_idempotent_table` |
| stale `finalize` (≤ frontier) is a no-op | `stale_finalize_is_noop_table` |
| write after commit/finalize is rejected | `cannot_write_after_commit_or_finalize_table` |
| no-op/error lifecycle exits release the semaphore reservation | `lifecycle_noop_paths_release_reservations_table` |

### 7.2 Visibility
| Behavior | Planned test |
| --- | --- |
| pending delta is visible only to its own txn | `pending_is_visible_only_to_its_txn_table` |
| committed deltas visible in txn order (≤ txn_id) | `committed_is_visible_in_txn_order_table` |
| later overlapping read blocks until earlier pending txn finalizes | `overlapping_read_blocks_until_earlier_finalize_table` |
| no visibility leakage from other txns' pending deltas | `no_pending_leakage_across_txns_table` |
| point read resolves inserts/deletes across the delta stack | `read_resolves_delta_stack_table` |

### 7.3 Recovery
| Behavior | Planned test |
| --- | --- |
| restart reconstructs visible state by reapplying committed deltas | `restart_reconstructs_committed_state_table` |
| prepared (uncommitted) pending delta is dropped on restart | `restart_drops_uncommitted_pending_table` |
| unresolved-finalize restart re-merges under Chain-driven replay | `unresolved_finalize_restart_replays_table` *(depends on `tc-chain`)* |

### 7.4 Integrity
| Behavior | Planned test |
| --- | --- |
| corrupt/malformed persisted state fails closed with structured error | `corrupt_state_fails_closed_table` |
| schema-mismatched `merge`/`delete_all` fails closed | `schema_mismatch_merge_fails_closed_table` |
| invalid key/value arity fails closed | `invalid_key_arity_fails_closed_table` |

### 7.5 Query correctness
| Behavior | Planned test |
| --- | --- |
| `slice(range)` returns only in-range rows | `slice_returns_in_range_rows_table` |
| `select(cols)` projects the requested columns | `select_projects_columns_table` |
| `limit(n)` caps the row stream | `limit_caps_row_stream_table` |
| `order_by(cols, rev)` orders via a supporting index | `order_by_uses_supporting_index_table` |
| `reverse` flips iteration order | `reverse_flips_order_table` |
| `count` matches streamed fold | `count_matches_streamed_fold_table` |
| `contains` (All/Key/Range) is correct | `contains_all_key_range_table` |
| range with no supporting index fails closed | `unsupported_range_fails_closed_table` |

### 7.6 Streaming / no-materialization
| Behavior | Planned test |
| --- | --- |
| streamed rows match a materialized expectation (visibility-aware) | `streamed_rows_match_materialized_table` |
| `truncate`/`update` use a temp scratch index, never buffering all rows | `truncate_update_use_scratch_not_buffer_table` |
| view composition stays lazy (no eager copy) | `view_composition_is_lazy_table` |

### 7.7 Performance
| Behavior | Planned test |
| --- | --- |
| read/write/range/scan within regression budget vs v1 baseline | `bench_read_write_range_scan_table` *(criterion; budget TBD vs v1)* |
| multi-index query plan selects the cheapest supporting index | `query_plan_selects_cheapest_index_table` |

### 7.8 Concurrency / locking
| Behavior | Planned test |
| --- | --- |
| canonical lock order (canon before state; primary before auxiliary) — no deadlock | `lock_order_no_deadlock_table` |
| concurrent read/write/finalize contention | `concurrent_read_write_finalize_table` |
| finalize+sync does not re-enter locks under guard | `finalize_sync_drops_guard_first_table` |

## 8. Unresolved questions / blockers

Recorded explicitly per the issue (*"Record unresolved questions as explicit
blockers; do not invent new semantics"*):

- **B1 — IR envelope / streaming encoder.** The v1 `TableView` / `IntoView`
  path and the `de::FromStream`/`IntoStream` codecs depend on the
  `tc_transact`/`destream` stack. The v2 crate uses `tc-ir` envelopes
  (`AGENTS.md`). The streaming codec for `Table` (analogous to
  `src/btree/codec.rs`) is **blocked** on confirming the `tc-ir` envelope
  encoding contract for row streams. No codec is invented here.
- **B2 — `Restore` / `CopyFrom`.** v1 implements `Restore<FE>` and
  `CopyFrom<FE, T>` for `TableFile`. The v2 BTree port has **not** yet ported
  these (only `Transact` + the in-process `PersistentStore` load). The Table
  `Restore`/`CopyFrom` port is **blocked** on the BTree port landing the
  equivalent, to keep the two variants symmetric. Tracked as a follow-up, not
  in scope for the parity core.
- **B3 — Chain-ordered event replay.** As with BTree (contract §"BTree parity
  gap status" item 4), explicit canonical-chain-event replay coverage depends
  on `tc-chain` integration and remains open. The Table port inherits the same
  blocker; `direct_mutation_flow`-style coverage is in scope, full replay is
  not.
- **B4 — Auxiliary index creation after table creation.** v1 `TableSchema` is
  fixed at create time (primary + named auxiliary indices). Adding/dropping
  indices at runtime is not a v1 behavior and is **out of scope** — do not
  invent it.
- **B5 — Performance regression budgets.** v1 baselines for
  read/write/range/scan must be captured before the perf gate can be enforced.
  Budgets are **TBD** pending a v1 baseline run; the test is a placeholder
  until then.

## 9. Acceptance criteria checklist

(From issue #4. This PR satisfies the spec deliverables; implementation issues
satisfy the green-test items.)

- [x] Every v1 Table module and public route has a v2 disposition ([§3](#3-file--module-migration-map), [§4](#4-route--api-matrix)).
- [x] Every required behavior maps to an executable validation case ([§7](#7-test-matrix)).
- [x] The no-materialization and transaction-authority boundaries are unambiguous ([§2](#2-boundary-decisions)).
- [x] No production Table implementation is included in this PR.
- [x] A checked-in Table parity/port document linked from `TRANSACTIONAL_COLLECTION_CONTRACT.md`.
- [x] A file/module migration map ([§3](#3-file--module-migration-map)).
- [x] A route/API matrix ([§4](#4-route--api-matrix)).
- [x] A test matrix covering lifecycle, visibility, recovery, integrity, query correctness, streaming, and performance ([§7](#7-test-matrix)).
- [x] Small fixture datasets reusable by later issues ([§6](#6-fixture-datasets), `tests/fixtures/`).
- [x] The pinned v1 revision is cited ([§1](#1-source-of-truth)).
- [x] Unresolved questions recorded as explicit blockers ([§8](#8-unresolved-questions--blockers)).
