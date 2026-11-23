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

Every application mutation of a persistent mutable collection must pass through
its owning Chain. Returned views grant read access, not mutation authority.
Chain records accepted mutations and owns their durable recovery; collections
apply local transactional changes. A caller transferring an unpublished owner
to Chain must relinquish mutable aliases and must not independently reopen its
storage. This is an exclusive-ownership contract for trusted native callers,
not a runtime alias-revocation mechanism.

Collection GET/POST results which alias an existing owner are read-only, including
full-range and identity views. Further transforms never restore mutation or
lifecycle authority. Selection writes target the original owner with explicit
bounds, so Chain observes the same PUT/DELETE that it records and replays.
Fresh request-local collections and explicit independent copies remain writable.
No unsupported read layout triggers an automatic whole-collection copy.

Schema construction from Value is fallible through `TryFrom`: semantic parsing
returns its constructor errors rather than testing castability on a cloned value.
Incremental codecs retain bounded field decoding and delegate semantic validation
to the same schema constructors.

Value-backed schemas represent dtype references as canonical hostless Links.
Their conversion delegates to `tc-value`'s type-reference conversion; collection
owners do not maintain a separate interpretation of URI-shaped scalar text.

Logical BTree views pair immutable column metadata with a matching typed native
owner through fallible construction. Public File/Slice variants are no longer
construction or mutation paths. Clones and slices share the metadata; terminal
projection moves a uniquely owned header and otherwise clones it for its independent
consumer. Untyped raw BTrees remain available directly. Native aliases retained
before handoff are not revoked; callers must relinquish them as described above.

Concrete BTree, Table, and Tensor values prepare their own terminal views, including
storage-error policy and guard ownership. Collection delegates this preparation;
codecs retain only wire structure.

## Table access

Table constructors delegate structural validation to b-table before native
allocation. Column names are unique across keys and values. Auxiliary declarations
contain nonempty, unique value-column lists; primary-key columns are appended
automatically and cannot be declared there. Index names are unique, valid native
names other than `primary`. Typed construction and Value decoding share these checks.

Table queries delegate to one supporting native index, choosing the primary index
before auxiliary indexes in declaration order. Constraints must form leading
equalities followed by at most one range; requested ordering must be a literal
index-column prefix. Unrestricted scans remain streamed. Unsupported combinations
fail before row traversal or deletion: use a matching composite index or rewrite
the predicate into the supported form. Covering projections and exact primary-row
lookups for noncovering results retain their ordinary ownership boundaries.

Table slice composition intersects existing constraints; it never widens an
existing view. Disjoint constraints return a structured error from the fallible
`TableSlice::slice` method. Unsupported combined predicates still require a
supporting native index. Limits and projections apply equally to row streams,
counts, emptiness, and callbacks, including lifecycle validation for a zero limit.
Table selections retain the original key columns first, in their original order,
then emit unique value columns in caller order. Duplicate columns, missing keys,
and layouts incompatible with this schema reject before traversal. A prepared
projection supplies both output metadata and field positions. Raw `Rows::select`
retains its general ordered, repeated-column behavior. Terminal operations consume
the composed row stream.

Table view counts and emptiness consume the same guarded row stream as iteration,
including order support checks for empty and zero-limit views. Untransformed local
Tables retain native predicates. Local view predicates may construct owned rows
that native predicates avoid; they do not collect the result or copy it wholesale.

Selection through a limited Table delegates to its concrete projection operation,
preserving the cap, output schema, and read-only ownership. Public row limits
accept finite, nonnegative integral values from integer and floating-point types
which fit `u64`, including zero. Other values reject before storage access instead
of being truncated or wrapped.

Table copying rejects duplicate semantic primary-key identities and preserves
complete canonical row order. Construction retains at most the previous owned row
and the next row, then moves completed rows into native storage. Numeric identity
is preserved across equal mixed representations without collecting a keyset.

Local and transactional replacements delegate to native upsert rather than
deleting an existing insert row before validation and replacement.

## Lifecycle

Logical BTree and Table values own lifecycle eligibility and concrete delegation;
Collection only dispatches to the selected value. Persistent owners delegate to
the shared lifecycle, local Tables have no transactional versions to decide, and
read-only views reject lifecycle calls.

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

Construction, reads, scans, and lifecycle operations propagate storage and
reservation errors. Poisoned bookkeeping is an invariant failure and panics;
poison is never cleared to resume use. Counts and emptiness checks consume the
same guard-owning streams as scans; storage errors never become missing rows or
empty results.

Locally detected storage integrity failures retain typed recovery provenance.
`StorageContext::storage_error` delegates recovery policy to the caller before
terminal projection erases error types, including errors from lazy collection
streams. Its default returns the error for direct callers to recover; the server
implementation terminates the host. Obsolete materialized workspace errors are
resolved by the existing handoff before this policy is applied. Collections
maintain no owner health flag or distributed decision policy.

Callers serialize lifecycle decisions for a collection. Before commit, they
prevent new mutations in that transaction and finish its mutating operations.
Immutable result snapshots and their streams may survive commit: publishing a
delta does not change its payload. Response projection therefore need not drain
or materialize those results before committing. Before rollback, callers finish
or cancel the transaction's operations and release its streams and guards.
Before finalization, they do so for transactions through the cutoff. Collections
do not check this precondition by acquiring new semaphore reservations; unrelated
later transactions may continue.

Later BTree/Table streams can retain native read guards that delay finalization.
Callers must drain, drop, or concurrently consume those streams while waiting;
finalization need not complete while a blocking stream remains unpolled. Tensor
snapshots retain their existing handoff from materialized deltas to canonical storage.

BTree, Table, and Tensor commit move the pending delta into committed visibility under the
shared state lock, then release reservations. Commit performs no persistence and
is not independently crash-durable. Empty decisions need no files. Failed decisions
retain reservations. Rollback discards pending work.

Write admission checks lifecycle eligibility before registering a reservation.
A rejected write to a committed or finalized transaction must not obstruct later
reads. Publication rechecks eligibility after asynchronous construction; callers
still seal and finish mutations before a lifecycle decision.

Finalization applies covered deltas in place to native canonical storage, prunes
covered transaction state, then releases reservations. The caller supplies durable
recovery evidence and coordinates canonical synchronization. Interrupted or failed
materialization must not be retried against uncertain live storage; the caller owns the
recovery-required boundary. Collection lifecycle methods return collection errors,
preserving native storage error classification. The transaction semaphore reports
only reservation and transaction-ordering failures.

## Visibility and ordering

New reads at or below the finalized cutoff return `Outdated`: canonical storage
may already include newer values. Read admission checks the cutoff before and
after waiting, and visible-state capture checks it under the same bookkeeping lock.
These checks add no retained version frontier or semaphore history.

Native BTree and Table root GET handlers return a native view for a null key.
`IntoView` acquires transaction consistency; encoding remains separate. BTree delegates
non-null slice arguments to its existing native slice behavior; no host-specific
snapshot endpoint is required. Native creation and strict loading both delegate
through `CollectionSchema` to the concrete owner. Creation awaits native cache
admission, including empty index roots, before publishing the constructed owner.
Strict loading never creates those roots.

`materialize` is the single public operation for copying a transaction-visible
collection into delegated native storage. It validates that the view preserves
its schema and canonical key order before allocating the destination. Lower-level
native storage copies and Tensor `copy_values` retain their separate purposes.
Restoration and ordinary reads share committed-base snapshot assembly under the
same lifecycle guard; only ordinary reads append their own pending version.

Native adapters preserve payload tags and validate child/separator cardinality.
Completed decoded payload vectors discard spare capacity once before native
cache admission, including the outer vectors of leaf keys and numeric blocks.
Storage containing old copied-minimum internal nodes must be recreated; loading
never migrates or repairs it. Collection wire values and the hash formula retain
their existing definitions; correcting tied-row ordering can change affected
BTree/Table digests.

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
- Streams retain their read guards and capacity permits until EOF, the first
  error, or drop. Lazy limits and projections use the same cleanup owner, which
  drops the source before its permits. Waiting remains cancellable and bounded
  by the request deadline.

Native BTree row and Tensor payload callbacks borrow decoded storage synchronously.
Native Table queries retain owned row streams; sparse Tensor storage uses BTree
callbacks directly, without a Table query layer. Lazy ndarray operations are
read-only; selected persistent writes remain owner-addressed collection operations.
They may validate or scatter into caller-owned buffers, but references do not
escape and callbacks do not await or acquire another lock domain. Retained
collection streams and State results own their values and transaction capabilities;
native page guards are released before materialization retry or response encoding.
Owning a row does not require copying it: transformations move consumed fields,
cloning only where repeated output positions require independent values.

Terminal Table handlers borrow their receiver for selection and invocation.
Construct handlers from `&Table`, or use `Route` on an explicitly constructed
Table; there is no separate generic routing adapter. Returned views acquire
independent ownership and remain read-only. `CollectionState::as_number` inspects
an embedded numeric value without coercion or cloning the surrounding State.
Tensor handlers consume their owned arguments after this inspection.

Every buffer, stream, cache, and task spawned by a collection must be bounded
and release its capacity on completion, cancellation, or drop. Collection-local
limits compose beneath any broader admission policy supplied by the caller.
Native file adapters inspect codec structure and delegate Value heap bounds to
its owning type before loading. They count native rows and children without
constructing payloads or applying Tensor geometry limits to generic nodes; native
writers reconcile actual retained size when mutable guards are released; loading
and complete-value publication retain admission. In-place mutation can exceed
the cache target, which is not a strict host-memory ceiling. Retained memory and
serialized byte counts are distinct, and synchronization does not change the
memory charge or implicitly dispose of empty directories.

## Persistence and recovery

Creation requires empty delegated storage. Loading strictly requires the existing
root and all schema-required indexes, and validates native tree structure and index
consistency. Loading reconstructs only materialized canonical contents. It does not
recover committed visibility, acceptance receipts, or a cutoff. The caller reconstructs
those from its retained WAL with original-ID capabilities and fresh workspaces.

Native fensor clones share live storage and retain no historical block payloads.
Transaction-visible sources, retained delta handles, and finalization handoff belong
to this crate. Native replacement only coordinates its current payload and indexes;
it does not publish a transaction decision.

Tensor numerical reads plan one bounded request in canonical block geometry, resolve
its visible sources newest-first, then delegate payload consumption to fensor's
native block reader. Source selection visits deltas newest-first and scans consecutive
unresolved requested IDs in ascending order, without scanning across gaps or newer
selections. It completes before payload reads: when several visited blocks are
corrupt, a newer membership error may precede an older block's membership or payload
error. Errors are never treated as absence. Shadowed payloads are not read. Native
scans still decode their containing pages, including rows outside the selected range.
Stored blocks retain exact canonical-length validation. Materialization invalidates a selected workspace result or error;
the affected bounded run is resolved again after releasing its native guards.

Tensor deltas retain a native BTree set of touched logical block IDs and delegate
replacement payloads and occupancy to fensor's `SparseBlocks` with canonical
geometry. Dense storage chunks one flat
row-major region; sparse-coordinate prefixes identify independent dense suffix
regions. The last block of each region may be short. Native BTree rows store
compact payloads, omitting all-zero blocks; stored lengths are validated exactly.
A touched all-zero block suppresses older contents; missing touched entries fall
through. Page sharing preserves block reservations, bounded copy-on-write, wire
formats, and semantic hashes. A lightly occupied block retains its full dense
payload rather than choosing another representation. After failed or cancelled native publication, the caller owns disposal and recovery
of the affected native handles. Interrupted collection publication requires caller-owned rollback; failed canonical
materialization requires caller-coordinated recovery.

Committed deltas retain their delegated transaction workspaces until finalization.
Each delta has one shared owner for its native handles and metadata, retained by
pending/committed state and independently acquired readers. Assignment stamps retain a separate generation token without
retaining that workspace owner. Snapshots share read permits only when one exists;
unreserved internal snapshots allocate no permit owner. Visible-history containers
and bounded traversal metadata remain owned allocations.

Host workspace removal follows successful recursive finalization. Retained readers
consult the owning Collection state's finalized frontier before and after native
reads, before propagating obsolete workspace errors. That frontier is published
only after all covered materialization succeeds; no separate materialized flags
are maintained. Later Tensor snapshots use canonical storage for retired layers;
sparse cursors preserve their logical position across this handoff. BTree/Table
stream setup performs the same retirement checks before retaining native streams.
Explicit `sync_all()` makes materialized canonical storage durable; `sync()` is buffered
writeback. Neither persists pending or committed workspace deltas as accepted work.

BTree, Table, and Tensor share one private lifecycle owner containing their transaction
state and existing semaphore. It implements visibility selection and decisions once,
delegating delta validation and application to concrete key, row, or logical-block operations. Concrete collections
retain creation, loading, workspace construction, hashing, copying and restoration.
There is no independent publication owner or acceptance metadata.

`CopyFrom::copy_from(txn, dir, source)` constructs an independent transactional
collection by recursively delegating to its concrete owner. Sources are borrowed
for the copy and read through the supplied transaction. The caller exclusively
delegates empty, unpublished destination storage; native constructors reject
nonempty storage without changing its contents. Copying streams contents and
synchronizes the result without deciding the source transaction. On failure or
cancellation, partial unpublished output belongs to caller cleanup. `Restore::restore`
instead replaces pending state in an existing owner.
Tensor selects native destination layout at this boundary: retain valid reported
sparse-axis hints, including shape-preserving unary and cast values; otherwise use
the schema's scalar-sparse default. Dense values use dense storage. This packing
policy is independent of semantic identity and does not change the native reader. Reordered views which canonical storage cannot faithfully represent fail with a structured error. Native schemas convert to and from a collection class path and a Value
for strict reopening. Collection identity hashes the class and semantic schema,
hashes ordered native contents, then combines those hashes with SHA-256. It
includes column and index definitions but excludes cache configuration and physical
layout; no serialization is used for hashing. Tensor content hashing consumes
bounded row-major numerical batches and reuses exact typed zero digests, including
independent signs of complex components. Its digest is identical to hashing each
logical element separately; sparse hashing still consumes logical cardinality.
The caller coordinates publication
and retention. `Restore::restore` validates kind and semantic schema, builds
insert/delete deltas in a separate delegated workspace, and installs the pending
replacement only after construction succeeds. Rollback preserves the previous
committed value. Self-restoration, including a cloned alias, is a no-op after
write eligibility is checked. Distinct-owner restoration acquires source read
admission without waiting while destination writes are reserved; conflicts return
structured errors before allocating staging storage. Source streams release their
native guards and permits before the staged replacement is installed. Collection
routes expose no snapshot restoration endpoint. Compatibility uses typed metadata
in the concrete owner before reservations or staging; it does not construct wire
Values for comparison. Table compatibility includes ordered columns and auxiliary
index definitions, excluding storage configuration. BTree views include column
names, dtypes, and optional size metadata. Tensor restoration requires dtype, shape,
density, and normalized sparse axis to agree; block capacities may differ.

Native trees mutate individual blocks and delete obsolete nodes in place. They
provide no immutable root generations or unreachable-version reclamation. Native
copying and loading retain corruption checks. Development fixtures from earlier
layouts must be recreated; no migration reader or fallback is provided.

Explicit materializations may accelerate access but are not an independent source
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

## Tensor values and numerical execution

Native geometry uses compact row-major chunks of at most 4,096 elements. Each
sparse-coordinate prefix has an independent dense suffix region, with a partial
final chunk when needed. Block capacity determines physical conflict boundaries
and may span rows; reopening preserves the stored geometry.

Tensor schemas encode `[dtype_path, shape]` for dense values and
`[dtype_path, shape, sparse_axis]` for sparse values. The optional zero-based axis
identifies the last sparse dimension; the final axis denotes scalar-sparse
storage. Sparse payloads retain ordered nonzero coordinate/value pairs regardless
of axis. This replaces the layout-string schema directly; callers and durable
records must use the current schema representation. Native storage metadata is
unchanged.

Complete dense wire payloads are `[[dtype_path, shape], values]`; sparse payloads
are `[[dtype_path, shape, sparse_axis], [[coordinate, value], ...]]`. Encoding
streams values and omits numerical zeros from sparse payloads. Omitting the axis
selects dense storage; no layout string is encoded. Typed payload tags determine
dtype on reopening, independently of casts.

Collection Tensor metadata requires a nonempty shape with positive dimensions and
at most 4096 axes. Typed construction, literals, decoding, transformed results,
schema extraction, and copy-layout preparation share this representability
boundary. Unsupported scalar, empty, or over-rank Tensor results return structured
errors; existing scalar GET and complete-reduction results remain Numbers. Native
fensor views retain their own capabilities. Validation does not materialize values
or change numerical policy.

Native literal constructors check supplied element bytes against the delegated
`materialized_tensor_bytes` allowance after allocation. This is not an aggregate
memory bound or a limit on streamed logical size. Streamed construction and copying
use fensor's storage-block capacity independently of this whole-value allowance.

Tensor POST handlers consume recognized parameters and reject leftover keys before
snapshot acquisition or numerical execution. Missing/invalid recognized arguments
retain their ordinary diagnostics, and optional null arguments retain their defaults.
The single-axis flip route accepts a scalar or singleton list and rejects other
list cardinalities. Parameter lookup borrows textual keys.

Tensor deltas replace complete bounded logical storage blocks, including explicit
all-zero replacements. Payload and occupied-block changes share the same delta
and lifecycle decision. A mutation interrupted after changing a pending delta
requires the caller to stop using the transaction and roll back before reuse.
Restoration constructs separate state and installs it only after success.
Selection mutations target the owner with explicit ranges, reserve compact
conservative logical-block intervals, stage only touched blocks, and merge their
patch into pending state. Disjoint patches cannot replace each other. A vacant
pending slot adopts its completed patch directly; an occupied slot merges touched
blocks outside the state lock. Full-owner fills stage complete bounded blocks;
selected fills retain geometric selection semantics without making a returned
view writable. Full-owner fill constructs exact-length replacement blocks from
validated geometry without reading old transactional payloads. Overwritten payloads
are not inspected; validation and integrity errors from visited native publication
storage remain unchanged. Assignment checks mutation generations across reservation
conversion; partial publication requires caller-owned rollback.

Fensor owns geometry, native storage, numerical operations, and bounded evaluation.
Supported backend/dtype combinations follow fensor and ha-ndarray. Native mean,
population standard deviation, and Euclidean norm return f64 for real inputs;
complex mean returns c64, while complex standard deviation and norm return f64.
Dense and sparse values use the same numerical kernels. Implicit zeros participate
in reductions as ordinary values. Pointwise operations that would produce a
nonzero implicit background require explicit lazy `as_dense` conversion. Tensor
values bind computed inputs to their transaction and
retain read permits until consumed or dropped. Deterministic visibility and
application of supplied values do not imply bitwise reproducibility of floating
point recomputation. Callers requiring exact numerical replay retain accepted
values under their durable history policy.

Native planning preserves consumer coordinates through supported transformed and
owned sources; collections continue supplying transaction-visible storage.
`copy_values` explicitly consumes every logical value through the same snapshot
and native construction, including implicit sparse zeros. Ordinary copying and
materialization do not select this operation automatically when indexed
traversal is unsupported.

Eligible sparse encoding, copying, and reductions traverse occupied blocks and
account for implicit zeros. Supported forward slices and permutations retain
indexed planning through owned expressions. Larger sparse selections with
unsupported mapping or ordering fail explicitly rather than scanning their whole
logical shape. Bounded point reads and explicit row-major reads remain available;
computed matrix/Fourier results retain their own logical evaluation. No operation
implicitly sorts or persists intermediate results.

Literal PUT dispatch consumes its inputs only on a matching constructor;
`Collection::from_put` returns `LiteralPut::Unmatched` with the original owned
key and value otherwise. Callers continue ordinary dispatch without copying them.

Native decoding, literals, and copying use unpublished construction owners. Dense
construction allocates files on first use; sparse construction assembles bounded
blocks and stores them in the destination's native BTree. Both consume input
once and publish metadata only after success. Decoders retain wire validation and
original decoding errors; native construction owns payload assembly. Shape and
coordinate decoding checks rank incrementally without trusting sequence-size hints;
extra fields are consumed and validated before rejection. This does not change Tensor
transaction visibility, touched-block semantics, or finalization. Delta copying
and finalization share the same bounded touched-block replacement traversal.

Tensor metadata describes one native storage format with no format-version field
or migration reader. Strict loading rejects malformed or extra metadata fields.
Physical page capacity does not change logical-block reservations or delta visibility. Payload adapters buffer
codec chunks with bounded working memory and flush before `FileSave` returns;
this does not add durability to commit or replace delegated `sync_all`.

Tensor assignment and streamed writes delegate request-to-block mapping to fensor.
Collection consumes bounded native update batches into unpublished deltas while
retaining responsibility for reservations, snapshot visibility, and generation
checks. Delta replacement traversal uses paginated touched IDs directly without
repeating membership lookups. Touched membership distinguishes explicit-zero
replacements from untouched blocks; native payload storage alone determines
whether a replacement contains values. Missing native payload rows yield zero
blocks, while only missing touched IDs permit falling through to older data.

Tensor occupied traversal owns bounded lookahead per visible transaction layer.
A flat frame stack merges these layers without nesting stream polls or drops.
It yields after bounded traversal work, including when native reads are ready;
the step count carries across emitted keys. Handoff is rechecked after yielding.
Pagination consumes that scan; expression consumers retain its lookahead across
output batches. A replacement suppresses every older occurrence of its touched
block, including explicit-zero replacements. Each layer retains its cursor and
discards prefetched workspace keys on materialization before resuming canonical
traversal. Native BTree owns touched-set insertion; collections publish payload
changes before inserting membership. Callers roll back interrupted publication.

Private unit-test implementations live under `tests/unit/` and shared fixtures
under `tests/common/`. Their existing Rust module names and targeted test commands
are unchanged; see [test ownership](CONTRIBUTING.md#test-ownership).

## Caller-owned failure handling

An error, cancellation, or unwind may leave a pending native mutation partially
applied. The caller must stop affected work, release its retained results and
permits, and roll back before reuse. Bulk requests require rollback of all their
staged work, including earlier successful rows. Collections do not remember an
interrupted future or automatically reject a subsequent commit. Input rejection
before mutation and failed unpublished construction leave existing state unchanged.

Failed or cancelled canonical materialization must not be retried on the same
owner. Discard that owner, its clones, and retained snapshots; recover using fresh
owners and caller-selected canonical state. Failed finalization retains committed
deltas and leaves the cutoff unchanged, but canonical contents may already have
changed. Rollback cannot repair them.

The server owns supervised execution and rollback, including cancellation. Failed
rollback or uncertain canonical materialization requires host termination and
caller-owned durable recovery evidence. Direct library callers must enforce the same cleanup
and safe-reuse obligations themselves. Collections return structured errors and
never terminate the process. Ordinary stream errors, EOF, and drop release sources
and permits; streams do not revoke previously acquired snapshots.

Tokio owns cooperative scheduling. Native storage owns local reads and mutations;
collections own transaction visibility and canonical
materialization handoff. Shared pending-delta installation prepares concrete
construction without awaiting under the state lock, then rechecks transaction
state before publication.

BTree and Table compose native index streams with one flat multiway merge.
Selection delegates to collate's bounded `MergeQueue` priority queue, with
logarithmic work per consumed entry. Native storage receives one immutable
`IndexOrder` configuration. The shared persistence boundary owns the default
Value ordering recipe: one `ValueCollator`, delegating numbers to
`NumberCollator` and recursing through tuples. Complex values with equal norms
compare by real and imaginary components, so unequal values occupy distinct
range positions. Range reservations retain the native owner's range comparator;
complete-key scans and overlays use its key comparator, which also accounts for
key length. The configuration itself is not a comparator, so each operation
selects its comparison role explicitly. B-tree owns complete-key composition;
b-table delegates selected-index row ordering to it. Table obtains its full-row
stream and overlay comparator together from native `Table::rows` with one index
selection, so comparison and traversal use the same column mapping and direction.
Native `Table::select` serves
projection and row-only scans without constructing an unused comparator.
Existing indexes whose order depends on the former equal-norm grouping must be
recreated; strict loading rejects incompatible order. There is no migration or
compatibility reader.

Each Table delta retains the pre-transaction row in deletes and the final row in
inserts. Repeated mutations preserve the original row, including deletion followed
by reinsertion. Every native index therefore carries the retraction needed when
an update moves a row outside a predicate. Scans merge those entries directly,
without probing newer layers for each returned row. Finalization deletes original
rows before inserting final rows. Point reads prefer pending insertions and then
tombstones. Point reads and first-write baseline lookups use one descending
visibility cursor, selecting a single shared delta under the state lock and
releasing the lock before native I/O. They do not clone the complete history or
probe the same layer twice. Absent point reads and first-write baseline lookups remain linear in
retained history. No key-location registry is maintained.

BTree retains an exact-key tombstone across reinsertion and restoration. An
existing insertion keeps its representation until deletion; a subsequent
insertion supplies the new representation. Finalization applies tombstones before
insertions, matching streamed visibility even for numerically equal values with
different floating-point bits. No original-row lookup is needed for set keys.

One iterative consumer avoids recursive polling or destruction through transaction
history. It delegates ready-source ordering to collate's shared `MergeQueue`;
Collection owns visibility, tombstones, and retained insertion representations.
Lookahead and retained read capabilities grow with visible history,
not the number of stored rows; Tokio supplies cooperative scheduling. Native
leaf batches use heap-backed outer storage so inline key capacity does not
multiply into large stream polling frames.

Table bulk updates and range deletion stage rows in separate caller-delegated
storage. They release the input streams before applying the staged rows to
pending state. Source failure leaves pending contents unchanged, while retaining
the transaction's reservation and unpublished cleanup obligations. Interrupted
application to pending state still requires rollback. This adds a temporary
workspace and native index roots per bulk operation rather than buffering rows
in memory.

Bulk staging retains operation-input rows alongside replacements. Composition
preserves any earlier transaction-original row before removing prior pending
insertions and applying the staged final insertions. Restoration installs a
complete committed-baseline-to-snapshot delta only after construction succeeds.
Restoration streams the committed baseline and the source through ordinary merged
readers into an empty replacement delta, rather than copying canonical nodes and
replaying versions separately. This rebuilds native indexes row by row and keeps
one visibility and retirement path. All input streams release before installation.

### Point-read visibility and errors

BTree and Table point lookups visit visible deltas newest first, checking inserts
before deletes within each delta. The first determining entry ends the lookup;
canonical storage is read only when no delta determines the result. Errors from
visited storage propagate, but shadowed older storage is not inspected. Ordered
scans and strict native validation retain their existing corruption checks.
Scan snapshots retain metadata proportional to visible history; point reads
retain one cursor and one selected delta. Native Table point operations acquire
only the primary index, including presence checks which do not copy row values.
