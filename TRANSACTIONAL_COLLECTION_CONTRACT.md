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

Native BTree construction requires an explicit schema.

Value-backed schemas represent dtype references as canonical hostless Links.
Their conversion delegates to `tc-value`'s type-reference conversion; collection
owners do not maintain a separate interpretation of URI-shaped scalar text.

`btree::BTree` represents a persistent owner or a read-only slice. Its native
schema owns the complete immutable column definitions; aliases and slices share
that schema. Construction validates columns once, before native allocation.
Terminal encoding copies its independent header while retaining guarded key
streaming. Native aliases retained before handoff are not revoked; callers must
relinquish them as described above.

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
`Table::slice` method. Unsupported combined predicates still require a
supporting native index. Limits and projections apply equally to row streams,
counts and emptiness, including lifecycle validation for a zero limit.
Table selections retain the original key columns first, in their original order,
then emit unique value columns in caller order. Duplicate columns, missing keys,
and layouts incompatible with this schema reject before traversal. A prepared
projection supplies both output metadata and field positions. Raw `Rows::select`
retains its general ordered, repeated-column behavior. Terminal operations consume
the composed row stream.

Table view counts and emptiness consume the same guarded row stream as iteration,
including order support checks for empty and zero-limit views. Predicates consume
owned rows through the transactional read stream without collecting the result
or copying it wholesale.

Table views compose through their source Table. Repeated limits take the smaller
cap; repeated projections collapse without restoring hidden columns. Projection
commutes through a limit, preserving the cap and keeping the limit outermost.
There is at most one projection and one limit. Slicing or ordering a projection
preserves its visible schema and accepts only visible columns. Slicing or ordering
a limited view is unsupported, since moving either operation before the limit
would change its meaning. Point lookup remains available only on persistent owners
and unrestricted slices. All view transforms preserve read-only ownership.

Public row limits accept finite, nonnegative integral values from integer and floating-point types
which fit `u64`, including zero. Other values reject before storage access instead
of being truncated or wrapped.

Table copying rejects duplicate semantic primary-key identities and preserves
complete canonical row order. Construction retains at most the previous owned row
and the next row, then moves completed rows into native storage. Numeric identity
is preserved across equal mixed representations without collecting a keyset.

Replacements delegate to native upsert rather than
deleting an existing insert row before validation and replacement.

## Lifecycle

Logical BTree and Table values own lifecycle eligibility and concrete delegation;
Collection only dispatches to the selected value. Persistent owners delegate to
the shared lifecycle, and read-only views reject lifecycle calls. Decoded Table
literals initialize the ordinary base directly while unpublished, without pending
deltas. Once published, their reads and mutations use the same version visibility,
rollback, and finalization rules as every other Table base.

Every transactional collection implements the shared fallible `Transact`
lifecycle:

- pending mutation is visible only to its transaction;
- commit durably publishes that version, then makes it eligible for ordered visibility;
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
implementation terminates the host. Interrupted durable publication marks the
live owner unusable until it is reopened. Collections choose no distributed
decision or repair policy.

Callers serialize lifecycle decisions for a collection. Before commit, they
prevent new mutations in that transaction and finish its mutating operations.
Immutable result snapshots and their streams may survive commit: publishing a
delta does not change its payload. Response projection therefore need not drain
or materialize those results before committing. Before rollback, callers finish
or cancel the transaction's operations and release its streams and guards.
Before finalization, they do so for transactions through the cutoff. Collections
do not check this precondition by acquiring new semaphore reservations; unrelated
later transactions may continue.

Committed native versions are immutable, and snapshots pin their sources. Finalization can
publish a new compacted descriptor while later readers continue using their
pinned inputs. Reclamation waits for the last version lease; no reader retries
against a rewritten canonical base.

BTree, Table, and Tensor commit synchronize the pending native delta, atomically
publish its original-ID receipt, move the delta into committed visibility, and
release reservations. Successful commit is independently crash-durable. Read-only
decisions need no files; their in-memory decision prevents late writes until
finalization. Failed durable publication requires reopening, including when
cancellation leaves the publication outcome uncertain. Rollback discards only
pending work and reclaims unpinned staging storage.

Write admission checks lifecycle eligibility before registering a reservation.
A rejected write to a committed or finalized transaction must not obstruct later
reads. Publication rechecks eligibility after asynchronous construction; callers
still seal and finish mutations before a lifecycle decision.

Finalization compacts covered immutable deltas and atomically replaces the
collection descriptor before pruning state and releasing reservations.
Published inputs remain untouched until that switch is durable.
Failure or cancellation leaves the live owner unusable; reopening selects the
complete old or new descriptor and resumes ordinary lifecycle processing.
Collection errors retain native storage classification. The transaction semaphore
reports only reservation and transaction-ordering failures.

## Visibility and ordering

New reads at or below the finalized cutoff return `Outdated`: canonical storage
may already include newer values. Read admission checks the cutoff before acquiring
a range permit, then validates and captures the read state together under the
bookkeeping lock. The returned permit and read state form one successful admission;
a failed capture releases its permit. Publication revalidates eligibility under
the state lock after asynchronous preparation. These are production validity
checks, with no additional retained frontier or semaphore history.

Native BTree and Table root GET handlers return a native view for a null key.
`IntoView` acquires transaction consistency; encoding remains separate. BTree delegates
non-null slice arguments to its existing native slice behavior; no host-specific
snapshot endpoint is required. Native creation and strict loading both delegate
through `CollectionSchema` to the concrete owner. Creation awaits native cache
admission, including empty index roots, before publishing the constructed owner.
Strict loading never creates those roots.

`CopyFrom::copy_from` is the public operation for copying a transaction-visible
collection into delegated native storage. It validates that the view preserves
its schema and canonical key order before allocating the destination.
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

Collections reject a new mutation with `Conflict` once a newer mutation is
durably visible on the same owner, including mutations to disjoint Tensor blocks.
The check and reservation share one short bookkeeping scope. Durable empty
replacements participate in this order.
Tensor assignment also checks ordering when capturing the pending generation,
before it reads source payloads or stages a replacement.
Read-only decisions release reservations without establishing a mutation boundary.
Previously admitted disjoint mutations can finish and commit. On an unpublished
owner, recovery can replay the exact accepted transaction whose native receipt
is missing, even when a newer receipt exists.

Iteration, ranges, projections, limits, updates, and truncation preserve this
same snapshot. Large result and mutation sets remain pull-driven rather than
being collected into unbounded intermediate vectors.

## Locking and backpressure

Diagnostic formatting never acquires transaction-state locks. It may report
immutable metadata but does not inspect mutable version counts or frontiers.

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
native storage guards are released before response encoding.
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

Creation requires empty delegated storage. Durability assumes the caller has
durably established its ancestor hierarchy. The native owner synchronizes its
root and contents; the enclosing directory owner publishes new ancestor names.
Request-local workspace storage remains subject to transaction cleanup.

Each native owner contains an immutable
initial `base`, a `versions` directory, a `commits` directory, and one atomically
replaced `collection` descriptor. Construction synchronizes the base before
publishing a valid descriptor. Pending deltas are staged directly in uniquely
named version directories beneath this delegated owner. Commit synchronizes that
directory before publishing a receipt keyed by the unchanged protocol TxnId.
Physical version names have no transaction-ordering meaning.

Loading strictly validates the descriptor, required native structures and indexes,
referenced version identities, and their recorded sizes. It reconstructs the
finalized frontier and committed visibility directly. Only after successful
validation may it reclaim unreachable versions and obsolete receipts. Old layouts
are rejected; recreate development data instead of migrating it.

Finalization keeps chronological compacted runs with strictly decreasing size
classes, measured by native delta entries or covered Tensor blocks. Adjacent
runs of the same or increasing class merge through the concrete collection's
ordinary delta traversal. At most 64 compacted runs remain, independently of
transaction count. Temporary merge outputs are reclaimed as they are superseded;
only surviving outputs are synchronized before the descriptor switch. Large
merges stream native rows. A complete BTree prefix discards deletion markers
for keys absent from its initial base, so cancelled insertions do not accumulate
historical keys. Existing committed inputs and retained snapshots
remain available throughout. This bounds finalized read layers without copying
the entire base at every cutoff; it trades amortized delta rewriting for those
read bounds.

`Collection::recover(id, replay)` is restricted to an unpublished, exclusively
owned recovery instance. It skips replay when that ID is already committed or
covered by the native frontier; otherwise it invokes the supplied callback once
and durably commits the result. Chain validates its required request history and
provides the original-ID capability inside that callback. Collections own no
request WAL, canonical-history selection, or replica-repair policy.

Native fensor clones share live nontransactional storage. Collection freezes
published native versions and owns visibility and leases; fensor does not publish
transaction decisions.

Tensor numerical reads plan one bounded request in canonical block geometry, resolve
its visible sources newest-first, then delegate payload consumption to fensor's
native block reader in request order. A patch visits consecutive unresolved IDs
under one zero-index guard, then releases it before checking payload occupancy.
Single-block reads borrow the first visible payload as part of the lookup.
Scalar reads reuse the block address that determined their reservation.
Native scans can decode containing nodes during source selection,
including rows outside the selected range. Selected payloads retain exact
canonical-length validation. Errors propagate through source selection and payload
consumption. Immutable version leases keep selection valid through finalization.

Tensor deltas delegate replacement payloads and occupancy to fensor's
`SparseBlocks` with canonical geometry. A partial version contains `payload/`
and a `zeros/` BTree set of explicit zero overrides. A logical block ID can occur
in only one of these trees. A whole version stores its
payloads in `replacement/` and covers every block. Dense storage chunks one flat
row-major region. A sparse coordinate
prefix selects a dense subtensor, which can span several storage blocks.
The last block of each subtensor may be short. Native BTree rows omit blocks
represented by implicit zero. Stored lengths are validated exactly.
A zero override or an absent block in a whole version supplies zeros and
suppresses older contents. Absence from both trees in a partial version
delegates to the next older source.
Strict reopening requires one exact version layout and checks the version
receipt against the coverage count. Partial versions validate both trees and
count the disjoint union of their keys. Whole versions derive their coverage
count from geometry.
Rows in the same leaf node still have independent block reservations
and delta visibility. A lightly occupied block retains its full dense
payload rather than choosing another representation. Interrupted pending mutation
requires caller-owned rollback; uncertain durable publication requires reopening.

Each delta has one shared lease for native handles and metadata, retained by
pending/committed state and independently acquired readers. Assignment stamps
retain a separate generation token without retaining the allocation. Snapshots
share read permits only when one exists; unreserved internal snapshots allocate
no permit owner. Visible-history containers and bounded traversal metadata remain
owned allocations. Reclamation never deletes a leased version. Source guards
release before their lease. A later lifecycle operation or reopening reclaims
versions whose last reader has finished.

Host workspace removal follows successful recursive finalization. Persistent
collection deltas reside under their owner, so workspace removal does not affect
retained native readers. Fresh literals and independent copies still receive
caller-delegated workspace roots. Collections expose no separate sync operation:
construction, commit, and finalization own their durability barriers.

BTree, Table, and Tensor share private version bookkeeping containing only pending
versions, committed decisions, compacted runs, publication health, and the finalized frontier. Canonical native handles
and semaphores belong to each concrete collection outside the version-state lock.
Immutable schema inspection and durability access never acquire that lock. Native
handles are shared by Arc, so cloning a collection does not clone its index maps.
Scoped operations borrow native handles; retained snapshots clone their Arc.

The shared bookkeeping implements visibility and lifecycle ordering once. Concrete
collections supply scoped callbacks for delta construction, immutable merging,
and reservation release, all outside the bookkeeping lock. The lock and pending
maps stay private to persistence. Tensor can synchronously inspect or compare a
pending generation under that lock; native merging runs after unlocking. Successful
duplicate decisions still release reservations, while stale finalization invokes
neither application nor release. Superseded deltas are dropped outside
the state lock. Concrete collections retain reservation admission, creation,
loading, owner-local staging, hashing, copying and restoration. Shared native
publication handles descriptor and receipt I/O without traversing payloads.

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
policy is independent of semantic identity. Native schemas convert to and from
a collection class path and a Value for strict reopening. Collection identity
combines class and semantic schema with ordered contents using SHA-256. It
includes column and index definitions and excludes physical layout. Sparse Tensor
identity hashes ordered nonzero entries. Dense Tensor identity and native capture
integrity hash bounded row-major value batches, preserving the signs of real and
complex zero components. Native capture integrity also includes the reopening
descriptor. All hashing operates on native values.

The caller coordinates publication and retention. `Restore::restore` validates kind and semantic schema, builds
insert/delete deltas in separate owner-local staging storage, and installs the pending
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
names, dtypes, and optional size metadata. Tensor restoration requires dtype,
shape, and logical density to agree. The destination retains its native geometry.
Tensor restoration delegates transaction-visible block reads and assembly in the
destination geometry to fensor. Dense reads preserve signed zeros and sparse
reads visit occupied source blocks. The transactional owner then records
complete replacement membership in sorted order. This includes implicit zeros
so a commit with an older transaction ID cannot restore a value in those blocks
later.

Native trees mutate individual blocks while constructing an unpublished delta.
Collection publishes that native delta immutably rather than changing a reachable
tree in place. Native copying and loading retain corruption checks. Development fixtures from earlier
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

Native sparse nodes use `b-tree` framing and fensor's bounded typed row codec.
See the [fensor native storage contract](https://github.com/TinyChain-Inc/fensor/blob/main/DESIGN.md#native-storage-and-sources).
Collection file adapters own dtype tags and byte-format selection, and delegate
node payloads directly. The legacy packed-byte sparse node tags are retired.
Stores written with those tags require recreation from logical Tensor values.
Public Tensor encoding streams logical coordinates and values through its
transaction-consistent view; it exposes no native storage fields.
fensor supplies ordered sparse contents from indexed candidates or from bounded
logical requests. Both use the same stream representation for view preparation.

`Tensor` directly enumerates its numerical dtypes. Each variant contains an
opaque `TensorAccess` selecting a transactional base, native geometric view, or
transaction-bound expression through one private representation. The base owns
storage, reservations, mutation, copying, restoration, and lifecycle application.
The public Tensor methods select the
dtype and delegate to that base; views reject mutations and lifecycle calls.
Standard typed-to-Tensor conversions wrap native operation results; fensor's
associated output types determine their numerical dtype.

Read-only views retain native fensor geometry or a fensor expression bound to its
original transaction ID. Their source handles and binding are private: matching
or cloning the public payload cannot recover write authority or rebind an
expression. Geometry and numerical execution delegate to fensor. Transforming a
view preserves read-only ownership even when the resulting geometry is an
identity. Explicit copying constructs an independent base.

The base owns point writes, whole-tensor writes, and writes to a selection in its
own coordinates. Write ranges accept fixed coordinates and forward stepped spans.
The base validates the range before reserving blocks or staging updates.

Native geometry uses compact row-major chunks of at most 4,096 elements. Each
sparse-coordinate prefix has an independent dense suffix region, with a partial
final chunk when needed. Block capacity determines physical conflict boundaries
and may span rows; reopening preserves the stored geometry.

Complete dense wire payloads are `[[dtype_path, shape], values]`; sparse payloads
are `[[dtype_path, shape], [[coordinate, value], ...]]`. Literal PUT uses the
same pair of fields. A sparse entry starts with a coordinate sequence, while a
complex number starts with a numeric component. An empty contents array denotes
an all-zero sparse Tensor. Sparse entries are ordered and unique, and absent
coordinates are zero. Sparse decoding and literals use the trailing axis.
Both input paths require canonical dtype paths, nonnegative integer dimensions
and coordinates, and numeric values. Complex pairs require two real numeric
components. The cast route uses the same canonical dtype references.
Replica comparison hashes the logical schema and ordered contents, including
only nonzero entries for sparse values.

Native reopening descriptors retain the selected zero-based sparse axis alongside
dtype and shape. Service declarations may supply that axis as an explicit storage
setting. The `"sparse"` tag in a declaration selects the trailing axis.

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
pending slot adopts its completed patch directly; an occupied slot merges changed
blocks outside the state lock. Full-owner fills and assignments stage a complete
replacement and install it as pending in one swap. Restore and full assignments
stream native blocks or pack ordered sparse entries where applicable. Fensor
appends those ordered blocks to the unpublished version in bounded batches. A whole
fill whose value has no stored payload stages an empty version. Dense signed
zeros that differ from the canonical zero retain payloads. Subsequent writes
within the same pending version preserve its whole coverage. Selected fills map
bounded constant batches directly into the owner's blocks. Batches that replace
an entire block need no values from older versions.
Full-owner fill constructs its payload from validated geometry and propagates
integrity errors from the storage it visits during publication. Assignment claims
its captured mutation generation under the publication lock after reservation
conversion and before applying any staged blocks.
Partial publication requires caller-owned rollback.

Fensor owns geometry, native storage, numerical operations, and bounded evaluation.
Supported backend/dtype combinations follow fensor and ha-ndarray. Native mean,
population standard deviation, and Euclidean norm return f64 for real inputs;
complex mean returns c64, while complex standard deviation and norm return f64.
Dense and sparse values use the same numerical kernels. Sparse zero elements
have the dtype's canonical zero, which participates in reductions as an ordinary
value. Dense elements retain signed zeros. Pointwise operations that would produce a
nonzero implicit background require explicit lazy `as_dense` conversion. Tensor
values bind computed inputs to their transaction and
retain read permits until consumed or dropped. Deterministic visibility and
application of supplied values do not imply bitwise reproducibility of floating
point recomputation. Callers requiring exact numerical replay retain accepted
values under their durable history policy.

Native planning preserves consumer coordinates through supported transformed and
owned sources; collections continue supplying transaction-visible storage.

Eligible sparse encoding, copying, and reductions traverse occupied blocks and
account for implicit zeros. Supported forward slices and permutations retain
indexed planning through owned expressions. Other mappings and computed
matrix or Fourier results stream logical requests in bounded batches.

Literal PUT dispatch consumes its inputs only on a matching constructor;
`Collection::from_put` returns `LiteralPut::Unmatched` with the original owned
key and value otherwise. Callers continue ordinary dispatch without copying them.

Tensor inputs use checked dtype conversion before storage or mutation. Sparse
literal and wire entries must remain nonzero in the declared dtype.
Native decoding, literals, and copying use unpublished construction owners.
Dense construction allocates files on first use; sparse construction assembles bounded
blocks and stores them in the destination's native BTree. Both consume input
once and publish metadata only after success. Decoders retain wire validation and
original decoding errors; native construction owns payload assembly. Shape and
coordinate decoding checks rank incrementally without trusting sequence-size hints;
extra fields are consumed and validated before rejection. This does not change Tensor
transaction visibility, block coverage, or finalization. Delta copying and
finalization share the same bounded replacement traversal.

Tensor metadata describes one native storage format with no format-version field
or migration reader. Strict loading rejects malformed or extra metadata fields.
Payload adapters buffer
codec chunks with bounded working memory and flush before `FileSave` returns;
the transactional owner supplies durability through construction and lifecycle.

Tensor assignment and streamed writes delegate request-to-block mapping to fensor.
Collection consumes bounded native update batches into unpublished deltas while
retaining responsibility for reservations, snapshot visibility, and generation
checks. Partial replacement traversal merges ordered zero and occupied IDs
using bounded batches. Whole replacement traversal visits only stored payloads.
Patch application reads stored payloads and removes a previous payload when the
replacement supplies zeros. Compaction validates stored replacements from both
inputs and borrows shadowed payloads for validation. A newer whole version
absorbs an older run by retaining its committed allocation after validation.
Merges with a newer partial version write to fresh storage. A partial version
over a whole version updates those contents and preserves whole coverage. Two
partial versions retain the union of their replacements, with newer replacements
taking precedence.

Tensor occupied traversal starts at the newest visible whole version or the
canonical base. It uses the same flat, cooperative merge as BTree and Table.
Each patch contributes an ordered stream of zero keys to remove and an ordered
stream of occupied keys to add. Stream refills check errors in input order, and
a visited corrupt zero index can fail the scan even when its keys suppress
nothing. Bounded lookahead and immutable version leases survive across output
batches. Patch writes update the payload first, then insert or remove the
corresponding zero marker based on the native storage result. Callers roll back
interrupted publication.

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

Failed or cancelled durable publication must not be retried on the same live
owner. Discard the owner and its clones, then strictly reopen its native
descriptor and receipts before replaying any required request history. Either
complete publication is recoverable; rollback cannot reverse an accepted receipt.

The server owns supervised execution and rollback, including cancellation. Failed
rollback or uncertain publication requires host termination and durable recovery.
Direct library callers enforce the same cleanup and safe-reuse obligations.
Collections return structured errors and never terminate the process. Ordinary
stream errors, EOF, and drop release sources and permits; streams do not revoke
previously acquired snapshots.

Tokio owns cooperative scheduling. Native storage owns local reads and mutations;
collections own transaction visibility and immutable version publication. Shared
pending-delta installation prepares concrete construction without awaiting under
the state lock, then rechecks transaction state before publication.

BTree and Table compose native index streams with one flat multiway merge.
Selection delegates to collate's bounded `MergeQueue` priority queue, with
logarithmic work per consumed entry. Native storage receives one `ValueCollator`,
delegating numbers to `NumberCollator` and recursing through tuples. Complex
values with equal norms compare by real and imaginary components, so unequal values occupy distinct
range positions. Range reservations retain the native owner's `RangeCollator`;
complete-key scans and overlays use its explicit `cmp_keys` or `cmp_key_iter`
methods, which also account for key length. B-tree owns complete-key composition;
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
releasing the lock before native I/O. Without a concurrent cutoff advance, they
visit each eligible layer at most once. A cutoff advance during a lookup can
replace the remaining layers, so a miss resumes selection from the new frontier.
Absent point reads and first-write baseline lookups remain linear in retained
history when the frontier is stable.

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

Table bulk updates and range deletion stage rows in separate owner-local
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
one visibility path. All input streams release before installation.

### Point-read visibility and errors

BTree and Table point lookups visit visible deltas newest first, checking inserts
before deletes within each delta. The first determining entry ends the lookup;
canonical storage is read only when no delta determines the result. Errors from
visited storage propagate, but shadowed older storage is not inspected. Ordered
scans and strict native validation retain their existing corruption checks.
Scan snapshots retain one native handle per visible delta in visibility order.
Point reads retain one selected delta at a time. Native Table point operations acquire
only the primary index, including presence checks which do not copy row values.
