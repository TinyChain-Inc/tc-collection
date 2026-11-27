# tc-collection roadmap

This file contains only unimplemented collection work. Current behavior is
specified by the crate README and transactional collection contract.

## Execution costs

Evaluate absent point reads and Table first-write baseline lookups under workloads
with long retained histories. These searches remain linear in visible layers;
any additional summary must justify its retained-memory and lifecycle costs with
measurements. Do not hide eager query materialization or truncate visible history.

Logical-length hashing remains linear with the existing digest definition, even
after batching and exact zero-digest reuse. A retained-content sparse digest would
require an explicit semantic change; it is not an implementation optimization.

Further sparse traversal optimizations belong to fensor's owning interfaces.
Reordered mappings and aggregate output traversal remain bounded but may visit
logical cardinality; they introduce no hidden sorting or materialization.
