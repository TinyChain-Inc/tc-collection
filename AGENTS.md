# tc-collection Agent Notes

`tc-collection` defines shard-aware collection behaviors. Keep routing logic shard-
local and lean, and push cross-host orchestration to client libraries.

## Routing and semantics

- Collections shard data across blocks; hosts execute shard-local handlers. Do not add
  cross-host routing logic here—client runtimes own cluster-aware dispatch.
- Keep v1 collection semantics (batching, auth headers, error envelopes) visible in
  docs and examples so clients can migrate without surprises.
- Express new handlers in terms of the shared IR envelopes; avoid adapter-specific
  request/response shapes so Python and WASM clients can reuse the same contracts.

## Documentation and validation

- Update the crate `README` and `ROADMAP` when adjusting placement rules, shard
  manifests, or health snapshot expectations. Design changes should land with concrete
  migration notes, not fallbacks.
- Add focused tests or examples alongside new routing behaviors to avoid introducing
  special-case pathways.
