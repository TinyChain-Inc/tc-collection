# Contributing to tc-collection

Read this repository's [invariants](AGENTS.md) and
[transactional contract](TRANSACTIONAL_COLLECTION_CONTRACT.md). The parent
workspace
[contributor guide](https://github.com/TinyChain-Inc/tcv2/blob/main/CONTRIBUTING.md)
is non-normative integration context for a superproject checkout.

Before opening a pull request, run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

Collection changes require focused streaming, transaction visibility, conflict,
finalization, cancellation, and lock-order tests as applicable. Public hosting,
durable history, and reconciliation are outside this repository.
