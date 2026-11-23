# Standalone CI

The repository workflow checks formatting, then runs strict Clippy and library
tests with default and complex support. Clippy enables `benchmarks` to compile
the timing targets; CI does not execute benchmarks. Doctests run once with
complex support. Build concurrency, timeouts, cached artifacts, and cancellation
of superseded runs keep validation bounded.

Cargo uses sibling path dependencies. `dependencies.json` pins their Git
repositories and exact commits, including transitive and optional path packages.
`bootstrap.py` populates the layout without parent-workspace scripts. It refuses
existing destinations, so it cannot reset or overwrite a developer's checkouts.
`tc-error` is an ordinary sibling repository. No parent contracts or CI scripts
are imported.

Private dependencies require the repository secret `DEPENDENCY_READ_TOKEN` with
read-only contents access to those repositories. CI configures GitHub CLI's Git
credential helper using that token, falling back to the current repository token
for public dependencies. Local bootstrap uses ordinary Git credentials. Fork pull
requests do not receive repository secrets; private dependency checks require an
authorized run rather than skipping validation.

To reproduce CI, clone this repository into a fresh directory named
`tc-collection` inside an otherwise empty parent, then run from the checkout:

```bash
python3 ci/bootstrap.py
cargo fmt -- --check
cargo clippy --locked --all-targets --features benchmarks -- -D warnings
cargo test --locked --lib
cargo clippy --locked --all-targets --features benchmarks,complex -- -D warnings
cargo test --locked --lib --features complex
cargo test --locked --doc --features complex
```

Use stable Rust with rustfmt and Clippy, Python 3, and Git. Match the workflow's
Cargo environment when sharing build artifacts. The matrix uses the CPU backend;
OpenCL deployment validation belongs to its owning numerical dependency.

When updating a dependency, publish its compatible commit and update its pin.
Uncommitted sibling changes are never included. Keep the path
dependency closure and the committed Cargo lockfile in sync, and validate from
the fresh layout: a passing build against dirty siblings does not certify CI.

Stable rustfmt does not enforce every import-grouping or logical-spacing rule;
those remain review responsibilities rather than source-pattern tests.
