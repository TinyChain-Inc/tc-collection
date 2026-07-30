# Table parity test fixtures

Small, deterministic datasets reused by the `Table` port test matrix in
[`TABLE_PARITY_PORT.md`](../../TABLE_PARITY_PORT.md) §7. Fixtures are CSV with a
header row; the key column(s) are noted below. Values are intentionally tiny so
tests stay fast. Column names are generic and structural — they describe the
table shape each fixture exercises, not any application domain.

## Schemas

### simple_pk.csv
- **Key:** `id` (Int)
- Values: `label` (String), `score` (Int), `tag` (String)
- 5 rows. Use for: primary-key reads, `upsert`/`delete`, single-column range on
  `id`, `count`, `select`/`order`/`limit` views.

### index_and_range.csv
- **Key:** `id` (Int)
- Values: `ref_id` (Int), `amount` (Float), `state` (String)
- 8 rows. Use for: multi-key visibility, `select`/`order`/`limit`, range slice
  on `id` or `amount`, auxiliary index on `ref_id` for cross-row lookup.

### composite_key.csv
- **Key:** `part_a` (Int), `part_b` (Int) — composite primary key
- Values: `val1` (Float), `val2` (Float)
- 12 rows. Use for: composite-key reads, auxiliary-index coverage, streaming /
  scan performance baselines, deterministic merge-order checks.

## Loading in tests

Implementation issues should load these via `include_str!` and parse with the
crate's value codec, or read from the path at test time. No fixture here implies
a schema or behavior beyond what `TABLE_PARITY_PORT.md` specifies.
