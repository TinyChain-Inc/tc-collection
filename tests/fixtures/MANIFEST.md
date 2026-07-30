# Table parity test fixtures

Small, deterministic datasets reused by the `Table` port test matrix in
[`TABLE_PARITY_PORT.md`](../../TABLE_PARITY_PORT.md) §7. Fixtures are CSV with a
header row; the first column(s) listed as the key in the spec are the primary
key. Values are intentionally tiny so tests stay fast.

## Schemas

### users.csv
- **Key:** `id` (Int)
- Values: `name` (String), `age` (Int), `city` (String)
- 5 rows. Use for: primary-key reads, `upsert`/`delete`, single-column range on
  `id`, `count`, `select`/`order`/`limit` views.

### orders.csv
- **Key:** `order_id` (Int)
- Values: `user_id` (Int), `total` (Float), `status` (String)
- 8 rows. Use for: multi-key visibility, `select`/`order`/`limit`, range slice
  on `order_id` or `total`, auxiliary index on `user_id` for cross-row lookup.

### sensor_readings.csv
- **Key:** `sensor_id` (Int), `ts` (Int) — composite primary key
- Values: `temp` (Float), `hum` (Float)
- 12 rows. Use for: composite-key reads, auxiliary-index coverage, streaming /
  scan performance baselines, deterministic merge-order checks.

## Loading in tests

Implementation issues should load these via `include_str!` and parse with the
crate's value codec, or read from the path at test time. No fixture here implies
a schema or behavior beyond what `TABLE_PARITY_PORT.md` specifies.
