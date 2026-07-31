//! Public API route handlers for a transactional [`PersistentTable`].
//!
//! Ports the v1 `table/public.rs` routing logic onto the v2 `tc-ir`
//! `Route`/`HandleGet`/`HandlePut`/`HandlePost`/`HandleDelete` traits.
//!
//! ## Module layout
//!
//! - [`handler`] — `TableRoute` enum and verb trait implementations
//! - [`selector`] — `KeyOrRange` and `cast_into_range` selector parsing
//! - [`response`] — `TableResponse` enum
//! - [`static_route`] — `TableStatic`/`TableStaticRoute` for `create`/`copy_from`
//! - [`router`] — `TableRouter` (per-instance) and `TableStatic` (class-level)
//!
//! ## Design
//!
//! The v2 `Route` trait returns `Option<&Self::Handler>` (a borrowed handler),
//! unlike v1 which returned `Box<dyn Handler>`.  [`TableRouter`] therefore
//! pre-constructs all route handler instances (one per known sub-route) when
//! it is built from a [`PersistentTable`].  Because `PersistentTable` is
//! `Arc`-based and cheap to clone, this is inexpensive.
//!
//! All handlers use [`Scalar`] as the IR request envelope and [`TableResponse`]
//! as the response type, per `AGENTS.md` ("express handlers in terms of the
//! shared IR envelopes").

mod handler;
mod response;
mod selector;
mod static_route;

pub use handler::{scalar_to_value, TableRoute};
pub use response::TableResponse;
pub use static_route::TableStaticRoute;

use std::fmt;

use freqfs::DirLock;
use pathlink::PathSegment;
use tc_ir::Route;

use super::file::PersistentTable;
use crate::btree::PersistentFile;

// ─── TableRouter (per-instance) ────────────────────────────────────────

/// Router for a single [`PersistentTable`] instance.
///
/// Pre-constructs all sub-route handlers when built from a table.  Implements
/// [`Route`] to resolve a path to the appropriate [`TableRoute`] handler.
///
/// Route table (parity port §4):
///
/// | Path | Handler |
/// |------|---------|
/// | `[]` | `Table` (read/slice/upsert/update/truncate/delete) |
/// | `["columns"]` | `Columns` |
/// | `["contains"]` | `Contains` |
/// | `["count"]` | `Count` |
/// | `["key_columns"]` | `KeyColumns` |
/// | `["key_names"]` | `KeyNames` |
/// | `["limit"]` | `Limit` |
/// | `["order"]` | `Order` |
/// | `["select"]` | `Select` |
pub struct TableRouter {
    pub(crate) table: TableRoute,
    pub(crate) columns: TableRoute,
    pub(crate) contains: TableRoute,
    pub(crate) count: TableRoute,
    pub(crate) key_columns: TableRoute,
    pub(crate) key_names: TableRoute,
    pub(crate) limit: TableRoute,
    pub(crate) order: TableRoute,
    pub(crate) select: TableRoute,
}

impl TableRouter {
    /// Build a router from a [`PersistentTable`].
    ///
    /// The table is cloned (cheaply — it is `Arc`-based) into each handler.
    pub fn new(table: PersistentTable) -> Self {
        Self {
            table: TableRoute::Table(table.clone()),
            columns: TableRoute::Columns(table.clone()),
            contains: TableRoute::Contains(table.clone()),
            count: TableRoute::Count(table.clone()),
            key_columns: TableRoute::KeyColumns(table.clone()),
            key_names: TableRoute::KeyNames(table.clone()),
            limit: TableRoute::Limit(table.clone()),
            order: TableRoute::Order(table.clone()),
            select: TableRoute::Select(table),
        }
    }
}

impl fmt::Debug for TableRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableRouter")
            .field("table", &self.table)
            .finish()
    }
}

impl Route for TableRouter {
    type Handler = TableRoute;

    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<&'a Self::Handler> {
        if path.is_empty() {
            Some(&self.table)
        } else if path.len() == 1 {
            match path[0].as_str() {
                "columns" => Some(&self.columns),
                "contains" => Some(&self.contains),
                "count" => Some(&self.count),
                "key_columns" => Some(&self.key_columns),
                "key_names" => Some(&self.key_names),
                "limit" => Some(&self.limit),
                "order" => Some(&self.order),
                "select" => Some(&self.select),
                _ => None,
            }
        } else {
            None
        }
    }
}

// ─── TableStatic (class-level: create / copy_from) ─────────────────────

/// Static router for the table class routes (`create`, `copy_from`).
///
/// Holds the root directory under which new tables are created.
///
/// Route table:
/// | Path | Handler |
/// |------|---------|
/// | `[]` | `Create` (GET: create a new table from a schema) |
/// | `["copy_from"]` | `CopyFrom` (POST: create + copy rows) |
pub struct TableStatic {
    create: TableStaticRoute,
    copy_from: TableStaticRoute,
}

impl TableStatic {
    /// Build a static router with the given root directory for new tables.
    pub fn new(root: DirLock<PersistentFile>) -> Self {
        Self {
            create: TableStaticRoute::Create { root: root.clone() },
            copy_from: TableStaticRoute::CopyFrom { root },
        }
    }
}

impl fmt::Debug for TableStatic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableStatic")
            .field("create", &self.create)
            .field("copy_from", &self.copy_from)
            .finish()
    }
}

impl Route for TableStatic {
    type Handler = TableStaticRoute;

    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<&'a Self::Handler> {
        if path.is_empty() {
            Some(&self.create)
        } else if path.len() == 1 && path[0].as_str() == "copy_from" {
            Some(&self.copy_from)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;
