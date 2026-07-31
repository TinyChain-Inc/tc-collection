//! Static route handlers for table construction: `create` and `copy_from`.
//!
//! Ports v1 `Static` in `table/public.rs`:
//! - `GET /state/collection/table` → `CreateHandler` (create a new table)
//! - `POST /state/collection/table/copy_from` → `CopyHandler`

use std::fmt;
use std::future::Future;
use std::pin::Pin;

use freqfs::DirLock;
use tc_error::{bad_request, TCError, TCResult};
use tc_ir::{
    HandleGet, HandlePost, Handler, Map, Method, Scalar, Transaction, TxnId,
};
use tc_value::Value;

use super::super::file::PersistentTable;
use super::super::schema::TableSchema;
use super::handler::scalar_to_value;
use super::response::TableResponse;
use crate::btree::PersistentFile;

type GetFut<'a> = Pin<Box<dyn Future<Output = TCResult<TableResponse>> + Send + 'a>>;
type PostFut<'a> = Pin<Box<dyn Future<Output = TCResult<TableResponse>> + Send + 'a>>;

/// Static route handler for table construction routes.
#[derive(Clone)]
pub enum TableStaticRoute {
    /// Create a new table from a schema value.
    Create {
        root: DirLock<PersistentFile>,
    },
    /// `copy_from` — create a new table from a schema and copy rows from a
    /// source table. The source is provided as an inline row tuple or as a
    /// reference that the host resolves before dispatching.
    CopyFrom {
        root: DirLock<PersistentFile>,
    },
}

impl fmt::Debug for TableStaticRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Create { .. } => f.write_str("Create"),
            Self::CopyFrom { .. } => f.write_str("CopyFrom"),
        }
    }
}

impl<T: Transaction + ?Sized> HandleGet<T> for TableStaticRoute {
    type Request = Scalar;
    type RequestContext = ();
    type Response = TableResponse;
    type Error = TCError;
    type Fut<'a>
        = GetFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn get<'a>(&'a self, txn: &'a T, request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::Create { root } => {
                let root = root.clone();
                let txn_id = txn.id();
                let value = scalar_to_value(request)?;
                Ok(Box::pin(async move {
                    let schema = TableSchema::try_from_value(value)?;
                    let table = create_table(&root, txn_id, schema).await?;
                    Ok(TableResponse::Table(table))
                }))
            }
            Self::CopyFrom { .. } => {
                Err(<Self as Handler<T>>::method_not_supported(Method::Get))
            }
        }
    }
}

impl<T: Transaction + ?Sized> HandlePost<T> for TableStaticRoute {
    type Request = Map<Scalar>;
    type RequestContext = ();
    type Response = TableResponse;
    type Error = TCError;
    type Fut<'a>
        = PostFut<'a>
    where
        Self: 'a,
        T: 'a,
        Self::Request: 'a;

    fn post<'a>(&'a self, txn: &'a T, mut request: Self::Request) -> TCResult<Self::Fut<'a>> {
        match self {
            Self::CopyFrom { root } => {
                let root = root.clone();
                let txn_id = txn.id();
                let schema_scalar = request.require("schema")?;
                let schema_value = scalar_to_value(schema_scalar)?;
                let source_scalar = request.optional("source")?;
                request.expect_empty()?;

                Ok(Box::pin(async move {
                    let schema = TableSchema::try_from_value(schema_value)?;
                    let table = create_table(&root, txn_id, schema).await?;

                    if let Some(source_scalar) = source_scalar {
                        let source_value = scalar_to_value(source_scalar)?;
                        copy_inline_rows(&table, txn_id, source_value).await?;
                    }

                    Ok(TableResponse::Table(table))
                }))
            }
            Self::Create { .. } => {
                Err(<Self as Handler<T>>::method_not_supported(Method::Post))
            }
        }
    }
}

/// Create a new [`PersistentTable`] under the given root directory.
///
/// Creates a unique subdirectory named after the transaction ID, with
/// "persistent" and "txn" subdirectories inside.
async fn create_table(
    root: &DirLock<PersistentFile>,
    txn_id: TxnId,
    schema: TableSchema,
) -> TCResult<PersistentTable> {
    let dir_name = format!("table-{txn_id}");

    let table_dir = {
        let mut root = root.write().await;
        root.get_or_create_dir(dir_name)
            .map_err(TCError::internal)?
    };

    let persistent_dir = {
        let mut table_dir = table_dir.write().await;
        table_dir
            .get_or_create_dir("persistent".to_string())
            .map_err(TCError::internal)?
    };

    let txn_dir = {
        let mut table_dir = table_dir.write().await;
        table_dir
            .get_or_create_dir("txn".to_string())
            .map_err(TCError::internal)?
    };

    Ok(PersistentTable::new(persistent_dir, txn_dir, schema))
}

/// Copy inline row data (a `Value::Tuple` of row tuples) into a new table.
///
/// This handles the case where `copy_from` receives the source data inline
/// rather than as a reference to another table.  When the source is a
/// reference (`Scalar::Ref`), the host is expected to resolve it and pass
/// the resolved table to [`PersistentTable::copy_from`] directly.
async fn copy_inline_rows(
    table: &PersistentTable,
    txn_id: TxnId,
    source: Value,
) -> TCResult<()> {
    let rows = match source {
        Value::Tuple(rows) => rows,
        Value::None => return Ok(()),
        other => {
            return Err(bad_request!(
                "copy_from source must be a tuple of rows or a table reference, got {other:?}"
            ))
        }
    };

    let key_len = table.schema().key().len();
    let value_len = table.schema().values().len();

    for row_value in rows {
        let row = match row_value {
            Value::Tuple(row) => row,
            other => {
                return Err(bad_request!(
                    "each source row must be a tuple, got {other:?}"
                ))
            }
        };

        if row.len() != key_len + value_len {
            return Err(bad_request!(
                "source row has {} columns but schema expects {} ({} key + {} value)",
                row.len(),
                key_len + value_len,
                key_len,
                value_len
            ));
        }

        let key: Vec<Value> = row[..key_len].to_vec();
        let values: Vec<Value> = row[key_len..].to_vec();
        table
            .upsert_row(txn_id, key, values)
            .await
            .map_err(|e| TCError::internal(e))?;
    }

    Ok(())
}
