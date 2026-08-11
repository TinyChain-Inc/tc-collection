#![forbid(unsafe_code)]
#![deny(clippy::needless_question_mark)]

mod persistent_file;
pub use persistent_file::PersistentFile;

mod stream;

mod context;
pub use context::CollectionDir;

mod txn;
pub use txn::StorageContext;

pub mod btree;
pub mod collection;
pub mod table;
pub mod tensor;

pub use collection::Collection;
pub use table::Table;
mod class;
pub use class::{BTreeType, CollectionType, TableType, TensorType};
mod decode;
pub use decode::decode_collection;
mod encode;
mod view;
pub use view::CollectionView;

pub mod route;
pub use route::CollectionState;

#[cfg(test)]
mod test {
    use futures::FutureExt;

    pub fn run_async_test(
        name: &str,
        test_fn: impl FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send
        + 'static,
    ) {
        std::thread::Builder::new()
            .name(name.to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_stack_size(16 * 1024 * 1024)
                    .enable_all()
                    .build()
                    .expect("create test runtime");
                let (send, recv) = std::sync::mpsc::sync_channel(0);

                runtime.spawn(async move {
                    let result = std::panic::AssertUnwindSafe(test_fn()).catch_unwind().await;
                    let _ = send.send(result);
                });

                match recv.recv().expect("test task completed") {
                    Ok(()) => {}
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            })
            .expect("spawn test thread")
            .join()
            .expect("join test thread");
    }
}

#[cfg(test)]
mod architecture_tests {
    #[test]
    fn routes_are_native_and_serialization_free() {
        for source in [
            include_str!("route.rs"),
            include_str!("btree/route.rs"),
            include_str!("table/public/mod.rs"),
            include_str!("table/public/handler.rs"),
            include_str!("tensor/route.rs"),
        ] {
            for forbidden in [
                "destream",
                "serde_json",
                "CollectionResponse",
                "RouteResponse",
                "HandleGet",
                "HandlePut",
                "HandlePost",
                "HandleDelete",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "collection routes must not depend on {forbidden}"
                );
            }
        }

        let table_handlers = include_str!("table/public/handler.rs");
        for duplicate in ["GetFut", "PutFut", "method_not_allowed"] {
            assert!(
                !table_handlers.contains(duplicate),
                "Table leaf handlers must delegate unsupported verbs to tc_ir::Handler"
            );
        }
    }

    #[test]
    fn table_literals_have_one_transaction_local_representation() {
        let obsolete_temp = ["Temp", "Table"].concat();
        let obsolete_static = ["Static", "Routes"].concat();
        for source in [
            include_str!("table/mod.rs"),
            include_str!("table/public/mod.rs"),
            include_str!("table/public/handler.rs"),
        ] {
            assert!(!source.contains(&obsolete_temp));
            assert!(!source.contains(&obsolete_static));
        }

        let codec = include_str!("table/codec.rs");
        assert!(codec.contains("Table::Local"));
        assert!(!codec.contains("PersistentTable"));

        let persistent = include_str!("table/file.rs");
        assert!(!persistent.contains("pub fn literal"));
        assert!(!persistent.contains("load_literal_row"));
    }

    #[test]
    fn views_and_codecs_have_separate_ownership() {
        let view = include_str!("view.rs");
        assert!(!view.contains("destream"));
        assert!(!view.contains("IntoStream"));

        let table_view = include_str!("table/view.rs");
        for fail_open in [".unwrap_or(", "let Ok(", ".expect("] {
            assert!(
                !table_view.contains(fail_open),
                "Table views must propagate storage and stream failures"
            );
        }

        let encode = include_str!("encode.rs");
        for forbidden in ["Handler", "Public", "Route<", ".route("] {
            assert!(
                !encode.contains(forbidden),
                "collection encoding must not depend on {forbidden}"
            );
        }
    }

    #[test]
    fn routes_delegate_without_handler_carrier_enums() {
        let ir = include_str!("../../tc-ir/src/handler.rs");
        assert!(!ir.contains("type Handler"));

        for source in [
            include_str!("route.rs"),
            include_str!("btree/route.rs"),
            include_str!("table/public/mod.rs"),
            include_str!("tensor/route.rs"),
        ] {
            for forbidden in [
                ["enum Collection", "Route"].concat(),
                ["enum Table", "Route"].concat(),
                ["enum BTree", "Route"].concat(),
                ["enum Tensor", "Route"].concat(),
                ["type Hand", "ler ="].concat(),
            ] {
                assert!(
                    !source.contains(&forbidden),
                    "native routes must not use {forbidden} carrier dispatch"
                );
            }
        }
    }

    #[test]
    fn transaction_and_stream_lifecycle_have_one_contract() {
        let txn = include_str!("txn.rs");
        assert!(!txn.contains("pub trait Transaction"));
        assert!(txn.contains("pub trait StorageContext"));

        for source in [
            include_str!("btree/file.rs"),
            include_str!("table/file.rs"),
            include_str!("collection.rs"),
        ] {
            assert!(!source.contains("type Commit"));
            assert!(!source.contains("commit failed\")"));
        }

        assert!(include_str!("btree/stream.rs").contains("GuardedStream"));
        assert!(include_str!("table/stream.rs").contains("ReadPermit"));
    }

    #[test]
    fn collection_transaction_types_have_no_default() {
        for source in [
            include_str!("collection.rs"),
            include_str!("btree/file.rs"),
            include_str!("table/file.rs"),
            include_str!("table/mod.rs"),
            include_str!("table/view.rs"),
        ] {
            assert!(
                !source.contains("Txn ="),
                "collection transaction parameters must always be explicit"
            );
        }
    }
}
