#![forbid(unsafe_code)]
#![deny(clippy::needless_question_mark)]

mod persistent_file;
pub use persistent_file::{CollectionFile, CollectionNode, PersistentFile};

mod stream;

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
