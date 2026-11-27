use freqfs::DirLock;

use super::Allocation;

impl<F> Allocation<F> {
    /// An unpublished allocation used by native delta tests.
    pub(crate) fn fixture(dir: DirLock<F>) -> Self {
        Self {
            id: "fixture".parse().unwrap(),
            dir,
        }
    }
}
