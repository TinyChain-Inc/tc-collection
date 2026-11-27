//! Native version publication. Payload traversal stays with each collection.
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::{Arc, Mutex, Weak};

use destream::{de, en};
use freqfs::{DirLock, FileLock};
use get_size::GetSize;
use tc_error::{TCError, TCResult};
use tc_ir::{Id, TxnId};

use crate::CollectionFile;

const BASE: &str = "base";

const VERSIONS: &str = "versions";

const COMMITS: &str = "commits";

const ROOT: &str = "collection";

const FORMAT: u8 = 1;

/// Opaque native publication metadata, delegated by the caller's file codec.
#[derive(Clone, Debug)]
pub struct CollectionMetadata(Record);

#[derive(Clone, Debug)]
enum Record {
    Creating,
    Root(Option<TxnId>, Vec<Run>),
    Commit(TxnId, Version),
}

#[derive(Clone, Debug)]
pub(super) struct Version {
    pub(super) id: Id,
    pub(super) units: u64,
}

#[derive(Clone, Debug)]
pub(super) struct Run {
    pub(super) through: TxnId,
    pub(super) version: Version,
}

impl GetSize for CollectionMetadata {
    fn get_heap_size(&self) -> usize {
        match &self.0 {
            Record::Creating => 0,
            Record::Root(_, runs) => {
                runs.capacity() * std::mem::size_of::<Run>()
                    + runs
                        .iter()
                        .map(|run| run.version.id.get_heap_size())
                        .sum::<usize>()
            }
            Record::Commit(_, version) => version.id.get_heap_size(),
        }
    }
}

impl<'en> en::ToStream<'en> for CollectionMetadata {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::IntoStream;
        match &self.0 {
            Record::Creating => (FORMAT, 2u8).into_stream(encoder),
            Record::Root(cutoff, runs) => (
                FORMAT,
                0u8,
                cutoff,
                en::SeqStream::from(futures::stream::iter(
                    runs.iter()
                        .map(|run| (run.through, &run.version.id, run.version.units)),
                )),
            )
                .into_stream(encoder),
            Record::Commit(id, version) => {
                (FORMAT, 1u8, id, &version.id, version.units).into_stream(encoder)
            }
        }
    }
}

impl de::FromStream for CollectionMetadata {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let mut fields = decoder.open_container(de::Kind::Seq, None).await?;
        let format = u8::from_stream((), decoder).await?;
        decoder.finish_child(&mut fields).await?;
        if format != FORMAT {
            return Err(de::Error::custom("unsupported collection metadata format"));
        }

        let tag = u8::from_stream((), decoder).await?;
        decoder.finish_child(&mut fields).await?;
        let record = match tag {
            0 => {
                let cutoff = Option::<TxnId>::from_stream((), decoder).await?;
                decoder.finish_child(&mut fields).await?;
                let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
                let mut runs: Vec<Run> = Vec::new();
                while sequence.slot() != de::Slot::End {
                    if runs.len() == u64::BITS as usize {
                        return Err(de::Error::custom("too many compacted collection versions"));
                    }

                    let (through, id, units) = <(TxnId, Id, u64)>::from_stream((), decoder).await?;
                    if units == 0
                        || cutoff.is_none_or(|cutoff| through > cutoff)
                        || runs.last().is_some_and(|prior| {
                            prior.through >= through || prior.version.units.ilog2() <= units.ilog2()
                        })
                    {
                        return Err(de::Error::custom("invalid compacted collection version"));
                    }

                    runs.push(Run {
                        through,
                        version: Version { id, units },
                    });

                    decoder.finish_child(&mut sequence).await?;
                }

                decoder.finish_child(&mut fields).await?;
                Record::Root(cutoff, runs)
            }
            1 => {
                let id = TxnId::from_stream((), decoder).await?;
                decoder.finish_child(&mut fields).await?;
                let version = Id::from_stream((), decoder).await?;
                decoder.finish_child(&mut fields).await?;
                let units = u64::from_stream((), decoder).await?;
                decoder.finish_child(&mut fields).await?;
                Record::Commit(id, Version { id: version, units })
            }
            2 => Record::Creating,
            _ => return Err(de::Error::custom("invalid collection metadata format")),
        };

        if fields.slot() != de::Slot::End {
            return Err(de::Error::custom("unexpected collection metadata field"));
        }

        Ok(Self(record))
    }
}

/// A directory lease pins a native version independently of transaction workspaces.
pub(crate) struct Allocation<F> {
    id: Id,
    dir: DirLock<F>,
}

impl<F> Allocation<F> {
    pub(crate) fn dir(&self) -> &DirLock<F> {
        &self.dir
    }

    pub(crate) fn id(&self) -> &Id {
        &self.id
    }
}

#[cfg(test)]
#[path = "../../tests/unit/persistence/storage.rs"]
mod tests;

pub(crate) struct Store<F> {
    root: DirLock<F>,
    base: DirLock<F>,
    versions: DirLock<F>,
    commits: DirLock<F>,
    manifest: FileLock<F>,
    leases: Mutex<BTreeMap<Id, Weak<Allocation<F>>>>,
}

fn missing(name: &str) -> TCError {
    freqfs::IntegrityError::invalid_data(format!("missing collection {name}")).into()
}

impl<F: CollectionFile> Store<F> {
    pub(crate) fn base(&self) -> &DirLock<F> {
        &self.base
    }

    pub async fn create(root: DirLock<F>) -> TCResult<Arc<Self>> {
        let (base, versions, commits, manifest) = {
            let mut dir = root.write().await;
            if !dir.is_empty() {
                return Err(TCError::bad_request(
                    "collection creation requires empty storage",
                ));
            }

            let base = dir.create_dir(BASE.into())?;
            let versions = dir.create_dir(VERSIONS.into())?;
            let commits = dir.create_dir(COMMITS.into())?;
            let manifest = dir
                .create_file(ROOT.into(), F::from(CollectionMetadata(Record::Creating)))
                .await?;

            (base, versions, commits, manifest)
        };

        Ok(Arc::new(Self {
            root,
            base,
            versions,
            commits,
            manifest,
            leases: Mutex::new(BTreeMap::new()),
        }))
    }

    pub async fn load(root: DirLock<F>) -> TCResult<Arc<Self>> {
        let (base, versions, commits, manifest) = {
            let dir = root.read().await;
            if dir.len() != 4 {
                return Err(missing("layout"));
            }

            (
                dir.get_dir(BASE).cloned().ok_or_else(|| missing(BASE))?,
                dir.get_dir(VERSIONS)
                    .cloned()
                    .ok_or_else(|| missing(VERSIONS))?,
                dir.get_dir(COMMITS)
                    .cloned()
                    .ok_or_else(|| missing(COMMITS))?,
                dir.get_file(ROOT).cloned().ok_or_else(|| missing(ROOT))?,
            )
        };

        let store = Arc::new(Self {
            root,
            base,
            versions,
            commits,
            manifest,
            leases: Mutex::new(BTreeMap::new()),
        });

        store.frontier().await?;
        Ok(store)
    }

    pub async fn initialize(&self) -> TCResult<()> {
        self.root.sync_all().await?;
        self.publish(None, Vec::new()).await
    }

    pub async fn allocate(&self) -> TCResult<Arc<Allocation<F>>> {
        let mut versions = self.versions.write().await;
        let (id, dir) = versions.create_dir_unique()?;
        let allocation = Arc::new(Allocation {
            id: id.to_string().parse().expect("UUID is an Id"),
            dir,
        });

        self.leases
            .lock()
            .expect("collection leases poisoned")
            .insert(allocation.id.clone(), Arc::downgrade(&allocation));

        Ok(allocation)
    }

    pub(super) async fn open(&self, version: &Version) -> TCResult<Arc<Allocation<F>>> {
        let dir = self
            .versions
            .read()
            .await
            .get_dir(version.id.as_str())
            .cloned()
            .ok_or_else(|| missing("version"))?;

        let allocation = Arc::new(Allocation {
            id: version.id.clone(),
            dir,
        });

        self.leases
            .lock()
            .expect("collection leases poisoned")
            .insert(allocation.id.clone(), Arc::downgrade(&allocation));

        Ok(allocation)
    }

    pub(super) async fn frontier(&self) -> TCResult<(Option<TxnId>, Vec<Run>)> {
        match &self.manifest.read::<CollectionMetadata>().await?.0 {
            Record::Root(cutoff, runs) => Ok((*cutoff, runs.clone())),
            _ => Err(missing("root descriptor")),
        }
    }

    pub(super) async fn receipts(
        &self,
        cutoff: Option<TxnId>,
    ) -> TCResult<BTreeMap<TxnId, Version>> {
        let mut receipts = BTreeMap::new();
        let mut after = None;
        loop {
            let next = {
                let commits = self.commits.read().await;
                commits
                    .range((after.as_deref().map_or(Unbounded, Excluded), Unbounded))
                    .next()
                    .map(|(name, entry)| (name.clone(), entry.clone()))
            };

            let Some((name, entry)) = next else {
                break;
            };

            let id: TxnId = name.parse().map_err(|_| missing("commit identity"))?;
            if name != id.to_string() {
                return Err(missing("canonical commit name"));
            }

            after = Some(name);
            if cutoff.is_some_and(|cutoff| id <= cutoff) {
                continue;
            }

            let freqfs::DirEntry::File(file) = entry else {
                return Err(missing("commit receipt"));
            };

            match &file.read::<CollectionMetadata>().await?.0 {
                Record::Commit(recorded, version) if *recorded == id => {
                    receipts.insert(id, version.clone());
                }
                _ => return Err(missing("commit identity")),
            }
        }

        Ok(receipts)
    }

    pub(super) async fn receipt(&self, id: TxnId) -> TCResult<Version> {
        let file = self
            .commits
            .read()
            .await
            .get_file(&id.to_string())
            .cloned()
            .ok_or_else(|| missing("commit receipt"))?;

        match &file.read::<CollectionMetadata>().await?.0 {
            Record::Commit(recorded, version) if *recorded == id => Ok(version.clone()),
            _ => Err(missing("commit identity")),
        }
    }

    pub async fn commit(&self, id: TxnId, allocation: &Allocation<F>, units: u64) -> TCResult<()> {
        allocation.dir.sync_all().await?;
        let metadata = CollectionMetadata(Record::Commit(
            id,
            Version {
                id: allocation.id.clone(),
                units,
            },
        ));

        let file = self
            .commits
            .write()
            .await
            .create_file(id.to_string(), F::from(metadata.clone()))
            .await?;

        file.replace_all(F::from(metadata)).await?;
        // Construction already durably established the containing commits directory.
        Ok(())
    }

    pub(super) async fn publish(&self, cutoff: Option<TxnId>, runs: Vec<Run>) -> TCResult<()> {
        self.manifest
            .replace_all(F::from(CollectionMetadata(Record::Root(cutoff, runs))))
            .await?;

        Ok(())
    }

    /// Remove a known disposable allocation only after its last reader releases it.
    pub async fn discard(&self, id: &Id) -> TCResult<()> {
        let mut versions = self.versions.write().await;
        let live = self
            .leases
            .lock()
            .expect("collection leases poisoned")
            .get(id)
            .is_some_and(|lease| lease.strong_count() != 0);

        if !live && versions.delete(id.as_str()).await {
            versions.sync_deleted().await?;
            self.leases
                .lock()
                .expect("collection leases poisoned")
                .remove(id);
        }

        Ok(())
    }

    pub async fn reclaim(&self) -> TCResult<()> {
        let (cutoff, runs) = self.frontier().await?;
        let protected: BTreeSet<_> = runs.into_iter().map(|run| run.version.id).collect();
        // Register allocations and inspect leases under this same directory guard.
        // Published runs survive even when no live reader currently retains them.
        {
            let mut versions = self.versions.write().await;
            let mut after = None;
            loop {
                let next = versions
                    .range((after.as_deref().map_or(Unbounded, Excluded), Unbounded))
                    .next()
                    .map(|(name, _)| name.clone());

                let Some(name) = next else {
                    break;
                };

                let live = self
                    .leases
                    .lock()
                    .expect("collection leases poisoned")
                    .get(name.as_str())
                    .is_some_and(|lease| lease.strong_count() != 0);

                if !protected.contains(name.as_str()) && !live {
                    versions.delete(&name).await;
                }

                after = Some(name);
            }

            self.leases
                .lock()
                .expect("collection leases poisoned")
                .retain(|_, lease| lease.strong_count() != 0);

            versions.sync_deleted().await?;
        }

        if let Some(cutoff) = cutoff {
            let mut commits = self.commits.write().await;
            let mut after = None;
            loop {
                let next = commits
                    .range((after.as_deref().map_or(Unbounded, Excluded), Unbounded))
                    .next()
                    .map(|(name, _)| name.clone());

                let Some(name) = next else {
                    break;
                };

                let id = name
                    .parse::<TxnId>()
                    .map_err(|_| missing("commit identity"))?;

                if id <= cutoff {
                    commits.delete(&name).await;
                }

                after = Some(name);
            }

            commits.sync_deleted().await?;
        }

        Ok(())
    }
}
