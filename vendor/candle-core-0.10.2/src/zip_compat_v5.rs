//! Preserve ZIP7's decoded-name index while using the patched ZIP9 parser.

use std::collections::{HashMap, hash_map::Entry};
use std::io::{Read, Seek};
use zip::read::{EntryArchive, UnicodeExtraFieldPolicy, ZipFile, ZipFileEntry};
use zip::result::{ZipError, ZipResult};

pub(crate) struct Archive<R> {
    entries: EntryArchive<R>,
    // Preserve the first insertion position, replacing its value on duplicates,
    // as ZIP7's IndexMap did. Names are decoded exactly once.
    indices: Vec<usize>,
    by_name: HashMap<String, usize>,
}

impl<R: Read + Seek> Archive<R> {
    pub(crate) fn new(reader: R) -> ZipResult<Self> {
        let entries = EntryArchive::new(reader, UnicodeExtraFieldPolicy::RequireValid)?;
        let mut indices = Vec::with_capacity(entries.len());
        let mut by_name = HashMap::with_capacity(entries.len());
        for index in 0..entries.len() {
            let name = entries.by_index_data(index)?.name()?.into_owned();
            match by_name.entry(name) {
                Entry::Vacant(slot) => {
                    slot.insert(indices.len());
                    indices.push(index);
                }
                Entry::Occupied(slot) => indices[*slot.get()] = index,
            }
        }
        Ok(Self {
            entries,
            indices,
            by_name,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn by_index(&mut self, index: usize) -> ZipResult<ZipFile<'_, R>> {
        let index = *self.indices.get(index).ok_or(ZipError::FileNotFound)?;
        self.entries.by_index(index)
    }

    pub(crate) fn by_index_data(&self, index: usize) -> ZipResult<ZipFileEntry<'_>> {
        let index = *self.indices.get(index).ok_or(ZipError::FileNotFound)?;
        self.entries.by_index_data(index)
    }

    pub(crate) fn by_name(&mut self, name: &str) -> ZipResult<ZipFile<'_, R>> {
        let index = *self.by_name.get(name).ok_or(ZipError::FileNotFound)?;
        self.by_index(index)
    }
}
