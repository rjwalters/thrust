//! Central-directory-order reader for consumers that must retain duplicate entries.
//!
//! The regular `ZipArchive` API and its default filename behavior are unchanged.

use super::{Config, ZipArchive, ZipFile, ZipFileEntry, ZipReadOptions};
use crate::result::{ZipError, ZipResult};
use crate::types::ZipFileData;
use std::borrow::Cow;
use std::io::{Read, Seek};

/// Treatment of Info-ZIP Unicode path/comment overrides.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeExtraFieldPolicy {
    /// ZIP9 behavior: use valid overrides, ignoring invalid CRC or UTF-8.
    #[default]
    Standard,
    /// Treat Unicode overrides as opaque extra fields, validating their outer
    /// lengths but not interpreting their contents. Honor the original declared
    /// filename encoding.
    /// Used by consumers migrating from ZIP0.6.
    Ignore,
    /// Require valid CRC and UTF-8 for overrides, honoring declared encoding otherwise.
    /// Used by consumers migrating from ZIP7.
    RequireValid,
}

/// A reader that retains every central-directory entry in its original order.
///
/// It shares ZIP9's central-directory parser, checked offsets, decompression
/// bounds and CRC readers. Consumers choose their own duplicate-name lookup
/// policy; iteration does not silently discard an earlier duplicate.
pub struct EntryArchive<R> {
    reader: R,
    files: Vec<(Box<[u8]>, ZipFileData)>,
}

impl<R: Read + Seek> EntryArchive<R> {
    /// Read entries using the given Unicode override policy.
    pub fn new(mut reader: R, policy: UnicodeExtraFieldPolicy) -> ZipResult<Self> {
        let (shared, _, _) = ZipArchive::read_directory(Config::default(), &mut reader, policy)?;
        Ok(Self {
            reader,
            files: shared.files,
        })
    }

    /// Number of entries, including duplicate names.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Read entry metadata without accessing compressed file contents.
    pub fn by_index_data(&self, index: usize) -> ZipResult<ZipFileEntry<'_>> {
        let (name, data) = self.files.get(index).ok_or(ZipError::FileNotFound)?;
        Ok(ZipFileEntry {
            file_name_raw: Cow::Borrowed(name),
            data: Cow::Borrowed(data),
        })
    }

    /// Read an entry through the same bounded decompressor and CRC checks used
    /// by `ZipArchive::by_index`.
    pub fn by_index(&mut self, index: usize) -> ZipResult<ZipFile<'_, R>> {
        let (name, data) = self.files.get(index).ok_or(ZipError::FileNotFound)?;
        let content = data.find_content(&mut self.reader)?;
        let reader = data.make_reader(content, ZipReadOptions::new())?;
        Ok(ZipFile {
            file_name_raw: Cow::Borrowed(name),
            data: Cow::Borrowed(data),
            reader,
        })
    }
}
