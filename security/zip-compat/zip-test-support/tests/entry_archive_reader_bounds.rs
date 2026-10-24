//! Exercise the new entry reader, including its two tensor filename policies.
//! The size-limit construction follows upstream declared_size_enforced.rs.
use std::io::{Cursor, Read, Write};
use zip::read::{EntryArchive, UnicodeExtraFieldPolicy};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

const SIZE: usize = 1 << 20;
const LIMIT: usize = 10;

fn archive(method: CompressionMethod) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer.start_file("a.bin", SimpleFileOptions::default().compression_method(method)).unwrap();
    writer.write_all(&vec![7; SIZE]).unwrap();
    writer.finish().unwrap().into_inner()
}

fn policies() -> [UnicodeExtraFieldPolicy; 2] {
    [UnicodeExtraFieldPolicy::Ignore, UnicodeExtraFieldPolicy::RequireValid]
}

fn methods() -> [CompressionMethod; 4] {
    [CompressionMethod::Stored, CompressionMethod::Deflated, CompressionMethod::Bzip2, CompressionMethod::Zstd]
}

#[test]
fn declared_size_limits_each_bounded_reader() {
    for policy in policies() {
        for method in methods() {
            let mut bytes = archive(method);
            let local = bytes.windows(4).position(|x| x == b"PK\x03\x04").unwrap();
            let central = bytes.windows(4).position(|x| x == b"PK\x01\x02").unwrap();
            for offset in [local + 22, central + 24] {
                bytes[offset..offset + 4].copy_from_slice(&(LIMIT as u32).to_le_bytes());
            }
            let mut reader = EntryArchive::new(Cursor::new(bytes), policy).unwrap();
            let mut entry = reader.by_index(0).unwrap();
            let mut buf = [0; 4096];
            let mut count = 0;
            let error = loop {
                match entry.read(&mut buf) {
                    Ok(0) => panic!("accepted understated size: {policy:?}/{method:?}"),
                    Ok(n) => count += n,
                    Err(error) => break error,
                }
            };
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert!(count <= LIMIT, "{policy:?}/{method:?}: emitted {count}");
        }
    }
}

#[test]
fn valid_compressed_entries_read_completely() {
    for policy in policies() {
        for method in methods() {
            let mut reader = EntryArchive::new(Cursor::new(archive(method)), policy).unwrap();
            let mut actual = Vec::new();
            reader.by_index(0).unwrap().read_to_end(&mut actual).unwrap();
            assert_eq!(actual, vec![7; SIZE]);
        }
    }
}

#[test]
fn crc_mismatch_is_rejected() {
    for policy in policies() {
        let mut bytes = archive(CompressionMethod::Stored);
        let central = bytes.windows(4).position(|x| x == b"PK\x01\x02").unwrap();
        // Corrupt payload, keeping both headers' declared CRC and length intact.
        bytes[central - 1] ^= 1;
        let mut reader = EntryArchive::new(Cursor::new(bytes), policy).unwrap();
        let mut actual = Vec::new();
        assert!(reader.by_index(0).unwrap().read_to_end(&mut actual).is_err());
    }
}
