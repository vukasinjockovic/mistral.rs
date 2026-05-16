//! LEECHQ24 v3 container reader (Forney-Q24 + 4-codebook tANS .leech format).
//!
//! Mirrors the Python writer/reader in
//! `antsquant/production/packer/q24_tans/core/container.py`.
//!
//! ```no_run
//! use mistralrs_leech_q24::{LeechQ24File, OpenOptions, Role};
//!
//! let f = LeechQ24File::open("/path/to/qwopus_ms18_v3.q24t.leech").unwrap();
//! println!("ms_default = {}", f.header().tile_size_default);
//! for entry in f.toc().iter().take(3) {
//!     println!("{} role={:?} shape={:?}", entry.name, entry.role, entry.shape);
//! }
//! ```
//!
//! The crate is pure-Rust, no CUDA. The CUDA decoder lives in
//! `mistralrs-quant/src/leech_q24/` and consumes the byte slices produced
//! here.

pub mod bucket_unpack;
pub mod codebook;
pub mod container;
pub mod error;
pub mod payload;

use std::path::Path;
use std::sync::Arc;

use byteorder::{ByteOrder, LittleEndian as LE};
use memmap2::Mmap;

pub use codebook::{parse_codebook_blob, CodebookSet, DecodeEntry, M_TABLE, N_CODEBOOKS, TABLE_LOG};
pub use container::{
    validate_role_dtype, DtypeTag, Header, Role, TocEntry, FIXED_PREFIX_SIZE, FORMAT_VERSION,
    HEADER_SIZE, MAGIC, SENTINEL_BUCKET, TOC_ENTRY_SIZE,
};
pub use error::{LeechQ24Error, Result};
pub use payload::LlvqTansPayload;

/// Mmapped LEECHQ24 file with parsed prefix, manifest, codebooks, and TOC.
pub struct LeechQ24File {
    mmap: Arc<Mmap>,
    header: Header,
    manifest_raw: Vec<u8>,
    codebooks: Vec<CodebookSet>,
    toc: Vec<TocEntry>,
}

/// Open-time validation knobs.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Run CRC32 over `[0, file_size - 4)` and compare to the trailing u32.
    /// Disabled by default because it touches the entire ~4.5 GB file.
    pub verify_crc: bool,
    /// Enforce `validate_role_dtype` on every TOC entry. Default true.
    pub validate_toc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            verify_crc: false,
            validate_toc: true,
        }
    }
}

impl LeechQ24File {
    /// Open with default options (no CRC verify, TOC validation on).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(path, OpenOptions::default())
    }

    pub fn open_with<P: AsRef<Path>>(path: P, opts: OpenOptions) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: Mmap is read-only and we hold the File for the Mmap's lifetime.
        // Callers must not mutate the file concurrently.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(Arc::new(mmap), opts)
    }

    fn from_mmap(mmap: Arc<Mmap>, opts: OpenOptions) -> Result<Self> {
        let bytes: &[u8] = &mmap[..];
        if bytes.len() < HEADER_SIZE + 4 {
            return Err(LeechQ24Error::Truncated {
                offset: 0,
                needed: HEADER_SIZE + 4,
                len: bytes.len() as u64,
            });
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if magic != MAGIC {
            return Err(LeechQ24Error::BadMagic { got: magic });
        }
        let header = Header::unpack(&bytes[8..8 + container::HEADER_STRUCT_SIZE])?;
        if header.format_version != FORMAT_VERSION {
            return Err(LeechQ24Error::UnsupportedVersion(header.format_version));
        }
        if header.header_size as usize != HEADER_SIZE {
            return Err(LeechQ24Error::BadTocEntry {
                idx: 0,
                reason: format!(
                    "header.header_size {} != {}",
                    header.header_size, HEADER_SIZE
                ),
            });
        }

        // Optional full-file CRC32.
        if opts.verify_crc {
            let stored = LE::read_u32(&bytes[bytes.len() - 4..]);
            let computed = crc32fast::hash(&bytes[..bytes.len() - 4]);
            if stored != computed {
                return Err(LeechQ24Error::CrcMismatch {
                    file: stored,
                    computed,
                });
            }
        }

        // Manifest (raw JSON bytes — consumers can deserialize their own schema).
        let m_off = header.manifest_offset as usize;
        let m_sz = header.manifest_size as usize;
        let manifest_raw = payload::slice_or_truncated(bytes, m_off, m_sz)?.to_vec();

        // Codebook blob.
        let codebooks =
            parse_codebook_blob(bytes, header.codebook_offset, header.codebook_size)?;

        // TOC.
        let toc_off = header.toc_offset as usize;
        let toc_count = header.toc_entry_count as usize;
        let toc_bytes_len = toc_count
            .checked_mul(TOC_ENTRY_SIZE)
            .ok_or(LeechQ24Error::BadTocEntry {
                idx: 0,
                reason: "toc_entry_count * TOC_ENTRY_SIZE overflow".to_owned(),
            })?;
        let toc_buf = payload::slice_or_truncated(bytes, toc_off, toc_bytes_len)?;
        let mut toc = Vec::with_capacity(toc_count);
        for i in 0..toc_count {
            let start = i * TOC_ENTRY_SIZE;
            let entry = TocEntry::unpack(&toc_buf[start..start + TOC_ENTRY_SIZE], i)?;
            if opts.validate_toc {
                validate_role_dtype(entry.role, entry.dtype_tag).map_err(|_| {
                    LeechQ24Error::BadTocEntry {
                        idx: i,
                        reason: format!(
                            "invalid (role={:?}, dtype={:?}) for tensor {:?}",
                            entry.role, entry.dtype_tag, entry.name
                        ),
                    }
                })?;
            }
            toc.push(entry);
        }

        Ok(LeechQ24File {
            mmap,
            header,
            manifest_raw,
            codebooks,
            toc,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Raw manifest JSON bytes — caller deserializes into their preferred schema.
    pub fn manifest_json(&self) -> &[u8] {
        &self.manifest_raw
    }

    /// Parse the manifest as `serde_json::Value` for ad-hoc inspection.
    pub fn manifest_value(&self) -> Result<serde_json::Value> {
        Ok(serde_json::from_slice(&self.manifest_raw)?)
    }

    pub fn toc(&self) -> &[TocEntry] {
        &self.toc
    }

    pub fn codebooks(&self) -> &[CodebookSet] {
        &self.codebooks
    }

    /// Locate a `CodebookSet` by its `symbol_set_id`.
    pub fn codebook_set(&self, symbol_set_id: u32) -> Option<&CodebookSet> {
        self.codebooks
            .iter()
            .find(|c| c.symbol_set_id == symbol_set_id)
    }

    /// Raw decode-tables blob for one symbol set (n_codebooks × M × u32 LE).
    pub fn decode_tables_bytes(&self, symbol_set_id: u32) -> Option<&[u8]> {
        let cb = self.codebook_set(symbol_set_id)?;
        let off = cb.decode_tables_offset as usize;
        let sz = cb.decode_tables_bytes as usize;
        if off + sz > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[off..off + sz])
    }

    /// Borrow the full mmapped file. Useful for raw verification / hashing.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Slice for the LLVQ_TANS payload of TOC entry `idx`. Errors if the
    /// entry is not LLVQ_TANS or offsets are out of range.
    pub fn llvq_tans_payload(&self, idx: usize) -> Result<LlvqTansPayload<'_>> {
        let entry = self.toc.get(idx).ok_or(LeechQ24Error::BadTocEntry {
            idx,
            reason: "index out of range".to_owned(),
        })?;
        payload::build_llvq_tans_payload(entry, &self.mmap[..])
    }

    /// Locate a tensor by exact name; returns `(toc_index, &TocEntry)`.
    pub fn find_tensor(&self, name: &str) -> Option<(usize, &TocEntry)> {
        self.toc
            .iter()
            .enumerate()
            .find(|(_, e)| e.name == name)
    }

    /// Raw bytes for a passthrough / overlay tensor.
    pub fn passthrough_bytes(&self, idx: usize) -> Result<&[u8]> {
        let entry = self.toc.get(idx).ok_or(LeechQ24Error::BadTocEntry {
            idx,
            reason: "index out of range".to_owned(),
        })?;
        if entry.is_llvq_tans() {
            return Err(LeechQ24Error::BadTocEntry {
                idx,
                reason: format!("tensor {:?} is LLVQ_TANS (not passthrough)", entry.name),
            });
        }
        payload::slice_or_truncated(
            &self.mmap[..],
            entry.passthrough_offset as usize,
            entry.passthrough_size as usize,
        )
    }

    /// Stored CRC32 from the last 4 bytes of the file.
    pub fn stored_crc(&self) -> u32 {
        let bytes = &self.mmap[..];
        LE::read_u32(&bytes[bytes.len() - 4..])
    }

    /// Compute the file's CRC32 (does not compare).
    pub fn compute_crc(&self) -> u32 {
        let bytes = &self.mmap[..];
        crc32fast::hash(&bytes[..bytes.len() - 4])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_options_default() {
        let o = OpenOptions::default();
        assert!(!o.verify_crc);
        assert!(o.validate_toc);
    }
}
