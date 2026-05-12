//! `.leech` container reader.
//!
//! Mmaps a `.leech` file, parses MAGIC + HEADER + manifest JSON + TOC, optionally
//! verifies the trailing CRC32 over `[0, file_size - 4)`, and exposes per-tensor
//! payload slices for higher layers (LeechLayer in `mistralrs-quant`, model
//! loader in `mistralrs-core`).
//!
//! Mirrors the Python prototype in `packer/`:
//!   container.rs ← packer/core/container.py
//!   payload.rs   ← packer/core/payload.py
//!   overlay.rs   ← packer/core/overlay.py
//!   fp8.rs       ← packer/core/fp8_decode.py
//!
//! The crate is pure-Rust, no CUDA. CUDA decode lives in `mistralrs-quant`
//! and reads the slices produced here.

pub mod container;
pub mod error;
pub mod fp8;
pub mod overlay;
pub mod payload;

use std::path::Path;
use std::sync::Arc;

use byteorder::{ByteOrder, LittleEndian as LE};
use memmap2::Mmap;
use serde::Deserialize;

pub use container::{
    DtypeTag, Header, Role, TocEntry, FIXED_PREFIX_SIZE, HEADER_SIZE, MAGIC, TOC_ENTRY_SIZE,
};
pub use error::{LeechError, Result};
pub use overlay::{parse_overlay_block, OverlayEntry, OverlayKind};
pub use payload::LlvqPayload;

/// Mmapped `.leech` file plus parsed prefix.
pub struct LeechFile {
    mmap: Arc<Mmap>,
    header: Header,
    manifest: Manifest,
    toc: Vec<TocEntry>,
}

/// Subset of the manifest JSON the loader actually consults.
/// Fields not listed here are ignored (forward-compat).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Manifest {
    pub format_version: u32,
    pub structural_schema_version: String,
    #[serde(rename = "global")]
    pub global: GlobalInfo,
    /// Full raw JSON value, for downstream tools that need everything.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct GlobalInfo {
    pub ms_bulk: Option<u32>,
    pub ms_embed: Option<u32>,
    pub ms_head: Option<u32>,
    pub offset_bits: Option<u32>,
    pub k_beta: Option<u32>,
    pub k_offset: Option<u32>,
    pub leech_group_dim: Option<u32>,
    pub tensor_count: Option<u32>,
    pub llvq_count: Option<u32>,
    pub passthrough_count: Option<u32>,
    pub overlay_count: Option<u32>,
    pub leftover_columns_zero: Option<bool>,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            format_version: 0,
            structural_schema_version: String::new(),
            global: GlobalInfo::default(),
            extra: Default::default(),
        }
    }
}

/// How aggressively to verify on open.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenOptions {
    /// Run CRC32 over `[0, file_size - 4)` and compare to the trailing u32.
    /// Disabled by default because it touches the whole file (~4 GB). Enable
    /// in tests and one-off integrity checks.
    pub verify_crc: bool,
    /// Enforce `structural_schema_version == "LLVQ_NIEMEIER_G24_v1"`.
    /// Default true — we don't read foreign lattices.
    pub require_schema: bool,
}

pub const EXPECTED_SCHEMA: &str = "LLVQ_NIEMEIER_G24_v1";

impl LeechFile {
    /// Open a `.leech` file with default options (no CRC verify, schema enforced).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(
            path,
            OpenOptions {
                verify_crc: false,
                require_schema: true,
            },
        )
    }

    /// Open with full options.
    pub fn open_with<P: AsRef<Path>>(path: P, opts: OpenOptions) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: we hold the File for the entire lifetime of the Mmap. The
        // file is treated as immutable; if someone mutates it underneath us,
        // we get undefined behavior. Callers should not point us at a file
        // that's being concurrently written.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);
        Self::from_mmap(mmap, opts)
    }

    fn from_mmap(mmap: Arc<Mmap>, opts: OpenOptions) -> Result<Self> {
        let bytes: &[u8] = &mmap[..];
        if bytes.len() < FIXED_PREFIX_SIZE + 4 {
            return Err(LeechError::Truncated {
                offset: 0,
                needed: FIXED_PREFIX_SIZE + 4,
                len: bytes.len() as u64,
            });
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if magic != MAGIC {
            return Err(LeechError::BadMagic { got: magic });
        }

        let header = Header::unpack(&bytes[8..8 + HEADER_SIZE])?;
        if header.format_version != container::FORMAT_VERSION {
            return Err(LeechError::UnsupportedVersion(header.format_version));
        }

        // Manifest JSON.
        let m_off = header.manifest_json_offset as usize;
        let m_size = header.manifest_json_size as usize;
        let manifest_bytes = slice_or_truncated(bytes, m_off, m_size)?;
        let manifest: Manifest = serde_json::from_slice(manifest_bytes)?;
        if opts.require_schema && manifest.structural_schema_version != EXPECTED_SCHEMA {
            return Err(LeechError::SchemaMismatch {
                got: manifest.structural_schema_version.clone(),
                expected: EXPECTED_SCHEMA.to_owned(),
            });
        }

        // TOC.
        let toc_off = header.toc_offset as usize;
        let toc_count = header.toc_entry_count as usize;
        let toc_bytes_len = toc_count
            .checked_mul(TOC_ENTRY_SIZE)
            .ok_or_else(|| LeechError::Truncated {
                offset: header.toc_offset,
                needed: usize::MAX,
                len: bytes.len() as u64,
            })?;
        let toc_buf = slice_or_truncated(bytes, toc_off, toc_bytes_len)?;
        let mut toc = Vec::with_capacity(toc_count);
        for i in 0..toc_count {
            let start = i * TOC_ENTRY_SIZE;
            let entry = TocEntry::unpack(&toc_buf[start..start + TOC_ENTRY_SIZE], i)?;
            container::validate_role_dtype(entry.role, entry.dtype_tag).map_err(|_| {
                LeechError::BadTocEntry {
                    idx: i,
                    reason: format!(
                        "invalid (role={:?}, dtype={:?}) combination",
                        entry.role, entry.dtype_tag
                    ),
                }
            })?;
            toc.push(entry);
        }

        // CRC verification — optional, expensive.
        if opts.verify_crc {
            let stored = LE::read_u32(&bytes[bytes.len() - 4..]);
            let computed = crc32fast::hash(&bytes[..bytes.len() - 4]);
            if stored != computed {
                return Err(LeechError::CrcMismatch {
                    file: stored,
                    computed,
                });
            }
        }

        Ok(LeechFile {
            mmap,
            header,
            manifest,
            toc,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn toc(&self) -> &[TocEntry] {
        &self.toc
    }

    /// Number of TOC entries.
    pub fn tensor_count(&self) -> usize {
        self.toc.len()
    }

    /// Raw mmapped byte slice over the entire file.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Bytes for the payload of TOC entry `idx`.
    pub fn payload(&self, idx: usize) -> Result<&[u8]> {
        let entry = self
            .toc
            .get(idx)
            .ok_or_else(|| LeechError::BadTocEntry {
                idx,
                reason: "index out of range".to_owned(),
            })?;
        let off = entry.payload_offset as usize;
        let size = entry.payload_size as usize;
        slice_or_truncated(&self.mmap[..], off, size)
    }

    /// Bytes for the per-tensor static table (LLVQ only; 0-size for passthroughs).
    pub fn static_table(&self, idx: usize) -> Result<&[u8]> {
        let entry = self
            .toc
            .get(idx)
            .ok_or_else(|| LeechError::BadTocEntry {
                idx,
                reason: "index out of range".to_owned(),
            })?;
        if entry.static_table_size == 0 {
            return Ok(&[]);
        }
        let base = self.header.static_tables_offset as usize;
        let off = base + entry.static_table_offset as usize;
        let size = entry.static_table_size as usize;
        slice_or_truncated(&self.mmap[..], off, size)
    }

    /// Parse the per-tensor LLVQ payload header (zero-copy slices).
    pub fn parse_llvq_payload(&self, idx: usize) -> Result<LlvqPayload<'_>> {
        let entry = &self.toc[idx];
        if entry.role != Role::Llvq {
            return Err(LeechError::BadTocEntry {
                idx,
                reason: format!("tensor {:?} is not LLVQ (role={:?})", entry.name, entry.role),
            });
        }
        let bytes = self.payload(idx)?;
        LlvqPayload::parse(bytes, &entry.name)
    }

    /// Bytes for the OVERLAY_BLOCK, or `None` if the file has no overlays.
    pub fn overlay_bytes(&self) -> Option<&[u8]> {
        if !self.header.has_overlays() || self.header.overlay_block_offset == 0 {
            return None;
        }
        let off = self.header.overlay_block_offset as usize;
        let bytes = &self.mmap[..];
        if off + overlay::OVERLAY_HEADER_SIZE > bytes.len() {
            return None;
        }
        let total = LE::read_u32(&bytes[off + 4..off + 8]) as usize;
        let end = off.checked_add(total)?;
        // Stop at (file end - 4) for the trailing CRC.
        let crc_floor = bytes.len().saturating_sub(4);
        if end > crc_floor {
            return None;
        }
        Some(&bytes[off..end])
    }

    /// Parse the overlay block, if present.
    pub fn overlays(&self) -> Result<Vec<OverlayEntry<'_>>> {
        match self.overlay_bytes() {
            Some(block) => parse_overlay_block(block),
            None => Ok(Vec::new()),
        }
    }

    /// Stored CRC32 from the last 4 bytes of the file.
    pub fn stored_crc(&self) -> u32 {
        let bytes = &self.mmap[..];
        LE::read_u32(&bytes[bytes.len() - 4..])
    }

    /// Compute and return the CRC32. Does NOT compare; use for diagnostics.
    pub fn compute_crc(&self) -> u32 {
        let bytes = &self.mmap[..];
        crc32fast::hash(&bytes[..bytes.len() - 4])
    }
}

fn slice_or_truncated(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| LeechError::Truncated {
            offset: offset as u64,
            needed: size,
            len: bytes.len() as u64,
        })?;
    if end > bytes.len() {
        return Err(LeechError::Truncated {
            offset: offset as u64,
            needed: size,
            len: bytes.len() as u64,
        });
    }
    Ok(&bytes[offset..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_constant_is_canonical() {
        assert_eq!(EXPECTED_SCHEMA, "LLVQ_NIEMEIER_G24_v1");
    }

    #[test]
    fn slice_truncated_detects_overflow() {
        let bytes = vec![0u8; 100];
        assert!(slice_or_truncated(&bytes, 50, 60).is_err());
        assert!(slice_or_truncated(&bytes, 50, 50).is_ok());
    }
}
