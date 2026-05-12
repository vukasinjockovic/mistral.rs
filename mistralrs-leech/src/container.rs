//! .leech v1 macro layout — mirror of `packer/core/container.py`.
//!
//! Bytes 0..8       MAGIC ("LEECHv01")
//! Bytes 8..72      HEADER (64 B, little-endian fields)
//! manifest_json    UTF-8 JSON at `manifest_json_offset`
//! TOC              `toc_entry_count × 128 B` entries at `toc_offset`
//! STATIC_TABLES    starts at `static_tables_offset` (per-tensor LLVQ tables)
//! PAYLOAD          per-tensor blobs at `payload_offset`
//! OVERLAY_BLOCK    at `overlay_block_offset` if flags bit 0 set
//! Last 4 bytes     CRC32 over `[0, file_size - 4)`

use crate::error::{LeechError, Result};
use byteorder::{ByteOrder, LittleEndian as LE};

pub const MAGIC: [u8; 8] = *b"LEECHv01";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 64;
pub const FIXED_PREFIX_SIZE: usize = 8 + HEADER_SIZE;
pub const TOC_ENTRY_SIZE: usize = 128;
pub const TOC_NAME_LEN: usize = 64;
pub const TOC_SHAPE_MAX_RANK: usize = 8;
pub const ALIGN: u64 = 64;

pub const FLAG_HAS_OVERLAYS: u32 = 1 << 0;

/// Tensor role tag (TOC byte +64).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Llvq = 0,
    Fp8E4m3 = 1,
    Bf16 = 2,
    Fp16Overlay = 3,
}

impl Role {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Role::Llvq),
            1 => Ok(Role::Fp8E4m3),
            2 => Ok(Role::Bf16),
            3 => Ok(Role::Fp16Overlay),
            other => Err(LeechError::BadTocEntry {
                idx: 0,
                reason: format!("unknown role {other}"),
            }),
        }
    }
}

/// Dtype tag (TOC byte +65).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeTag {
    Bf16 = 0,
    Fp8E4m3 = 1,
    Fp16 = 2,
    Fp32 = 3,
    Int8 = 4,
    Packed = 5,
}

impl DtypeTag {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(DtypeTag::Bf16),
            1 => Ok(DtypeTag::Fp8E4m3),
            2 => Ok(DtypeTag::Fp16),
            3 => Ok(DtypeTag::Fp32),
            4 => Ok(DtypeTag::Int8),
            5 => Ok(DtypeTag::Packed),
            other => Err(LeechError::BadTocEntry {
                idx: 0,
                reason: format!("unknown dtype_tag {other}"),
            }),
        }
    }

    /// Bytes per element, or `None` for packed (LLVQ stream — size depends on payload header).
    pub fn elem_size(self) -> Option<usize> {
        match self {
            DtypeTag::Bf16 => Some(2),
            DtypeTag::Fp8E4m3 => Some(1),
            DtypeTag::Fp16 => Some(2),
            DtypeTag::Fp32 => Some(4),
            DtypeTag::Int8 => Some(1),
            DtypeTag::Packed => None,
        }
    }
}

/// 64-byte HEADER. Matches `HEADER_FMT = "<IIQQQQQIIII"`.
#[derive(Debug, Clone)]
pub struct Header {
    pub format_version: u32,
    pub header_size: u32,
    pub manifest_json_offset: u64,
    pub manifest_json_size: u64,
    pub toc_offset: u64,
    pub toc_entry_count: u64,
    pub payload_offset: u64,
    pub static_tables_offset: u32,
    pub overlay_block_offset: u64, // assembled from lo/hi u32 fields
    pub flags: u32,
}

impl Header {
    pub fn unpack(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(LeechError::Truncated {
                offset: 8,
                needed: HEADER_SIZE,
                len: buf.len() as u64,
            });
        }
        let format_version = LE::read_u32(&buf[0..4]);
        let header_size = LE::read_u32(&buf[4..8]);
        let manifest_json_offset = LE::read_u64(&buf[8..16]);
        let manifest_json_size = LE::read_u64(&buf[16..24]);
        let toc_offset = LE::read_u64(&buf[24..32]);
        let toc_entry_count = LE::read_u64(&buf[32..40]);
        let payload_offset = LE::read_u64(&buf[40..48]);
        let static_tables_offset = LE::read_u32(&buf[48..52]);
        let overlay_lo = LE::read_u32(&buf[52..56]) as u64;
        let overlay_hi = LE::read_u32(&buf[56..60]) as u64;
        let flags = LE::read_u32(&buf[60..64]);
        Ok(Header {
            format_version,
            header_size,
            manifest_json_offset,
            manifest_json_size,
            toc_offset,
            toc_entry_count,
            payload_offset,
            static_tables_offset,
            overlay_block_offset: overlay_lo | (overlay_hi << 32),
            flags,
        })
    }

    pub fn has_overlays(&self) -> bool {
        self.flags & FLAG_HAS_OVERLAYS != 0
    }
}

/// 128-byte TOC entry. Matches `TOC_FMT = "<64sBBH8IQQIII"`.
#[derive(Debug, Clone)]
pub struct TocEntry {
    pub name: String,
    pub role: Role,
    pub dtype_tag: DtypeTag,
    pub rank: u16,
    pub shape: Vec<u32>, // only the first `rank` entries are meaningful
    pub payload_offset: u64,
    pub payload_size: u64,
    pub static_table_offset: u32,
    pub static_table_size: u32,
    pub crc32: u32,
}

impl TocEntry {
    pub fn unpack(buf: &[u8], idx: usize) -> Result<Self> {
        if buf.len() < TOC_ENTRY_SIZE {
            return Err(LeechError::BadTocEntry {
                idx,
                reason: format!("entry too short: {} < {}", buf.len(), TOC_ENTRY_SIZE),
            });
        }
        let name_raw = &buf[0..TOC_NAME_LEN];
        let name_end = name_raw.iter().position(|&b| b == 0).unwrap_or(TOC_NAME_LEN);
        let name = std::str::from_utf8(&name_raw[..name_end])
            .map_err(LeechError::from)?
            .to_owned();
        let role = Role::from_u8(buf[64]).map_err(|_| LeechError::BadTocEntry {
            idx,
            reason: format!("unknown role byte {}", buf[64]),
        })?;
        let dtype_tag = DtypeTag::from_u8(buf[65]).map_err(|_| LeechError::BadTocEntry {
            idx,
            reason: format!("unknown dtype_tag byte {}", buf[65]),
        })?;
        let rank = LE::read_u16(&buf[66..68]);
        if rank as usize > TOC_SHAPE_MAX_RANK {
            return Err(LeechError::BadTocEntry {
                idx,
                reason: format!("rank {rank} exceeds {TOC_SHAPE_MAX_RANK}"),
            });
        }
        let mut shape = vec![0u32; rank as usize];
        for i in 0..rank as usize {
            shape[i] = LE::read_u32(&buf[68 + i * 4..72 + i * 4]);
        }
        let payload_offset = LE::read_u64(&buf[100..108]);
        let payload_size = LE::read_u64(&buf[108..116]);
        let static_table_offset = LE::read_u32(&buf[116..120]);
        let static_table_size = LE::read_u32(&buf[120..124]);
        let crc32 = LE::read_u32(&buf[124..128]);
        Ok(TocEntry {
            name,
            role,
            dtype_tag,
            rank,
            shape,
            payload_offset,
            payload_size,
            static_table_offset,
            static_table_size,
            crc32,
        })
    }
}

/// Verify role × dtype_tag matrix from `format_spec.md` §5.
pub fn validate_role_dtype(role: Role, dtype: DtypeTag) -> Result<()> {
    let ok = matches!(
        (role, dtype),
        (Role::Llvq, DtypeTag::Packed)
            | (Role::Fp8E4m3, DtypeTag::Fp8E4m3)
            | (Role::Bf16, DtypeTag::Bf16)
            | (Role::Fp16Overlay, DtypeTag::Fp16)
    );
    if !ok {
        return Err(LeechError::BadTocEntry {
            idx: 0,
            reason: format!("invalid (role={role:?}, dtype={dtype:?}) combination"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_bytes() {
        assert_eq!(&MAGIC, b"LEECHv01");
    }

    #[test]
    fn header_size_constants() {
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(FIXED_PREFIX_SIZE, 72);
        assert_eq!(TOC_ENTRY_SIZE, 128);
    }

    #[test]
    fn role_roundtrip() {
        for v in 0..=3u8 {
            let r = Role::from_u8(v).unwrap();
            assert_eq!(r as u8, v);
        }
        assert!(Role::from_u8(99).is_err());
    }

    #[test]
    fn dtype_tag_roundtrip() {
        for v in 0..=5u8 {
            let d = DtypeTag::from_u8(v).unwrap();
            assert_eq!(d as u8, v);
        }
        assert!(DtypeTag::from_u8(99).is_err());
    }
}
