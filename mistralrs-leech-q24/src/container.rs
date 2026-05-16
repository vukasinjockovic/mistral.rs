//! LEECHQ24 v3 macro layout — mirror of
//! `antsquant/production/packer/q24_tans/core/container.py`.
//!
//! Bytes 0..8       MAGIC ("LEECHQ24")
//! Bytes 8..136     HEADER struct (128 B little-endian)
//! Bytes 136..256   header pad to HEADER_SIZE
//! manifest_json    UTF-8 JSON at `manifest_offset`
//! codebook_blob    per-symbol-set FSE decode tables at `codebook_offset`
//! TOC              `toc_entry_count × 448 B` entries at `toc_offset`
//! PAYLOAD          per-tensor blobs at `payload_offset` (64-byte aligned blocks)
//! Last 4 bytes     CRC32 over `[0, file_size - 4)`
//!
//! Endianness: little-endian throughout. All payload offsets in TOC entries are
//! absolute file offsets, aligned to 64 bytes.

use crate::error::{LeechQ24Error, Result};
use byteorder::{ByteOrder, LittleEndian as LE};

pub const MAGIC: [u8; 8] = *b"LEECHQ24";
pub const FORMAT_VERSION: u32 = 3;
pub const HEADER_SIZE: usize = 256;
pub const HEADER_STRUCT_SIZE: usize = 128;
pub const FIXED_PREFIX_SIZE: usize = MAGIC.len() + HEADER_STRUCT_SIZE;
pub const TOC_ENTRY_SIZE: usize = 448;
pub const TOC_STRUCT_SIZE: usize = 428;
pub const TENSOR_NAME_BYTES: usize = 200;
pub const SHAPE_MAX_DIMS: usize = 8;
pub const ALIGN: u64 = 64;
pub const SENTINEL_BUCKET: u16 = 0xFFFF;

/// Tensor role tag (TOC byte +200).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Forney-Q24 + 4-codebook (parity, pat) tANS.
    LlvqTans = 0,
    Fp8Passthrough = 1,
    Bf16Passthrough = 2,
    Fp16Overlay = 3,
}

impl Role {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Role::LlvqTans),
            1 => Ok(Role::Fp8Passthrough),
            2 => Ok(Role::Bf16Passthrough),
            3 => Ok(Role::Fp16Overlay),
            other => Err(LeechQ24Error::BadTocEntry {
                idx: 0,
                reason: format!("unknown role {other}"),
            }),
        }
    }
}

/// Dtype tag (TOC byte +201).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeTag {
    Bf16 = 0,
    Fp8E4m3 = 1,
    Fp16 = 2,
    Fp32 = 3,
    Int8 = 4,
    Packed = 5,
    Uint8 = 6,
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
            6 => Ok(DtypeTag::Uint8),
            other => Err(LeechQ24Error::BadTocEntry {
                idx: 0,
                reason: format!("unknown dtype_tag {other}"),
            }),
        }
    }

    pub fn elem_size(self) -> Option<usize> {
        match self {
            DtypeTag::Bf16 => Some(2),
            DtypeTag::Fp8E4m3 => Some(1),
            DtypeTag::Fp16 => Some(2),
            DtypeTag::Fp32 => Some(4),
            DtypeTag::Int8 => Some(1),
            DtypeTag::Uint8 => Some(1),
            DtypeTag::Packed => None,
        }
    }
}

/// HEADER struct (128 bytes after the 8-byte magic). Mirrors `HEADER_STRUCT`
/// in container.py.
#[derive(Debug, Clone)]
pub struct Header {
    pub format_version: u32,
    pub header_size: u32,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub codebook_offset: u64,
    pub codebook_size: u64,
    pub toc_offset: u64,
    pub toc_size: u64,
    pub toc_entry_count: u32,
    pub tile_size_default: u32,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub n_codebook_sets: u32,
    pub n_llvq_tensors: u32,
    pub n_fp8_tensors: u32,
    pub n_bf16_tensors: u32,
    pub n_overlay_tensors: u32,
    pub reserved_u32: u32,
    pub total_blocks: u64,
    /// BLAKE3-16 hash of the universal pattern_table baked into the decoder.
    /// Used by the loader to confirm the kernel was compiled against the same
    /// table the encoder used.
    pub pattern_table_blake3: [u8; 16],
}

impl Header {
    pub fn unpack(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_STRUCT_SIZE {
            return Err(LeechQ24Error::Truncated {
                offset: 8,
                needed: HEADER_STRUCT_SIZE,
                len: buf.len() as u64,
            });
        }
        let format_version = LE::read_u32(&buf[0..4]);
        let header_size = LE::read_u32(&buf[4..8]);
        let manifest_offset = LE::read_u64(&buf[8..16]);
        let manifest_size = LE::read_u64(&buf[16..24]);
        let codebook_offset = LE::read_u64(&buf[24..32]);
        let codebook_size = LE::read_u64(&buf[32..40]);
        let toc_offset = LE::read_u64(&buf[40..48]);
        let toc_size = LE::read_u64(&buf[48..56]);
        let toc_entry_count = LE::read_u32(&buf[56..60]);
        let tile_size_default = LE::read_u32(&buf[60..64]);
        let payload_offset = LE::read_u64(&buf[64..72]);
        let payload_size = LE::read_u64(&buf[72..80]);
        let n_codebook_sets = LE::read_u32(&buf[80..84]);
        let n_llvq_tensors = LE::read_u32(&buf[84..88]);
        let n_fp8_tensors = LE::read_u32(&buf[88..92]);
        let n_bf16_tensors = LE::read_u32(&buf[92..96]);
        let n_overlay_tensors = LE::read_u32(&buf[96..100]);
        let reserved_u32 = LE::read_u32(&buf[100..104]);
        let total_blocks = LE::read_u64(&buf[104..112]);
        let mut pattern_table_blake3 = [0u8; 16];
        pattern_table_blake3.copy_from_slice(&buf[112..128]);
        Ok(Header {
            format_version,
            header_size,
            manifest_offset,
            manifest_size,
            codebook_offset,
            codebook_size,
            toc_offset,
            toc_size,
            toc_entry_count,
            tile_size_default,
            payload_offset,
            payload_size,
            n_codebook_sets,
            n_llvq_tensors,
            n_fp8_tensors,
            n_bf16_tensors,
            n_overlay_tensors,
            reserved_u32,
            total_blocks,
            pattern_table_blake3,
        })
    }
}

/// One TOC entry (448 bytes on disk; 428 used + 20 reserved pad).
/// Mirrors `TOC_STRUCT` in container.py.
#[derive(Debug, Clone)]
pub struct TocEntry {
    pub name: String,
    pub role: Role,
    pub dtype_tag: DtypeTag,
    pub rank: u16,
    pub shape: Vec<u32>, // length == rank

    pub flags: u32,
    pub symbol_set_id: u32,
    pub tile_size: u32,
    pub k_beta: u32,
    pub k_offset: u32,
    pub r: u32,
    pub b: u32,

    // LLVQ_TANS-only:
    pub n_blocks: u64,
    pub n_tiles: u64,
    pub buckets_offset: u64,
    pub buckets_packed_bytes: u64,
    pub sentinel_indices_offset: u64,
    pub sentinel_count: u32,
    pub tile_states_offset: u64,
    pub tile_nb_totals_offset: u64,
    pub tile_bitstream_offset: u64,
    pub tile_bitstream_words: u64,
    pub beta_idx_offset: u64,
    pub beta_idx_bytes: u64,
    pub offset_idx_offset: u64,
    pub offset_idx_bytes: u64,
    pub beta_lloyd_offset: u64,
    pub offset_lloyd_offset: u64,
    pub beta_lloyd_bytes: u64,
    pub offset_lloyd_bytes: u64,

    // Passthrough/overlay only:
    pub passthrough_offset: u64,
    pub passthrough_size: u64,
}

impl TocEntry {
    pub fn unpack(buf: &[u8], idx: usize) -> Result<Self> {
        if buf.len() < TOC_STRUCT_SIZE {
            return Err(LeechQ24Error::BadTocEntry {
                idx,
                reason: format!("entry too short: {} < {}", buf.len(), TOC_STRUCT_SIZE),
            });
        }
        let name_raw = &buf[0..TENSOR_NAME_BYTES];
        let name_end = name_raw
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(TENSOR_NAME_BYTES);
        let name = std::str::from_utf8(&name_raw[..name_end])
            .map_err(LeechQ24Error::from)?
            .to_owned();

        let role = Role::from_u8(buf[200]).map_err(|_| LeechQ24Error::BadTocEntry {
            idx,
            reason: format!("unknown role byte {}", buf[200]),
        })?;
        let dtype_tag =
            DtypeTag::from_u8(buf[201]).map_err(|_| LeechQ24Error::BadTocEntry {
                idx,
                reason: format!("unknown dtype_tag byte {}", buf[201]),
            })?;
        let rank = LE::read_u16(&buf[202..204]);
        if rank as usize > SHAPE_MAX_DIMS {
            return Err(LeechQ24Error::BadTocEntry {
                idx,
                reason: format!("rank {rank} exceeds {SHAPE_MAX_DIMS}"),
            });
        }
        let mut shape_full = [0u32; SHAPE_MAX_DIMS];
        for i in 0..SHAPE_MAX_DIMS {
            shape_full[i] = LE::read_u32(&buf[204 + i * 4..208 + i * 4]);
        }
        let shape: Vec<u32> = shape_full[..rank as usize].to_vec();

        let flags = LE::read_u32(&buf[236..240]);
        let symbol_set_id = LE::read_u32(&buf[240..244]);
        let tile_size = LE::read_u32(&buf[244..248]);
        let k_beta = LE::read_u32(&buf[248..252]);
        let k_offset = LE::read_u32(&buf[252..256]);
        let r = LE::read_u32(&buf[256..260]);
        let b = LE::read_u32(&buf[260..264]);
        // [264..268] reserved_u32
        let n_blocks = LE::read_u64(&buf[268..276]);
        let n_tiles = LE::read_u64(&buf[276..284]);
        let buckets_offset = LE::read_u64(&buf[284..292]);
        let buckets_packed_bytes = LE::read_u64(&buf[292..300]);
        let sentinel_indices_offset = LE::read_u64(&buf[300..308]);
        let sentinel_count = LE::read_u32(&buf[308..312]);
        // [312..316] reserved_u32_2
        let tile_states_offset = LE::read_u64(&buf[316..324]);
        let tile_nb_totals_offset = LE::read_u64(&buf[324..332]);
        let tile_bitstream_offset = LE::read_u64(&buf[332..340]);
        let tile_bitstream_words = LE::read_u64(&buf[340..348]);
        let beta_idx_offset = LE::read_u64(&buf[348..356]);
        let beta_idx_bytes = LE::read_u64(&buf[356..364]);
        let offset_idx_offset = LE::read_u64(&buf[364..372]);
        let offset_idx_bytes = LE::read_u64(&buf[372..380]);
        let beta_lloyd_offset = LE::read_u64(&buf[380..388]);
        let offset_lloyd_offset = LE::read_u64(&buf[388..396]);
        let beta_lloyd_bytes = LE::read_u64(&buf[396..404]);
        let offset_lloyd_bytes = LE::read_u64(&buf[404..412]);
        let passthrough_offset = LE::read_u64(&buf[412..420]);
        let passthrough_size = LE::read_u64(&buf[420..428]);

        Ok(TocEntry {
            name,
            role,
            dtype_tag,
            rank,
            shape,
            flags,
            symbol_set_id,
            tile_size,
            k_beta,
            k_offset,
            r,
            b,
            n_blocks,
            n_tiles,
            buckets_offset,
            buckets_packed_bytes,
            sentinel_indices_offset,
            sentinel_count,
            tile_states_offset,
            tile_nb_totals_offset,
            tile_bitstream_offset,
            tile_bitstream_words,
            beta_idx_offset,
            beta_idx_bytes,
            offset_idx_offset,
            offset_idx_bytes,
            beta_lloyd_offset,
            offset_lloyd_offset,
            beta_lloyd_bytes,
            offset_lloyd_bytes,
            passthrough_offset,
            passthrough_size,
        })
    }

    pub fn is_llvq_tans(&self) -> bool {
        self.role == Role::LlvqTans
    }
}

/// Validate role × dtype combinations the packer is allowed to emit.
pub fn validate_role_dtype(role: Role, dtype: DtypeTag) -> Result<()> {
    let ok = matches!(
        (role, dtype),
        (Role::LlvqTans, DtypeTag::Packed)
            | (Role::Fp8Passthrough, DtypeTag::Fp8E4m3)
            | (Role::Bf16Passthrough, DtypeTag::Bf16)
            | (Role::Bf16Passthrough, DtypeTag::Fp16)
            | (Role::Bf16Passthrough, DtypeTag::Fp32)
            | (Role::Fp16Overlay, _)
    );
    if !ok {
        return Err(LeechQ24Error::BadTocEntry {
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
        assert_eq!(&MAGIC, b"LEECHQ24");
    }

    #[test]
    fn sizes() {
        assert_eq!(HEADER_SIZE, 256);
        assert_eq!(HEADER_STRUCT_SIZE, 128);
        assert_eq!(TOC_ENTRY_SIZE, 448);
        assert_eq!(TOC_STRUCT_SIZE, 428);
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
        for v in 0..=6u8 {
            let d = DtypeTag::from_u8(v).unwrap();
            assert_eq!(d as u8, v);
        }
        assert!(DtypeTag::from_u8(99).is_err());
    }
}
