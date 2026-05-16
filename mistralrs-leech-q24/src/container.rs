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
/// v4 format: K-parallel FSE sub-streams per tile (see plan
/// `2026-05-16_q24t_v4_parallel_substreams.md`).
pub const FORMAT_VERSION: u32 = 4;
/// v3 files are still accepted via a compat shim that forces `num_streams=1`
/// and aliases the new `substream_*` offsets to the v3 `tile_*` arrays.
pub const FORMAT_VERSION_V3_COMPAT: u32 = 3;
pub const HEADER_SIZE: usize = 256;
pub const HEADER_STRUCT_SIZE: usize = 128;
pub const FIXED_PREFIX_SIZE: usize = MAGIC.len() + HEADER_STRUCT_SIZE;
pub const TOC_ENTRY_SIZE: usize = 448;
/// Used bytes within a TOC entry for v4 (was 428 in v3; the new
/// `substream_states_offset` / `substream_nb_totals_offset` u64s occupy
/// bytes 428..444).
pub const TOC_STRUCT_SIZE: usize = 444;
pub const TOC_V3_STRUCT_SIZE: usize = 428;
pub const TENSOR_NAME_BYTES: usize = 200;
pub const SHAPE_MAX_DIMS: usize = 8;
pub const ALIGN: u64 = 64;
pub const SENTINEL_BUCKET: u16 = 0xFFFF;

/// v4 `num_streams` is restricted to powers of two ≤ 32 that divide every
/// supported `tile_size`. The CUDA dispatch + warp-reduce path is only
/// instantiated for these values.
pub const SUPPORTED_NUM_STREAMS: &[u32] = &[1, 2, 4, 8, 16, 32];

/// Check whether `K` is a valid v4 `num_streams`.
pub fn is_supported_num_streams(k: u32) -> bool {
    SUPPORTED_NUM_STREAMS.iter().any(|&v| v == k)
}

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

/// One TOC entry (448 bytes on disk).
/// v3: 428 used + 20 pad. v4: 444 used + 4 pad. Mirrors `TOC_STRUCT` /
/// `TOC_V3_STRUCT` in container.py.
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
    /// v4 only: K = number of FSE sub-streams per tile. Always 1 for v3 files
    /// (forced by the compat shim regardless of byte content at offset 264..268).
    /// LLVQ tensors must have `num_streams ∈ SUPPORTED_NUM_STREAMS`; other
    /// roles ignore this field.
    pub num_streams: u32,

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

    /// v4 NEW: file offset of the `[n_tiles, K] u16` substream starting states.
    /// In v3 compat mode this aliases `tile_states_offset`.
    pub substream_states_offset: u64,
    /// v4 NEW: file offset of the `[n_tiles, K] u16` per-substream nb totals.
    /// In v3 compat mode this aliases `tile_nb_totals_offset`.
    pub substream_nb_totals_offset: u64,
}

impl TocEntry {
    /// Parse one TOC entry. `format_version` MUST be one of
    /// [`FORMAT_VERSION`] (v4) or [`FORMAT_VERSION_V3_COMPAT`] (v3); the
    /// loader checks this before calling. The version drives the compat shim:
    ///
    /// - v3: `num_streams` is forced to 1 regardless of the byte content at
    ///   offset 264..268 (which held `reserved_u32 = 0` in v3 writers). The
    ///   two v4 substream offsets alias `tile_states_offset` /
    ///   `tile_nb_totals_offset` (so a downstream payload reader can use the
    ///   same code path for both versions).
    /// - v4: `num_streams` is read from offset 264..268 and asserted to be in
    ///   [`SUPPORTED_NUM_STREAMS`] for LLVQ tensors. The new offsets are read
    ///   from bytes 428..436 and 436..444.
    pub fn unpack(buf: &[u8], idx: usize, format_version: u32) -> Result<Self> {
        let min_size = if format_version == FORMAT_VERSION_V3_COMPAT {
            TOC_V3_STRUCT_SIZE
        } else {
            TOC_STRUCT_SIZE
        };
        if buf.len() < min_size {
            return Err(LeechQ24Error::BadTocEntry {
                idx,
                reason: format!("entry too short: {} < {}", buf.len(), min_size),
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
        // [264..268] is `reserved_u32` in v3 (always 0) and `num_streams` in v4.
        let raw_num_streams = LE::read_u32(&buf[264..268]);
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

        // ── v3 vs v4 compat shim ─────────────────────────────────────
        // Version-check FIRST (the format_version arg), then byte-read SECOND.
        // For v3 we ignore raw_num_streams (it's `reserved_u32 = 0` and would
        // cause div-by-zero downstream) and force num_streams = 1, aliasing
        // the new substream_* offsets to the v3 tile_* arrays. This makes a
        // v3 file decode-equivalent to a v4 K=1 file from the consumer's view.
        let (num_streams, substream_states_offset, substream_nb_totals_offset) =
            if format_version == FORMAT_VERSION_V3_COMPAT {
                (1u32, tile_states_offset, tile_nb_totals_offset)
            } else {
                // v4: read the new offsets and validate num_streams.
                let sub_states = LE::read_u64(&buf[428..436]);
                let sub_nb = LE::read_u64(&buf[436..444]);
                if role == Role::LlvqTans {
                    debug_assert!(
                        raw_num_streams > 0,
                        "v4 TocEntry must have num_streams > 0"
                    );
                    if !is_supported_num_streams(raw_num_streams) {
                        return Err(LeechQ24Error::BadTocEntry {
                            idx,
                            reason: format!(
                                "v4 num_streams must be in {{1,2,4,8,16,32}}, got {raw_num_streams}"
                            ),
                        });
                    }
                    if tile_size > 0 && (tile_size as u32) % raw_num_streams != 0 {
                        return Err(LeechQ24Error::BadTocEntry {
                            idx,
                            reason: format!(
                                "v4 num_streams {raw_num_streams} must divide tile_size {tile_size}"
                            ),
                        });
                    }
                    (raw_num_streams, sub_states, sub_nb)
                } else {
                    // Non-LLVQ entries: the K field is meaningless; normalize to 1.
                    let k = if raw_num_streams == 0 { 1 } else { raw_num_streams };
                    (k, sub_states, sub_nb)
                }
            };

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
            num_streams,
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
            substream_states_offset,
            substream_nb_totals_offset,
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
        // v4 TOC entry: 444 used + 4 pad. v3 compat: 428 used + 20 pad.
        assert_eq!(TOC_STRUCT_SIZE, 444);
        assert_eq!(TOC_V3_STRUCT_SIZE, 428);
    }

    #[test]
    fn supported_num_streams_set() {
        for &k in SUPPORTED_NUM_STREAMS {
            assert!(is_supported_num_streams(k));
        }
        for k in [0u32, 3, 5, 6, 7, 9, 12, 24, 33, 64] {
            assert!(!is_supported_num_streams(k), "K={k} unexpectedly supported");
        }
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
