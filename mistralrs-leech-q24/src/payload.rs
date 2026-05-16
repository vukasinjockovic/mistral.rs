//! Zero-copy accessors for one LLVQ_TANS tensor's payload sections.
//!
//! All offsets come from the per-tensor `TocEntry`; this module just bounds-
//! checks them against the mmapped file and exposes typed slices. Bit-packed
//! sections (packed buckets, beta_idx, offset_idx) come out as raw `&[u8]` —
//! decode them on the consumer side or with [`crate::bucket_unpack`].

use crate::container::TocEntry;
use crate::error::{LeechQ24Error, Result};
use byteorder::{ByteOrder, LittleEndian as LE};

/// Borrowed view of one LLVQ_TANS tensor's encoded payload. All slices alias
/// the mmapped file — no copies.
#[derive(Debug, Clone)]
pub struct LlvqTansPayload<'a> {
    /// Packed bucket stream, 13 bits per block, length = `packed_buckets_byte_count(n_blocks)`.
    pub buckets_packed: &'a [u8],
    /// Raw little-endian `u32` indices into `buckets` for sentinel blocks.
    /// Empty when `sentinel_count == 0`.
    pub sentinel_indices: &'a [u8],
    /// `tile_states[n_tiles]` — final tANS state offset per tile, u16 LE.
    pub tile_states: &'a [u8],
    /// `tile_nb_totals[n_tiles]` — bitstream bit-count per tile, u16 LE.
    pub tile_nb_totals: &'a [u8],
    /// `tile_bitstream[tile_bitstream_words]` — concatenated u64 LE words.
    pub tile_bitstream: &'a [u8],
    /// β codebook indices, bit-packed at 3 bits/elem.
    pub beta_idx_packed: &'a [u8],
    /// Offset codebook indices, bit-packed at 3 bits/elem. Empty when
    /// `k_offset == 0`.
    pub offset_idx_packed: &'a [u8],
    /// β Lloyd centroids — `R × K_beta` f32 LE values.
    pub beta_lloyd: &'a [u8],
    /// Offset Lloyd centroids — `R × K_offset` f32 LE values. Empty when
    /// `k_offset == 0`.
    pub offset_lloyd: &'a [u8],

    pub n_blocks: u64,
    pub n_tiles: u64,
    pub tile_bitstream_words: u64,
    pub sentinel_count: u32,
    pub symbol_set_id: u32,
    pub tile_size: u32,
    pub r: u32,
    pub b: u32,
    pub k_beta: u32,
    pub k_offset: u32,
}

impl<'a> LlvqTansPayload<'a> {
    /// Read one β centroid value (f32, row `r`, codebook index `k`).
    pub fn beta_lloyd_at(&self, r: u32, k: u32) -> Option<f32> {
        if r >= self.r || k >= self.k_beta {
            return None;
        }
        let off = ((r as usize) * (self.k_beta as usize) + k as usize) * 4;
        if off + 4 > self.beta_lloyd.len() {
            return None;
        }
        Some(f32::from_bits(LE::read_u32(
            &self.beta_lloyd[off..off + 4],
        )))
    }

    /// Read one offset centroid value (f32). Returns `None` when the tensor
    /// has no offsets or indices are out of range.
    pub fn offset_lloyd_at(&self, r: u32, k: u32) -> Option<f32> {
        if self.k_offset == 0 || r >= self.r || k >= self.k_offset {
            return None;
        }
        let off = ((r as usize) * (self.k_offset as usize) + k as usize) * 4;
        if off + 4 > self.offset_lloyd.len() {
            return None;
        }
        Some(f32::from_bits(LE::read_u32(
            &self.offset_lloyd[off..off + 4],
        )))
    }

    /// Number of beta_idx entries (= n_blocks).
    pub fn n_beta_idx(&self) -> u64 {
        self.n_blocks
    }
}

/// Build a payload view from a TOC entry, slicing into the full mmapped buffer.
pub(crate) fn build_llvq_tans_payload<'a>(
    entry: &TocEntry,
    file_bytes: &'a [u8],
) -> Result<LlvqTansPayload<'a>> {
    if !entry.is_llvq_tans() {
        return Err(LeechQ24Error::NotLlvqTans {
            name: entry.name.clone(),
            role: entry.role as u8,
        });
    }
    let sl = |off: u64, sz: u64| -> Result<&'a [u8]> {
        slice_or_truncated(file_bytes, off as usize, sz as usize)
    };

    let buckets_packed = sl(entry.buckets_offset, entry.buckets_packed_bytes)?;
    let sentinel_indices = if entry.sentinel_count > 0 {
        sl(entry.sentinel_indices_offset, entry.sentinel_count as u64 * 4)?
    } else {
        &[]
    };
    let tile_states = sl(entry.tile_states_offset, entry.n_tiles * 2)?;
    let tile_nb_totals = sl(entry.tile_nb_totals_offset, entry.n_tiles * 2)?;
    let tile_bitstream = sl(entry.tile_bitstream_offset, entry.tile_bitstream_words * 8)?;
    let beta_idx_packed = if entry.beta_idx_bytes > 0 {
        sl(entry.beta_idx_offset, entry.beta_idx_bytes)?
    } else {
        &[]
    };
    let offset_idx_packed = if entry.offset_idx_bytes > 0 {
        sl(entry.offset_idx_offset, entry.offset_idx_bytes)?
    } else {
        &[]
    };
    let beta_lloyd = if entry.beta_lloyd_bytes > 0 {
        sl(entry.beta_lloyd_offset, entry.beta_lloyd_bytes)?
    } else {
        &[]
    };
    let offset_lloyd = if entry.offset_lloyd_bytes > 0 {
        sl(entry.offset_lloyd_offset, entry.offset_lloyd_bytes)?
    } else {
        &[]
    };

    Ok(LlvqTansPayload {
        buckets_packed,
        sentinel_indices,
        tile_states,
        tile_nb_totals,
        tile_bitstream,
        beta_idx_packed,
        offset_idx_packed,
        beta_lloyd,
        offset_lloyd,
        n_blocks: entry.n_blocks,
        n_tiles: entry.n_tiles,
        tile_bitstream_words: entry.tile_bitstream_words,
        sentinel_count: entry.sentinel_count,
        symbol_set_id: entry.symbol_set_id,
        tile_size: entry.tile_size,
        r: entry.r,
        b: entry.b,
        k_beta: entry.k_beta,
        k_offset: entry.k_offset,
    })
}

/// Slice helper that surfaces a typed `Truncated` error.
pub(crate) fn slice_or_truncated(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(size)
        .ok_or(LeechQ24Error::Truncated {
            offset: offset as u64,
            needed: size,
            len: bytes.len() as u64,
        })?;
    if end > bytes.len() {
        return Err(LeechQ24Error::Truncated {
            offset: offset as u64,
            needed: size,
            len: bytes.len() as u64,
        });
    }
    Ok(&bytes[offset..end])
}
