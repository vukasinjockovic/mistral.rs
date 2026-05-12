//! LLVQ tensor payload parser — mirror of `packer/core/payload.py`.
//!
//! Per-tensor blob layout (role=LLVQ, dtype_tag=Packed):
//!     +0   u32 R                      // row count
//!     +4   u32 n_done                 // packed-column count (multiple of 24)
//!     +8   u32 B                      // block count = n_done / 24
//!     +12  u8  ms_used                // max shell observed, ≤ 18
//!     +13  u8  K_beta                 // β codebook entries (8)
//!     +14  u8  K_offset               // offset codebook entries (8 or 0)
//!     +15  u8  has_offset             // 0/1
//!     +16  u16 idx_bits               // bits for i_global per block
//!     +18  u8  leftover_kind          // 0=none, 1=zero-elided, 2=bf16 verbatim
//!     +19  u8[5] pad
//!     +24  beta_codebook   R * K_beta * fp16
//!     +(?) offset_codebook R * K_offset * fp16 (only if has_offset)
//!     +(?) packed_stream   ceil(R * B * (idx_bits + 3 + has_offset*3) / 8) bytes
//!     +(?) leftover        R * leftover_cols * bf16 (only if leftover_kind == 2)

use crate::error::{LeechError, Result};
use byteorder::{ByteOrder, LittleEndian as LE};
use half::f16;

pub const PAYLOAD_HEADER_SIZE: usize = 24;
pub const BETA_BITS: u8 = 3;
pub const OFFSET_BITS: u8 = 3;

/// Parsed payload header + slices into the original payload buffer.
///
/// All slices reference the input `payload: &[u8]` zero-copy.
#[derive(Debug, Clone)]
pub struct LlvqPayload<'a> {
    pub r: u32,
    pub n_done: u32,
    pub b: u32,
    pub ms_used: u8,
    pub k_beta: u8,
    pub k_offset: u8,
    pub has_offset: bool,
    pub idx_bits: u16,
    pub leftover_kind: u8,
    /// Per-block bit-width = idx_bits + 3 + (3 if has_offset else 0).
    pub per_block_bits: u32,

    pub beta_codebook: &'a [u8], // R * K_beta * 2 bytes (fp16 LE)
    pub offset_codebook: &'a [u8], // R * K_offset * 2 bytes (fp16 LE) — empty if !has_offset
    pub packed_stream: &'a [u8], // MSB-first packed bit stream
    pub leftover: &'a [u8],      // bf16 LE if leftover_kind == 2, else empty
}

impl<'a> LlvqPayload<'a> {
    pub fn parse(payload: &'a [u8], tensor_name: &str) -> Result<Self> {
        if payload.len() < PAYLOAD_HEADER_SIZE {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: format!(
                    "payload too short for header: {} < {}",
                    payload.len(),
                    PAYLOAD_HEADER_SIZE
                ),
            });
        }
        let r = LE::read_u32(&payload[0..4]);
        let n_done = LE::read_u32(&payload[4..8]);
        let b = LE::read_u32(&payload[8..12]);
        let ms_used = payload[12];
        let k_beta = payload[13];
        let k_offset = payload[14];
        let has_offset_u8 = payload[15];
        let has_offset = has_offset_u8 != 0;
        let idx_bits = LE::read_u16(&payload[16..18]);
        let leftover_kind = payload[18];
        // payload[19..24] is pad — ignored.

        if idx_bits == 0 || idx_bits > 64 {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: format!("idx_bits out of range: {idx_bits}"),
            });
        }
        if leftover_kind > 2 {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: format!("leftover_kind out of range: {leftover_kind}"),
            });
        }
        if n_done % 24 != 0 || (n_done / 24) != b {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: format!("n_done {n_done} / B {b} mismatch (must satisfy n_done = B * 24)"),
            });
        }
        if k_beta == 0 {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: "k_beta must be > 0".to_owned(),
            });
        }
        if has_offset && k_offset == 0 {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: "has_offset=1 but k_offset=0".to_owned(),
            });
        }
        if !has_offset && k_offset != 0 {
            return Err(LeechError::BadPayloadHeader {
                name: tensor_name.to_owned(),
                reason: format!("has_offset=0 but k_offset={k_offset}"),
            });
        }

        let per_block_bits =
            idx_bits as u32 + BETA_BITS as u32 + if has_offset { OFFSET_BITS as u32 } else { 0 };

        let mut cur = PAYLOAD_HEADER_SIZE;
        let beta_size = r as usize * k_beta as usize * 2;
        let beta_codebook = take(payload, &mut cur, beta_size, tensor_name, "beta_codebook")?;

        let offset_codebook = if has_offset {
            let off_size = r as usize * k_offset as usize * 2;
            take(payload, &mut cur, off_size, tensor_name, "offset_codebook")?
        } else {
            &payload[cur..cur] // empty slice
        };

        let n_blocks = r as usize * b as usize;
        let total_bits = n_blocks * per_block_bits as usize;
        let stream_bytes = (total_bits + 7) / 8;
        let packed_stream = take(payload, &mut cur, stream_bytes, tensor_name, "packed_stream")?;

        let leftover = if leftover_kind == 2 {
            // The remainder of the payload is the bf16 leftover region.
            // Caller validates it against (C - n_done) * R * 2 once the full shape is known.
            &payload[cur..]
        } else {
            &payload[cur..cur]
        };

        Ok(LlvqPayload {
            r,
            n_done,
            b,
            ms_used,
            k_beta,
            k_offset,
            has_offset,
            idx_bits,
            leftover_kind,
            per_block_bits,
            beta_codebook,
            offset_codebook,
            packed_stream,
            leftover,
        })
    }

    /// Interpret `beta_codebook` bytes as `&[f16]` (length = R * K_beta).
    /// Caller may reshape to `(R, K_beta)` row-major.
    pub fn beta_codebook_f16(&self) -> Vec<f16> {
        slice_le_f16(self.beta_codebook)
    }

    /// Interpret `offset_codebook` bytes as `&[f16]` (empty if !has_offset).
    pub fn offset_codebook_f16(&self) -> Vec<f16> {
        slice_le_f16(self.offset_codebook)
    }
}

fn take<'a>(
    buf: &'a [u8],
    cur: &mut usize,
    n: usize,
    tensor_name: &str,
    field: &str,
) -> Result<&'a [u8]> {
    let end = cur
        .checked_add(n)
        .ok_or_else(|| LeechError::BadPayloadHeader {
            name: tensor_name.to_owned(),
            reason: format!("size overflow for {field}"),
        })?;
    if end > buf.len() {
        return Err(LeechError::BadPayloadHeader {
            name: tensor_name.to_owned(),
            reason: format!(
                "payload truncated reading {field}: need {n} at {}, have {}",
                cur,
                buf.len() - *cur
            ),
        });
    }
    let slice = &buf[*cur..end];
    *cur = end;
    Ok(slice)
}

fn slice_le_f16(bytes: &[u8]) -> Vec<f16> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = LE::read_u16(&bytes[i * 2..i * 2 + 2]);
        out.push(f16::from_bits(bits));
    }
    out
}

/// Compute `idx_bits` from the maximum shell index, mirroring
/// `idx_bits_for_ms_flat` in payload.py — used for validation only.
/// Decoders should trust the value stored in the payload header.
pub fn idx_bits_for_total(n_total: u64) -> u16 {
    if n_total <= 1 {
        1
    } else {
        // ceil(log2(n_total))
        (64 - (n_total - 1).leading_zeros()) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_locked() {
        assert_eq!(PAYLOAD_HEADER_SIZE, 24);
    }

    #[test]
    fn idx_bits_basics() {
        assert_eq!(idx_bits_for_total(1), 1);
        assert_eq!(idx_bits_for_total(2), 1);
        assert_eq!(idx_bits_for_total(3), 2);
        assert_eq!(idx_bits_for_total(4), 2);
        assert_eq!(idx_bits_for_total(5), 3);
        // ms=18 lives in the (2^53, 2^54] range → 54 bits.
        assert_eq!(idx_bits_for_total((1u64 << 53) + 1), 54);
    }
}
