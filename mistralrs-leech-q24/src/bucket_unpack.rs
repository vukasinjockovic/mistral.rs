//! 13-bit bucket unpacker (CPU reference path).
//!
//! Each bucket is a 13-bit value `(parity:1, h:6, f:6)` packed LSB-first
//! into a contiguous byte stream. The packer (`pack_buckets_13bit` in
//! `q24_tans/core/bucket_pack.py`) pads the stream by 1 trailing byte so
//! the 3-byte read span at the end never OOBs.
//!
//! Sentinel buckets (0xFFFF, marking non-Λ24 blocks) are stored OUT-OF-BAND
//! as a `u32` index list. Use [`patch_sentinels`] after unpacking to restore
//! them; the packed stream itself contains zeros at those positions.
//!
//! The CUDA decoder will not call into this module — it does its own inline
//! 13-bit extraction (Path B). This is the host-side reference for tests
//! and CPU-only paths.

use crate::error::{LeechQ24Error, Result};
use byteorder::{ByteOrder, LittleEndian as LE};

pub const BUCKET_BITS: usize = 13;
pub const BUCKET_MASK: u16 = (1 << BUCKET_BITS) - 1; // 0x1FFF

/// Bytes needed to hold `n_blocks` 13-bit buckets, including the safety byte.
pub fn packed_buckets_byte_count(n_blocks: usize) -> usize {
    (n_blocks * BUCKET_BITS + 7) / 8 + 1
}

/// Unpack a single bucket. `packed` must include the trailing safety byte.
#[inline]
pub fn unpack_bucket(packed: &[u8], i: usize) -> Result<u16> {
    let bit_pos = i * BUCKET_BITS;
    let bi = bit_pos >> 3;
    let bo = bit_pos & 7;
    if bi + 2 >= packed.len() {
        return Err(LeechQ24Error::Truncated {
            offset: bi as u64,
            needed: 3,
            len: packed.len() as u64,
        });
    }
    let v = (packed[bi] as u32) >> bo
        | (packed[bi + 1] as u32) << (8 - bo as u32 % 8).wrapping_rem(32);
    // Branch-free 3-byte span: always combine 3 bytes.
    let v3 = (packed[bi] as u32)
        | ((packed[bi + 1] as u32) << 8)
        | ((packed[bi + 2] as u32) << 16);
    let _ = v;
    Ok(((v3 >> bo) & BUCKET_MASK as u32) as u16)
}

/// Unpack all `n` buckets into `out`. `out.len()` must be `n`.
pub fn unpack_buckets_13bit(packed: &[u8], n: usize, out: &mut [u16]) -> Result<()> {
    if out.len() != n {
        return Err(LeechQ24Error::BucketUnpackOob {
            idx: out.len(),
            n,
        });
    }
    let required = packed_buckets_byte_count(n);
    if packed.len() < required {
        return Err(LeechQ24Error::Truncated {
            offset: 0,
            needed: required,
            len: packed.len() as u64,
        });
    }
    for i in 0..n {
        let bit_pos = i * BUCKET_BITS;
        let bi = bit_pos >> 3;
        let bo = bit_pos & 7;
        // Span: bo + 13 bits → up to 3 bytes (max bo=7, span = 7+13 = 20 bits).
        let v = (packed[bi] as u32)
            | ((packed[bi + 1] as u32) << 8)
            | ((packed[bi + 2] as u32) << 16);
        out[i] = ((v >> bo) & BUCKET_MASK as u32) as u16;
    }
    Ok(())
}

/// After unpacking, restore the sentinel value `0xFFFF` at every index given
/// by `sentinel_indices` (raw little-endian `u32` bytes from the file).
pub fn patch_sentinels(buckets: &mut [u16], sentinel_indices_bytes: &[u8]) -> Result<()> {
    if sentinel_indices_bytes.len() % 4 != 0 {
        return Err(LeechQ24Error::BadCodebookHeader {
            offset: 0,
            reason: format!(
                "sentinel_indices byte length {} not multiple of 4",
                sentinel_indices_bytes.len()
            ),
        });
    }
    let n_sentinels = sentinel_indices_bytes.len() / 4;
    for k in 0..n_sentinels {
        let idx = LE::read_u32(&sentinel_indices_bytes[k * 4..k * 4 + 4]) as usize;
        if idx >= buckets.len() {
            return Err(LeechQ24Error::BucketUnpackOob {
                idx,
                n: buckets.len(),
            });
        }
        buckets[idx] = crate::container::SENTINEL_BUCKET;
    }
    Ok(())
}

/// Extract `(parity, h, f)` triple from one 13-bit bucket. Mirrors the
/// encoder's bit layout in `q24_tans/core/codec.py::decompose_w_int8_kernel`.
#[inline]
pub fn split_bucket(bucket: u16) -> (u8, u8, u8) {
    let parity = ((bucket >> 12) & 1) as u8;
    let h = ((bucket >> 6) & 0x3F) as u8;
    let f = (bucket & 0x3F) as u8;
    (parity, h, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bytes_formula_matches_python() {
        assert_eq!(packed_buckets_byte_count(0), 1);
        assert_eq!(packed_buckets_byte_count(1), 3); // (13+7)/8 + 1 = 2 + 1
        assert_eq!(packed_buckets_byte_count(8), 14); // (104+7)/8 + 1 = 13 + 1
        assert_eq!(packed_buckets_byte_count(287_000_000), 466_375_001);
    }

    #[test]
    fn roundtrip_aligned_values() {
        // Mirror packer's pack_buckets_13bit for a small set, then unpack.
        let vals = [0u16, 1, 8191, 0x1ABC, 42, 0x0FFF];
        let n = vals.len();
        let mut packed = vec![0u8; packed_buckets_byte_count(n)];
        for (i, &v) in vals.iter().enumerate() {
            let v = (v & BUCKET_MASK) as u64;
            let bit_pos = i * BUCKET_BITS;
            let bi = bit_pos >> 3;
            let bo = bit_pos & 7;
            packed[bi] |= ((v << bo) & 0xFF) as u8;
            if bo + BUCKET_BITS > 8 {
                packed[bi + 1] |= ((v >> (8 - bo)) & 0xFF) as u8;
            }
            if bo + BUCKET_BITS > 16 {
                packed[bi + 2] |= ((v >> (16 - bo)) & 0xFF) as u8;
            }
        }
        let mut out = vec![0u16; n];
        unpack_buckets_13bit(&packed, n, &mut out).unwrap();
        let expected: Vec<u16> = vals.iter().map(|&v| v & BUCKET_MASK).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn split_bucket_decomposes() {
        // bucket = (parity<<12) | (h<<6) | f
        let parity = 1u16;
        let h = 0x2Au16;
        let f = 0x15u16;
        let bucket = (parity << 12) | (h << 6) | f;
        let (p, hh, ff) = split_bucket(bucket);
        assert_eq!(p, parity as u8);
        assert_eq!(hh, h as u8);
        assert_eq!(ff, f as u8);
    }

    #[test]
    fn patch_sentinels_restores_ffff() {
        let mut buckets = vec![0u16, 1, 2, 3, 4];
        let idx_bytes = {
            let mut v = Vec::new();
            v.extend_from_slice(&1u32.to_le_bytes());
            v.extend_from_slice(&3u32.to_le_bytes());
            v
        };
        patch_sentinels(&mut buckets, &idx_bytes).unwrap();
        assert_eq!(buckets, vec![0, 0xFFFF, 2, 0xFFFF, 4]);
    }
}
