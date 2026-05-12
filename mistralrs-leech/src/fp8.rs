//! fp8_e4m3 → bf16 conversion — mirror of `packer/core/fp8_decode.py:fp8_e4m3_to_bf16_numpy`.
//!
//! 256-entry uint16 LUT matching the CUDA PTX `cvt.rn.bf16.f8e4m3` semantics.
//! Used for the `lm_head.weight` and `model.embed_tokens.weight` tensors that
//! ride the `.leech` container as raw fp8 bytes (role=Fp8E4m3, dtype_tag=Fp8E4m3).
//!
//! Encoding:
//!   sign (1 bit) | exp (4 bits, bias 7) | mantissa (3 bits)
//!
//! Special values:
//!   exp = 0xF, mant = 0x7  → NaN  (only NaN encoding in e4m3fn)
//!   exp = 0,   mant = 0    → ±0
//!   exp = 0,   mant != 0   → subnormal (mant * 2^-9)
//!   else                   → normal (1.mant * 2^(exp - 7))
//!
//! bf16 layout:
//!   sign (1) | exp (8, bias 127) | mantissa (7)
//!
//! Normalization adjustment: bf16_exp = fp8_exp - 7 + 127 = fp8_exp + 120.

use half::bf16;
use std::sync::OnceLock;

static LUT: OnceLock<[u16; 256]> = OnceLock::new();

fn build_lut() -> [u16; 256] {
    let mut lut = [0u16; 256];
    for byte in 0u32..256 {
        let sign = (byte >> 7) & 1;
        let exp4 = (byte >> 3) & 0xF;
        let mant3 = byte & 0x7;
        let bits: u16 = if exp4 == 0xF && mant3 == 0x7 {
            // NaN. bf16 NaN is 0x7FC0 (or 0xFFC0 with sign).
            (0x7FC0 | ((sign as u16) << 15)) as u16
        } else if exp4 == 0 {
            if mant3 == 0 {
                // ±0
                (sign as u16) << 15
            } else {
                // Subnormal: value = mant3 / 8 * 2^-6 = mant3 * 2^-9.
                let mag = (mant3 as f32) * (1.0f32 / 512.0); // 2^-9
                let val = if sign != 0 { -mag } else { mag };
                f32_to_bf16_bits(val)
            }
        } else {
            // Normal range.
            let bf16_exp = (exp4 as u16) + 120; // exp - 7 + 127
            let bf16_mant = (mant3 as u16) << 4;
            ((sign as u16) << 15) | (bf16_exp << 7) | bf16_mant
        };
        lut[byte as usize] = bits;
    }
    lut
}

/// Round-to-nearest-even f32 → bf16 raw bits, matching numpy's behavior used in the encoder LUT.
fn f32_to_bf16_bits(val: f32) -> u16 {
    let f32_bits = val.to_bits();
    let bf16_bits = (f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16;
    bf16_bits as u16
}

fn lut() -> &'static [u16; 256] {
    LUT.get_or_init(build_lut)
}

/// Decode one fp8_e4m3 byte to bf16 raw bits.
#[inline]
pub fn fp8_byte_to_bf16_bits(b: u8) -> u16 {
    lut()[b as usize]
}

/// Decode one fp8_e4m3 byte to a `bf16` value.
#[inline]
pub fn fp8_byte_to_bf16(b: u8) -> bf16 {
    bf16::from_bits(fp8_byte_to_bf16_bits(b))
}

/// Bulk-convert a slice of fp8_e4m3 bytes into a freshly-allocated `Vec<bf16>`.
pub fn fp8_bytes_to_bf16(bytes: &[u8]) -> Vec<bf16> {
    let table = lut();
    let mut out = Vec::with_capacity(bytes.len());
    for &b in bytes {
        out.push(bf16::from_bits(table[b as usize]));
    }
    out
}

/// Bulk-convert into a caller-provided destination slice (length must match).
pub fn fp8_bytes_to_bf16_into(bytes: &[u8], dst: &mut [bf16]) {
    assert_eq!(bytes.len(), dst.len(), "fp8 length mismatch");
    let table = lut();
    for (b, slot) in bytes.iter().zip(dst.iter_mut()) {
        *slot = bf16::from_bits(table[*b as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_neg_zero() {
        assert_eq!(fp8_byte_to_bf16_bits(0x00), 0x0000);
        assert_eq!(fp8_byte_to_bf16_bits(0x80), 0x8000);
    }

    #[test]
    fn nan_encoding() {
        // 0x7F = 0 1111 111 → NaN
        let pos = fp8_byte_to_bf16(0x7F);
        let neg = fp8_byte_to_bf16(0xFF);
        assert!(pos.is_nan());
        assert!(neg.is_nan());
    }

    #[test]
    fn one_dot_zero() {
        // fp8 1.0 = 0 0111 000 = 0x38
        let v = fp8_byte_to_bf16(0x38);
        assert_eq!(v, bf16::from_f32(1.0));
    }

    #[test]
    fn negative_one() {
        // fp8 -1.0 = 1 0111 000 = 0xB8
        let v = fp8_byte_to_bf16(0xB8);
        assert_eq!(v, bf16::from_f32(-1.0));
    }

    #[test]
    fn power_of_two() {
        // fp8 2.0 = 0 1000 000 = 0x40
        let v = fp8_byte_to_bf16(0x40);
        assert_eq!(v, bf16::from_f32(2.0));
        // fp8 0.5 = 0 0110 000 = 0x30
        let v = fp8_byte_to_bf16(0x30);
        assert_eq!(v, bf16::from_f32(0.5));
    }

    #[test]
    fn subnormal_smallest() {
        // fp8 smallest positive subnormal = 0 0000 001 = 0x01 → 2^-9
        let v = fp8_byte_to_bf16(0x01);
        let expected = bf16::from_f32(1.0f32 / 512.0);
        assert_eq!(v, expected);
    }

    #[test]
    fn lut_is_initialized_lazily() {
        // Force two consecutive accesses; ensure they return the same LUT pointer.
        let _ = fp8_byte_to_bf16_bits(0);
        let _ = fp8_byte_to_bf16_bits(1);
    }
}
