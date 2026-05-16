//! Bit-extract equivalence test for leech_q24 GEMV Path A.
//!
//! The v0 GEMV inner loop reads `nb` bits one-by-one from `tile_bitstream`,
//! producing 1 dependent 64-bit load per inner iteration (the 721-load
//! scoreboard chain identified by Nsight). Path A replaces this with a
//! single u64 read (rarely 2, on cross-word straddles) per `nb`-bit symbol.
//!
//! This test asserts the batch extractor is **bit-equal** to v0's serial
//! loop on:
//!   1. The hand-verified oracle from the plan §2.6.
//!   2. ~1M random `(bit_off, nb_left, nb)` triples spanning realistic
//!      ranges and intentionally exercising cross-word straddles.
//!
//! Pure Rust, no CUDA dependency. Mirrors the CUDA helper
//! `extract_nb_bits_from_window` in `kernels/leech_q24/leech_q24_decode.cu`.

/// Exact port of the v0 serial loop in
/// `leech_q24_decode.cu:279-286` (the GEMV variant) and `:121-129`
/// (the decode-only variant). Reads `nb` bits ending at absolute bit
/// position `bit_off + nb_left - 1` (inclusive), MSB-first into the
/// returned u32. `tile_bitstream` is little-endian u64 words, with
/// bit at absolute position `p` living at
/// `(tile_bitstream[p >> 6] >> (p & 63)) & 1`.
fn v0_per_bit_loop(bitstream: &[u64], bit_off: u64, nb_left: i32, nb: u32) -> u32 {
    let mut bits_val: u32 = 0;
    for k in 0..nb {
        let pos = nb_left - 1 - (k as i32);
        let abs_bit = bit_off + (pos as u64);
        let word = bitstream[(abs_bit >> 6) as usize];
        let bit_idx = (abs_bit & 63) as u32;
        let bit = ((word >> bit_idx) & 1) as u32;
        bits_val = (bits_val << 1) | bit;
    }
    bits_val
}

/// Bit-exact equivalent of `v0_per_bit_loop`, but issuing a single u64
/// load (plus one more on cross-word straddles) instead of `nb` dependent
/// 64-bit loads. Identical algorithm and bit-direction to the
/// `__device__` helper `extract_nb_bits_from_window` in
/// `kernels/leech_q24/leech_q24_decode.cu`.
fn batch_extract(bitstream: &[u64], bit_off: u64, nb_left: i32, nb: u32) -> u32 {
    let low_pos: u64 = bit_off + (nb_left as u64) - (nb as u64);
    let word_idx: u64 = low_pos >> 6;
    let bit_idx: u32 = (low_pos & 63) as u32;
    let w0: u64 = bitstream[word_idx as usize];
    let bits: u64 = if bit_idx + nb <= 64 {
        (w0 >> bit_idx) & ((1u64 << nb) - 1)
    } else {
        let w1: u64 = bitstream[(word_idx + 1) as usize];
        let lo_n: u32 = 64 - bit_idx;
        let lo_bits: u64 = w0 >> bit_idx;
        let hi_bits: u64 = (w1 & ((1u64 << (nb - lo_n)) - 1)) << lo_n;
        (lo_bits | hi_bits) & ((1u64 << nb) - 1)
    };
    bits as u32
}

/// Minimal xorshift64* RNG so this test has no external deps.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid zero state.
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn gen_range_u32(&mut self, lo: u32, hi_inclusive: u32) -> u32 {
        let span = (hi_inclusive - lo + 1) as u64;
        ((self.next_u64() % span) as u32) + lo
    }
    fn gen_range_u64(&mut self, lo: u64, hi_inclusive: u64) -> u64 {
        let span = hi_inclusive - lo + 1;
        (self.next_u64() % span) + lo
    }
}

#[test]
fn oracle_3ab_nb3() {
    // From plan §2.6 (and re-derived in the commit message):
    //   bitstream = [0x3AB], bit_off=0, nb_left=10, nb=3
    //   bits = positions [7, 8, 9] of 0b 11_1010_1011 = 0b111 = 7
    let bitstream: &[u64] = &[0x3AB];
    let v0 = v0_per_bit_loop(bitstream, 0, 10, 3);
    let bx = batch_extract(bitstream, 0, 10, 3);
    assert_eq!(v0, 7, "v0 oracle: 0x3AB / nb_left=10 / nb=3 must be 7");
    assert_eq!(bx, 7, "batch oracle: 0x3AB / nb_left=10 / nb=3 must be 7");
    assert_eq!(v0, bx);
}

#[test]
fn cross_boundary_at_word_seam() {
    // Construct a bitstream where the next 8 bits begin at absolute
    // position 60 — straddling the 64-bit word boundary (bits 60..67).
    // Low word bits 60..63 = 0b1011, high word bits 0..3 = 0b1101.
    // Whole 8-bit slab packed LSB-first = 0b1101_1011 = 0xDB.
    let mut bitstream = [0u64; 2];
    bitstream[0] = 0b1011u64 << 60;
    bitstream[1] = 0b1101u64;
    // Frame as nb_left=68, nb=8 so low_pos = 0 + 68 - 8 = 60 (straddle).
    let v0 = v0_per_bit_loop(&bitstream, 0, 68, 8);
    let bx = batch_extract(&bitstream, 0, 68, 8);
    assert_eq!(v0, 0xDB);
    assert_eq!(bx, 0xDB);
}

#[test]
fn nb_eq_1_is_lsb_of_top_position() {
    // For nb=1, bits_val is just the bit at absolute position
    // (bit_off + nb_left - 1).
    let bitstream: &[u64] = &[0xAA55_AA55_AA55_AA55u64, 0x0123_4567_89AB_CDEFu64];
    for nb_left in 1..=128i32 {
        for bit_off in [0u64, 1, 7, 32, 63, 64, 65, 100] {
            // Skip out-of-range frames.
            if (bit_off + nb_left as u64) > 128 {
                continue;
            }
            let v0 = v0_per_bit_loop(bitstream, bit_off, nb_left, 1);
            let bx = batch_extract(bitstream, bit_off, nb_left, 1);
            assert_eq!(v0, bx, "mismatch at nb_left={} bit_off={}", nb_left, bit_off);
        }
    }
}

#[test]
fn random_million_triples_match_v0() {
    // Bitstream: 64 u64 words = 4096 bits. Enough to exercise mid- and
    // end-of-buffer accesses without huge memory traffic. Random content.
    let mut rng = XorShift64::new(0xC0FFEE_BABE_CAFEu64);
    const N_WORDS: usize = 64;
    const N_BITS: u64 = (N_WORDS as u64) * 64;
    let mut bitstream = [0u64; N_WORDS];
    for w in bitstream.iter_mut() {
        *w = rng.next_u64();
    }

    // Hot-path symbol widths in q24-tANS are typically 1..=8. We test the
    // full range that the v0 loop legitimately handles (nb is read from
    // an 8-bit field but `(1 << nb) - 1` overflows at nb=64; the kernel
    // never sees nb > ~11). Cap at 31 so `(1u64 << nb) - 1` is well-
    // defined in u32 destination.
    let n_iters = 1_000_000u32;
    let mut straddle_count = 0u32;
    for _ in 0..n_iters {
        let nb: u32 = rng.gen_range_u32(1, 11);
        // nb_left ≥ nb, bit_off + nb_left ≤ N_BITS so the high bit
        // (bit_off + nb_left - 1) is in range. low_pos = bit_off + nb_left - nb
        // is ≥ bit_off ≥ 0, so always valid.
        let max_nb_left = N_BITS - 1; // leave room for bit_off=0
        let nb_left = rng.gen_range_u64(nb as u64, max_nb_left.min(256)) as i32;
        let max_bit_off = N_BITS - (nb_left as u64);
        let bit_off = rng.gen_range_u64(0, max_bit_off);

        let v0 = v0_per_bit_loop(&bitstream, bit_off, nb_left, nb);
        let bx = batch_extract(&bitstream, bit_off, nb_left, nb);
        if v0 != bx {
            panic!(
                "mismatch: bit_off={} nb_left={} nb={} v0={:#x} bx={:#x}",
                bit_off, nb_left, nb, v0, bx
            );
        }

        // Count straddles to confirm the test exercises the two-word path.
        let low_pos = bit_off + (nb_left as u64) - (nb as u64);
        if (low_pos & 63) as u32 + nb > 64 {
            straddle_count += 1;
        }
    }

    // ~nb/64 of triples should straddle on average; for nb ∈ [1,11] mean
    // is roughly 5% — assert at least a few thousand to confirm coverage.
    assert!(
        straddle_count > 5_000,
        "expected substantial cross-word coverage, got {}",
        straddle_count
    );
}
