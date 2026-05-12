// Branchless bit extraction from the .leech body stream.
//
// Layout (matches packer/core/bit_pack_njit.py):
//   Per block: [ i_global (idx_bits) | beta_idx (3) | offset_idx (3 or 0) ]
//   Bits packed MSB-first into bytes; bytes little-endian within u64 words.
//
// The bit_off for block `block_id` is `block_id * W` where W = idx_bits + 3
// + (has_offset ? 3 : 0). To extract W ≤ 60 bits starting at any bit position,
// we load two consecutive u64 words, byte-reverse each (to convert
// MSB-first-within-byte / LE-byte-order to a contiguous MSB-first bitstream),
// then do a 128-bit logical shift.
//
// All ops branchless except the bit-spans-boundary check.

#pragma once
#include <cstdint>

namespace leech {

// ──────────────────────────────────────────────────────────────────────────
// Byte-reverse a uint64_t in 4 PTX prmt ops.
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    // For now use the portable form. Replace with prmt.b32 inline-asm in tune phase.
    uint64_t r = 0;
    r |= ((x >> 56) & 0xFFull) << 0;
    r |= ((x >> 48) & 0xFFull) << 8;
    r |= ((x >> 40) & 0xFFull) << 16;
    r |= ((x >> 32) & 0xFFull) << 24;
    r |= ((x >> 24) & 0xFFull) << 32;
    r |= ((x >> 16) & 0xFFull) << 40;
    r |= ((x >>  8) & 0xFFull) << 48;
    r |= ((x >>  0) & 0xFFull) << 56;
    return r;
}

// ──────────────────────────────────────────────────────────────────────────
// Extract W ≤ 60 bits from the packed stream starting at bit position bit_off.
// Returns the value right-justified in a u64.
//
// Stream layout (MSB-first within each byte):
//   byte0 bit7 | byte0 bit6 | ... | byte0 bit0 | byte1 bit7 | ...
//
// Implementation: two aligned 64-bit loads, byte-swap each to get an MSB-first
// bitstream in `lo`/`hi`, then shift-extract W bits starting at bit_in_w
// within `lo`.
__device__ __forceinline__ uint64_t extract_bits(
    const uint8_t* __restrict__ packed,
    uint64_t bit_off,
    int W
) {
    uint64_t word_idx = bit_off >> 6;            // 64-bit word index
    int bit_in_w = static_cast<int>(bit_off & 63);  // 0..63

    // Load two consecutive u64 words. Cast is safe: the stream is at least
    // (n_blocks * W + 63) / 64 + 1 words long if we tail-pad by 8 bytes.
    const uint64_t* words = reinterpret_cast<const uint64_t*>(packed);
    uint64_t lo = bswap64(words[word_idx]);
    uint64_t hi = bswap64(words[word_idx + 1]);

    uint64_t val;
    if (bit_in_w + W <= 64) {
        val = (lo >> (64 - bit_in_w - W)) & ((W == 64) ? ~0ull : ((1ull << W) - 1ull));
    } else {
        int take_lo = 64 - bit_in_w;
        int take_hi = W - take_lo;
        uint64_t lo_part = lo & (((bit_in_w == 0) ? ~0ull : ((1ull << take_lo) - 1ull)));
        uint64_t hi_part = hi >> (64 - take_hi);
        val = (lo_part << take_hi) | hi_part;
    }
    return val;
}

// ──────────────────────────────────────────────────────────────────────────
// Convenience: pull (i_global, beta_idx, offset_idx) from one block.
// idx_bits is template-parameterized so the compiler can fold the masks.
template<int IDX_BITS, int BETA_BITS, int OFFSET_BITS>
__device__ __forceinline__ void unpack_block_indices(
    const uint8_t* __restrict__ packed,
    uint32_t block_id,
    uint64_t& i_global,
    uint32_t& beta_idx,
    uint32_t& offset_idx
) {
    constexpr int W = IDX_BITS + BETA_BITS + OFFSET_BITS;
    uint64_t bit_off = static_cast<uint64_t>(block_id) * static_cast<uint64_t>(W);
    uint64_t val = extract_bits(packed, bit_off, W);

    // val LSB-first contains: offset_idx (OFFSET_BITS) | beta_idx (BETA_BITS) | i_global (IDX_BITS)
    if constexpr (OFFSET_BITS > 0) {
        offset_idx = static_cast<uint32_t>(val & ((1ull << OFFSET_BITS) - 1));
        val >>= OFFSET_BITS;
    } else {
        offset_idx = 0;
    }
    beta_idx = static_cast<uint32_t>(val & ((1ull << BETA_BITS) - 1));
    val >>= BETA_BITS;
    i_global = val & ((IDX_BITS == 64) ? ~0ull : ((1ull << IDX_BITS) - 1ull));
}

}  // namespace leech
