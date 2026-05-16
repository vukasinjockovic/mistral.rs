// 13-bit bucket extraction from the packed bucket stream — Path B (inline, no
// prelude unpack).
//
// Layout (matches q24_tans/core/bucket_pack.py::pack_buckets_13bit):
//   Each bucket occupies 13 contiguous bits, LSB-first within bytes and across
//   byte boundaries. The stream is padded by ≥1 byte after the last bucket so
//   the 3-byte read window at the final bucket never OOBs.
//
// One bucket decodes as:
//   bucket = (parity << 12) | (h << 6) | f
// where parity ∈ {0,1}, h, f ∈ [0, 64). Sentinel blocks were packed as 0 by
// the encoder; their indices live in a separate side-list and must be patched
// in by the caller before kernel launch (host-side `is_sentinel` bitmap or
// equivalent). The Q24-tANS production V6 bundle has zero sentinels.

#pragma once
#include <cstdint>

namespace leech_q24 {

constexpr uint32_t BUCKET_BITS = 13;
constexpr uint32_t BUCKET_MASK = (1u << BUCKET_BITS) - 1u; // 0x1FFF

// Extract bucket `i` as a 13-bit value (caller masks to (parity, h, f) parts).
// `packed` must be padded by ≥1 trailing byte for the final bucket.
__device__ __forceinline__ uint32_t extract_bucket_13(
    const uint8_t* __restrict__ packed,
    uint32_t i
) {
    uint32_t bit_pos = i * BUCKET_BITS;
    uint32_t bi = bit_pos >> 3;
    uint32_t bo = bit_pos & 7u;
    // Load 3 contiguous bytes (worst case spans 3 bytes when bo + 13 > 16).
    uint32_t v = (uint32_t)packed[bi]
               | ((uint32_t)packed[bi + 1] << 8)
               | ((uint32_t)packed[bi + 2] << 16);
    return (v >> bo) & BUCKET_MASK;
}

__device__ __forceinline__ void split_bucket(
    uint32_t bucket,
    uint32_t& parity,
    uint32_t& h,
    uint32_t& f
) {
    parity = (bucket >> 12) & 1u;
    h = (bucket >> 6) & 0x3Fu;
    f = bucket & 0x3Fu;
}

} // namespace leech_q24
