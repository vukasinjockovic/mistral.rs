// Combinadic-by-value-desc multiset unrank — port of `_unrank_combinadic_v2` from
// packer/core/leech_decode_njit_v2_combinadic.py.
//
// REPLACES the previous string-lex `unrank_multiset` (vector #2A) wholesale —
// no preservation of legacy bijection. The container .leech MUST have been
// packed under `--bijection combinadic` for the output to be valid; mixing
// produces garbage v_int.
//
// Algorithm (CO-LEX / standard CNS, one value at a time):
//   1. Compute per-stage sizes[v] = C(remaining_n, counts[v]) and the
//      cumulative product products[v] = Π_{u≥v} sizes[u]. O(k).
//   2. For each value v in descending order:
//        a. Extract digit r_v = (rank / products[v+1]) % sizes[v].
//        b. CNS-decode r_v → c_v ascending reduced positions in [0, n_avail).
//        c. Lift each reduced position to original via the j-th set bit of
//           the snapshotted free_mask, then clear from the live mask.
//
// First-cut port: byte-for-byte mirror of the Python reference. No CUDA
// intrinsics (no __popc) so semantics are guaranteed identical. After we
// confirm correctness, swap loop-popcount → __popc and add binary-search
// nth_set_bit for speed.

#pragma once
#include <cstdint>

namespace leech {

// Precomputed binomial(n, k) table for n ≤ 24.
__constant__ int64_t c_binom_table[25][25];

__device__ __forceinline__ int64_t binom_small(int64_t n, int64_t k) {
    if (k < 0 || k > n || n > 24) return 0;
    return c_binom_table[n][k];
}

// Multinomial helper — retained for any caller still wanting it; unused by
// combinadic unrank itself.
__device__ __forceinline__ int64_t perms_rem_k(
    const int8_t* rem_scratch, int k, int64_t n_rem
) {
    int64_t result = 1;
    int64_t rem = n_rem;
    for (int idx = 0; idx < k; ++idx) {
        int64_t c = static_cast<int64_t>(rem_scratch[idx]);
        result *= binom_small(rem, c);
        rem -= c;
    }
    return result;
}

// Combinadic unrank. Signature kept identical to legacy unrank_multiset.
// `rem_scratch` is ignored (kept for ABI compatibility with call sites).
// NOTE: NOT __forceinline__ — local int64 sizes[12]/products[13] arrays plus
// reduced_pos[24] need their own stack frame so inlined call-site state isn't
// disturbed by the spill of these arrays. First-cut correctness; reinstate
// inline after the algorithm passes correctness gates.
__device__ __forceinline__ void unrank_multiset(
    int64_t  rank,
    const int8_t*  dist_vals,
    const uint8_t* counts,
    int      k,
    int      n,
    int8_t*  out,
    int8_t*  /*rem_scratch*/
) {
    // Stage 1: stage sizes & cumulative products. k ≤ 8 for Niemeier Λ24
    // leaders; reserve 12 for headroom.
    int64_t sizes[12];
    int64_t products[13];

    int remaining = n;
    for (int v = 0; v < k; ++v) {
        int c = static_cast<int>(counts[v]);
        sizes[v] = binom_small(remaining, c);
        remaining -= c;
    }
    products[k] = 1;
    for (int v = k - 1; v >= 0; --v) {
        products[v] = products[v + 1] * sizes[v];
    }

    // Stage 2+3: free_mask = low n bits set; per-value digit + CNS + lift.
    // int64 mask suffices for n ≤ 24 (we only use the low 24 bits).
    int64_t free_mask = (static_cast<int64_t>(1) << n) - static_cast<int64_t>(1);

    int reduced_pos[24];

    for (int v = 0; v < k; ++v) {
        int c_v = static_cast<int>(counts[v]);
        if (c_v == 0) continue;

        int64_t r_v = (rank / products[v + 1]) % sizes[v];

        // n_avail = popcount of low n bits of free_mask. Vector #A:
        // single PTX popc vs 24-iter predicated-add loop.
        int n_avail = __popc(static_cast<uint32_t>(free_mask) & 0x00FFFFFFu);

        // CNS decode r_v → c_v ascending reduced positions.
        int64_t r_rem = r_v;
        for (int t = c_v; t > 0; --t) {
            int j = t - 1;
            while (j + 1 <= n_avail - 1 && binom_small(j + 1, t) <= r_rem) {
                ++j;
            }
            reduced_pos[t - 1] = j;
            r_rem -= binom_small(j, t);
        }

        // Lift each reduced position to original via target-th set bit of
        // snapshot, then clear that bit from the live free_mask.
        int64_t snapshot = free_mask;
        for (int t = 0; t < c_v; ++t) {
            int target = static_cast<int>(reduced_pos[t]);
            int count = 0;
            int pos = -1;
            for (int jj = 0; jj < n; ++jj) {
                if ((snapshot >> jj) & static_cast<int64_t>(1)) {
                    if (count == target) { pos = jj; break; }
                    ++count;
                }
            }
            out[pos] = dist_vals[v];
            free_mask &= ~(static_cast<int64_t>(1) << pos);
        }
    }
}

// 24-bit popcount — single PTX popc. Used by leech_decode.cu codeword logic.
__device__ __forceinline__ int popcount24(uint32_t b) {
    return __popc(b & 0x00FFFFFFu);
}

}  // namespace leech
