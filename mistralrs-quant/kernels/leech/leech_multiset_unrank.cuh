// Factoradic multiset unrank — port of `_unrank_multiset_v2` from
// packer/core/leech_decode_njit_v2.py:60-82.
//
// Given a rank `r ∈ [0, multinomial(n; counts))`, produce one permutation of a
// multiset whose distinct values are `dist_vals[]` with multiplicities
// `counts[]`. Output is written to `out[0..n]`.
//
// The only branch is `if r < block: break`, which is warp-uniform within a
// block-decode (every lane has its own r/block per iteration, but each lane
// stays branch-coherent within its own unrank loop). With parity-sort at
// tensor-load time (Phase 4), the outer parity dispatch is also warp-uniform.
//
// `rem_scratch` is a small per-thread int64 array of length ≥ k (k ≤ 8 for
// Niemeier Λ24 leaders) used as the running multiplicity buffer.

#pragma once
#include <cstdint>

namespace leech {

// Precomputed binomial(n, k) table for n ≤ 24. Initialized once per process via
// leech_init_binom() (called from leech_init_tables_ffi). All values fit in
// int64. Table size: 25 × 25 × 8 = 5000 bytes (well within __constant__ budget).
__constant__ int64_t c_binom_table[25][25];

// Branchless binomial(n, k) for small k (k ≤ 24). One __constant__ load —
// previously was a loop of length min(k, n-k).
__device__ __forceinline__ int64_t binom_small(int64_t n, int64_t k) {
    if (k < 0 || k > n || n > 24) return 0;
    return c_binom_table[n][k];
}

// Multinomial(rem_scratch[0..k]; n_rem) via successive binomials.
// rem_scratch is int8 (counts ≤ 24 always fit) — saves 8x register footprint
// vs the original int64 implementation.
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

// Unrank into out[0..n]. dist_vals/counts are int64 device-resident tables;
// out and rem_scratch are int8 caller scratch (values fit — Leech lattice
// coords are bounded by ±32 and counts by 24). This 8x reduction in scratch
// type avoids local-memory spill on the per-thread scratch arrays.
__device__ __forceinline__ void unrank_multiset(
    int64_t  rank,
    const int64_t* dist_vals,
    const int64_t* counts,
    int      k,
    int      n,
    int8_t*  out,
    int8_t*  rem_scratch
) {
    // counts and dist_vals are __device__ read-only — use __ldg to route
    // through the read-only/texture cache (separate from L1).
    int8_t dist_cache[16];  // k ≤ 8 typically; 16 is safe upper bound
    int8_t counts_cache[16];
    for (int i = 0; i < k; ++i) {
        dist_cache[i] = static_cast<int8_t>(__ldg(&dist_vals[i]));
        counts_cache[i] = static_cast<int8_t>(__ldg(&counts[i]));
        rem_scratch[i] = counts_cache[i];
    }
    int64_t r = rank;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            int8_t cnt = rem_scratch[j];
            int8_t avail = (cnt > 0) ? 1 : 0;
            rem_scratch[j] = static_cast<int8_t>(cnt - avail);
            int64_t block = (avail == 1) ? perms_rem_k(rem_scratch, k, n - i - 1) : 0;
            if (avail == 1 && r < block) {
                out[i] = dist_cache[j];
                goto next_i;
            }
            r -= block * avail;
            rem_scratch[j] = static_cast<int8_t>(rem_scratch[j] + avail);
        }
        next_i:;
    }
}

// 24-bit popcount — single PTX popc.
__device__ __forceinline__ int popcount24(uint32_t b) {
    return __popc(b & 0x00FFFFFFu);
}

}  // namespace leech
