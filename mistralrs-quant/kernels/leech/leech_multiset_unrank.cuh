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

// Branchless binomial(n, k) for small k (k ≤ 24).
__device__ __forceinline__ int64_t binom_small(int64_t n, int64_t k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    int64_t r = 1;
    for (int64_t i = 0; i < k; ++i) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

// Multinomial(rem_scratch[0..k]; n_rem) via successive binomials.
__device__ __forceinline__ int64_t perms_rem_k(
    const int64_t* rem_scratch, int k, int64_t n_rem
) {
    int64_t result = 1;
    int64_t rem = n_rem;
    for (int idx = 0; idx < k; ++idx) {
        int64_t c = rem_scratch[idx];
        result *= binom_small(rem, c);
        rem -= c;
    }
    return result;
}

// Unrank into out[0..n]. dist_vals/counts must be the per-class slices.
// rem_scratch length ≥ k.
__device__ __forceinline__ void unrank_multiset(
    int64_t  rank,
    const int64_t* dist_vals,
    const int64_t* counts,
    int      k,
    int      n,
    int64_t* out,
    int64_t* rem_scratch
) {
    for (int i = 0; i < k; ++i) rem_scratch[i] = counts[i];
    int64_t r = rank;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            int64_t cnt = rem_scratch[j];
            int64_t avail = (cnt > 0) ? 1 : 0;
            rem_scratch[j] -= avail;
            int64_t block = (avail == 1) ? perms_rem_k(rem_scratch, k, n - i - 1) : 0;
            if (avail == 1 && r < block) {
                out[i] = dist_vals[j];
                // mark break: keep rem_scratch[j] decremented and exit j-loop
                goto next_i;
            }
            r -= block * avail;
            rem_scratch[j] += avail;
        }
        next_i:;
    }
}

// 24-bit popcount — single PTX popc.
__device__ __forceinline__ int popcount24(uint32_t b) {
    return __popc(b & 0x00FFFFFFu);
}

}  // namespace leech
