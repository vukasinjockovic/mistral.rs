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
// Byte-packed array accessors. Storing 24 int8 values as 6 uint32 keeps the
// per-thread state in registers (vs spilling to local memory). Access uses
// shift/mask which nvcc lowers to PRMT (~1 cycle) on Ampere+.
__device__ __forceinline__ int8_t pk_get(const uint32_t* a, int i) {
    return static_cast<int8_t>((a[i >> 2] >> ((i & 3) << 3)) & 0xFFu);
}
__device__ __forceinline__ void pk_set(uint32_t* a, int i, int v) {
    int shift = (i & 3) << 3;
    uint32_t mask = ~(0xFFu << shift);
    a[i >> 2] = (a[i >> 2] & mask) | ((static_cast<uint32_t>(v) & 0xFFu) << shift);
}

// perms_rem_k variant on packed rem_scratch.
__device__ __forceinline__ int64_t perms_rem_k_packed(
    const uint32_t* rem_packed, int k, int64_t n_rem
) {
    int64_t result = 1;
    int64_t rem = n_rem;
    for (int idx = 0; idx < k; ++idx) {
        int64_t c = static_cast<int64_t>(pk_get(rem_packed, idx));
        result *= binom_small(rem, c);
        rem -= c;
    }
    return result;
}

// Unrank into packed `out` (uint32_t* with 6 elements covering 24 bytes).
// rem_scratch_packed has 2 elements (8 bytes) — k ≤ 8 fits.
__device__ __forceinline__ void unrank_multiset(
    int64_t  rank,
    const int64_t* dist_vals,
    const int64_t* counts,
    int      k,
    int      n,
    uint32_t* out_packed,
    uint32_t* rem_packed
) {
    // Pack dist_cache as uint32_t[2] (8 bytes — k ≤ 8 fits). Stays in
    // registers vs the int8[16] which the compiler used to spill to stack.
    uint32_t dist_pk[2] = {0, 0};
    for (int i = 0; i < k; ++i) {
        int8_t cnt_val = static_cast<int8_t>(__ldg(&counts[i]));
        int8_t dist_val = static_cast<int8_t>(__ldg(&dist_vals[i]));
        pk_set(dist_pk, i, dist_val);
        pk_set(rem_packed, i, cnt_val);
    }
    int64_t r = rank;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            int cnt = static_cast<int>(pk_get(rem_packed, j));
            int avail = (cnt > 0) ? 1 : 0;
            pk_set(rem_packed, j, cnt - avail);
            int64_t block = (avail == 1) ? perms_rem_k_packed(rem_packed, k, n - i - 1) : 0;
            if (avail == 1 && r < block) {
                pk_set(out_packed, i, static_cast<int>(pk_get(dist_pk, j)));
                goto next_i;
            }
            r -= block * avail;
            pk_set(rem_packed, j, static_cast<int>(pk_get(rem_packed, j)) + avail);
        }
        next_i:;
    }
}

// 24-bit popcount — single PTX popc.
__device__ __forceinline__ int popcount24(uint32_t b) {
    return __popc(b & 0x00FFFFFFu);
}

}  // namespace leech
