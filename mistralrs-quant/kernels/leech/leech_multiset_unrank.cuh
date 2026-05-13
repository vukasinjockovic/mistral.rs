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

// Precomputed binomial(n, k) table for n ≤ 24.
__constant__ int64_t c_binom_table[25][25];

// Branchless binomial(n, k) — one __constant__ load.
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

// Unrank into out[0..n]. dist_vals (int8) and counts (uint8) are narrowed
// device-resident tables (attack vector #4); out and rem_scratch are int8
// caller scratch. Values fit: Leech-lattice coords |v| ≤ 32, counts ≤ 24.
//
// ─── Vector #2 Option A: incremental multinomial maintenance ──────────────
// Per LLVQ paper §3.3 step 5: "small static tables, integer prefix-sum scans,
// integer division and modulo, and local combinatorial reconstruction" —
// explicitly NOT iterative subtraction with O(k) recomputed multinomials.
//
// We maintain the running multinomial M = (rem_n)! / Π rem_scratch[i]! as an
// invariant. After picking value j: M_new = M * rem_scratch[j] / rem_n. This
// drops per-iteration work from O(k) (the old `perms_rem_k(...)` recompute)
// to O(1) and shortens the dependency chain to a single mul/div per step,
// enabling much better ILP under nvcc.
//
// Complexity: O(n·k) ≈ 24·8 = 192 ops/call (vs prior O(n·k²) ≈ 1500).
// Stub experiment showed unrank is ~82% of decode time; this should drop
// 1.91 ms decode → ~0.5 ms (3-4×), bounded above by the stubbed 350 µs.
__device__ __forceinline__ void unrank_multiset(
    int64_t  rank,
    const int8_t*  dist_vals,
    const uint8_t* counts,
    int      k,
    int      n,
    int8_t*  out,
    int8_t*  rem_scratch
) {
    // Initialize remaining counts.
    for (int i = 0; i < k; ++i) rem_scratch[i] = static_cast<int8_t>(counts[i]);

    // Compute initial multinomial M = n! / Π counts[i]! via the existing
    // perms_rem_k helper. ONE call, O(k) — not per-iteration.
    int64_t M = perms_rem_k(rem_scratch, k, n);

    int64_t r = rank;
    int rem_n = n;

    for (int i = 0; i < n; ++i) {
        // Prefix-sum scan over the remaining distinct values: at each step,
        // block_j = (# completions if we pick j next) = M * rem_scratch[j] / rem_n.
        // Pick the smallest j such that cumulative > r.
        int64_t cum = 0;
        for (int j = 0; j < k; ++j) {
            int8_t cnt = rem_scratch[j];
            if (cnt == 0) continue;
            int64_t block_j = M * static_cast<int64_t>(cnt) / static_cast<int64_t>(rem_n);
            int64_t cum_next = cum + block_j;
            if (r < cum_next) {
                out[i] = dist_vals[j];
                r -= cum;
                M = block_j;                                    // M_new = M_old * c_j / rem_n
                rem_scratch[j] = static_cast<int8_t>(cnt - 1);  // c_j_new = c_j - 1
                break;
            }
            cum = cum_next;
        }
        rem_n -= 1;
    }
}

// 24-bit popcount — single PTX popc.
__device__ __forceinline__ int popcount24(uint32_t b) {
    return __popc(b & 0x00FFFFFFu);
}

}  // namespace leech
