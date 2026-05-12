// .leech LLVQ decode-only CUDA kernel.
//
// Phase 3 deliverable: byte-equal CUDA decoder for one LLVQ tensor's body.
// One block-index per thread; produces int8[R, B, 24] in global memory.
// No GEMM, no β·v + offset epilogue — that's Phase 4 (fused decode+GEMM).
//
// The kernel is template-specialized on `ms_used ∈ {13, 18}`, selecting
// between the two constexpr-baked Leech table namespaces emitted by
// antsquant/tools/gen_leech_tables.py.
//
// Algorithm mirrors `packer/core/leech_decode_njit_v2.py` op-for-op:
//   1. Extract i_global from packed_stream bits
//   2. Shell lookup: linear scan over N_cumulative[]
//   3. Class lookup within shell: linear scan over class_cum_offset[]
//   4. Parity dispatch: even (algebraic sign unrank) vs odd (XOR with codeword)
//   5. Place values via F0/F1/multiset unrank
//   6. Write 24 int8 values to out_v_int
//
// All tables come from the constexpr headers leech_tables_ms{13,18}.h.
// Per-class scalars → __constant__ (≤47 KB at ms=18, well within 64 KB).
// Ragged arrays (codewords, multisets) → __device__ (~12 MB at ms=18).

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "leech_bit_extract.cuh"
#include "leech_sign_unrank.cuh"
#include "leech_multiset_unrank.cuh"

// Pick the table set at compile time. Phase 3 targets ms=18 (V6-base).
// Future: kernel template on this so both ms=13 and ms=18 specializations exist.
#include "leech_tables_ms18.h"
namespace ltab = leech::ms18;

namespace leech {

// ──────────────────────────────────────────────────────────────────────────
// __constant__ mirrors of the per-class scalars (~47 KB at ms=18).
// Initialized via the constexpr arrays at compile time.
// ──────────────────────────────────────────────────────────────────────────
__constant__ uint64_t c_N_cumulative[ltab::N_CUMULATIVE_LEN];
__constant__ int32_t  c_shell_class_start[ltab::N_SHELLS_PLUS1];
__constant__ int32_t  c_shell_class_count[ltab::N_SHELLS_PLUS1];
__constant__ int64_t  c_A[ltab::N_CLASSES];
__constant__ int64_t  c_two_B[ltab::N_CLASSES];
__constant__ int64_t  c_orbit_F1[ltab::N_CLASSES];
__constant__ uint8_t  c_parity[ltab::N_CLASSES];
__constant__ int64_t  c_class_cum_offset[ltab::N_CLASSES];
__constant__ uint32_t c_V2_mask[ltab::N_CLASSES];
__constant__ int8_t   c_dep_bit[ltab::N_CLASSES];
__constant__ uint8_t  c_T[ltab::N_CLASSES];

// ──────────────────────────────────────────────────────────────────────────
// __device__ mirrors of the ragged arrays (~12 MB at ms=18).
// Loaded once per process, accessed through L1.
// ──────────────────────────────────────────────────────────────────────────
__device__ uint32_t d_codewords_flat[sizeof(ltab::codewords_flat) / sizeof(uint32_t)];
__device__ int64_t  d_codewords_ofs [sizeof(ltab::codewords_ofs)  / sizeof(int64_t)];
__device__ int64_t  d_f0_dist_flat  [sizeof(ltab::f0_distinct_flat) / sizeof(int64_t)];
__device__ int64_t  d_f0_dist_ofs   [sizeof(ltab::f0_distinct_ofs)  / sizeof(int64_t)];
__device__ int64_t  d_f0_cnt_flat   [sizeof(ltab::f0_counts_flat)   / sizeof(int64_t)];
__device__ int64_t  d_f1_dist_flat  [sizeof(ltab::f1_distinct_flat) / sizeof(int64_t)];
__device__ int64_t  d_f1_dist_ofs   [sizeof(ltab::f1_distinct_ofs)  / sizeof(int64_t)];
__device__ int64_t  d_f1_cnt_flat   [sizeof(ltab::f1_counts_flat)   / sizeof(int64_t)];
__device__ int64_t  d_multi_dist_flat[sizeof(ltab::multiset_distinct_flat)/sizeof(int64_t)];
__device__ int64_t  d_multi_ofs     [sizeof(ltab::multi_ofs)         / sizeof(int64_t)];
__device__ int64_t  d_multi_cnt_flat[sizeof(ltab::multiset_counts_flat)/sizeof(int64_t)];
__device__ int64_t  d_nz_flat       [sizeof(ltab::nz_distinct_desc_flat)/sizeof(int64_t)];
__device__ int64_t  d_nz_ofs        [sizeof(ltab::nz_ofs)            / sizeof(int64_t)];

// One-shot host-side initializer. The host caller copies the constexpr data
// into the __constant__/__device__ arrays before the first kernel launch.
//
// Declared inline-friendly so the Rust FFI side can drive it without a
// separate .cu file.
__host__ inline void leech_init_tables() {
    cudaMemcpyToSymbol(c_N_cumulative,        ltab::N_cumulative,        sizeof(ltab::N_cumulative));
    cudaMemcpyToSymbol(c_shell_class_start,   ltab::shell_class_start,   sizeof(ltab::shell_class_start));
    cudaMemcpyToSymbol(c_shell_class_count,   ltab::shell_class_count,   sizeof(ltab::shell_class_count));
    cudaMemcpyToSymbol(c_A,                   ltab::A,                   sizeof(ltab::A));
    cudaMemcpyToSymbol(c_two_B,               ltab::two_B,               sizeof(ltab::two_B));
    cudaMemcpyToSymbol(c_orbit_F1,            ltab::orbit_F1,            sizeof(ltab::orbit_F1));
    cudaMemcpyToSymbol(c_parity,              ltab::parity,              sizeof(ltab::parity));
    cudaMemcpyToSymbol(c_class_cum_offset,    ltab::class_cum_offset,    sizeof(ltab::class_cum_offset));
    cudaMemcpyToSymbol(c_V2_mask,             ltab::V2_mask,             sizeof(ltab::V2_mask));
    cudaMemcpyToSymbol(c_dep_bit,             ltab::dep_bit,             sizeof(ltab::dep_bit));
    cudaMemcpyToSymbol(c_T,                   ltab::T,                   sizeof(ltab::T));
    cudaMemcpyToSymbol(d_codewords_flat,      ltab::codewords_flat,      sizeof(ltab::codewords_flat));
    cudaMemcpyToSymbol(d_codewords_ofs,       ltab::codewords_ofs,       sizeof(ltab::codewords_ofs));
    cudaMemcpyToSymbol(d_f0_dist_flat,        ltab::f0_distinct_flat,    sizeof(ltab::f0_distinct_flat));
    cudaMemcpyToSymbol(d_f0_dist_ofs,         ltab::f0_distinct_ofs,     sizeof(ltab::f0_distinct_ofs));
    cudaMemcpyToSymbol(d_f0_cnt_flat,         ltab::f0_counts_flat,      sizeof(ltab::f0_counts_flat));
    cudaMemcpyToSymbol(d_f1_dist_flat,        ltab::f1_distinct_flat,    sizeof(ltab::f1_distinct_flat));
    cudaMemcpyToSymbol(d_f1_dist_ofs,         ltab::f1_distinct_ofs,     sizeof(ltab::f1_distinct_ofs));
    cudaMemcpyToSymbol(d_f1_cnt_flat,         ltab::f1_counts_flat,      sizeof(ltab::f1_counts_flat));
    cudaMemcpyToSymbol(d_multi_dist_flat,     ltab::multiset_distinct_flat, sizeof(ltab::multiset_distinct_flat));
    cudaMemcpyToSymbol(d_multi_ofs,           ltab::multi_ofs,           sizeof(ltab::multi_ofs));
    cudaMemcpyToSymbol(d_multi_cnt_flat,      ltab::multiset_counts_flat,sizeof(ltab::multiset_counts_flat));
    cudaMemcpyToSymbol(d_nz_flat,             ltab::nz_distinct_desc_flat, sizeof(ltab::nz_distinct_desc_flat));
    cudaMemcpyToSymbol(d_nz_ofs,              ltab::nz_ofs,              sizeof(ltab::nz_ofs));
}

// ──────────────────────────────────────────────────────────────────────────
// Shell lookup: find m such that N_cumulative[m] ≤ i_global < N_cumulative[m+1].
// Linear scan over ≤20 entries — cheap, predictable. Phase 4 will swap for
// warp-cooperative ballot search.
// ──────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ int lookup_shell(uint64_t i_global) {
    #pragma unroll
    for (int m = 0; m < ltab::MS_MAX + 1; ++m) {
        if (c_N_cumulative[m + 1] > i_global) {
            return m;
        }
    }
    return ltab::MS_MAX;  // shouldn't reach
}

__device__ __forceinline__ int lookup_class(int m, int64_t I_shell) {
    int sc_start = c_shell_class_start[m];
    int sc_count = c_shell_class_count[m];
    int64_t shell_total = static_cast<int64_t>(c_N_cumulative[m + 1] - c_N_cumulative[m]);
    for (int jj = 0; jj < sc_count; ++jj) {
        int g = sc_start + jj;
        int64_t next_off = (jj + 1 < sc_count)
                         ? c_class_cum_offset[g + 1]
                         : shell_total;
        if (I_shell < next_off) {
            return g;
        }
    }
    return sc_start;  // unreachable on valid input
}

// ──────────────────────────────────────────────────────────────────────────
// EVEN-class decode. Maps `_decode_local_even_v2` from leech_decode_njit_v2.py.
// Algebraic sign unrank replaces the legacy valid_signs_flat[] lookup.
// ──────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void decode_even(
    int64_t i_local, int g,
    int8_t* out_x, int8_t* perm_F0, int8_t* perm_F1, int8_t* abs_x,
    int8_t* rem_scratch
) {
    int64_t A   = c_A[g];
    int64_t two_B = c_two_B[g];
    int64_t oF1 = c_orbit_F1[g];

    int64_t r       = i_local % A;
    int64_t rest    = i_local / A;
    int64_t s_idx   = rest % two_B;
    int64_t I_perm  = rest / two_B;
    int64_t rank_F0 = I_perm / oF1;
    int64_t rank_F1 = I_perm % oF1;

    int64_t cw_lo = d_codewords_ofs[g];
    uint32_t b = d_codewords_flat[cw_lo + r];
    int w = popcount24(b);

    // F_0 placement: 24 - w slots
    int n_f0 = 24 - w;
    for (int i = 0; i < 24; ++i) perm_F0[i] = 0;
    if (n_f0 > 0) {
        int64_t f0_lo = d_f0_dist_ofs[g];
        int64_t f0_hi = d_f0_dist_ofs[g + 1];
        int k = static_cast<int>(f0_hi - f0_lo);
        unrank_multiset(
            rank_F0,
            d_f0_dist_flat + f0_lo,
            d_f0_cnt_flat  + f0_lo,
            k, n_f0,
            perm_F0,
            rem_scratch
        );
    }

    // F_1 placement: w slots
    for (int i = 0; i < 24; ++i) perm_F1[i] = 0;
    if (w > 0) {
        int64_t f1_lo = d_f1_dist_ofs[g];
        int64_t f1_hi = d_f1_dist_ofs[g + 1];
        int k = static_cast<int>(f1_hi - f1_lo);
        unrank_multiset(
            rank_F1,
            d_f1_dist_flat + f1_lo,
            d_f1_cnt_flat  + f1_lo,
            k, w,
            perm_F1,
            rem_scratch
        );
    }

    // Interleave abs_x by codeword bits (branchless multiplex).
    int f0_cursor = 0;
    int f1_cursor = 0;
    for (int i = 0; i < 24; ++i) {
        int bit = (b >> i) & 1;
        int8_t v_if_zero = perm_F0[f0_cursor];
        int8_t v_if_one  = (w > 0) ? perm_F1[f1_cursor] : (int8_t)0;
        abs_x[i] = (bit == 1) ? v_if_one : v_if_zero;
        f0_cursor += (1 - bit);
        f1_cursor += bit;
    }

    // Algebraic sign reconstruction. No table lookup.
    uint64_t sign_bits = even_sign_unrank(
        static_cast<uint64_t>(s_idx),
        c_V2_mask[g],
        c_dep_bit[g],
        c_T[g]
    );
    int64_t nz_lo = d_nz_ofs[g];
    int64_t nz_hi = d_nz_ofs[g + 1];
    for (int i = 0; i < 24; ++i) out_x[i] = 0;
    int bit_idx = 0;
    for (int64_t vi_idx = nz_lo; vi_idx < nz_hi; ++vi_idx) {
        int8_t vi = static_cast<int8_t>(d_nz_flat[vi_idx]);
        for (int i = 0; i < 24; ++i) {
            int match = (abs_x[i] == vi) ? 1 : 0;
            int sign = static_cast<int>((sign_bits >> bit_idx) & 1ull) & match;
            int sign_factor = 2 * sign - 1;
            out_x[i] = static_cast<int8_t>(out_x[i] + abs_x[i] * match * sign_factor);
            bit_idx += match;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ODD-class decode. Maps `_decode_local_odd_v2`.
// Sign reconstruction is pure XOR — paper §3.3 step 4.
// ──────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void decode_odd(
    int64_t i_local, int g,
    int8_t* out_x, int8_t* abs_x,
    int8_t* rem_scratch
) {
    int64_t A = c_A[g];
    int64_t r       = i_local % A;
    int64_t I_perm  = i_local / A;

    int64_t cw_lo = d_codewords_ofs[g];
    uint32_t b = d_codewords_flat[cw_lo + r];

    int64_t m_lo = d_multi_ofs[g];
    int64_t m_hi = d_multi_ofs[g + 1];
    int k = static_cast<int>(m_hi - m_lo);
    for (int i = 0; i < 24; ++i) abs_x[i] = 0;
    unrank_multiset(
        I_perm,
        d_multi_dist_flat + m_lo,
        d_multi_cnt_flat  + m_lo,
        k, 24,
        abs_x,
        rem_scratch
    );

    // Branchless XOR sign reconstruction (paper §3.3 step 4).
    for (int i = 0; i < 24; ++i) {
        int v = abs_x[i];
        int parity_low = (v >> 1) & 1;
        int b_bit = static_cast<int>((b >> i) & 1u);
        int sign_neg = parity_low ^ b_bit;
        int sign_factor = 1 - 2 * sign_neg;
        out_x[i] = static_cast<int8_t>(v * sign_factor);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Main kernel: one thread per block-index. Phase 3 — decode-only, no GEMM.
// ──────────────────────────────────────────────────────────────────────────
template<int IDX_BITS, int BETA_BITS, int OFFSET_BITS>
__global__ void leech_decode_v_int_kernel(
    const uint8_t* __restrict__ packed_stream,
    int8_t*        __restrict__ out_v_int,
    uint32_t n_blocks
) {
    uint32_t block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) return;

    // 1. Extract i_global from the packed stream.
    uint64_t i_global;
    uint32_t beta_idx_unused;
    uint32_t offset_idx_unused;
    unpack_block_indices<IDX_BITS, BETA_BITS, OFFSET_BITS>(
        packed_stream, block_id,
        i_global, beta_idx_unused, offset_idx_unused
    );

    // 2. Shell + class lookup.
    int m = lookup_shell(i_global);
    int64_t I_shell = static_cast<int64_t>(i_global - c_N_cumulative[m]);
    int g = lookup_class(m, I_shell);
    int64_t i_local = I_shell - c_class_cum_offset[g];

    // 3. Parity dispatch.
    int8_t out_x[24];
    int8_t abs_x[24];
    int8_t perm_F0[24];
    int8_t perm_F1[24];
    int8_t rem_scratch[8];

    if (c_parity[g] == 0) {
        decode_even(i_local, g, out_x, perm_F0, perm_F1, abs_x, rem_scratch);
    } else {
        decode_odd(i_local, g, out_x, abs_x, rem_scratch);
    }

    // 4. Write 24 int8 values.
    int8_t* row_out = out_v_int + static_cast<size_t>(block_id) * 24;
    #pragma unroll
    for (int k = 0; k < 24; ++k) {
        row_out[k] = out_x[k];
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Phase 4.0a kernel: decode + β·v + offset → bf16 weight tile.
//
// Eliminates the int8 HBM roundtrip from the Phase 3 path. Each thread:
//   1. Extracts (i_global, beta_idx, offset_idx) from packed_stream
//   2. Decodes the 24 int8 v_int values (same as Phase 3)
//   3. Applies LOCKED epilogue: bf16 = RNE(fp32(β * v_int + offset))
//   4. Writes 24 bf16 values to the output weight tile at [row, col_block*24..]
//
// Output layout: bf16[R, B*24] row-major, where (R, B*24) is the dequantized
// weight matrix (excluding any leftover columns — caller concatenates those).
// ──────────────────────────────────────────────────────────────────────────
template<int IDX_BITS, int BETA_BITS, int OFFSET_BITS, bool HAS_OFFSET>
__global__ void leech_decode_bf16_kernel(
    const uint8_t*       __restrict__ packed_stream,
    const __half*        __restrict__ beta_codebook,    // [R, K_beta]
    const __half*        __restrict__ offset_codebook,  // [R, K_offset] or nullptr
    __nv_bfloat16*       __restrict__ out_weight,       // [R, B*24] row-major
    uint32_t r_rows,
    uint32_t b_blocks,
    uint32_t k_beta,
    uint32_t k_offset
) {
    uint32_t block_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_blocks = r_rows * b_blocks;
    if (block_id >= n_blocks) return;

    uint32_t row_idx = block_id / b_blocks;
    uint32_t col_block_idx = block_id - row_idx * b_blocks;

    // 1. Extract i_global, beta_idx, offset_idx for this block.
    uint64_t i_global;
    uint32_t beta_idx;
    uint32_t offset_idx;
    unpack_block_indices<IDX_BITS, BETA_BITS, OFFSET_BITS>(
        packed_stream, block_id, i_global, beta_idx, offset_idx);

    // 2. Shell + class lookup.
    int m = lookup_shell(i_global);
    int64_t I_shell = static_cast<int64_t>(i_global - c_N_cumulative[m]);
    int g = lookup_class(m, I_shell);
    int64_t i_local = I_shell - c_class_cum_offset[g];

    // 3. Decode 24 v_int values.
    int8_t out_x[24];
    int8_t abs_x[24];
    int8_t perm_F0[24];
    int8_t perm_F1[24];
    int8_t rem_scratch[8];

    if (c_parity[g] == 0) {
        decode_even(i_local, g, out_x, perm_F0, perm_F1, abs_x, rem_scratch);
    } else {
        decode_odd(i_local, g, out_x, abs_x, rem_scratch);
    }

    // 4. Apply LOCKED epilogue per CUDA_KERNEL_SPEC §3.1:
    //      w_fp32 = beta * v_int + offset
    //      w_bf16 = RNE(w_fp32)
    // β / offset come from per-row codebooks indexed by the block's beta_idx /
    // offset_idx fields. fp16 → fp32 cast happens via __half2float.
    float beta_f = __half2float(beta_codebook[row_idx * k_beta + beta_idx]);
    float offset_f = 0.0f;
    if constexpr (HAS_OFFSET) {
        offset_f = __half2float(offset_codebook[row_idx * k_offset + offset_idx]);
    }

    size_t row_stride = static_cast<size_t>(b_blocks) * 24;
    __nv_bfloat16* row_out =
        out_weight + static_cast<size_t>(row_idx) * row_stride
                   + static_cast<size_t>(col_block_idx) * 24;

    #pragma unroll
    for (int k = 0; k < 24; ++k) {
        float v_f = beta_f * static_cast<float>(out_x[k]) + offset_f;
        row_out[k] = __float2bfloat16_rn(v_f);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Phase 4.0b kernel: fused decode + β·v + offset + dot product → bf16 output.
//
// One thread = one (m_idx, n_idx) output element. Each thread walks all
// b_blocks weight blocks for its assigned weight row, decoding each block,
// applying the LOCKED epilogue, and accumulating the dot product against
// activations from the m_idx-th batch row. No HBM roundtrip for decoded
// weights — they live in registers/local-mem inside the thread.
//
// Layout:
//   a:   [M, b_blocks*24]   bf16 activations (row-major)
//   out: [M, N_rows]        bf16 output (row-major)
//
// Optimal at batch=1 (M=1): no decode redundancy across threads. At M>1, each
// weight block is decoded M times (one per batch row) — wasteful, but Phase
// 4.0a dequantize+matmul wins there once M >= ~16.
// ──────────────────────────────────────────────────────────────────────────
// Warp-cooperative GEMV: one warp produces one output element. 32 lanes
// split the K-block range (each lane handles b_blocks/32 blocks). At the end,
// the warp reduces partial sums via shfl_xor.
//
// 4 warps per CTA = 128 threads → 4 outputs per CTA. With N=12288, 3072 CTAs.
// On 170 SMs with 8 CTAs/SM, ~2 waves. Plenty of parallelism for latency
// hiding when the decode path stalls on __device__ table reads.
//
// Activations are staged once per CTA in shared memory so all 4 warps share
// the same activation row.
template<int IDX_BITS, int BETA_BITS, int OFFSET_BITS, bool HAS_OFFSET>
__global__ void __launch_bounds__(128, 8) leech_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ a_act,         // [m, b_blocks*24]
    const uint8_t*       __restrict__ packed_stream,
    const __half*        __restrict__ beta_codebook, // [n_rows, k_beta]
    const __half*        __restrict__ offset_codebook, // [n_rows, k_offset] or null
    __nv_bfloat16*       __restrict__ out_y,         // [m, n_rows]
    uint32_t m,
    uint32_t n_rows,
    uint32_t b_blocks,
    uint32_t k_beta,
    uint32_t k_offset
) {
    constexpr int WARPS_PER_CTA = 4;
    constexpr int THREADS_PER_WARP = 32;
    constexpr int THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP;

    int warp_id = threadIdx.x / THREADS_PER_WARP;
    int lane    = threadIdx.x % THREADS_PER_WARP;

    uint32_t n_idx = blockIdx.x * WARPS_PER_CTA + warp_id;
    uint32_t m_idx = blockIdx.y;
    if (n_idx >= n_rows) return;

    // Shared memory layout:
    //   [0 .. K_TOTAL)                          bf16 activations (per CTA, one row)
    //   [K_TOTAL .. K_TOTAL + WARPS*K_CB)       fp32 beta codebook (per warp)
    //   [next .. + WARPS*K_CB)                  fp32 offset codebook (per warp)
    // K_CB caps at 8 (current encoder K_beta = K_offset = 8). Stored as fp32
    // so the hot inner loop reads register-cheap floats.
    constexpr int K_CB_MAX = 8;
    extern __shared__ __nv_bfloat16 a_smem[];
    uint32_t k_total = b_blocks * 24;
    size_t a_stride = static_cast<size_t>(b_blocks) * 24;
    const __nv_bfloat16* a_row = a_act + static_cast<size_t>(m_idx) * a_stride;

    // Cooperative bulk load: each thread strides every THREADS_PER_CTA-th elt.
    for (uint32_t i = threadIdx.x; i < k_total; i += THREADS_PER_CTA) {
        a_smem[i] = a_row[i];
    }

    // Per-warp codebook caches (fp32) live in shared right after the act row.
    float* beta_smem = reinterpret_cast<float*>(a_smem + k_total);
    float* offset_smem = beta_smem + WARPS_PER_CTA * K_CB_MAX;
    if (lane < (int)k_beta) {
        beta_smem[warp_id * K_CB_MAX + lane] =
            __half2float(beta_codebook[n_idx * k_beta + lane]);
    }
    if constexpr (HAS_OFFSET) {
        if (lane < (int)k_offset) {
            offset_smem[warp_id * K_CB_MAX + lane] =
                __half2float(offset_codebook[n_idx * k_offset + lane]);
        }
    }
    __syncthreads();
    const float* beta_row = beta_smem + warp_id * K_CB_MAX;
    const float* offset_row = offset_smem + warp_id * K_CB_MAX;

    int8_t out_x[24];
    int8_t abs_x[24];
    int8_t perm_F0[24];
    int8_t perm_F1[24];
    int8_t rem_scratch[8];

    float partial = 0.0f;

    // Each lane handles every WARP_SIZE-th k_block of this warp's output row.
    for (uint32_t k_block = lane; k_block < b_blocks; k_block += THREADS_PER_WARP) {
        uint32_t block_id = n_idx * b_blocks + k_block;

        uint64_t i_global;
        uint32_t beta_idx;
        uint32_t offset_idx;
        leech::unpack_block_indices<IDX_BITS, BETA_BITS, OFFSET_BITS>(
            packed_stream, block_id, i_global, beta_idx, offset_idx);

        int m_shell = leech::lookup_shell(i_global);
        int64_t I_shell = static_cast<int64_t>(i_global - leech::c_N_cumulative[m_shell]);
        int g = leech::lookup_class(m_shell, I_shell);
        int64_t i_local = I_shell - leech::c_class_cum_offset[g];

        if (leech::c_parity[g] == 0) {
            leech::decode_even(i_local, g, out_x, perm_F0, perm_F1, abs_x, rem_scratch);
        } else {
            leech::decode_odd(i_local, g, out_x, abs_x, rem_scratch);
        }

        float beta_f = beta_row[beta_idx];
        float offset_f = 0.0f;
        if constexpr (HAS_OFFSET) {
            offset_f = offset_row[offset_idx];
        }

        const __nv_bfloat16* a_slice = a_smem + static_cast<size_t>(k_block) * 24;
        #pragma unroll
        for (int k = 0; k < 24; ++k) {
            float w_f = beta_f * static_cast<float>(out_x[k]) + offset_f;
            __nv_bfloat16 w_bf = __float2bfloat16_rn(w_f);
            float w_back = __bfloat162float(w_bf);
            float a_f = __bfloat162float(a_slice[k]);
            partial += a_f * w_back;
        }
    }

    // Warp reduction via butterfly shuffle.
    #pragma unroll
    for (int offset_s = 16; offset_s > 0; offset_s /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset_s);
    }

    if (lane == 0) {
        out_y[m_idx * n_rows + n_idx] = __float2bfloat16_rn(partial);
    }
}

}  // namespace leech

// ──────────────────────────────────────────────────────────────────────────
// C ABI launchers — called from Rust FFI in mistralrs-quant/src/leech/ffi.rs.
//
// Caller is responsible for:
//   1. Calling leech_init_tables() ONCE per process before any decode call.
//   2. Sizing out_v_int to n_blocks * 24 bytes.
//   3. Padding packed_stream by ≥ 8 bytes so the two-u64-load in extract_bits
//      doesn't OOB on the final block.
// ──────────────────────────────────────────────────────────────────────────
extern "C" {

void leech_init_tables_ffi() {
    leech::leech_init_tables();
}

// Decode-only entry point. ms_used selects the template specialization;
// has_offset toggles whether 3 extra bits per block are present.
void leech_decode_v_int_cuda(
    const uint8_t* packed_stream,
    int8_t*        out_v_int,
    uint32_t       n_blocks,
    int            idx_bits,
    int            has_offset,
    cudaStream_t   stream
) {
    constexpr int BLOCK = 128;
    uint32_t grid = (n_blocks + BLOCK - 1) / BLOCK;

    // Compile-time dispatch on (idx_bits, has_offset).
    // ms=18 → idx_bits=54; ms=13 → idx_bits=48. We pick ms=18 for now and
    // template-specialize further in Phase 4.
    if (idx_bits == 54 && has_offset) {
        leech::leech_decode_v_int_kernel<54, 3, 3><<<grid, BLOCK, 0, stream>>>(
            packed_stream, out_v_int, n_blocks);
    } else if (idx_bits == 54 && !has_offset) {
        leech::leech_decode_v_int_kernel<54, 3, 0><<<grid, BLOCK, 0, stream>>>(
            packed_stream, out_v_int, n_blocks);
    } else if (idx_bits == 48 && has_offset) {
        leech::leech_decode_v_int_kernel<48, 3, 3><<<grid, BLOCK, 0, stream>>>(
            packed_stream, out_v_int, n_blocks);
    } else if (idx_bits == 48 && !has_offset) {
        leech::leech_decode_v_int_kernel<48, 3, 0><<<grid, BLOCK, 0, stream>>>(
            packed_stream, out_v_int, n_blocks);
    }
    // Other idx_bits values: silently skipped. Phase 4 adds full dispatch.
}

// Phase 4.0a entry point: decode + β·v + offset → bf16 weight tile.
//
// Caller responsibilities (same as decode-only, plus codebooks):
//   1. Call leech_init_tables_ffi() once per process.
//   2. packed_stream tail-padded by ≥ 8 bytes.
//   3. beta_codebook/offset_codebook are R * K_beta / R * K_offset fp16 entries.
//   4. out_weight is sized R * B * 24 bf16 entries (= 2 * R * B * 24 bytes).
//
// ms=18 → idx_bits=54; ms=13 → idx_bits=48. Same {IDX_BITS, HAS_OFFSET}
// dispatch matrix as the decode-only launcher.
void leech_decode_bf16_cuda(
    const uint8_t* packed_stream,
    const void*    beta_codebook,
    const void*    offset_codebook,
    void*          out_weight_bf16,
    uint32_t       r_rows,
    uint32_t       b_blocks,
    uint32_t       k_beta,
    uint32_t       k_offset,
    int            idx_bits,
    int            has_offset,
    cudaStream_t   stream
) {
    constexpr int BLOCK = 128;
    uint32_t n_blocks = r_rows * b_blocks;
    uint32_t grid = (n_blocks + BLOCK - 1) / BLOCK;

    const __half* beta_h   = reinterpret_cast<const __half*>(beta_codebook);
    const __half* offset_h = reinterpret_cast<const __half*>(offset_codebook);
    __nv_bfloat16* out_bf  = reinterpret_cast<__nv_bfloat16*>(out_weight_bf16);

    if (idx_bits == 54 && has_offset) {
        leech::leech_decode_bf16_kernel<54, 3, 3, true>
            <<<grid, BLOCK, 0, stream>>>(packed_stream, beta_h, offset_h, out_bf,
                                          r_rows, b_blocks, k_beta, k_offset);
    } else if (idx_bits == 54 && !has_offset) {
        leech::leech_decode_bf16_kernel<54, 3, 0, false>
            <<<grid, BLOCK, 0, stream>>>(packed_stream, beta_h, nullptr, out_bf,
                                          r_rows, b_blocks, k_beta, 0);
    } else if (idx_bits == 48 && has_offset) {
        leech::leech_decode_bf16_kernel<48, 3, 3, true>
            <<<grid, BLOCK, 0, stream>>>(packed_stream, beta_h, offset_h, out_bf,
                                          r_rows, b_blocks, k_beta, k_offset);
    } else if (idx_bits == 48 && !has_offset) {
        leech::leech_decode_bf16_kernel<48, 3, 0, false>
            <<<grid, BLOCK, 0, stream>>>(packed_stream, beta_h, nullptr, out_bf,
                                          r_rows, b_blocks, k_beta, 0);
    }
}

// Phase 4.0b: fused decode + epilogue + GEMV (no decoded-weight HBM roundtrip).
//
// Best at batch=1; degrades to M-fold redundant decode at large M.
void leech_gemv_bf16_cuda(
    const void*    a_act_bf16,      // [m, b_blocks*24] bf16
    const uint8_t* packed_stream,
    const void*    beta_codebook,   // fp16
    const void*    offset_codebook, // fp16 or null
    void*          out_y_bf16,      // [m, n_rows] bf16
    uint32_t       m,
    uint32_t       n_rows,
    uint32_t       b_blocks,
    uint32_t       k_beta,
    uint32_t       k_offset,
    int            idx_bits,
    int            has_offset,
    cudaStream_t   stream
) {
    // 1 warp = 1 output, 4 warps/CTA → 4 outputs/CTA.
    constexpr int WARPS_PER_CTA = 4;
    constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
    constexpr int K_CB_MAX = 8;
    uint32_t n_chunks = (n_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    dim3 grid(n_chunks, m, 1);
    dim3 block(THREADS_PER_CTA, 1, 1);
    uint32_t smem_bytes = b_blocks * 24 * sizeof(__nv_bfloat16)
                        + WARPS_PER_CTA * K_CB_MAX * sizeof(float) * 2;

    const __nv_bfloat16* a_h  = reinterpret_cast<const __nv_bfloat16*>(a_act_bf16);
    const __half*        b_h  = reinterpret_cast<const __half*>(beta_codebook);
    const __half*        o_h  = reinterpret_cast<const __half*>(offset_codebook);
    __nv_bfloat16*       y_h  = reinterpret_cast<__nv_bfloat16*>(out_y_bf16);

    if (idx_bits == 54 && has_offset) {
        leech::leech_gemv_bf16_kernel<54, 3, 3, true>
            <<<grid, block, smem_bytes, stream>>>(a_h, packed_stream, b_h, o_h, y_h,
                                          m, n_rows, b_blocks, k_beta, k_offset);
    } else if (idx_bits == 54 && !has_offset) {
        leech::leech_gemv_bf16_kernel<54, 3, 0, false>
            <<<grid, block, smem_bytes, stream>>>(a_h, packed_stream, b_h, nullptr, y_h,
                                          m, n_rows, b_blocks, k_beta, 0);
    } else if (idx_bits == 48 && has_offset) {
        leech::leech_gemv_bf16_kernel<48, 3, 3, true>
            <<<grid, block, smem_bytes, stream>>>(a_h, packed_stream, b_h, o_h, y_h,
                                          m, n_rows, b_blocks, k_beta, k_offset);
    } else if (idx_bits == 48 && !has_offset) {
        leech::leech_gemv_bf16_kernel<48, 3, 0, false>
            <<<grid, block, smem_bytes, stream>>>(a_h, packed_stream, b_h, nullptr, y_h,
                                          m, n_rows, b_blocks, k_beta, 0);
    }
}

}  // extern "C"
