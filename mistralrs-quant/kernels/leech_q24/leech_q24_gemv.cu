// LEECHQ24 v3 fused GEMV kernel — Phase B.
//
// Computes y[m, r] = Σ_block β[r, β_idx[bi]] · v[bi, j] · a[m, k_base + j]
//                    + Σ_block offset[r, offset_idx[bi]] · a[m, k_base + j]
// where (β·v + offset) is the bf16 reconstruction of the LLVQ weight per block.
//
// Mapping: ONE THREAD PER TILE. Each thread:
//   1. Walks the tile's 32 blocks in order, decoding the tANS chained state.
//   2. For each block, determines the weight row (= block_idx / B) and the
//      block's column position within that row.
//   3. Computes β[row, β_idx] · v + offset[row, offset_idx], dot products
//      against the activation slice a[m, k_base : k_base + 24].
//   4. Accumulates into up to 2 row-keyed f32 slots (a tile spans at most 2
//      weight rows because TILE_SIZE=32 ≤ typical B≥32; verified for all
//      tensors in the artifact).
//   5. atomicAdd-flushes the partial sums to y_acc_f32[m * R + row].
//
// A small finalize kernel converts y_acc_f32 → bf16. The two-pass setup is
// the standard "atomic accumulate in f32, narrow at the end" pattern used by
// most GEMV implementations to avoid bf16 atomic precision issues.
//
// Phase B.0 (this file): batch M=1, ONE thread per tile, two row slots.
// Phase B.1 (next): parity-sort, larger M, vectorized activation loads.

#include <cstdint>
#include <cuda_bf16.h>
#include "leech_q24_bucket_extract.cuh"
#include "leech_q24_pattern_table.h"

namespace leech_q24 {

constexpr int TABLE_LOG_GEMV = 10;
constexpr int M_TABLE_GEMV = 1 << TABLE_LOG_GEMV;
constexpr uint32_t M_MASK_GEMV = (uint32_t)(M_TABLE_GEMV - 1);
constexpr int N_CODEBOOKS_GEMV = 4;
constexpr int COORDS_PER_BLOCK_GEMV = 24;

// Declared in leech_q24_decode.cu; reused here.
extern __constant__ uint32_t c_decode_tables[N_CODEBOOKS_GEMV * M_TABLE_GEMV];
extern __device__ uint8_t d_pattern_table[PATTERN_TABLE_BYTES];

// ─── Helpers: 3-bit idx unpack (beta_idx / offset_idx) ───────────────────
__device__ __forceinline__ uint32_t extract_3bit(
    const uint8_t* __restrict__ packed, uint32_t i)
{
    uint32_t bit_pos = i * 3u;
    uint32_t bi = bit_pos >> 3;
    uint32_t bo = bit_pos & 7u;
    // 3 bits never span > 2 bytes.
    uint32_t v = (uint32_t)packed[bi] | ((uint32_t)packed[bi + 1] << 8);
    return (v >> bo) & 0x7u;
}

// ─── Phase B.0 fused GEMV kernel (M = 1) ─────────────────────────────────
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,    // may be null when has_offset == 0
    float*          __restrict__ y_acc_f32,       // [R] f32 (M=1)
    uint32_t r_rows,
    uint32_t b_blocks,
    uint32_t n_blocks,
    uint32_t n_tiles,
    uint32_t k_beta,
    uint32_t k_offset,
    int32_t  w_offset,
    int32_t  has_offset
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tiles) return;

    uint32_t state = (uint32_t)tile_states[tid];
    int32_t  nb_left = (int32_t)tile_nb_totals[tid];
    uint64_t bit_off = tile_bit_offsets[tid];

    uint32_t tile_start = tid * (uint32_t)TILE_SIZE;
    uint32_t tile_end = tile_start + (uint32_t)TILE_SIZE;
    if (tile_end > n_blocks) tile_end = n_blocks;

    // Up to 2 row slots per tile (TILE_SIZE ≤ smallest B in production).
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;

    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK_GEMV];

        // (row, col_block) for this block (row-major flat).
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK_GEMV;

        // β and offset lookups (per-block, per-row).
        uint32_t beta_idx_val = extract_3bit(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }

        // Find or allocate the accumulator slot for this row.
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
                // Should never happen given TILE_SIZE ≤ B in production. Flush
                // slot 0 (oldest) and reuse it.
                atomicAdd(&y_acc_f32[acc_row[0]], acc_val[0]);
                acc_row[0] = acc_row[1];
                acc_val[0] = acc_val[1];
                slot = 1;
            }
            acc_row[slot] = row;
            acc_val[slot] = 0.0f;
            if (acc_count < 2) acc_count++;
        }

        float partial = 0.0f;

        #pragma unroll
        for (int j = 0; j < COORDS_PER_BLOCK_GEMV; ++j) {
            uint32_t pat_j = pat_row[j];
            uint32_t cb = (parity << 1) | pat_j;
            uint32_t entry = c_decode_tables[cb * M_TABLE_GEMV + state];
            uint32_t sym  = entry & 0xFFu;
            uint32_t nb   = (entry >> 8) & 0xFFu;
            uint32_t base = entry >> 16;

            uint32_t bits_val = 0;
            for (uint32_t k = 0; k < nb; ++k) {
                int32_t  pos = nb_left - 1 - (int32_t)k;
                uint64_t abs_bit = bit_off + (uint64_t)pos;
                uint64_t word = tile_bitstream[abs_bit >> 6];
                uint32_t bit_idx = (uint32_t)(abs_bit & 63ull);
                uint32_t bit = (uint32_t)((word >> bit_idx) & 1ull);
                bits_val = (bits_val << 1) | bit;
            }
            nb_left -= (int32_t)nb;
            state = (base | bits_val) & M_MASK_GEMV;

            // v reconstruction (Phase A fused path).
            int32_t w_int = (int32_t)sym - w_offset;
            int32_t c_low = (parity == 0u)
                ? ((int32_t)pat_j << 1)
                : ((pat_j != 0u) ? -1 : 1);
            int32_t v_int = c_low + 4 * w_int;

            // weight = β·v + offset, dot with bf16 activation. Activation is
            // single-batch (M=1) so a_act[k] is a 1-D slice indexed by
            // absolute K position within the weight row.
            float w_val = beta_val * (float)v_int + offset_val;
            float a_f32 = __bfloat162float(a_act[k_base + j]);
            partial += w_val * a_f32;
        }

        acc_val[slot] += partial;
    }

    // Flush remaining row slots.
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
}

__global__ void leech_q24_finalize_f32_to_bf16_kernel(
    const float* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(in[i]);
}

// ─── Host launchers ──────────────────────────────────────────────────────
extern "C" void leech_q24_gemv_bf16_cuda(
    const void*     a_act_bf16,
    const uint8_t*  packed_buckets,
    const uint16_t* tile_states,
    const uint16_t* tile_nb_totals,
    const uint64_t* tile_bitstream,
    const uint64_t* tile_bit_offsets,
    const uint8_t*  beta_idx_packed,
    const uint8_t*  offset_idx_packed,
    const float*    beta_lloyd,
    const float*    offset_lloyd,
    float*          y_acc_f32,
    void*           out_y_bf16,
    uint32_t r_rows,
    uint32_t b_blocks,
    uint32_t n_blocks,
    uint32_t n_tiles,
    uint32_t k_beta,
    uint32_t k_offset,
    int32_t  w_offset,
    int32_t  tile_size,
    int32_t  has_offset,
    void*    stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // Zero the f32 accumulator (M=1 → R floats).
    cudaMemsetAsync(y_acc_f32, 0, (size_t)r_rows * sizeof(float), s);

    constexpr int THREADS = 128;
    uint32_t grid = (n_tiles + THREADS - 1) / THREADS;
    (void)tile_size;
    leech_q24_gemv_bf16_kernel<32><<<grid, THREADS, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),
        packed_buckets, tile_states, tile_nb_totals,
        tile_bitstream, tile_bit_offsets,
        beta_idx_packed, offset_idx_packed,
        beta_lloyd, offset_lloyd,
        y_acc_f32,
        r_rows, b_blocks, n_blocks, n_tiles,
        k_beta, k_offset,
        w_offset, has_offset
    );

    // Narrow f32 → bf16 in a tiny finalize kernel.
    uint32_t fgrid = (r_rows + THREADS - 1) / THREADS;
    leech_q24_finalize_f32_to_bf16_kernel<<<fgrid, THREADS, 0, s>>>(
        y_acc_f32,
        reinterpret_cast<__nv_bfloat16*>(out_y_bf16),
        r_rows
    );
}

} // namespace leech_q24
