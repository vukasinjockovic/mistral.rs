// LEECHQ24 v3 decode-only kernel.
//
// Phase A deliverable: byte-equal CUDA decoder for one LLVQ_TANS tensor's
// body. One TILE per thread; TILE_SIZE blocks × 24 coords decoded with a
// chained tANS state. Produces int8[R, B, 24] = int8[N_blocks, 24] in HBM.
//
// Algorithm (mirrors q24_tans/core/codec.py::decode_tile_kernel and
// reconstruct_v_from_w fused):
//   1. extract bucket = (parity, h, f) from the 13-bit packed stream
//      (Path B inline extract, no prelude unpack).
//   2. for j in 0..24:
//        cb = (parity << 1) | pattern_table[h, f, j]
//        entry = decode_tables[cb, state]
//        sym  = entry & 0xFF
//        nb   = (entry >> 8) & 0xFF
//        base = entry >> 16
//        bits = read_back_nb_bits(tile_bitstream, bit_off, nb_left, nb)
//        nb_left -= nb
//        state = (base | bits) & (M - 1)
//        w = (int8)sym - W_OFFSET
//        c_low = (parity == 0) ? (2 * pat_j) : (pat_j ? -1 : 1)
//        v = c_low + 4 * w
//        out_v[bi, j] = v
//
// Constants:
//   TABLE_LOG = 10  →  M = 1024
//   N_CODEBOOKS = 4
//   PATTERN_TABLE: 64 × 64 × 24 uint8 (compile-time baked via header)
//   The pattern_table hash MUST match the header field; the Rust loader
//   verifies this before calling init_tables.

#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include "leech_q24_bucket_extract.cuh"
#include "leech_q24_pattern_table.h"

namespace leech_q24 {

constexpr int TABLE_LOG = 10;
constexpr int M_TABLE = 1 << TABLE_LOG;  // 1024
constexpr uint32_t M_MASK = (uint32_t)(M_TABLE - 1);
constexpr int N_CODEBOOKS = 4;
constexpr int COORDS_PER_BLOCK = 24;

// ─── Constant-memory mirror of the FSE decode tables ─────────────────────
// 4 codebooks × 1024 entries × 4 bytes = 16 KB. Fits __constant__ window
// (64 KB total, this is a quarter).
__constant__ uint32_t c_decode_tables[N_CODEBOOKS * M_TABLE];

// ─── Device-memory mirror of the universal pattern_table ─────────────────
// 64 × 64 × 24 = 96 KB. Too big for __constant__ (limit 64 KB), so place in
// __device__ memory and rely on L1/L2 caching. Access pattern within a single
// thread is: same (h, f), sweeping j ∈ [0, 24). One u8 per coord = 24 B per
// block, 768 B per tile, well within L1.
__device__ uint8_t d_pattern_table[PATTERN_TABLE_BYTES];

// Host-callable initializer. Mirrors leech_init_tables_ffi in the baseline.
// Idempotent; the Rust caller wraps it with a OnceLock.
extern "C" void leech_q24_init_tables_ffi(
    const uint32_t* decode_tables_host, // N_CODEBOOKS * M_TABLE entries
    uint32_t symbol_set_id_unused        // reserved (placeholder for sid switch)
) {
    (void)symbol_set_id_unused;
    cudaMemcpyToSymbol(c_decode_tables, decode_tables_host,
                       N_CODEBOOKS * M_TABLE * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_pattern_table, pattern_table,
                       PATTERN_TABLE_BYTES);
}

// ─── Tile decode kernel ─────────────────────────────────────────────────
//
// One thread per tile. Tile = TILE_SIZE consecutive blocks, sharing a chained
// tANS state. Parallelism comes from many tiles (typical tensor: thousands).
//
// Templating on TILE_SIZE lets the compiler unroll the per-tile loops; in
// practice we always launch with TILE_SIZE=32 (matches codec.DEFAULT_TILE_SIZE).
template<int TILE_SIZE>
__global__ void leech_q24_decode_kernel(
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets, // prefix-sum of nb_totals
    int8_t*  __restrict__ out_v,
    uint32_t n_blocks,
    uint32_t n_tiles,
    int32_t  w_offset                              // S=7 → 3, S=9 → 4
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tiles) return;

    uint32_t state = (uint32_t)tile_states[tid];   // already in [0, M)
    int32_t  nb_left = (int32_t)tile_nb_totals[tid];
    uint64_t bit_off = tile_bit_offsets[tid];

    uint32_t tile_start = tid * (uint32_t)TILE_SIZE;
    uint32_t tile_end = tile_start + (uint32_t)TILE_SIZE;
    if (tile_end > n_blocks) tile_end = n_blocks;

    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);

        // Base offset into the pattern table for this (h, f). j sweeps 0..24.
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];

        #pragma unroll
        for (int j = 0; j < COORDS_PER_BLOCK; ++j) {
            uint32_t pat_j = pat_row[j];
            uint32_t cb = (parity << 1) | pat_j;
            uint32_t entry = c_decode_tables[cb * M_TABLE + state];
            uint32_t sym  = entry & 0xFFu;
            uint32_t nb   = (entry >> 8) & 0xFFu;
            uint32_t base = entry >> 16;

            // Read `nb` bits from the back of the tile's bit window.
            // codec.py does: bits = MSB-first over (nb_left-1, nb_left-2, ...).
            // Each bit's absolute file position = bit_off + (nb_left - 1 - k).
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
            state = (base | bits_val) & M_MASK;

            // Reconstruct v from w (fused — saves an HBM int8 roundtrip).
            int32_t w_int = (int32_t)sym - w_offset;
            int32_t c_low = (parity == 0u)
                ? ((int32_t)pat_j << 1)               // 0 or 2
                : ((pat_j != 0u) ? -1 : 1);
            int32_t v_int = c_low + 4 * w_int;
            out_v[bi * COORDS_PER_BLOCK + j] = (int8_t)v_int;
        }
    }
}

// Host-callable launcher. The Rust FFI calls this. TILE_SIZE is fixed at 32
// for the production V6 bundle; future variants will template more.
extern "C" void leech_q24_decode_v_int_cuda(
    const uint8_t*  packed_buckets,
    const uint16_t* tile_states,
    const uint16_t* tile_nb_totals,
    const uint64_t* tile_bitstream,
    const uint64_t* tile_bit_offsets,
    int8_t*  out_v,
    uint32_t n_blocks,
    uint32_t n_tiles,
    int32_t  w_offset,
    int32_t  tile_size,
    void*    stream                                  // cudaStream_t
) {
    constexpr int THREADS_PER_BLOCK = 128;
    uint32_t grid = (n_tiles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    // Runtime dispatch over supported TILE_SIZE values. Encoded artifacts may
    // choose T ∈ {4, 8, 16, 32}; the kernel is templated on T so all four
    // variants are instantiated here. Unsupported values fall back to T=32.
    switch (tile_size) {
        case 4:
            leech_q24_decode_kernel<4><<<grid, THREADS_PER_BLOCK, 0, s>>>(
                packed_buckets, tile_states, tile_nb_totals, tile_bitstream,
                tile_bit_offsets, out_v, n_blocks, n_tiles, w_offset
            );
            break;
        case 8:
            leech_q24_decode_kernel<8><<<grid, THREADS_PER_BLOCK, 0, s>>>(
                packed_buckets, tile_states, tile_nb_totals, tile_bitstream,
                tile_bit_offsets, out_v, n_blocks, n_tiles, w_offset
            );
            break;
        case 16:
            leech_q24_decode_kernel<16><<<grid, THREADS_PER_BLOCK, 0, s>>>(
                packed_buckets, tile_states, tile_nb_totals, tile_bitstream,
                tile_bit_offsets, out_v, n_blocks, n_tiles, w_offset
            );
            break;
        case 32:
            leech_q24_decode_kernel<32><<<grid, THREADS_PER_BLOCK, 0, s>>>(
                packed_buckets, tile_states, tile_nb_totals, tile_bitstream,
                tile_bit_offsets, out_v, n_blocks, n_tiles, w_offset
            );
            break;
        default:
            fprintf(stderr,
                "leech_q24: unsupported tile_size=%d in decode_v_int, "
                "falling back to T=32\n", tile_size);
            leech_q24_decode_kernel<32><<<grid, THREADS_PER_BLOCK, 0, s>>>(
                packed_buckets, tile_states, tile_nb_totals, tile_bitstream,
                tile_bit_offsets, out_v, n_blocks, n_tiles, w_offset
            );
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase B.0 fused decode + β·v + offset + GEMV → bf16 (batch M = 1).
//
// One thread per tile. Reuses c_decode_tables and d_pattern_table from above
// (single-TU; cross-TU __constant__/__device__ sharing requires -rdc=true,
// which we don't enable).
// ─────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t extract_3bit_q24(
    const uint8_t* __restrict__ packed, uint32_t i)
{
    uint32_t bit_pos = i * 3u;
    uint32_t bi = bit_pos >> 3;
    uint32_t bo = bit_pos & 7u;
    // 3 bits span ≤ 2 bytes. Caller must pad packed[] by ≥1 trailing byte
    // for the i = n_blocks-1 case (host upload does this).
    uint32_t v = (uint32_t)packed[bi] | ((uint32_t)packed[bi + 1] << 8);
    return (v >> bo) & 0x7u;
}

// Path A: single-u64 batch read of `nb` bits ending at absolute bit
// position (bit_off + nb_left - 1), inclusive. Equivalent to v0's
// per-bit serial loop with bit_idx LSB-first within u64 word and
// MSB-first assembly into bits_val. Replaces the ~721
// LDG.E.64.CONSTANT-per-thread scoreboard chain in the GEMV inner loop
// with one batch read per symbol (plus one more on cross-word
// straddles).
//
// CPU bit-equivalence proven in
// `mistralrs-quant/tests/leech_q24_bit_extract.rs` over 1M random
// (bit_off, nb_left, nb) triples and the §2.6 oracle.
__device__ __forceinline__ uint32_t extract_nb_bits_from_window(
    const uint64_t* __restrict__ tile_bitstream,
    uint64_t bit_off,
    int32_t  nb_left,
    uint32_t nb)
{
    uint64_t low_pos  = bit_off + (uint64_t)nb_left - (uint64_t)nb;
    uint64_t word_idx = low_pos >> 6;
    uint32_t bit_idx  = (uint32_t)(low_pos & 63ull);
    uint64_t w0       = tile_bitstream[word_idx];
    uint64_t bits;
    if (bit_idx + nb <= 64u) {
        bits = (w0 >> bit_idx) & ((1ull << nb) - 1ull);
    } else {
        uint64_t w1     = tile_bitstream[word_idx + 1];
        uint32_t lo_n   = 64u - bit_idx;
        uint64_t lo_bits = (w0 >> bit_idx);
        uint64_t hi_bits = (w1 & ((1ull << (nb - lo_n)) - 1ull)) << lo_n;
        bits = lo_bits | hi_bits;
    }
    return (uint32_t)bits;
}

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
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];

        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;

        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }

        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
                // Should not happen with TILE_SIZE=32 ≤ B ≥ 32 in production.
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

        // bf16x2-vectorized a_act loads: one __nv_bfloat162 (32-bit LDG.E.32)
        // serves coords (j2, j2+1). The FSE state chain remains coord-by-coord
        // serial — only the activation load and FMA input are vectorized.
        // Alignment: k_base = col_block * 24, 24 is even → &a_act[k_base+j2]
        // for j2 ∈ {0,2,...,22} is 4-byte aligned. COORDS_PER_BLOCK=24 is
        // even → no tail iteration needed.
        #pragma unroll
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);

            // ── Coord j2 ──────────────────────────────────────────────
            {
                int j = j2;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;

                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;

                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;

                float w_val = beta_val * (float)v_int + offset_val;
                partial += w_val * a_f32_lo;
            }

            // ── Coord j2+1 ────────────────────────────────────────────
            {
                int j = j2 + 1;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;

                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;

                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;

                float w_val = beta_val * (float)v_int + offset_val;
                partial += w_val * a_f32_hi;
            }
        }

        acc_val[slot] += partial;
    }

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
    cudaMemsetAsync(y_acc_f32, 0, (size_t)r_rows * sizeof(float), s);
    constexpr int THREADS = 128;
    uint32_t grid = (n_tiles + THREADS - 1) / THREADS;
    // Runtime dispatch over supported TILE_SIZE values. Encoded artifacts may
    // choose T ∈ {4, 8, 16, 32}; the kernel is templated on T so all four
    // variants are instantiated here. Unsupported values fall back to T=32.
    switch (tile_size) {
        case 4:
            leech_q24_gemv_bf16_kernel<4><<<grid, THREADS, 0, s>>>(
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
            break;
        case 8:
            leech_q24_gemv_bf16_kernel<8><<<grid, THREADS, 0, s>>>(
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
            break;
        case 16:
            leech_q24_gemv_bf16_kernel<16><<<grid, THREADS, 0, s>>>(
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
            break;
        case 32:
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
            break;
        default:
            fprintf(stderr,
                "leech_q24: unsupported tile_size=%d in gemv_bf16, "
                "falling back to T=32\n", tile_size);
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
            break;
    }
    uint32_t fgrid = (r_rows + THREADS - 1) / THREADS;
    leech_q24_finalize_f32_to_bf16_kernel<<<fgrid, THREADS, 0, s>>>(
        y_acc_f32,
        reinterpret_cast<__nv_bfloat16*>(out_y_bf16),
        r_rows
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Instrumented variant: per-stage clock64() profiling of the v0 GEMV kernel.
//
// Brackets 8 per-coord-step operations with clock64() and accumulates cycles
// into a per-thread uint64_t cyc[8] array. At end-of-kernel each thread
// atomicAdds its accumulators into a single global [8] array. Output is
// IDENTICAL to v0 (no semantic changes; only timing brackets added).
//
// Stages (per coord unless noted):
//   0: bucket_extract + split_bucket (per BLOCK, not per coord)
//   1: pat_row[j] load
//   2: c_decode_tables[cb * M_TABLE + state] lookup
//   3: extract_nb_bits_from_window (the Path A batch u64 read)
//   4: state = (base | bits_val) & M_MASK
//   5: a_act[k_base + j] load + __bfloat162float cast
//   6: partial += w_val * a_f32
//   7: end-of-tile atomicAdd (per TILE, not per coord)
//
// Anti-reorder strategy: We use `asm volatile("" : : "l"(x) : "memory")` as a
// barrier between the producer op and the clock64 read, forcing the compiler
// to materialize the op before the timer stops. clock64() compiles to a
// non-reorderable SR.CLOCKLO/HI read on Blackwell sm_120.
//
// Overhead: ~6-10 cycles per clock64. With 8 reads/coord × 24 coords ×
// TILE_SIZE blocks × 522,240 tiles at T=4 → ~3.2 G clock reads → ~16 µs added
// to wall time at 1.4 GHz / 170 SMs. Negligible vs 1427 µs envelope.
// ─────────────────────────────────────────────────────────────────────────

namespace detail {
__device__ __forceinline__ void clk_barrier_u32(uint32_t v) {
    // Force the compiler to materialize `v` before this point; opaque to LLVM.
    asm volatile("" : : "r"(v) : "memory");
}
__device__ __forceinline__ void clk_barrier_u64(uint64_t v) {
    asm volatile("" : : "l"(v) : "memory");
}
__device__ __forceinline__ void clk_barrier_f32(float v) {
    asm volatile("" : : "f"(v) : "memory");
}
} // namespace detail

template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_timed_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
    uint32_t r_rows,
    uint32_t b_blocks,
    uint32_t n_blocks,
    uint32_t n_tiles,
    uint32_t k_beta,
    uint32_t k_offset,
    int32_t  w_offset,
    int32_t  has_offset,
    uint64_t* __restrict__ stage_cycles_out      // [N_STAGES] global accumulator
) {
    constexpr int N_STAGES = 8;
    uint64_t cyc[N_STAGES] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tiles) return;

    uint32_t state = (uint32_t)tile_states[tid];
    int32_t  nb_left = (int32_t)tile_nb_totals[tid];
    uint64_t bit_off = tile_bit_offsets[tid];

    uint32_t tile_start = tid * (uint32_t)TILE_SIZE;
    uint32_t tile_end = tile_start + (uint32_t)TILE_SIZE;
    if (tile_end > n_blocks) tile_end = n_blocks;

    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;

    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        // ── Stage 0: bucket extract + split (PER BLOCK) ──────────────────
        uint64_t t0_pre = clock64();
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        detail::clk_barrier_u32(parity);
        detail::clk_barrier_u32(h);
        detail::clk_barrier_u32(f);
        uint64_t t0_post = clock64();
        cyc[0] += t0_post - t0_pre;

        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];

        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;

        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }

        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
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

        // NOTE: NOT unrolled here — unrolling makes 24 separate clock64()
        // brackets per coord, blowing up per-stage cycle counts in ways that
        // are hard to reason about. The compiler may still partially unroll;
        // we accept that for measurement purposes.
        //
        // bf16x2-vectorized variant: outer loop runs 12 pair-iterations. The
        // stage 5 bracket covers ONE __nv_bfloat162 load + two casts per pair
        // (vs one bf16 load + one cast per coord in the scalar form). The
        // test code divides cyc[5] by `blocks*24` (per-coord-equivalent), so
        // the reported number is the per-coord-equivalent cost of the paired
        // load — directly comparable to the pre-vectorize stage 5 number.
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            // ── Stage 5 (pair): a_act bf16x2 load + 2× bf16→f32 cast ────
            uint64_t t5_pre = clock64();
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);
            detail::clk_barrier_f32(a_f32_lo);
            detail::clk_barrier_f32(a_f32_hi);
            uint64_t t5_post = clock64();
            cyc[5] += t5_post - t5_pre;

            // ── Coord j2 ──────────────────────────────────────────────
            {
                int j = j2;

                // ── Stage 1: pat_row[j] load ────────────────────────────
                uint64_t t1_pre = clock64();
                uint32_t pat_j = pat_row[j];
                detail::clk_barrier_u32(pat_j);
                uint64_t t1_post = clock64();
                cyc[1] += t1_post - t1_pre;

                uint32_t cb = (parity << 1) | pat_j;

                // ── Stage 2: c_decode_tables lookup ────────────────────
                uint64_t t2_pre = clock64();
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                detail::clk_barrier_u32(entry);
                uint64_t t2_post = clock64();
                cyc[2] += t2_post - t2_pre;

                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;

                // ── Stage 3: extract_nb_bits_from_window ───────────────
                uint64_t t3_pre = clock64();
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                detail::clk_barrier_u32(bits_val);
                uint64_t t3_post = clock64();
                cyc[3] += t3_post - t3_pre;

                nb_left -= (int32_t)nb;

                // ── Stage 4: state update ──────────────────────────────
                uint64_t t4_pre = clock64();
                state = (base | bits_val) & M_MASK;
                detail::clk_barrier_u32(state);
                uint64_t t4_post = clock64();
                cyc[4] += t4_post - t4_pre;

                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;

                // ── Stage 6: FMA ───────────────────────────────────────
                uint64_t t6_pre = clock64();
                partial += w_val * a_f32_lo;
                detail::clk_barrier_f32(partial);
                uint64_t t6_post = clock64();
                cyc[6] += t6_post - t6_pre;
            }

            // ── Coord j2+1 ────────────────────────────────────────────
            {
                int j = j2 + 1;

                // ── Stage 1: pat_row[j] load ────────────────────────────
                uint64_t t1_pre = clock64();
                uint32_t pat_j = pat_row[j];
                detail::clk_barrier_u32(pat_j);
                uint64_t t1_post = clock64();
                cyc[1] += t1_post - t1_pre;

                uint32_t cb = (parity << 1) | pat_j;

                // ── Stage 2: c_decode_tables lookup ────────────────────
                uint64_t t2_pre = clock64();
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                detail::clk_barrier_u32(entry);
                uint64_t t2_post = clock64();
                cyc[2] += t2_post - t2_pre;

                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;

                // ── Stage 3: extract_nb_bits_from_window ───────────────
                uint64_t t3_pre = clock64();
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                detail::clk_barrier_u32(bits_val);
                uint64_t t3_post = clock64();
                cyc[3] += t3_post - t3_pre;

                nb_left -= (int32_t)nb;

                // ── Stage 4: state update ──────────────────────────────
                uint64_t t4_pre = clock64();
                state = (base | bits_val) & M_MASK;
                detail::clk_barrier_u32(state);
                uint64_t t4_post = clock64();
                cyc[4] += t4_post - t4_pre;

                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;

                // ── Stage 6: FMA ───────────────────────────────────────
                uint64_t t6_pre = clock64();
                partial += w_val * a_f32_hi;
                detail::clk_barrier_f32(partial);
                uint64_t t6_post = clock64();
                cyc[6] += t6_post - t6_pre;
            }
        }

        acc_val[slot] += partial;
    }

    // ── Stage 7: end-of-tile atomicAdd (PER TILE) ──────────────────────
    uint64_t t7_pre = clock64();
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
    uint64_t t7_post = clock64();
    cyc[7] += t7_post - t7_pre;

    // ── Flush per-thread accumulators into the global [8] array ───────
    #pragma unroll
    for (int s = 0; s < N_STAGES; ++s) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&stage_cycles_out[s]),
            (unsigned long long)cyc[s]
        );
    }
}

extern "C" void leech_q24_gemv_bf16_timed_cuda(
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
    uint64_t* stage_cycles_out,                  // device [8]
    void*    stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaMemsetAsync(y_acc_f32, 0, (size_t)r_rows * sizeof(float), s);
    cudaMemsetAsync(stage_cycles_out, 0, 8 * sizeof(uint64_t), s);
    constexpr int THREADS = 128;
    uint32_t grid = (n_tiles + THREADS - 1) / THREADS;
    switch (tile_size) {
        case 4:
            leech_q24_gemv_bf16_timed_kernel<4><<<grid, THREADS, 0, s>>>(
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),
                packed_buckets, tile_states, tile_nb_totals,
                tile_bitstream, tile_bit_offsets,
                beta_idx_packed, offset_idx_packed,
                beta_lloyd, offset_lloyd,
                y_acc_f32,
                r_rows, b_blocks, n_blocks, n_tiles,
                k_beta, k_offset,
                w_offset, has_offset,
                stage_cycles_out
            );
            break;
        case 8:
            leech_q24_gemv_bf16_timed_kernel<8><<<grid, THREADS, 0, s>>>(
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),
                packed_buckets, tile_states, tile_nb_totals,
                tile_bitstream, tile_bit_offsets,
                beta_idx_packed, offset_idx_packed,
                beta_lloyd, offset_lloyd,
                y_acc_f32,
                r_rows, b_blocks, n_blocks, n_tiles,
                k_beta, k_offset,
                w_offset, has_offset,
                stage_cycles_out
            );
            break;
        case 16:
            leech_q24_gemv_bf16_timed_kernel<16><<<grid, THREADS, 0, s>>>(
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),
                packed_buckets, tile_states, tile_nb_totals,
                tile_bitstream, tile_bit_offsets,
                beta_idx_packed, offset_idx_packed,
                beta_lloyd, offset_lloyd,
                y_acc_f32,
                r_rows, b_blocks, n_blocks, n_tiles,
                k_beta, k_offset,
                w_offset, has_offset,
                stage_cycles_out
            );
            break;
        case 32:
            leech_q24_gemv_bf16_timed_kernel<32><<<grid, THREADS, 0, s>>>(
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),
                packed_buckets, tile_states, tile_nb_totals,
                tile_bitstream, tile_bit_offsets,
                beta_idx_packed, offset_idx_packed,
                beta_lloyd, offset_lloyd,
                y_acc_f32,
                r_rows, b_blocks, n_blocks, n_tiles,
                k_beta, k_offset,
                w_offset, has_offset,
                stage_cycles_out
            );
            break;
        default:
            fprintf(stderr,
                "leech_q24: unsupported tile_size=%d in gemv_bf16_timed, "
                "falling back to T=32\n", tile_size);
            leech_q24_gemv_bf16_timed_kernel<32><<<grid, THREADS, 0, s>>>(
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),
                packed_buckets, tile_states, tile_nb_totals,
                tile_bitstream, tile_bit_offsets,
                beta_idx_packed, offset_idx_packed,
                beta_lloyd, offset_lloyd,
                y_acc_f32,
                r_rows, b_blocks, n_blocks, n_tiles,
                k_beta, k_offset,
                w_offset, has_offset,
                stage_cycles_out
            );
            break;
    }
    uint32_t fgrid = (r_rows + THREADS - 1) / THREADS;
    leech_q24_finalize_f32_to_bf16_kernel<<<fgrid, THREADS, 0, s>>>(
        y_acc_f32,
        reinterpret_cast<__nv_bfloat16*>(out_y_bf16),
        r_rows
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Subtractive profile variants.
//
// Each variant is a near-clone of `leech_q24_gemv_bf16_kernel<TILE_SIZE>` with
// ONE per-coord-step operation neutralized. Output is INCORRECT — these
// variants exist only to measure wall-time delta when a single stage is
// removed. The variant with the largest wall drop (vs the full v0 kernel) IS
// the actual critical-path bottleneck.
//
// All variants share the v0 FFI signature so the bench harness can swap them
// in cheaply. DCE-prevention strategy: each neutralized op is replaced with a
// j/state/cb-dependent expression that still forces dependent registers to be
// materialized, so the surrounding work is not folded away. The kernels are
// NOT meant to validate correctness.
// ─────────────────────────────────────────────────────────────────────────

// V_NO_PAT — replace pat_row[j] load with `j & 1u`. No memory traffic for the
// pattern table; everything else identical.
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_no_pat_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;
    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        // pat_row pointer NOT computed — pat_j source replaced below.
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;
        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
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
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int j = j2 + kk;
                // V_NO_PAT — DCE-resistant j-dependent expression in place of pat_row[j]
                uint32_t pat_j = ((uint32_t)j) & 1u;
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;
                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;
                float a_use = (kk == 0) ? a_f32_lo : a_f32_hi;
                partial += w_val * a_use;
            }
        }
        acc_val[slot] += partial;
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
}

// V_NO_DECODE — replace c_decode_tables[cb*M+state] lookup with a state/cb
// dependent ALU expression. No __constant__ memory traffic, but `state`
// still flows through every iteration so the chain still executes.
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_no_decode_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;
    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;
        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
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
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int j = j2 + kk;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                // V_NO_DECODE — DCE-resistant ALU expression in place of c_decode_tables[]
                uint32_t entry = (state ^ (cb * 0x9E3779B1u)) | 0x00050100u;
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;
                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;
                float a_use = (kk == 0) ? a_f32_lo : a_f32_hi;
                partial += w_val * a_use;
            }
        }
        acc_val[slot] += partial;
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
}

// V_NO_BITS — replace extract_nb_bits_from_window with bits_val = 0.
// nb_left is still decremented (preserves loop semantics), state chain
// still depends on `bits_val` (= 0) so the per-iter dependency stays,
// but no LDG.E.64 to tile_bitstream.
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_no_bits_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;
    // Defeat DCE on the unused bitstream pointer: touch one word per tile so
    // the kernel still has the same arg-binding cost as v0.
    uint64_t bs_stub = tile_bitstream[bit_off >> 6];
    asm volatile("" : : "l"(bs_stub) : "memory");
    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;
        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
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
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int j = j2 + kk;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;
                // V_NO_BITS — no LDG.E.64; bits_val = 0
                uint32_t bits_val = 0u;
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;
                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;
                float a_use = (kk == 0) ? a_f32_lo : a_f32_hi;
                partial += w_val * a_use;
            }
        }
        acc_val[slot] += partial;
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
}

// V_NO_AACT — replace a_act bf16x2 load with a constant pair; no LDG.E.32
// to the activation buffer. tid-dependent constants prevent CSE across
// blocks.
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_no_aact_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;
    // Defeat DCE on the unused a_act pointer: touch one half per tile.
    __nv_bfloat16 a_stub = a_act[tid % 16u];
    asm volatile("" : : "h"(*reinterpret_cast<unsigned short*>(&a_stub)) : "memory");
    // Use tid-derived constants to avoid CSE across tiles.
    float a_const_lo = 1.0f;
    float a_const_hi = -1.0f;
    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
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
        // col_block silenced; suppress unused warning.
        (void)col_block;
        #pragma unroll
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            // V_NO_AACT — constant activation values; no a_act LDG.E.32.
            float a_f32_lo = a_const_lo;
            float a_f32_hi = a_const_hi;
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int j = j2 + kk;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;
                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;
                float a_use = (kk == 0) ? a_f32_lo : a_f32_hi;
                partial += w_val * a_use;
            }
        }
        acc_val[slot] += partial;
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
}

// V_NO_ATOMIC — replace end-of-tile atomicAdd with a non-atomic store. There
// IS still a global write per tile (preserves bandwidth) but no atomic
// serialization. May race; output is wrong; that is fine.
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_no_atomic_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;
    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;
        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
                // V_NO_ATOMIC — non-atomic store on overflow path too.
                y_acc_f32[acc_row[0]] = acc_val[0];
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
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int j = j2 + kk;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;
                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;
                float a_use = (kk == 0) ? a_f32_lo : a_f32_hi;
                partial += w_val * a_use;
            }
        }
        acc_val[slot] += partial;
    }
    // V_NO_ATOMIC — non-atomic global store
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            y_acc_f32[acc_row[s]] = acc_val[s];
        }
    }
}

// V_NO_STATE — break the FSE state-chain dependency. After each update,
// reset state to 0. All loads still execute (state still flows into
// c_decode_tables[]) but the chain dependency length is reduced to 1.
template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_no_state_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
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
    float    acc_val[2] = {0.0f, 0.0f};
    uint32_t acc_row[2] = {UINT32_MAX, UINT32_MAX};
    int32_t  acc_count = 0;
    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        uint32_t bucket = extract_bucket_13(packed_buckets, bi);
        uint32_t parity, h, f;
        split_bucket(bucket, parity, h, f);
        const uint8_t* pat_row =
            &d_pattern_table[(h * 64 + f) * COORDS_PER_BLOCK];
        uint32_t row = bi / b_blocks;
        uint32_t col_block = bi - row * b_blocks;
        uint32_t k_base = col_block * (uint32_t)COORDS_PER_BLOCK;
        uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
        float beta_val = beta_lloyd[row * k_beta + beta_idx_val];
        float offset_val = 0.0f;
        if (has_offset) {
            uint32_t offset_idx_val = extract_3bit_q24(offset_idx_packed, bi);
            offset_val = offset_lloyd[row * k_offset + offset_idx_val];
        }
        int32_t slot = -1;
        #pragma unroll
        for (int32_t s = 0; s < 2; ++s) {
            if (s < acc_count && acc_row[s] == row) { slot = s; }
        }
        if (slot < 0) {
            slot = acc_count;
            if (slot >= 2) {
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
        for (int j2 = 0; j2 < COORDS_PER_BLOCK; j2 += 2) {
            __nv_bfloat162 a_pair = *reinterpret_cast<const __nv_bfloat162*>(
                &a_act[k_base + j2]);
            float a_f32_lo = __low2float(a_pair);
            float a_f32_hi = __high2float(a_pair);
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int j = j2 + kk;
                uint32_t pat_j = pat_row[j];
                uint32_t cb = (parity << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym  = entry & 0xFFu;
                uint32_t nb   = (entry >> 8) & 0xFFu;
                uint32_t base = entry >> 16;
                uint32_t bits_val = extract_nb_bits_from_window(
                    tile_bitstream, bit_off, nb_left, nb);
                nb_left -= (int32_t)nb;
                // V_NO_STATE — break the FSE state-chain dependency.
                // The expression still uses base|bits_val so the compiler
                // keeps both live, but state never accumulates across coords.
                state = 0u;
                (void)base; (void)bits_val;
                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                float w_val = beta_val * (float)v_int + offset_val;
                float a_use = (kk == 0) ? a_f32_lo : a_f32_hi;
                partial += w_val * a_use;
            }
        }
        acc_val[slot] += partial;
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        if (s < acc_count) {
            atomicAdd(&y_acc_f32[acc_row[s]], acc_val[s]);
        }
    }
}

// ──── Launchers (one extern "C" per variant; v0 FFI-compatible) ──────────

#define LEECH_Q24_SUBTRACTIVE_LAUNCHER(NAME)                                   \
extern "C" void leech_q24_gemv_bf16_##NAME##_cuda(                              \
    const void*     a_act_bf16,                                                \
    const uint8_t*  packed_buckets,                                            \
    const uint16_t* tile_states,                                               \
    const uint16_t* tile_nb_totals,                                            \
    const uint64_t* tile_bitstream,                                            \
    const uint64_t* tile_bit_offsets,                                          \
    const uint8_t*  beta_idx_packed,                                           \
    const uint8_t*  offset_idx_packed,                                         \
    const float*    beta_lloyd,                                                \
    const float*    offset_lloyd,                                              \
    float*          y_acc_f32,                                                 \
    void*           out_y_bf16,                                                \
    uint32_t r_rows,                                                           \
    uint32_t b_blocks,                                                         \
    uint32_t n_blocks,                                                         \
    uint32_t n_tiles,                                                          \
    uint32_t k_beta,                                                           \
    uint32_t k_offset,                                                         \
    int32_t  w_offset,                                                         \
    int32_t  tile_size,                                                        \
    int32_t  has_offset,                                                       \
    void*    stream                                                            \
) {                                                                            \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    cudaMemsetAsync(y_acc_f32, 0, (size_t)r_rows * sizeof(float), s);          \
    constexpr int THREADS = 128;                                               \
    uint32_t grid = (n_tiles + THREADS - 1) / THREADS;                         \
    switch (tile_size) {                                                       \
        case 4:                                                                \
            leech_q24_gemv_bf16_##NAME##_kernel<4><<<grid, THREADS, 0, s>>>(   \
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),            \
                packed_buckets, tile_states, tile_nb_totals,                   \
                tile_bitstream, tile_bit_offsets,                              \
                beta_idx_packed, offset_idx_packed,                            \
                beta_lloyd, offset_lloyd,                                      \
                y_acc_f32,                                                     \
                r_rows, b_blocks, n_blocks, n_tiles,                           \
                k_beta, k_offset,                                              \
                w_offset, has_offset                                           \
            );                                                                 \
            break;                                                             \
        case 8:                                                                \
            leech_q24_gemv_bf16_##NAME##_kernel<8><<<grid, THREADS, 0, s>>>(   \
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),            \
                packed_buckets, tile_states, tile_nb_totals,                   \
                tile_bitstream, tile_bit_offsets,                              \
                beta_idx_packed, offset_idx_packed,                            \
                beta_lloyd, offset_lloyd,                                      \
                y_acc_f32,                                                     \
                r_rows, b_blocks, n_blocks, n_tiles,                           \
                k_beta, k_offset,                                              \
                w_offset, has_offset                                           \
            );                                                                 \
            break;                                                             \
        case 16:                                                               \
            leech_q24_gemv_bf16_##NAME##_kernel<16><<<grid, THREADS, 0, s>>>(  \
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),            \
                packed_buckets, tile_states, tile_nb_totals,                   \
                tile_bitstream, tile_bit_offsets,                              \
                beta_idx_packed, offset_idx_packed,                            \
                beta_lloyd, offset_lloyd,                                      \
                y_acc_f32,                                                     \
                r_rows, b_blocks, n_blocks, n_tiles,                           \
                k_beta, k_offset,                                              \
                w_offset, has_offset                                           \
            );                                                                 \
            break;                                                             \
        case 32:                                                               \
        default:                                                               \
            leech_q24_gemv_bf16_##NAME##_kernel<32><<<grid, THREADS, 0, s>>>(  \
                reinterpret_cast<const __nv_bfloat16*>(a_act_bf16),            \
                packed_buckets, tile_states, tile_nb_totals,                   \
                tile_bitstream, tile_bit_offsets,                              \
                beta_idx_packed, offset_idx_packed,                            \
                beta_lloyd, offset_lloyd,                                      \
                y_acc_f32,                                                     \
                r_rows, b_blocks, n_blocks, n_tiles,                           \
                k_beta, k_offset,                                              \
                w_offset, has_offset                                           \
            );                                                                 \
            break;                                                             \
    }                                                                          \
    uint32_t fgrid = (r_rows + THREADS - 1) / THREADS;                         \
    leech_q24_finalize_f32_to_bf16_kernel<<<fgrid, THREADS, 0, s>>>(           \
        y_acc_f32,                                                             \
        reinterpret_cast<__nv_bfloat16*>(out_y_bf16),                          \
        r_rows                                                                 \
    );                                                                         \
}

LEECH_Q24_SUBTRACTIVE_LAUNCHER(no_pat)
LEECH_Q24_SUBTRACTIVE_LAUNCHER(no_decode)
LEECH_Q24_SUBTRACTIVE_LAUNCHER(no_bits)
LEECH_Q24_SUBTRACTIVE_LAUNCHER(no_aact)
LEECH_Q24_SUBTRACTIVE_LAUNCHER(no_atomic)
LEECH_Q24_SUBTRACTIVE_LAUNCHER(no_state)

#undef LEECH_Q24_SUBTRACTIVE_LAUNCHER

// ─────────────────────────────────────────────────────────────────────────
// Phase B.1 — warp-cooperative fused GEMV.
//
// Design: one warp owns one tile. Lane 0 drives the serial FSE state chain
// (768 dependent updates per tile) into per-block scratch in SMEM; lanes
// 0..23 each consume one coord per block via an FMA into a per-lane f32
// register, then warp-reduce + atomicAdd into y_acc_f32[row] at tile end.
//
// 4 warps/block, 128 threads/block, grid = ceil(n_tiles / 4). Replaces
// v0's 510-block × 128-thread grid with 16,320 × 128 = 32× more blocks,
// taking warp residency from 18.75% → 75%.
//
// Commit 2 (current): Mode B work distribution with the v0 per-bit
// extract loop still on lane 0. Bench target: 600-800 µs.
// Commit 3 will replace the per-bit loop with a uint4 shift-window.
//
// v0 (`leech_q24_gemv_bf16_kernel`, `leech_q24_gemv_bf16_cuda`) stays
// intact throughout the rewrite; runtime selection lives in the Rust
// caller via the `LEECHQ24_WARPCOOP` env var.
// ─────────────────────────────────────────────────────────────────────────

constexpr int WARPCOOP_BLOCK_THREADS = 128;
constexpr int WARPCOOP_WARPS_PER_BLOCK = WARPCOOP_BLOCK_THREADS / 32;
constexpr int WARPCOOP_K_BETA_MAX = 8;     // production k_beta == 8
constexpr int WARPCOOP_K_OFFSET_MAX = 8;   // production k_offset == 8
constexpr int WARPCOOP_TILE_SIZE = 32;     // production tile_size

// Per-warp SMEM slab. One per warp in a block.
struct alignas(16) WarpTile {
    // coord_pack[j] holds the int8 v_int for coord j of the *current* block
    // being consumed (packed into a u32 for store/load convenience). Only
    // entries [0..24) are read by the consumer lanes; [24..32) are padding
    // for alignment.
    uint32_t coord_pack[32];                            // 128 B

    // β / offset Lloyd row centroids, refreshed by lane 0 on row change.
    float    beta_lloyd_row[WARPCOOP_K_BETA_MAX];       //  32 B
    float    offset_lloyd_row[WARPCOOP_K_OFFSET_MAX];   //  32 B
};
static_assert(sizeof(WarpTile) <= 2048,
              "per-warp SMEM exceeds 2 KB budget");

template<int TILE_SIZE>
__global__ void leech_q24_gemv_bf16_warpcoop_kernel(
    const __nv_bfloat16* __restrict__ a_act,
    const uint8_t*  __restrict__ packed_buckets,
    const uint16_t* __restrict__ tile_states,
    const uint16_t* __restrict__ tile_nb_totals,
    const uint64_t* __restrict__ tile_bitstream,
    const uint64_t* __restrict__ tile_bit_offsets,
    const uint8_t*  __restrict__ beta_idx_packed,
    const uint8_t*  __restrict__ offset_idx_packed,
    const float*    __restrict__ beta_lloyd,
    const float*    __restrict__ offset_lloyd,
    float*          __restrict__ y_acc_f32,
    uint32_t r_rows,
    uint32_t b_blocks,
    uint32_t n_blocks,
    uint32_t n_tiles,
    uint32_t k_beta,
    uint32_t k_offset,
    int32_t  w_offset,
    int32_t  has_offset
) {
    extern __shared__ uint8_t s_raw[];
    const int WARP_ID = (int)(threadIdx.x >> 5);
    const int LANE    = (int)(threadIdx.x & 31);
    const uint32_t TILE = blockIdx.x * (uint32_t)WARPCOOP_WARPS_PER_BLOCK
                          + (uint32_t)WARP_ID;
    if (TILE >= n_tiles) return;

    WarpTile* WT = reinterpret_cast<WarpTile*>(s_raw) + WARP_ID;

    // Lane 0 owns the serial state chain; non-zero lanes only do FMA.
    uint32_t state    = 0;
    int32_t  nb_left  = 0;
    uint64_t bit_off  = 0;
    if (LANE == 0) {
        state    = (uint32_t)tile_states[TILE];
        nb_left  = (int32_t)tile_nb_totals[TILE];
        bit_off  = tile_bit_offsets[TILE];
    }

    // Per-lane f32 row partial. Lanes 24..31 stay 0 (warp-reduce harmless).
    float    lane_acc = 0.0f;
    // Row currently held in WT->beta_lloyd_row[] / WT->offset_lloyd_row[].
    // UINT32_MAX = sentinel "no row loaded yet"; the first block of every
    // tile trips the straddle path and loads the centroids for row 0 of
    // the tile through the same code path as a mid-tile reload (single-
    // path centroid loading; see plan §2.10.2).
    uint32_t cur_row = UINT32_MAX;

    const uint32_t tile_start = TILE * (uint32_t)TILE_SIZE;
    uint32_t tile_end_raw = tile_start + (uint32_t)TILE_SIZE;
    if (tile_end_raw > n_blocks) tile_end_raw = n_blocks;
    const uint32_t tile_end = tile_end_raw;

    for (uint32_t bi = tile_start; bi < tile_end; ++bi) {
        // ── Lane-0 phase 1: bucket + row/col compute (β/offset DEFERRED) ─
        uint32_t row_b = 0, col_b = 0;
        uint32_t parity_b = 0, h_b = 0, f_b = 0;
        if (LANE == 0) {
            uint32_t bucket = extract_bucket_13(packed_buckets, bi);
            split_bucket(bucket, parity_b, h_b, f_b);
            row_b = bi / b_blocks;
            col_b = bi - row_b * b_blocks;
        }
        // Broadcast row_b BEFORE the straddle handshake; needed by every
        // lane to decide whether to participate in the butterfly reduce.
        row_b = __shfl_sync(0xFFFFFFFFu, row_b, 0);

        // ── Mid-tile row-straddle handshake (uniform across warp) ────────
        // On the first block of the tile, cur_row == UINT32_MAX, so this
        // path fires once and loads centroids for row_b through the same
        // code path as a mid-tile reload. The atomicAdd is suppressed via
        // the cur_row != UINT32_MAX guard.
        bool straddle = (row_b != cur_row);
        if (straddle) {
            float v = lane_acc;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            }
            if (LANE == 0 && cur_row != UINT32_MAX) {
                atomicAdd(&y_acc_f32[cur_row], v);
            }
            lane_acc = 0.0f;
            __syncwarp(0xFFFFFFFFu);
            if (LANE == 0) {
                #pragma unroll
                for (int k = 0; k < WARPCOOP_K_BETA_MAX; ++k) {
                    WT->beta_lloyd_row[k] = (k < (int)k_beta)
                        ? beta_lloyd[row_b * k_beta + (uint32_t)k]
                        : 0.0f;
                }
                if (has_offset) {
                    #pragma unroll
                    for (int k = 0; k < WARPCOOP_K_OFFSET_MAX; ++k) {
                        WT->offset_lloyd_row[k] = (k < (int)k_offset)
                            ? offset_lloyd[row_b * k_offset + (uint32_t)k]
                            : 0.0f;
                    }
                }
                cur_row = row_b;
            }
            __syncwarp(0xFFFFFFFFu);
            cur_row = __shfl_sync(0xFFFFFFFFu, cur_row, 0);
        }

        // ── Lane-0 phase 2: β/offset SMEM read (now fresh) + state chain ─
        float beta_b = 0.0f, offset_b = 0.0f;
        if (LANE == 0) {
            uint32_t beta_idx_val = extract_3bit_q24(beta_idx_packed, bi);
            beta_b = WT->beta_lloyd_row[beta_idx_val];
            offset_b = 0.0f;
            if (has_offset) {
                uint32_t oi = extract_3bit_q24(offset_idx_packed, bi);
                offset_b = WT->offset_lloyd_row[oi];
            }

            // Locate the pattern row for this (h, f).
            const uint8_t* pat_row =
                &d_pattern_table[(h_b * 64 + f_b) * COORDS_PER_BLOCK];

            // 24-coord serial state chain. Per-bit extract loop is the v0
            // form; Commit 3 replaces it with a uint4 shift-window.
            #pragma unroll
            for (int j = 0; j < COORDS_PER_BLOCK; ++j) {
                uint32_t pat_j = pat_row[j];
                uint32_t cb    = (parity_b << 1) | pat_j;
                uint32_t entry = c_decode_tables[cb * M_TABLE + state];
                uint32_t sym   = entry & 0xFFu;
                uint32_t nb    = (entry >>  8) & 0xFFu;
                uint32_t base  = entry >> 16;

                uint32_t bits_val = 0;
                for (uint32_t k = 0; k < nb; ++k) {
                    int32_t  pos     = nb_left - 1 - (int32_t)k;
                    uint64_t abs_bit = bit_off + (uint64_t)pos;
                    uint64_t word    = tile_bitstream[abs_bit >> 6];
                    uint32_t bit_idx = (uint32_t)(abs_bit & 63ull);
                    uint32_t bit     = (uint32_t)((word >> bit_idx) & 1ull);
                    bits_val = (bits_val << 1) | bit;
                }
                nb_left -= (int32_t)nb;
                state = (base | bits_val) & M_MASK;

                int32_t w_int = (int32_t)sym - w_offset;
                int32_t c_low = (parity_b == 0u)
                    ? ((int32_t)pat_j << 1)
                    : ((pat_j != 0u) ? -1 : 1);
                int32_t v_int = c_low + 4 * w_int;
                // Store as sign-extended int32 (consumer re-narrows). High
                // bits are don't-care; |v_int| ≤ 23 so cast-back is safe.
                WT->coord_pack[j] = (uint32_t)(int32_t)v_int;
            }
        }

        // ── Publish coord_pack[] + broadcast scalars to consumer lanes ───
        __syncwarp(0xFFFFFFFFu);
        col_b    = __shfl_sync(0xFFFFFFFFu, col_b, 0);
        beta_b   = __shfl_sync(0xFFFFFFFFu, beta_b, 0);
        offset_b = __shfl_sync(0xFFFFFFFFu, offset_b, 0);

        // ── Consumer phase: 24 lanes each do one FMA ─────────────────────
        if (LANE < COORDS_PER_BLOCK) {
            int32_t v_int = (int32_t)WT->coord_pack[LANE];
            float   w_val = beta_b * (float)v_int + offset_b;
            uint32_t k_idx = col_b * (uint32_t)COORDS_PER_BLOCK + (uint32_t)LANE;
            float   a_f32 = __bfloat162float(a_act[k_idx]);
            lane_acc += w_val * a_f32;
        }
    }
    __syncwarp(0xFFFFFFFFu);

    // ── Epilogue: butterfly-reduce lane_acc, atomicAdd into y_acc_f32 ────
    float v = lane_acc;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    if (LANE == 0 && cur_row != UINT32_MAX) {
        atomicAdd(&y_acc_f32[cur_row], v);
    }
}

extern "C" void leech_q24_gemv_bf16_warpcoop_cuda(
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
    cudaMemsetAsync(y_acc_f32, 0, (size_t)r_rows * sizeof(float), s);
    (void)tile_size;

    const uint32_t grid =
        (n_tiles + (uint32_t)WARPCOOP_WARPS_PER_BLOCK - 1)
        / (uint32_t)WARPCOOP_WARPS_PER_BLOCK;
    const size_t smem_bytes =
        (size_t)WARPCOOP_WARPS_PER_BLOCK * sizeof(WarpTile);
    leech_q24_gemv_bf16_warpcoop_kernel<WARPCOOP_TILE_SIZE>
        <<<grid, WARPCOOP_BLOCK_THREADS, smem_bytes, s>>>(
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

    constexpr int FINAL_THREADS = 128;
    const uint32_t fgrid = (r_rows + FINAL_THREADS - 1) / FINAL_THREADS;
    leech_q24_finalize_f32_to_bf16_kernel<<<fgrid, FINAL_THREADS, 0, s>>>(
        y_acc_f32,
        reinterpret_cast<__nv_bfloat16*>(out_y_bf16),
        r_rows
    );
}

} // namespace leech_q24
