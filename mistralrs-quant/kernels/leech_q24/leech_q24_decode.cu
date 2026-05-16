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
    // Production path: TILE_SIZE = 32. Other sizes fall through to 32 with a
    // warning at the Rust layer (no other size in production today).
    (void)tile_size;
    leech_q24_decode_kernel<32><<<grid, THREADS_PER_BLOCK, 0, s>>>(
        packed_buckets, tile_states, tile_nb_totals, tile_bitstream,
        tile_bit_offsets, out_v, n_blocks, n_tiles, w_offset
    );
}

} // namespace leech_q24
