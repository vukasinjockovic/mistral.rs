//! `extern "C"` declarations for the CUDA-backed Q24-tANS decoder.
//!
//! Implemented in `mistralrs-quant/kernels/leech_q24/leech_q24_decode.cu`,
//! compiled and linked by `mistralrs-quant/build.rs` via the
//! `kernels/*/*.cu` glob.

use std::os::raw::c_void;

#[allow(dead_code)]
extern "C" {
    /// Copy the FSE decode_tables and the universal pattern_table into the
    /// kernel's `__constant__` / `__device__` regions. Must be called once
    /// per process before any decode call. The Rust wrapper enforces this
    /// via a `OnceLock`.
    ///
    /// `decode_tables_host` is a host pointer to `4 × 1024 × u32 = 16 KB`.
    /// `symbol_set_id` is reserved for future multi-symbol-set support and
    /// currently ignored — only one codebook set is loaded at a time.
    pub(crate) fn leech_q24_init_tables_ffi(
        decode_tables_host: *const u32,
        symbol_set_id: u32,
    );

    /// Decode `n_tiles` tiles from `packed_buckets` + `tile_*` into
    /// `out_v` (int8[n_blocks, 24]). All buffers are DEVICE pointers.
    ///
    /// # Args
    /// - `packed_buckets`: device, 13-bit packed bucket stream, tail-padded
    ///   by ≥1 byte for the 3-byte read window at the last bucket.
    /// - `tile_states`: device, `n_tiles × u16` final state offsets.
    /// - `tile_nb_totals`: device, `n_tiles × u16` bit counts per tile.
    /// - `tile_bitstream`: device, concatenated tile bitstreams as u64 LE.
    /// - `tile_bit_offsets`: device, `n_tiles × u64` precomputed prefix sum
    ///   of `tile_nb_totals` (host computes this at load time).
    /// - `out_v`: device, `n_blocks × 24 × int8` output.
    /// - `n_blocks`: total LLVQ blocks for this tensor.
    /// - `n_tiles`: ceil(n_blocks / tile_size).
    /// - `w_offset`: 3 (S=7, ms=13) or 4 (S=9, ms=18).
    /// - `tile_size`: production = 32; the kernel currently hard-codes 32.
    /// - `stream`: optional cudaStream_t (raw pointer, may be null).
    pub(crate) fn leech_q24_decode_v_int_cuda(
        packed_buckets: *const u8,
        tile_states: *const u16,
        tile_nb_totals: *const u16,
        tile_bitstream: *const u64,
        tile_bit_offsets: *const u64,
        out_v: *mut i8,
        n_blocks: u32,
        n_tiles: u32,
        w_offset: i32,
        tile_size: i32,
        stream: *mut c_void,
    );

    /// Phase B.0 fused decode + β·v + offset + GEMV → bf16 output (batch M=1).
    ///
    /// All buffers are device pointers. The kernel atomicAdds into a per-row
    /// f32 accumulator, then a tiny finalize kernel converts f32 → bf16 into
    /// `out_y_bf16` (R bf16 values). The accumulator scratch is caller-owned
    /// (device-allocated, sized for R floats).
    pub(crate) fn leech_q24_gemv_bf16_cuda(
        a_act_bf16: *const c_void,
        packed_buckets: *const u8,
        tile_states: *const u16,
        tile_nb_totals: *const u16,
        tile_bitstream: *const u64,
        tile_bit_offsets: *const u64,
        beta_idx_packed: *const u8,
        offset_idx_packed: *const u8,
        beta_lloyd: *const f32,
        offset_lloyd: *const f32,
        y_acc_f32: *mut f32,
        out_y_bf16: *mut c_void,
        r_rows: u32,
        b_blocks: u32,
        n_blocks: u32,
        n_tiles: u32,
        k_beta: u32,
        k_offset: u32,
        w_offset: i32,
        tile_size: i32,
        has_offset: i32,
        stream: *mut c_void,
    );

    /// Instrumented v0 fused GEMV — same semantics as `leech_q24_gemv_bf16_cuda`
    /// but accumulates per-stage cycle counts into `stage_cycles_out` (device
    /// pointer to a `[u64; 8]` buffer). Stages, in order:
    ///
    /// 0. bucket_extract + split_bucket (per BLOCK)
    /// 1. pat_row[j] load
    /// 2. c_decode_tables[cb * M_TABLE + state] lookup
    /// 3. extract_nb_bits_from_window (Path A batch u64 read)
    /// 4. state = (base | bits_val) & M_MASK
    /// 5. a_act load + bf16→f32 cast
    /// 6. partial += w_val * a_f32
    /// 7. end-of-tile atomicAdd (per TILE)
    ///
    /// Output is identical to v0 (timing brackets only). Wall time is ~5-10%
    /// slower than v0 due to clock64 reads (~16 µs added at T=4 scale).
    pub(crate) fn leech_q24_gemv_bf16_timed_cuda(
        a_act_bf16: *const c_void,
        packed_buckets: *const u8,
        tile_states: *const u16,
        tile_nb_totals: *const u16,
        tile_bitstream: *const u64,
        tile_bit_offsets: *const u64,
        beta_idx_packed: *const u8,
        offset_idx_packed: *const u8,
        beta_lloyd: *const f32,
        offset_lloyd: *const f32,
        y_acc_f32: *mut f32,
        out_y_bf16: *mut c_void,
        r_rows: u32,
        b_blocks: u32,
        n_blocks: u32,
        n_tiles: u32,
        k_beta: u32,
        k_offset: u32,
        w_offset: i32,
        tile_size: i32,
        has_offset: i32,
        stage_cycles_out: *mut u64,
        stream: *mut c_void,
    );

    /// Phase B.1 warp-cooperative fused GEMV. Same signature as
    /// `leech_q24_gemv_bf16_cuda`. Selectable at runtime by the caller
    /// (env-var `LEECHQ24_WARPCOOP` in tests / `LeechLayer` flag in the
    /// future). v0 stays as the kill-switch path.
    pub(crate) fn leech_q24_gemv_bf16_warpcoop_cuda(
        a_act_bf16: *const c_void,
        packed_buckets: *const u8,
        tile_states: *const u16,
        tile_nb_totals: *const u16,
        tile_bitstream: *const u64,
        tile_bit_offsets: *const u64,
        beta_idx_packed: *const u8,
        offset_idx_packed: *const u8,
        beta_lloyd: *const f32,
        offset_lloyd: *const f32,
        y_acc_f32: *mut f32,
        out_y_bf16: *mut c_void,
        r_rows: u32,
        b_blocks: u32,
        n_blocks: u32,
        n_tiles: u32,
        k_beta: u32,
        k_offset: u32,
        w_offset: i32,
        tile_size: i32,
        has_offset: i32,
        stream: *mut c_void,
    );
}
