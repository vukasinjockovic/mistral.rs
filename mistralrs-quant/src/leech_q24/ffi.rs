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
}
