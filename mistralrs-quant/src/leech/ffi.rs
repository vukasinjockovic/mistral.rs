//! `extern "C"` declarations for the CUDA-backed leech decoder.
//!
//! Implemented in `mistralrs-quant/kernels/leech/leech_decode.cu`. Compiled
//! and linked by `mistralrs-quant/build.rs` via the `kernels/*/*.cu` glob.

use std::os::raw::{c_int, c_void};

#[allow(dead_code)]
extern "C" {
    /// Copy all universal Leech tables into __constant__ / __device__ memory.
    /// Idempotent — safe to call multiple times. Must be called once per
    /// process before any decode call.
    pub(crate) fn leech_init_tables_ffi();

    /// Decode `n_blocks` LLVQ blocks from `packed_stream` to `out_v_int`.
    ///
    /// # Args
    /// - `packed_stream`: device pointer to the body bitstream. MUST be padded
    ///   by ≥8 bytes past the last meaningful byte so the kernel's two-u64
    ///   load doesn't OOB on the final block.
    /// - `out_v_int`: device pointer to `n_blocks * 24` int8 cells.
    /// - `n_blocks`: number of LLVQ blocks to decode (= R * B for a tensor).
    /// - `idx_bits`: per-block i_global width. ms=18 → 54, ms=13 → 48.
    /// - `has_offset`: 0 or 1. When 1, each block has 3 extra trailing bits
    ///   (offset codebook index).
    /// - `stream`: CUDA stream handle. Pass null for the default stream.
    pub(crate) fn leech_decode_v_int_cuda(
        packed_stream: *const u8,
        out_v_int: *mut i8,
        n_blocks: u32,
        idx_bits: c_int,
        has_offset: c_int,
        stream: *mut c_void, // cudaStream_t — opaque to Rust
    );

    /// Phase 4.0a: decode + β·v + offset → bf16 weight tile.
    ///
    /// One kernel that combines the Phase 3 decode with the LOCKED epilogue
    /// `bf16 = RNE(fp32(β * v_int + offset))`. Eliminates the int8 HBM
    /// roundtrip from the Phase 3 path (used by `dequantize_w`).
    ///
    /// # Args
    /// - `packed_stream`: device, tail-padded by ≥ 8 bytes.
    /// - `beta_codebook`: device, `R * K_beta` fp16 entries.
    /// - `offset_codebook`: device, `R * K_offset` fp16 entries; pass null if
    ///   `has_offset == 0`.
    /// - `out_weight_bf16`: device, `R * B * 24` bf16 entries.
    /// - `r_rows`: row count of the decoded weight matrix.
    /// - `b_blocks`: LLVQ blocks per row.
    /// - `k_beta`: β codebook entries per row (typically 8).
    /// - `k_offset`: offset codebook entries per row (0 or 8).
    /// - `idx_bits`: 48 (ms=13) or 54 (ms=18).
    /// - `has_offset`: 0 or 1.
    /// - `stream`: cudaStream_t. Pass null for the default stream.
    pub(crate) fn leech_decode_bf16_cuda(
        packed_stream: *const u8,
        beta_codebook: *const c_void,
        offset_codebook: *const c_void,
        out_weight_bf16: *mut c_void,
        r_rows: u32,
        b_blocks: u32,
        k_beta: u32,
        k_offset: u32,
        idx_bits: c_int,
        has_offset: c_int,
        stream: *mut c_void,
    );

    /// Phase 4.0b: fused decode + β·v + offset + GEMV → bf16 output. No HBM
    /// roundtrip for decoded weights. Best at batch=1 generation. M-fold
    /// redundant decode at large M — use `leech_decode_bf16_cuda` +
    /// candle matmul for prefill / batch >= ~16.
    pub(crate) fn leech_gemv_bf16_cuda(
        a_act_bf16: *const c_void,
        packed_stream: *const u8,
        beta_codebook: *const c_void,
        offset_codebook: *const c_void,
        parity_perm: *const c_void,
        out_y_bf16: *mut c_void,
        m: u32,
        n_rows: u32,
        b_blocks: u32,
        k_beta: u32,
        k_offset: u32,
        idx_bits: c_int,
        has_offset: c_int,
        stream: *mut c_void,
    );

    /// Compute per-block decode parity (0=even, 1=odd). Used by host code to
    /// build the parity-sort permutation passed to leech_gemv_bf16_cuda.
    pub(crate) fn leech_compute_block_parity_cuda(
        packed_stream: *const u8,
        out_parity: *mut u8,
        n_blocks: u32,
        idx_bits: c_int,
        has_offset: c_int,
        stream: *mut c_void,
    );
}
