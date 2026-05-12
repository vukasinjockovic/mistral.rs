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
}
