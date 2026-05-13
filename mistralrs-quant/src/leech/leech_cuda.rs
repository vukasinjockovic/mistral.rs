//! CUDA-backed `.leech` decode-only entry point.
//!
//! Drives the kernel in `mistralrs-quant/kernels/leech/leech_decode.cu` from
//! safe Rust. The actual kernel launch is in C; this layer handles the
//! Rust-side argument validation, stream wiring, and error reporting.
//!
//! Phase 3 scope: decode-only. The full `QuantMethod` integration (with β·v
//! + offset epilogue and fused GEMM) is Phase 4.

use std::ffi::c_void;
use std::fmt;
use std::sync::OnceLock;

use crate::leech::ffi::{
    leech_compute_block_parity_cuda, leech_decode_bf16_cuda, leech_decode_v_int_cuda,
    leech_gemv_bf16_cuda, leech_init_tables_ffi,
};

/// Sticky one-shot init.
static TABLES_INITIALIZED: OnceLock<()> = OnceLock::new();

#[derive(Debug)]
pub enum LeechDecodeError {
    /// Output buffer was not sized for `n_blocks * 24` int8 cells.
    OutputBufferTooSmall { needed: usize, got: usize },
    /// Packed bitstream is too short for the requested block count.
    PackedStreamTooShort { needed: usize, got: usize },
    /// idx_bits outside the supported {48, 54} set.
    UnsupportedIdxBits(u32),
}

impl fmt::Display for LeechDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeechDecodeError::OutputBufferTooSmall { needed, got } => write!(
                f,
                "out_v_int too small: need {needed} bytes, got {got}"
            ),
            LeechDecodeError::PackedStreamTooShort { needed, got } => write!(
                f,
                "packed_stream too short: need ≥ {needed} bytes (incl. 8B tail-pad), got {got}"
            ),
            LeechDecodeError::UnsupportedIdxBits(b) => write!(
                f,
                "unsupported idx_bits {b}: kernel templates exist for {{48, 54}} only"
            ),
        }
    }
}

impl std::error::Error for LeechDecodeError {}

/// Copy all universal Leech tables into the kernel's __constant__ / __device__
/// memory regions. Idempotent — safe to call repeatedly. The first call pays
/// ~30 ms for the 12 MB device-side copy at ms=18; subsequent calls are O(1).
///
/// Must be called once before any [`leech_decode_v_int`] call. Layers that
/// want to be lazy can call from their construction path.
pub fn init_tables() -> Result<(), LeechDecodeError> {
    TABLES_INITIALIZED.get_or_init(|| {
        // SAFETY: `leech_init_tables_ffi` invokes `cudaMemcpyToSymbol` on
        // statically-allocated symbols; it cannot violate Rust memory safety.
        unsafe { leech_init_tables_ffi() };
    });
    Ok(())
}

/// Decode `n_blocks` LLVQ blocks from a packed bitstream into 24-int8-per-block
/// rows. Both buffers must be DEVICE pointers (allocated via CUDA, not host
/// memory) — Phase 3 is decode-only and does not move data between host and
/// device.
///
/// # Args
/// - `packed_stream`: device-side body bitstream. MUST be padded by ≥ 8 bytes
///   past the last meaningful byte so the kernel's 2× u64 load on the final
///   block doesn't OOB.
/// - `out_v_int`: device-side output buffer, sized for `n_blocks * 24` cells.
/// - `n_blocks`: block count (= R * B for one tensor).
/// - `idx_bits`: 48 (ms=13) or 54 (ms=18). Selects the kernel template.
/// - `has_offset`: whether each block has the 3-bit offset codebook index.
/// - `stream`: optional CUDA stream pointer (raw cudaStream_t). `null` runs
///   on the default stream.
///
/// # Safety
/// `packed_stream` and `out_v_int` MUST point to device-allocated memory at
/// least as large as the declared sizes. The Rust slice lengths are inspected
/// only as size hints — callers can shadow a `CudaSlice<T>` with a `&[u8]` /
/// `&mut [i8]` of matching length for this argument.
pub fn leech_decode_v_int(
    packed_stream: &[u8],
    out_v_int: &mut [i8],
    n_blocks: u32,
    idx_bits: u32,
    has_offset: bool,
    stream: *mut c_void,
) -> Result<(), LeechDecodeError> {
    init_tables()?;

    let needed_out = (n_blocks as usize) * 24;
    if out_v_int.len() < needed_out {
        return Err(LeechDecodeError::OutputBufferTooSmall {
            needed: needed_out,
            got: out_v_int.len(),
        });
    }
    let per_block_bits = idx_bits + 3 + if has_offset { 3 } else { 0 };
    let needed_bits = (n_blocks as u64) * (per_block_bits as u64);
    let needed_pad = ((needed_bits + 7) / 8) as usize + 8;
    if packed_stream.len() < needed_pad {
        return Err(LeechDecodeError::PackedStreamTooShort {
            needed: needed_pad,
            got: packed_stream.len(),
        });
    }
    if idx_bits != 48 && idx_bits != 54 {
        return Err(LeechDecodeError::UnsupportedIdxBits(idx_bits));
    }

    // SAFETY: kernel runs on device-side raw pointers — Rust slice lengths
    // are advisory here. The kernel reads `(n_blocks * per_block_bits + 7) / 8`
    // bytes from `packed_stream` and writes `n_blocks * 24` to `out_v_int`,
    // both of which we just bounded.
    unsafe {
        leech_decode_v_int_cuda(
            packed_stream.as_ptr(),
            out_v_int.as_mut_ptr(),
            n_blocks,
            idx_bits as std::os::raw::c_int,
            if has_offset { 1 } else { 0 },
            stream,
        );
    }
    Ok(())
}

/// Phase 4.0a: decode + β·v + offset → bf16 weight tile in one kernel.
///
/// All buffers are DEVICE pointers. This is the fused replacement for the
/// `decode_v_int → fp32 FMA → bf16` two-pass chain — eliminates the int8 HBM
/// roundtrip and runs the LOCKED epilogue (`bf16 = RNE(fp32(β·v + offset))`)
/// inline per block.
///
/// # Args
/// - `packed_stream`: device-side body bitstream, tail-padded by ≥ 8 bytes.
/// - `beta_codebook_ptr`: device pointer to `R * K_beta` fp16 entries.
/// - `offset_codebook_ptr`: device pointer to `R * K_offset` fp16 entries; pass
///   null if `has_offset` is false.
/// - `out_weight_bf16_ptr`: device pointer to `R * B * 24` bf16 entries.
/// - `r_rows` / `b_blocks` / `k_beta` / `k_offset`: tensor dimensions.
/// - `idx_bits`: 48 (ms=13) or 54 (ms=18).
/// - `has_offset`: whether per-block offset bits are present.
/// - `stream`: optional CUDA stream pointer.
///
/// # Safety
/// All pointers must be device-allocated and sized correctly. Callers should
/// hold their `CudaSlice<T>` guards alive across the call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_decode_bf16(
    packed_stream: *const u8,
    beta_codebook_ptr: *const c_void,
    offset_codebook_ptr: *const c_void,
    out_weight_bf16_ptr: *mut c_void,
    r_rows: u32,
    b_blocks: u32,
    k_beta: u32,
    k_offset: u32,
    idx_bits: u32,
    has_offset: bool,
    stream: *mut c_void,
) -> Result<(), LeechDecodeError> {
    init_tables()?;
    if idx_bits != 48 && idx_bits != 54 {
        return Err(LeechDecodeError::UnsupportedIdxBits(idx_bits));
    }
    unsafe {
        leech_decode_bf16_cuda(
            packed_stream,
            beta_codebook_ptr,
            offset_codebook_ptr,
            out_weight_bf16_ptr,
            r_rows,
            b_blocks,
            k_beta,
            k_offset,
            idx_bits as std::os::raw::c_int,
            if has_offset { 1 } else { 0 },
            stream,
        );
    }
    Ok(())
}

/// Phase 4.0b: fused decode + β·v + offset + GEMV → bf16 output. All buffers
/// are DEVICE pointers. Each output element is computed by one thread that
/// walks its weight row's blocks in-register. Best at batch=1 generation.
///
/// # Args
/// - `a_act_bf16`: device pointer to `[m, b_blocks*24]` bf16 activations.
/// - `packed_stream`: device, weight bitstream tail-padded by ≥ 8 bytes.
/// - `beta_codebook_ptr`: device pointer to `[n_rows, k_beta]` fp16 entries.
/// - `offset_codebook_ptr`: device pointer to `[n_rows, k_offset]` fp16 entries;
///   pass null if `has_offset` is false.
/// - `out_y_bf16`: device pointer to `[m, n_rows]` bf16 output.
/// - `m`: batch rows.
/// - `n_rows`: weight output features (= `LeechLayer::rows`).
/// - `b_blocks`: blocks per weight row.
/// - `k_beta` / `k_offset`: codebook widths.
/// - `idx_bits`: 48 (ms=13) or 54 (ms=18).
/// - `has_offset`: whether per-block offset bits are present.
/// - `stream`: optional CUDA stream pointer.
///
/// # Safety
/// All pointers must be device-allocated; caller holds CudaSlice guards alive.
#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_gemv_bf16(
    a_act_bf16_ptr: *const c_void,
    packed_stream: *const u8,
    beta_codebook_ptr: *const c_void,
    offset_codebook_ptr: *const c_void,
    parity_perm_ptr: *const c_void,
    out_y_bf16_ptr: *mut c_void,
    m: u32,
    n_rows: u32,
    b_blocks: u32,
    k_beta: u32,
    k_offset: u32,
    idx_bits: u32,
    has_offset: bool,
    stream: *mut c_void,
) -> Result<(), LeechDecodeError> {
    init_tables()?;
    if idx_bits != 48 && idx_bits != 54 {
        return Err(LeechDecodeError::UnsupportedIdxBits(idx_bits));
    }
    unsafe {
        leech_gemv_bf16_cuda(
            a_act_bf16_ptr,
            packed_stream,
            beta_codebook_ptr,
            offset_codebook_ptr,
            parity_perm_ptr,
            out_y_bf16_ptr,
            m,
            n_rows,
            b_blocks,
            k_beta,
            k_offset,
            idx_bits as std::os::raw::c_int,
            if has_offset { 1 } else { 0 },
            stream,
        );
    }
    Ok(())
}

/// Compute per-block decode parity into out_parity_ptr (device, u8[n_blocks]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_compute_block_parity(
    packed_stream: *const u8,
    out_parity_ptr: *mut u8,
    n_blocks: u32,
    idx_bits: u32,
    has_offset: bool,
    stream: *mut c_void,
) -> Result<(), LeechDecodeError> {
    init_tables()?;
    if idx_bits != 48 && idx_bits != 54 {
        return Err(LeechDecodeError::UnsupportedIdxBits(idx_bits));
    }
    unsafe {
        leech_compute_block_parity_cuda(
            packed_stream,
            out_parity_ptr,
            n_blocks,
            idx_bits as std::os::raw::c_int,
            if has_offset { 1 } else { 0 },
            stream,
        );
    }
    Ok(())
}
