//! CUDA-backed Q24-tANS decode-only entry point.
//!
//! Drives the kernel in `mistralrs-quant/kernels/leech_q24/leech_q24_decode.cu`
//! from safe Rust. Phase A is decode-only — fused decode+β·v+offset+GEMV is
//! Phase B and lives in a sibling file.

use std::ffi::c_void;
use std::fmt;
use std::sync::OnceLock;

use crate::leech_q24::ffi::{
    leech_q24_decode_v_int_cuda, leech_q24_gemv_bf16_cuda,
    leech_q24_gemv_bf16_warpcoop_cuda, leech_q24_init_tables_ffi,
};

/// Sticky one-shot init guard. Subsequent `init_tables` calls with the SAME
/// `symbol_set_id` are a no-op; with a DIFFERENT `symbol_set_id`, the second
/// call re-runs the `cudaMemcpyToSymbol`s. The latter case is rare (only when
/// a process loads multiple .leech files with different ms variants), and
/// nothing else depends on the previous state.
static TABLES_INITIALIZED: OnceLock<()> = OnceLock::new();

#[derive(Debug)]
pub enum LeechQ24DecodeError {
    OutputBufferTooSmall { needed: usize, got: usize },
    UnsupportedTileSize(i32),
    UnsupportedWOffset(i32),
    TileBitOffsetsTooSmall { needed: usize, got: usize },
    DecodeTablesWrongSize { needed: usize, got: usize },
}

impl fmt::Display for LeechQ24DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeechQ24DecodeError::OutputBufferTooSmall { needed, got } => {
                write!(f, "out_v too small: need {needed} bytes, got {got}")
            }
            LeechQ24DecodeError::UnsupportedTileSize(t) => {
                write!(f, "unsupported tile_size {t}: production kernel hard-codes 32")
            }
            LeechQ24DecodeError::UnsupportedWOffset(w) => {
                write!(
                    f,
                    "unsupported w_offset {w}: ms=13 → 3, ms=18 → 4 are the only supported values"
                )
            }
            LeechQ24DecodeError::TileBitOffsetsTooSmall { needed, got } => {
                write!(
                    f,
                    "tile_bit_offsets too small: need {needed} u64s, got {got}"
                )
            }
            LeechQ24DecodeError::DecodeTablesWrongSize { needed, got } => {
                write!(
                    f,
                    "decode_tables wrong size: need {needed} u32s (4 × 1024), got {got}"
                )
            }
        }
    }
}

impl std::error::Error for LeechQ24DecodeError {}

/// Compute the host-side prefix sum of `tile_nb_totals` (u16) into
/// `tile_bit_offsets` (u64). The output's element `i` holds the cumulative
/// bit count of all tiles before `i`, so `out[0] = 0` and
/// `out[n] = Σ tile_nb_totals[0..n]`. The kernel uses this as the absolute
/// bit-offset base for each tile.
pub fn compute_tile_bit_offsets(tile_nb_totals: &[u16]) -> Vec<u64> {
    let mut out = Vec::with_capacity(tile_nb_totals.len());
    let mut acc: u64 = 0;
    for &nb in tile_nb_totals {
        out.push(acc);
        acc += nb as u64;
    }
    out
}

/// One-shot init. Copies the FSE decode_tables (4 × 1024 × u32) and the
/// universal pattern_table baked at compile time into the kernel's
/// `__constant__` / `__device__` arrays.
///
/// `decode_tables` is a host slice of `4 × 1024 = 4096` u32 entries
/// (n_codebooks × M_TABLE). The pattern_table is taken from the compile-time
/// baked header — the LEECHQ24 file's `pattern_table_blake3` header field
/// MUST match the constant in `leech_q24_pattern_table.h`.
pub fn init_tables(decode_tables: &[u32], symbol_set_id: u32) -> Result<(), LeechQ24DecodeError> {
    const NEEDED: usize = 4 * 1024;
    if decode_tables.len() != NEEDED {
        return Err(LeechQ24DecodeError::DecodeTablesWrongSize {
            needed: NEEDED,
            got: decode_tables.len(),
        });
    }
    TABLES_INITIALIZED.get_or_init(|| {
        // SAFETY: `decode_tables_host` points to a host array of NEEDED u32s,
        // the kernel copies into a static __constant__ symbol of equal size.
        unsafe { leech_q24_init_tables_ffi(decode_tables.as_ptr(), symbol_set_id) };
    });
    Ok(())
}

/// Decode `n_tiles` tiles from the Q24-tANS bitstream → `out_v` (int8[n_blocks, 24]).
///
/// All large buffers are device-allocated raw pointers; the Rust slice lengths
/// passed in are used as size hints for bounds checks at the host boundary
/// but the kernel reads / writes purely through the raw pointers.
///
/// # Args
/// - `packed_buckets`: device, 13-bit packed buckets, ≥1 trailing byte pad.
/// - `tile_states`: device, `[n_tiles] u16` LE.
/// - `tile_nb_totals`: device, `[n_tiles] u16` LE.
/// - `tile_bitstream`: device, `[bs_words] u64` LE.
/// - `tile_bit_offsets`: device, `[n_tiles] u64`, precomputed via
///   [`compute_tile_bit_offsets`] on host.
/// - `out_v`: device, `[n_blocks × 24] i8`.
/// - `n_blocks` / `n_tiles`: tensor block / tile counts.
/// - `w_offset`: 3 (S=7, ms=13) or 4 (S=9, ms=18).
/// - `tile_size`: production hard-codes 32; pass 32 here.
/// - `stream`: cudaStream_t (raw pointer). Pass null for default stream.
///
/// # Safety
/// All pointers must reference device memory of the declared size. The Rust
/// slice lengths are size hints only.
#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_q24_decode_v_int(
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
) -> Result<(), LeechQ24DecodeError> {
    if tile_size != 32 {
        return Err(LeechQ24DecodeError::UnsupportedTileSize(tile_size));
    }
    if !(w_offset == 3 || w_offset == 4) {
        return Err(LeechQ24DecodeError::UnsupportedWOffset(w_offset));
    }
    unsafe {
        leech_q24_decode_v_int_cuda(
            packed_buckets,
            tile_states,
            tile_nb_totals,
            tile_bitstream,
            tile_bit_offsets,
            out_v,
            n_blocks,
            n_tiles,
            w_offset,
            tile_size,
            stream,
        );
    }
    Ok(())
}

/// Phase B.0 fused decode + β·v + offset + GEMV → bf16 (batch M=1).
///
/// All large buffers are device pointers. The Rust side does not interpret
/// any of them — it only enforces tile_size and w_offset are supported and
/// forwards to the C kernel.
///
/// # Args
/// - `a_act_bf16_ptr`: device, `[K_total]` bf16 (K_total = b_blocks * 24).
/// - `packed_buckets`, `tile_states`, `tile_nb_totals`, `tile_bitstream`,
///   `tile_bit_offsets`: same as the decode-only path.
/// - `beta_idx_packed`: device, `[n_blocks]` 3-bit packed.
/// - `offset_idx_packed`: device, `[n_blocks]` 3-bit packed; pass null when
///   `has_offset == false`.
/// - `beta_lloyd_ptr`: device, `[R, k_beta]` f32.
/// - `offset_lloyd_ptr`: device, `[R, k_offset]` f32; pass null when no offset.
/// - `y_acc_f32_ptr`: device scratch, `[R]` f32. The kernel zero-fills before
///   the atomicAdd pass; caller need not pre-zero.
/// - `out_y_bf16_ptr`: device, `[R]` bf16. The narrowed result.
/// - `r_rows`, `b_blocks`, `n_blocks`, `n_tiles`, `k_beta`, `k_offset`:
///   tensor dimensions.
/// - `w_offset`: 3 (S=7, ms=13) or 4 (S=9, ms=18).
/// - `tile_size`: production hard-codes 32.
/// - `has_offset`: 0 or 1.
/// - `stream`: cudaStream_t; null for default.
///
/// # Safety
/// All pointers must reference device memory of the declared size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_q24_gemv_bf16(
    a_act_bf16_ptr: *const c_void,
    packed_buckets: *const u8,
    tile_states: *const u16,
    tile_nb_totals: *const u16,
    tile_bitstream: *const u64,
    tile_bit_offsets: *const u64,
    beta_idx_packed: *const u8,
    offset_idx_packed: *const u8,
    beta_lloyd_ptr: *const f32,
    offset_lloyd_ptr: *const f32,
    y_acc_f32_ptr: *mut f32,
    out_y_bf16_ptr: *mut c_void,
    r_rows: u32,
    b_blocks: u32,
    n_blocks: u32,
    n_tiles: u32,
    k_beta: u32,
    k_offset: u32,
    w_offset: i32,
    tile_size: i32,
    has_offset: bool,
    stream: *mut c_void,
) -> Result<(), LeechQ24DecodeError> {
    if tile_size != 32 {
        return Err(LeechQ24DecodeError::UnsupportedTileSize(tile_size));
    }
    if !(w_offset == 3 || w_offset == 4) {
        return Err(LeechQ24DecodeError::UnsupportedWOffset(w_offset));
    }
    unsafe {
        leech_q24_gemv_bf16_cuda(
            a_act_bf16_ptr,
            packed_buckets,
            tile_states,
            tile_nb_totals,
            tile_bitstream,
            tile_bit_offsets,
            beta_idx_packed,
            offset_idx_packed,
            beta_lloyd_ptr,
            offset_lloyd_ptr,
            y_acc_f32_ptr,
            out_y_bf16_ptr,
            r_rows,
            b_blocks,
            n_blocks,
            n_tiles,
            k_beta,
            k_offset,
            w_offset,
            tile_size,
            if has_offset { 1 } else { 0 },
            stream,
        );
    }
    Ok(())
}

/// Phase B.1 warp-cooperative fused GEMV. Identical surface to
/// [`leech_q24_gemv_bf16`]; the caller chooses which kernel to invoke
/// (typically via the `LEECHQ24_WARPCOOP` env var in tests, or a runtime
/// flag in `LeechLayer`).
///
/// # Safety
/// All pointers must reference device memory of the declared size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_q24_gemv_bf16_warpcoop(
    a_act_bf16_ptr: *const c_void,
    packed_buckets: *const u8,
    tile_states: *const u16,
    tile_nb_totals: *const u16,
    tile_bitstream: *const u64,
    tile_bit_offsets: *const u64,
    beta_idx_packed: *const u8,
    offset_idx_packed: *const u8,
    beta_lloyd_ptr: *const f32,
    offset_lloyd_ptr: *const f32,
    y_acc_f32_ptr: *mut f32,
    out_y_bf16_ptr: *mut c_void,
    r_rows: u32,
    b_blocks: u32,
    n_blocks: u32,
    n_tiles: u32,
    k_beta: u32,
    k_offset: u32,
    w_offset: i32,
    tile_size: i32,
    has_offset: bool,
    stream: *mut c_void,
) -> Result<(), LeechQ24DecodeError> {
    if tile_size != 32 {
        return Err(LeechQ24DecodeError::UnsupportedTileSize(tile_size));
    }
    if !(w_offset == 3 || w_offset == 4) {
        return Err(LeechQ24DecodeError::UnsupportedWOffset(w_offset));
    }
    unsafe {
        leech_q24_gemv_bf16_warpcoop_cuda(
            a_act_bf16_ptr,
            packed_buckets,
            tile_states,
            tile_nb_totals,
            tile_bitstream,
            tile_bit_offsets,
            beta_idx_packed,
            offset_idx_packed,
            beta_lloyd_ptr,
            offset_lloyd_ptr,
            y_acc_f32_ptr,
            out_y_bf16_ptr,
            r_rows,
            b_blocks,
            n_blocks,
            n_tiles,
            k_beta,
            k_offset,
            w_offset,
            tile_size,
            if has_offset { 1 } else { 0 },
            stream,
        );
    }
    Ok(())
}
