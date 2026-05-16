//! CPU-build stub for `leech_q24`. The Q24-tANS decoder is GPU-only in
//! mistral.rs — the production CPU reference lives in
//! `antsquant/production/packer/q24_tans/core/codec.py::decode_tile_kernel`.
//!
//! This module exists so the crate compiles without the `cuda` feature; any
//! call into the decoder on CPU panics with a clear message.

use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum LeechQ24DecodeError {
    CpuNotImplemented,
}

impl fmt::Display for LeechQ24DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeechQ24DecodeError::CpuNotImplemented => write!(
                f,
                "Q24-tANS CPU decoder is not provided by this crate — \
                 build with `--features cuda` or use the Python reference in \
                 production/packer/q24_tans/core/codec.py"
            ),
        }
    }
}

impl std::error::Error for LeechQ24DecodeError {}

pub fn compute_tile_bit_offsets(tile_nb_totals: &[u16]) -> Vec<u64> {
    let mut out = Vec::with_capacity(tile_nb_totals.len());
    let mut acc: u64 = 0;
    for &nb in tile_nb_totals {
        out.push(acc);
        acc += nb as u64;
    }
    out
}

pub fn init_tables(_decode_tables: &[u32], _symbol_set_id: u32) -> Result<(), LeechQ24DecodeError> {
    Err(LeechQ24DecodeError::CpuNotImplemented)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_q24_decode_v_int(
    _packed_buckets: *const u8,
    _tile_states: *const u16,
    _tile_nb_totals: *const u16,
    _tile_bitstream: *const u64,
    _tile_bit_offsets: *const u64,
    _out_v: *mut i8,
    _n_blocks: u32,
    _n_tiles: u32,
    _w_offset: i32,
    _tile_size: i32,
    _stream: *mut c_void,
) -> Result<(), LeechQ24DecodeError> {
    Err(LeechQ24DecodeError::CpuNotImplemented)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_q24_gemv_bf16(
    _a_act_bf16_ptr: *const c_void,
    _packed_buckets: *const u8,
    _tile_states: *const u16,
    _tile_nb_totals: *const u16,
    _tile_bitstream: *const u64,
    _tile_bit_offsets: *const u64,
    _beta_idx_packed: *const u8,
    _offset_idx_packed: *const u8,
    _beta_lloyd_ptr: *const f32,
    _offset_lloyd_ptr: *const f32,
    _y_acc_f32_ptr: *mut f32,
    _out_y_bf16_ptr: *mut c_void,
    _r_rows: u32,
    _b_blocks: u32,
    _n_blocks: u32,
    _n_tiles: u32,
    _k_beta: u32,
    _k_offset: u32,
    _w_offset: i32,
    _tile_size: i32,
    _has_offset: bool,
    _stream: *mut c_void,
) -> Result<(), LeechQ24DecodeError> {
    Err(LeechQ24DecodeError::CpuNotImplemented)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn leech_q24_gemv_bf16_warpcoop(
    _a_act_bf16_ptr: *const c_void,
    _packed_buckets: *const u8,
    _tile_states: *const u16,
    _tile_nb_totals: *const u16,
    _tile_bitstream: *const u64,
    _tile_bit_offsets: *const u64,
    _beta_idx_packed: *const u8,
    _offset_idx_packed: *const u8,
    _beta_lloyd_ptr: *const f32,
    _offset_lloyd_ptr: *const f32,
    _y_acc_f32_ptr: *mut f32,
    _out_y_bf16_ptr: *mut c_void,
    _r_rows: u32,
    _b_blocks: u32,
    _n_blocks: u32,
    _n_tiles: u32,
    _k_beta: u32,
    _k_offset: u32,
    _w_offset: i32,
    _tile_size: i32,
    _has_offset: bool,
    _stream: *mut c_void,
) -> Result<(), LeechQ24DecodeError> {
    Err(LeechQ24DecodeError::CpuNotImplemented)
}
