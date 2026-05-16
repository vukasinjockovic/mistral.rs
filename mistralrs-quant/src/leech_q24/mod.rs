//! Q24-tANS .leech v3 decoder bindings.
//!
//! Phase A (decode-only): produces int8[N, 24] from the Q24-tANS bitstream.
//! Phase B (planned): fused decode + β·v + offset + GEMV → bf16 output.
//!
//! The container is parsed by the sibling crate `mistralrs-leech-q24`. This
//! module owns the CUDA kernel and the Rust FFI surface.

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod leech_q24_cuda;

#[cfg(not(feature = "cuda"))]
mod leech_q24_cpu;

#[cfg(feature = "cuda")]
pub use leech_q24_cuda::{
    compute_tile_bit_offsets, init_tables, leech_q24_decode_v_int, leech_q24_gemv_bf16,
    leech_q24_gemv_bf16_warpcoop, LeechQ24DecodeError,
};

#[cfg(not(feature = "cuda"))]
pub use leech_q24_cpu::{
    compute_tile_bit_offsets, init_tables, leech_q24_decode_v_int, leech_q24_gemv_bf16,
    leech_q24_gemv_bf16_warpcoop, LeechQ24DecodeError,
};
