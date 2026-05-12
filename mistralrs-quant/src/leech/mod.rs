//! `.leech` LLVQ-quantized weight format support for mistral.rs.
//!
//! Phase 3 scope: a decode-only CUDA kernel + thin Rust wrapper. The full
//! `QuantMethod` integration (fused dequant+GEMM, mistral.rs dispatch arms)
//! lands in Phase 4/5 per `LEECH_INTEGRATION_PLAN.md`.
//!
//! Module structure mirrors `gptq/mod.rs`:
//!
//!   - `cuda` feature: `leech_cuda` exports the live CUDA-backed decoder.
//!   - no `cuda`: `leech_cpu` exports a stub that returns a "rebuild with
//!     --features cuda" error. This keeps `cargo check` clean on machines
//!     without nvcc.
//!
//! All `.leech` container parsing happens in the sibling crate `mistralrs-leech`
//! (Phase 1). This module is the GPU-side decoder only.

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod leech_cuda;

#[cfg(not(feature = "cuda"))]
mod leech_cpu;

#[cfg(feature = "cuda")]
pub use leech_cuda::{init_tables, leech_decode_v_int, LeechDecodeError};

#[cfg(not(feature = "cuda"))]
pub use leech_cpu::{init_tables, leech_decode_v_int, LeechDecodeError};
