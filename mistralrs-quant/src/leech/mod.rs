//! `.leech` LLVQ-quantized weight format support for mistral.rs.
//!
//! Module surface:
//!   - [`leech_layer`] / [`LeechLayer`]: the `QuantMethod` impl.
//!   - [`leech_linear`] / [`leech_linear_from_tensors`]: constructors plumbed
//!     into the `mistralrs-quant::linear*` dispatch.
//!   - [`init_tables`] / [`leech_decode_v_int`]: low-level CUDA decoder
//!     (Phase 3) — kept on the public surface for the Phase 4 fused-kernel
//!     plumbing and the GPU integration test.
//!
//! All container parsing lives in the sibling crate `mistralrs-leech`.
//! All universal Leech tables (codebooks, sign metadata, etc.) are
//! compile-time baked into the kernel TU via
//! `kernels/leech/leech_tables_ms{13,18}.h`.

pub mod leech_layer;
pub mod leech_linear;

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod leech_cuda;

#[cfg(not(feature = "cuda"))]
mod leech_cpu;

pub use leech_layer::LeechLayer;
pub use leech_linear::{leech_linear, leech_linear_from_tensors};

#[cfg(feature = "cuda")]
pub use leech_cuda::{
    init_tables, leech_decode_bf16, leech_decode_v_int, leech_gemv_bf16, LeechDecodeError,
};

#[cfg(not(feature = "cuda"))]
pub use leech_cpu::{init_tables, leech_decode_v_int, LeechDecodeError};
