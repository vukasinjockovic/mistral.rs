//! CPU stub for non-CUDA builds.
//!
//! Every entry point returns `LeechDecodeError::CudaRequired`. To actually
//! decode `.leech` weights, rebuild with `--features cuda` and an nvcc toolchain.
//!
//! Future: a numba-free pure-Rust CPU decoder could live here, ported from
//! `packer/core/leech_decode_njit_v2.py`. Not in scope for Phase 3 — see
//! `LEECH_INTEGRATION_PLAN.md` Q3.

use std::fmt;

#[derive(Debug)]
pub enum LeechDecodeError {
    CudaRequired,
}

impl fmt::Display for LeechDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeechDecodeError::CudaRequired => write!(
                f,
                ".leech decode requires the `cuda` feature; rebuild with `--features cuda`"
            ),
        }
    }
}

impl std::error::Error for LeechDecodeError {}

/// No-op on non-CUDA builds.
pub fn init_tables() -> Result<(), LeechDecodeError> {
    Err(LeechDecodeError::CudaRequired)
}

/// Errors out — see module docstring.
pub fn leech_decode_v_int(
    _packed_stream: &[u8],
    _out_v_int: &mut [i8],
    _n_blocks: u32,
    _idx_bits: u32,
    _has_offset: bool,
) -> Result<(), LeechDecodeError> {
    Err(LeechDecodeError::CudaRequired)
}
