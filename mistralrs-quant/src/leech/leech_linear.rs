//! `leech_linear` — constructor that produces an `Arc<dyn QuantMethod>` from
//! a `QuantizedConfig::Leech` + a `ShardedVarBuilder`.
//!
//! Called from the dispatch arms in `mistralrs-quant/src/lib.rs` (`linear`,
//! `linear_no_bias`, `linear_b`). Sources the per-tensor LLVQ payload from
//! the `ShardedVarBuilder` — naming convention:
//!
//!     <prefix>.packed_stream                u8   [stream_bytes + 8 (tail-pad)]
//!     <prefix>.beta_codebook                f16  [R, K_beta]
//!     <prefix>.offset_codebook              f16  [R, K_offset]  (optional)
//!     <prefix>.leftover_bf16                bf16 [R, leftover_cols] (optional)
//!     <prefix>.bias                         bf16 [R]              (optional)
//!
//! These tensors are populated by the Phase 5 sidecar loader in
//! `mistralrs-core` (next slot in the integration plan), which converts a
//! `.leech` container's payload bytes into named tensors before handing the
//! `ShardedVarBuilder` to model construction.
//!
//! No GEMM yet — `forward_raw` still bails until Phase 4 lands. The
//! constructor is what tells us the wiring is sound.

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::leech::leech_layer::LeechLayer;
use crate::{QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder};

/// Build a `LeechLayer` from a `QuantizedConfig::Leech` + variable builder.
///
/// `in_dim` / `out_dim` are the *unquantized* weight shape (rows × cols).
/// `with_bias` is true when the surrounding `linear` (vs `linear_no_bias`)
/// dispatched here.
pub fn leech_linear(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    with_bias: bool,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let (ms_used, idx_bits, has_offset) = match config {
        QuantizedConfig::Leech {
            ms_used,
            idx_bits,
            has_offset,
        } => (*ms_used, *idx_bits, *has_offset),
        _ => candle_core::bail!("leech_linear: expected QuantizedConfig::Leech, got {:?}", config),
    };

    // Phase 5 prep stub: the actual `ShardedVarBuilder` integration with
    // `.leech` payload tensors lives in the sidecar loader on the
    // `mistralrs-core` side. Right now we don't have a loader producing
    // these named tensors, so this constructor errors out — but the
    // dispatch arms in `linear*` are wired and compile clean.
    //
    // When Phase 5 (loader) lands, replace this body with:
    //   let packed_stream = vb.get_unchecked("packed_stream")?;
    //   let beta_codebook = vb.get_unchecked("beta_codebook")?;
    //   let offset_codebook = has_offset.then(|| vb.get_unchecked("offset_codebook")).transpose()?;
    //   let leftover_bf16  = vb.contains_tensor("leftover_bf16").then(...).transpose()?;
    //   let bias = with_bias.then(|| vb.get_unchecked("bias")).transpose()?;
    //   let layer = <LeechLayer as QuantMethod>::new(QuantMethodConfig::Leech { ... })?;
    //   Ok(Arc::new(layer) as Arc<dyn QuantMethod>)

    let _ = (in_dim, out_dim, with_bias, vb, ms_used, idx_bits, has_offset);
    candle_core::bail!(
        "leech_linear: full loader integration is Phase 5b (mistralrs-core sidecar). \
         Dispatch wiring is in place; the .leech → tensor pump is not."
    )
}

/// Build a `LeechLayer` directly from already-loaded tensors. Used by tests
/// and by the eventual sidecar loader bypassing the `ShardedVarBuilder`
/// indirection.
#[allow(clippy::too_many_arguments)]
pub fn leech_linear_from_tensors(
    packed_stream: Tensor,
    beta_codebook: Tensor,
    offset_codebook: Option<Tensor>,
    rows: u32,
    blocks_per_row: u32,
    idx_bits: u8,
    has_offset: bool,
    ms_used: u8,
    leftover_bf16: Option<Tensor>,
    bias: Option<Tensor>,
    _device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    let layer = <LeechLayer as QuantMethod>::new(QuantMethodConfig::Leech {
        packed_stream,
        beta_codebook,
        offset_codebook,
        rows,
        blocks_per_row,
        idx_bits,
        has_offset,
        ms_used,
        leftover_bf16,
        bias,
    })?;
    Ok(Arc::new(layer) as Arc<dyn QuantMethod>)
}
