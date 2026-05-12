//! `LeechLayer` — `QuantMethod` impl for `.leech` LLVQ-quantized weights.
//!
//! Phase 5 scope: the trait surface + constructor + dequantize path. The
//! actual `forward_raw` (fused decode+GEMM) lands in Phase 4 — for now it
//! returns a "not yet implemented" error.
//!
//! The intended Phase 4 forward path:
//!     activation (bf16) ──┐
//!                          ├──→ leech_fused_gemm ──→ output (bf16)
//!     packed_stream  ────┤    (decode + β·v + offset + GEMM in one kernel)
//!     β / offset cb ────┤
//!     parity_perm   ────┘
//!
//! For Phase 5 today, `dequantize_w()` calls the Phase 3 decode-only CUDA
//! kernel + the encoder bake formula on CPU side to produce a bf16 weight
//! Tensor. This lets ISQ / LoRA-merge / debug paths work even without the
//! fused kernel.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    DistributedKind, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
    QuantizedSerdeType,
};

/// `.leech` LLVQ-quantized linear layer.
///
/// Field tensors are device-resident — the sidecar loader (Phase 5b in
/// mistralrs-core) is responsible for moving the `.leech` payload bytes into
/// CUDA memory before constructing this layer.
#[derive(Debug)]
pub struct LeechLayer {
    /// Packed LLVQ block bit-stream (u8 device buffer). Has 8 B tail-pad.
    packed_stream: Tensor,
    /// Per-row β codebook (fp16, `[rows, K_beta]`).
    beta_codebook: Tensor,
    /// Per-row offset codebook (fp16, `[rows, K_offset]`). Absent if `!has_offset`.
    offset_codebook: Option<Tensor>,
    /// Row count of the decoded weight matrix.
    rows: u32,
    /// LLVQ blocks per row (= n_done / 24 = (in_features - leftover_cols) / 24).
    blocks_per_row: u32,
    /// Per-block i_global bit width. 54 at ms=18, 48 at ms=13.
    idx_bits: u8,
    /// Whether each block carries a 3-bit offset codebook index.
    has_offset: bool,
    /// Encoder ms_used — selects the kernel template specialization.
    ms_used: u8,
    /// Optional bf16 leftover slab for in_features % 24 != 0 tensors.
    leftover_bf16: Option<Tensor>,
    /// Optional bias (bf16 / fp16).
    bias: Option<Tensor>,
    /// Cached weight dtype/device for the QuantMethod trait.
    dtype: DType,
    device: Device,
}

impl QuantizedSerde for LeechLayer {
    fn name(&self) -> &'static str {
        "leech"
    }

    fn isq_serde_supported(&self) -> bool {
        false
    }
}

impl QuantMethod for LeechLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Leech {
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
            } => {
                if has_offset != offset_codebook.is_some() {
                    candle_core::bail!(
                        "leech: has_offset ({has_offset}) inconsistent with offset_codebook presence ({})",
                        offset_codebook.is_some()
                    );
                }
                if idx_bits != 48 && idx_bits != 54 {
                    candle_core::bail!(
                        "leech: unsupported idx_bits {idx_bits} — kernel templates are {{48, 54}}"
                    );
                }
                if ms_used != 13 && ms_used != 18 {
                    candle_core::bail!(
                        "leech: unsupported ms_used {ms_used} — kernel templates are {{13, 18}}"
                    );
                }
                let dtype = bias
                    .as_ref()
                    .map(|b| b.dtype())
                    .unwrap_or(DType::BF16);
                let device = packed_stream.device().clone();
                Ok(LeechLayer {
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
                    dtype,
                    device,
                })
            }
            _ => candle_core::bail!("LeechLayer::new called with wrong QuantMethodConfig variant"),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        // Phase 5 prep: this is the slow but correct path used by ISQ /
        // LoRA-merge / debug. It mirrors `payload.py:reconstruct_bf16_from_streams`:
        //   1. Run the decode-only CUDA kernel → int8[R, B, 24] device tensor.
        //   2. Apply β·v + offset (fp32 fma) → bf16 cast.
        //   3. Concatenate the optional leftover slab onto the right edge.
        //
        // Phase 4's `forward_raw` fuses steps 1+2 plus the activation GEMM —
        // this path stays around for ISQ even after that lands.
        candle_core::bail!(
            "leech: dequantize_w not yet wired — Phase 4 will compose decode-only kernel \
             with bf16 epilogue. Phase 5 prep stub."
        )
    }

    fn forward_raw(&self, _a: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "leech: forward_raw (fused decode+GEMM) lands in Phase 4 — not yet implemented"
        )
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::BF16)
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.device.clone())
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("LeechLayer does not support LoRA merge (use overlay block instead).")
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!(
            "LeechLayer does not support ISQ. .leech files are pre-quantized; re-encode offline if needed."
        )
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        None
    }
}

// Convenience constants for accessing the serde tag from the outside.
pub const SERDE_TAG: QuantizedSerdeType = QuantizedSerdeType::Leech;

// ─── Read-only accessors used by Phase 4 fused-kernel wiring ──────────────

impl LeechLayer {
    pub fn packed_stream(&self) -> &Tensor {
        &self.packed_stream
    }
    pub fn beta_codebook(&self) -> &Tensor {
        &self.beta_codebook
    }
    pub fn offset_codebook(&self) -> Option<&Tensor> {
        self.offset_codebook.as_ref()
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn blocks_per_row(&self) -> u32 {
        self.blocks_per_row
    }
    pub fn idx_bits(&self) -> u8 {
        self.idx_bits
    }
    pub fn has_offset(&self) -> bool {
        self.has_offset
    }
    pub fn ms_used(&self) -> u8 {
        self.ms_used
    }
    pub fn leftover_bf16(&self) -> Option<&Tensor> {
        self.leftover_bf16.as_ref()
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    /// Total in-features represented by this weight = blocks_per_row * 24 + leftover_cols.
    pub fn in_features(&self) -> usize {
        let n_done = (self.blocks_per_row as usize) * 24;
        let leftover_cols = self
            .leftover_bf16
            .as_ref()
            .map(|t| t.dims().last().copied().unwrap_or(0))
            .unwrap_or(0);
        n_done + leftover_cols
    }
    /// Output features = rows.
    pub fn out_features(&self) -> usize {
        self.rows as usize
    }
}
