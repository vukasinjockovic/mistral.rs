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
    /// Parity-sort permutation: `u16[rows * blocks_per_row]` device buffer.
    /// For each row r, parity_perm[r * b_blocks .. (r+1) * b_blocks] lists
    /// the k_block indices in parity-sorted order (all even-parity blocks
    /// first, then odd). Eliminates warp divergence in the GEMV decode branch.
    /// Lazily computed on first forward_raw call. Stored as a raw CudaSlice
    /// because candle doesn't expose u16 as a tensor dtype.
    #[cfg(feature = "cuda")]
    parity_perm: std::sync::OnceLock<candle_core::cuda::cudarc::driver::CudaSlice<u16>>,
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
                    #[cfg(feature = "cuda")]
                    parity_perm: std::sync::OnceLock::new(),
                    dtype,
                    device,
                })
            }
            _ => candle_core::bail!("LeechLayer::new called with wrong QuantMethodConfig variant"),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            self.dequantize_w_cuda()
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::bail!(
                "leech: dequantize_w requires the cuda feature — CPU path not yet implemented"
            )
        }
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            self.forward_raw_cuda(a)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = a;
            candle_core::bail!(
                "leech: forward_raw requires the cuda feature — CPU path not yet implemented"
            )
        }
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

// ─── CUDA forward / dequantize implementation (Phase 4.0a) ────────────────

#[cfg(feature = "cuda")]
impl LeechLayer {
    /// Lazily build the parity-sort permutation. Runs the compute_block_parity
    /// kernel on the packed_stream to tag each block's parity, then on host
    /// sorts each row's block indices so all even-parity blocks come first.
    ///
    /// Returns a device pointer to the u16 permutation tensor, or null if
    /// construction failed (caller falls back to non-parity-sorted iteration).
    fn parity_perm_device_ptr(&self) -> *const std::ffi::c_void {
        use candle_core::cuda::cudarc::driver::DevicePtr;
        use candle_core::Storage;
        use std::ffi::c_void;

        use crate::leech::leech_compute_block_parity;
        use crate::utils::slice_ptr;

        let cuda = match &self.device {
            Device::Cuda(d) => d,
            _ => return std::ptr::null(),
        };

        let perm_slice = self.parity_perm.get_or_init(|| {
            let n_rows = self.rows as usize;
            let b_blocks = self.blocks_per_row as usize;
            let n_blocks = n_rows * b_blocks;

            // 1. Compute per-block parity via CUDA kernel.
            let parity_buf = cuda.alloc_zeros::<u8>(n_blocks).expect("alloc parity");
            let stream = cuda.cuda_stream();
            let stream_raw = stream.cu_stream() as *mut c_void;

            let (ps_s, ps_l) = self.packed_stream.storage_and_layout();
            let Storage::Cuda(ps_s) = &*ps_s else { panic!("packed_stream not CUDA") };
            let ps_slice = ps_s.as_cuda_slice::<u8>().expect("u8 slice");
            let (ps_ptr, _ps_guard) = slice_ptr(ps_slice, ps_l.start_offset());
            let (par_ptr_u64, _par_guard) = parity_buf.device_ptr(&stream);

            unsafe {
                leech_compute_block_parity(
                    ps_ptr as *const u8,
                    par_ptr_u64 as *mut u8,
                    n_blocks as u32,
                    self.idx_bits as u32,
                    self.has_offset,
                    stream_raw,
                )
                .expect("compute_block_parity");
            }
            drop(_par_guard);
            drop(_ps_guard);

            // 2. Pull parity to host.
            let mut parity_host: Vec<u8> = vec![0u8; n_blocks];
            cuda.memcpy_dtoh(&parity_buf, &mut parity_host).expect("dtoh parity");

            // 3. Build per-row permutation: evens first, then odds.
            let mut perm_host: Vec<u16> = Vec::with_capacity(n_blocks);
            for r in 0..n_rows {
                let row_par = &parity_host[r * b_blocks..(r + 1) * b_blocks];
                for (k, &p) in row_par.iter().enumerate() {
                    if p == 0 {
                        perm_host.push(k as u16);
                    }
                }
                for (k, &p) in row_par.iter().enumerate() {
                    if p == 1 {
                        perm_host.push(k as u16);
                    }
                }
            }

            // 4. Upload to device.
            let mut perm_dev = unsafe { cuda.alloc::<u16>(n_blocks).expect("alloc perm") };
            cuda.memcpy_htod(&perm_host, &mut perm_dev).expect("htod perm");
            perm_dev
        });

        let stream = cuda.cuda_stream();
        let (ptr, _g) = perm_slice.device_ptr(&stream);
        ptr as *const c_void
    }

    /// Phase 4.0a dequantize: one kernel that decodes + applies the LOCKED
    /// epilogue `bf16 = RNE(fp32(β · v_int + offset))` into a contiguous
    /// `[rows, blocks_per_row * 24]` bf16 weight tensor. If `leftover_bf16` is
    /// present it gets concatenated on the right.
    fn dequantize_w_cuda(&self) -> Result<Tensor> {
        use candle_core::cuda::cudarc::driver::DevicePtr;
        use candle_core::{CudaStorage, Shape, Storage};
        use half::{bf16, f16};
        use std::ffi::c_void;

        use crate::leech::leech_decode_bf16;
        use crate::utils::slice_ptr;

        let cuda = match &self.device {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("LeechLayer::dequantize_w_cuda: device is not CUDA"),
        };

        // packed_stream is a Tensor of u8 (contiguous device buffer).
        let (ps_s, ps_l) = self.packed_stream.storage_and_layout();
        let Storage::Cuda(ps_s) = &*ps_s else {
            candle_core::bail!("packed_stream not on CUDA");
        };
        let ps_slice = ps_s.as_cuda_slice::<u8>()?;
        let (ps_ptr, _ps_guard) = slice_ptr(ps_slice, ps_l.start_offset());

        // beta_codebook: f16, shape [rows, k_beta].
        let (bc_s, bc_l) = self.beta_codebook.storage_and_layout();
        let Storage::Cuda(bc_s) = &*bc_s else {
            candle_core::bail!("beta_codebook not on CUDA");
        };
        let bc_slice = bc_s.as_cuda_slice::<f16>()?;
        let (bc_ptr, _bc_guard) = slice_ptr(bc_slice, bc_l.start_offset());

        // Derive K_beta and K_offset from codebook shapes.
        let k_beta = self.beta_codebook.dim(1)? as u32;
        let k_offset = self
            .offset_codebook
            .as_ref()
            .map(|t| t.dim(1).unwrap() as u32)
            .unwrap_or(0);

        // Allocate the dequantized weight buffer: [rows, blocks_per_row * 24] bf16.
        let n_done = self.blocks_per_row as usize * 24;
        let out_elems = self.rows as usize * n_done;
        let out_buf = cuda.alloc_zeros::<bf16>(out_elems)?;
        let stream = cuda.cuda_stream();
        let (out_ptr_u64, _out_guard) = out_buf.device_ptr(&stream);
        let stream_raw = stream.cu_stream() as *mut c_void;

        // Kernel call needs all guards live. Doing it inside the optional-offset
        // match keeps the offset_codebook guard alive across the launch without
        // a self-referential struct.
        match self.offset_codebook.as_ref() {
            Some(t) => {
                let (oc_s, oc_l) = t.storage_and_layout();
                let Storage::Cuda(oc_s) = &*oc_s else {
                    candle_core::bail!("offset_codebook not on CUDA");
                };
                let oc_slice = oc_s.as_cuda_slice::<f16>()?;
                let (offset_ptr_u64, _oc_guard) = slice_ptr(oc_slice, oc_l.start_offset());
                unsafe {
                    leech_decode_bf16(
                        ps_ptr as *const u8,
                        bc_ptr as *const c_void,
                        offset_ptr_u64 as *const c_void,
                        out_ptr_u64 as *mut c_void,
                        self.rows,
                        self.blocks_per_row,
                        k_beta,
                        k_offset,
                        self.idx_bits as u32,
                        self.has_offset,
                        stream_raw,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("leech_decode_bf16: {e}")))?;
                }
                drop(_oc_guard);
            }
            None => {
                unsafe {
                    leech_decode_bf16(
                        ps_ptr as *const u8,
                        bc_ptr as *const c_void,
                        std::ptr::null::<c_void>(),
                        out_ptr_u64 as *mut c_void,
                        self.rows,
                        self.blocks_per_row,
                        k_beta,
                        k_offset,
                        self.idx_bits as u32,
                        self.has_offset,
                        stream_raw,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("leech_decode_bf16: {e}")))?;
                }
            }
        }

        drop(_out_guard);
        drop(_bc_guard);
        drop(_ps_guard);

        let out_storage = CudaStorage::wrap_cuda_slice(out_buf, cuda.clone());
        let w_main = Tensor::from((
            Storage::Cuda(out_storage),
            Shape::from((self.rows as usize, n_done)),
        ));

        // If there's a leftover slab, concat on the right edge (dim=1).
        match self.leftover_bf16.as_ref() {
            Some(left) => {
                let left_cast = if left.dtype() != DType::BF16 {
                    left.to_dtype(DType::BF16)?
                } else {
                    left.clone()
                };
                Tensor::cat(&[w_main, left_cast], 1)
            }
            None => Ok(w_main),
        }
    }

    /// Phase 4.0 forward: dispatch on M.
    /// - M ≤ 8 (typical generation): use the fused GEMV kernel — no decoded
    ///   weight HBM roundtrip, ~6.4 ms per large matmul on RTX 5090.
    /// - M > 8 (prefill / batched): dequantize once, then standard bf16 matmul
    ///   — amortizes decode across many output rows.
    fn forward_raw_cuda(&self, a: &Tensor) -> Result<Tensor> {
        let a_bf16 = if a.dtype() != DType::BF16 {
            a.to_dtype(DType::BF16)?
        } else {
            a.clone()
        };
        let a_dims = a_bf16.dims().to_vec();
        let m: usize = a_dims[..a_dims.len().saturating_sub(1)]
            .iter()
            .product::<usize>()
            .max(1);

        // Currently we only support in_features = blocks_per_row * 24 in the
        // GEMV kernel (leftover columns not yet handled in the fused path).
        let supports_gemv = self.leftover_bf16.is_none() && m <= 8;

        let y = if supports_gemv {
            self.forward_raw_gemv(&a_bf16, m, &a_dims)?
        } else {
            let w = self.dequantize_w_cuda()?; // [out_features, in_features]
            a_bf16.broadcast_matmul(&w.t()?)?
        };
        match self.bias.as_ref() {
            Some(b) => {
                let b_cast = if b.dtype() != DType::BF16 {
                    b.to_dtype(DType::BF16)?
                } else {
                    b.clone()
                };
                y.broadcast_add(&b_cast)
            }
            None => Ok(y),
        }
    }

    /// Phase 4.0b fused GEMV path.
    fn forward_raw_gemv(&self, a_bf16: &Tensor, m: usize, a_dims: &[usize]) -> Result<Tensor> {
        use candle_core::cuda::cudarc::driver::DevicePtr;
        use candle_core::{CudaStorage, Shape, Storage};
        use half::bf16;
        use std::ffi::c_void;

        use crate::leech::leech_gemv_bf16;
        use crate::utils::slice_ptr;

        let cuda = match &self.device {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("LeechLayer::forward_raw_gemv: not CUDA"),
        };

        // Ensure activation is contiguous and shaped [M, K_total].
        let k_total = self.blocks_per_row as usize * 24;
        let a_flat = a_bf16.reshape((m, k_total))?.contiguous()?;
        let (a_s, a_l) = a_flat.storage_and_layout();
        let Storage::Cuda(a_s) = &*a_s else {
            candle_core::bail!("activation not on CUDA");
        };
        let a_slice = a_s.as_cuda_slice::<bf16>()?;
        let (a_ptr, _a_guard) = slice_ptr(a_slice, a_l.start_offset());

        let (ps_s, ps_l) = self.packed_stream.storage_and_layout();
        let Storage::Cuda(ps_s) = &*ps_s else {
            candle_core::bail!("packed_stream not on CUDA");
        };
        let ps_slice = ps_s.as_cuda_slice::<u8>()?;
        let (ps_ptr, _ps_guard) = slice_ptr(ps_slice, ps_l.start_offset());

        let (bc_s, bc_l) = self.beta_codebook.storage_and_layout();
        let Storage::Cuda(bc_s) = &*bc_s else {
            candle_core::bail!("beta_codebook not on CUDA");
        };
        let bc_slice = bc_s.as_cuda_slice::<half::f16>()?;
        let (bc_ptr, _bc_guard) = slice_ptr(bc_slice, bc_l.start_offset());

        let k_beta = self.beta_codebook.dim(1)? as u32;
        let k_offset = self
            .offset_codebook
            .as_ref()
            .map(|t| t.dim(1).unwrap() as u32)
            .unwrap_or(0);

        let n_rows = self.rows;
        let out_elems = m * n_rows as usize;
        let out_buf = cuda.alloc_zeros::<bf16>(out_elems)?;
        let stream = cuda.cuda_stream();
        let (out_ptr_u64, _out_guard) = out_buf.device_ptr(&stream);
        let stream_raw = stream.cu_stream() as *mut c_void;

        match self.offset_codebook.as_ref() {
            Some(t) => {
                let (oc_s, oc_l) = t.storage_and_layout();
                let Storage::Cuda(oc_s) = &*oc_s else {
                    candle_core::bail!("offset_codebook not on CUDA");
                };
                let oc_slice = oc_s.as_cuda_slice::<half::f16>()?;
                let (offset_ptr_u64, _oc_guard) = slice_ptr(oc_slice, oc_l.start_offset());
                let pp_ptr = self.parity_perm_device_ptr();
                unsafe {
                    leech_gemv_bf16(
                        a_ptr as *const c_void,
                        ps_ptr as *const u8,
                        bc_ptr as *const c_void,
                        offset_ptr_u64 as *const c_void,
                        pp_ptr,
                        out_ptr_u64 as *mut c_void,
                        m as u32,
                        n_rows,
                        self.blocks_per_row,
                        k_beta,
                        k_offset,
                        self.idx_bits as u32,
                        self.has_offset,
                        stream_raw,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("leech_gemv_bf16: {e}")))?;
                }
                drop(_oc_guard);
            }
            None => {
                let pp_ptr = self.parity_perm_device_ptr();
                unsafe {
                    leech_gemv_bf16(
                        a_ptr as *const c_void,
                        ps_ptr as *const u8,
                        bc_ptr as *const c_void,
                        std::ptr::null::<c_void>(),
                        pp_ptr,
                        out_ptr_u64 as *mut c_void,
                        m as u32,
                        n_rows,
                        self.blocks_per_row,
                        k_beta,
                        k_offset,
                        self.idx_bits as u32,
                        self.has_offset,
                        stream_raw,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("leech_gemv_bf16: {e}")))?;
                }
            }
        }
        drop(_out_guard);
        drop(_bc_guard);
        drop(_ps_guard);
        drop(_a_guard);

        let out_storage = CudaStorage::wrap_cuda_slice(out_buf, cuda.clone());
        let mut out_shape: Vec<usize> = a_dims[..a_dims.len().saturating_sub(1)].to_vec();
        out_shape.push(n_rows as usize);
        Ok(Tensor::from((
            Storage::Cuda(out_storage),
            Shape::from(out_shape),
        )))
    }
}
