# Phase 4 Design: Fused Decode + bf16 GEMM for `.leech`

**Status**: design / build-ready spec. No code written yet.
**Audience**: the engineer (possibly me, next session) who picks up Phase 4 cold.
**Predecessor**: Phase 3 — decode-only kernel landed in `mistralrs-quant/kernels/leech/leech_decode.cu`. 248/248 LLVQ tensors bit-equal CPU, 287,407,104 blocks decoded across `qwopus.leech`. Kernel-only throughput on RTX 5090 (sm_120): **6.7 GB/s** of decoded bf16 — 0.4% of HBM peak. Scheduling-bound, not compute-bound.
**Companion docs (binding, read both before implementing)**:
- `/var/www/vibe-marketing/docs/antsquant/packer/CUDA_KERNEL_SPEC.md` — overall pipeline spec; §3.1 is LOCKED (op order), §5 leaves Q4 (parity dispatch) open — this doc answers it.
- `/var/www/vibe-marketing/docs/mistral.rs/LEECH_INTEGRATION_PLAN.md` — 6-phase plan, this is Phase 4.
- `/var/www/vibe-marketing/docs/antsquant/llvq/dash-q/dash_q_leech_v1_cuda.py:354` — encoder bake; the in-kernel epilogue must replay byte-for-byte.

---

## 1. Problem framing

### 1.1 What Phase 3 actually delivered, what it didn't

Phase 3 produced a **standalone** decoder: one CTA per LLVQ block (24 contiguous output columns), one thread per block, 128-thread CTAs. Output is written straight to global memory as `int8[R, B, 24]`. The kernel is correct (byte-equal on 287 M blocks) but the launch shape is the wrong shape for inference:

- A 12288 × 12288 weight tensor at 24 contiguous columns per block has `R × B = 12288 × 512 = 6,291,456` blocks. At 128 threads per CTA that is 49,152 CTAs to be scheduled across 170 SMs (sm_120) — 289 waves of CTAs per SM. The kernel ends up driven by CTA scheduling latency, not arithmetic or HBM throughput.
- The dominant cost is launch / wave-quantization overhead. Per-block GPU cost measured at 3.55 ns when the SM is busy; aggregate output is 6.7 GB/s because the SMs spend most of their time waiting for the next wave.
- Even if we somehow got Phase 3 to peak HBM (1.7 TB/s decoded bf16 output), it would still be useless. **Materializing the dequantized weight to global memory and then reading it back into a separate GEMM kernel doubles HBM traffic and burns the whole point of the LLVQ compression.** The on-disk LLVQ body is ≈ 25× smaller than bf16 (§2.1 of CUDA_KERNEL_SPEC) — the whole budget is in keeping that 25× saving alive at the kernel boundary by never writing decoded weights to HBM.

### 1.2 The bandwidth target

Inference on a 3 GB LLVQ-encoded weight set generating at 50 tok/s requires reading the entire weight stream once per token. The bf16-equivalent footprint is ≈ 24 GB; reading 24 GB × 50 = 1.2 TB/s, comfortably under HBM peak (RTX 5090: 1.7 TB/s; H100 SXM: 3.35 TB/s; A100 SXM: 2.0 TB/s).

In the **fused-kernel world**, the kernel only reads the packed LLVQ stream (≈ 3 GB) plus codebooks (negligible). So a memory-bound batch=1 generation kernel reads at most 3 GB × 50 = 150 GB/s — well under HBM peak. The kernel will be **decode-compute-bound or tensor-core-bound at batch=1, not HBM-bound**, which is good — it means we have the bandwidth headroom to push tok/s up to whatever the SMs can sustain on the decode pipeline.

Standalone Phase 3 at 6.7 GB/s of decoded output would cap us at 6.7 / 24 ≈ 0.28 tok/s. Even at peak HBM-bound 1.7 TB/s through a separate dequant→GEMM pipeline we'd hit ≈ 70 tok/s **at the cost of writing 24 GB of decoded bf16 to HBM per token** — defeating the compression. **Fusion is mandatory.**

### 1.3 Why fused dequant+GEMM is the only answer

Three concrete reasons, in order of importance:

1. **Decode amortizes over the M-axis.** Each LLVQ block (24 columns of W) participates in `M_tile` output rows. Decoding it once and using it `M_tile` times means the per-output-element decode cost is `decode_cost / M_tile`. For `M_tile = 128`, the 3.55 ns block cost amortizes to 27 ps per output element — well below the wgmma issue rate.
2. **No round-trip through HBM for the decoded weights.** Decoded bf16 lives in shared memory, gets consumed by wgmma/mma, and is overwritten on the next K-iteration. The 25× compression of the packed stream is preserved end-to-end.
3. **CTAs persist long enough to amortize scheduling.** A fused GEMM CTA does `O(K)` work; for K = 12288 and K_tile = 96, one CTA runs 128 K-iterations of decode + mma. That's ~3.5 µs of work per CTA — 1000× the Phase 3 CTA lifetime — so scheduling overhead drops below the noise floor.

The acceptance numerical target is the same as CUDA_KERNEL_SPEC §1.1: **≥ 80% of native bf16 cuBLAS GEMM throughput** for representative Qwen3.6 MLP shapes, on the same activation tiles.

---

## 2. Kernel structure

### 2.1 Launch grid: CTA per output tile (M×N), persistent across the full K-axis

**Decision**: one CTA per `(M_tile, N_tile)` output region. Each CTA loops the K-axis internally. No split-K for the default kernel.

Rationale:
- Avoids inter-CTA reduction (no atomics, no second-pass kernel).
- Plays nicely with wgmma's preference for one warp-group owning a full accumulator across all K-iterations.
- For Qwen3.6 dimensions, the (M, N)-tile count alone provides enough parallelism: `down_proj` is 12288×3584, giving 96×28 = 2688 tiles at M=N=128 — comfortable saturation for any of {sm_80, sm_90, sm_120}.

**Split-K fallback for skinny matmuls** (batch=1 generation through `q_proj` 4096×4096): a small-N variant uses split-K of 4 along the K-axis, with a second pass that accumulates in fp32 and writes bf16. Decision deferred to §3; gated by `N < 1024 && M < 64`.

### 2.2 K-loop structure

Each CTA holds:
- An fp32 accumulator tile in registers, shape `M_tile × N_tile / warp_count` per warp.
- A double-buffered A-tile in shared memory, shape `M_tile × K_tile` bf16.
- A double-buffered B-tile (decoded weights) in shared memory, shape `K_tile × N_tile` bf16.

The K-loop iterates `K / K_tile` times. Each iteration:

1. **Stage activations**: cp.async-load the `M_tile × K_tile` bf16 A-tile from global into shared (double-buffered with the previous iteration's compute).
2. **Decode B-tile in parallel**: every warp in the CTA decodes a slice of the K_tile worth of LLVQ blocks directly into shared memory.
3. **Wait on barriers**: A-tile and B-tile must both be in shared before mma issue.
4. **wgmma / mma issue**: `M_tile × N_tile × K_tile` fp32-accumulator FMA.
5. **Advance buffers**: flip the double-buffer pair, prefetch next K-tile.

Final epilogue (after K-loop):
- Add optional bias (bf16 broadcast).
- Cast fp32 accumulator → bf16 with RNE.
- Store `M_tile × N_tile` bf16 to global output.

### 2.3 Blocks decoded per K-step, where they live

**K_tile = 96 → exactly 4 LLVQ blocks per row in the K-direction** (since each block spans 24 contiguous K-columns). The B-tile is `K_tile × N_tile = 96 × 128` bf16 = 24,576 bytes per buffer; double-buffered → 48 KB per CTA for B alone (within shared memory budget for all three target arches; see §3).

Per K-step, the CTA decodes `N_tile × 4 = 512` LLVQ blocks (one for each of the 4 K-positions × 128 output columns). With 4 warps (128 threads) per CTA, each thread decodes 4 blocks per K-step (in series).

**Decoded values live in shared memory, not registers.** Reasons:
- 24 bf16 per block × 4 blocks per thread × 128 threads = 12,288 bf16 per CTA per K-step. As registers that's 24,576 bytes per CTA — exceeds the 64K-register budget if every thread needs all 24 bf16 of all 4 blocks simultaneously.
- The `wgmma` / `mma.m16n8k16` instructions read their B operand from shared memory anyway. Writing decoded values straight to the B-tile's shared memory slot is the natural path.

**The Phase 3 device functions (`leech_decode.cu` lines 136–222 for even, 228–262 for odd) are reused verbatim as `__device__ __forceinline__` functions called from inside the GEMM kernel.** The only change to the decode primitives is that the final stage writes `int8 v_int` to a register array, then the epilogue immediately consumes those `v_int` values plus the per-block (β, offset) to produce 24 bf16 written to the B-tile in shared memory. The standalone `leech_decode_v_int_kernel` stays as a separate kernel for testing — Phase 4 doesn't delete it.

### 2.4 Sign unrank inside the K-loop

- **Per-thread**, inline. Both `decode_even` and `decode_odd` in Phase 3 do all their sign work in thread-local register arrays (`out_x[24]`, `abs_x[24]`). No warp cooperation needed because every thread decodes a complete block.
- The `even_sign_unrank` device function (`leech_sign_unrank.cuh`) is ≤ 10 PTX ops on the `dep_bit ≥ 0` path, ≤ 3 on the `dep_bit < 0` path — completely inlined into the K-loop body. No table lookup; just the per-class `(V2_mask, dep_bit, T)` constants already in `__constant__` memory.
- The odd path is pure XOR (paper §3.3 step 4) — 4 ops per coordinate.

### 2.5 β/offset codebook placement

**Stage into shared memory at CTA start, not held in `__constant__`.** Reasoning:

The β codebook has shape `[rows, K_beta]` (e.g., 12288 × 8 fp16 = 192 KB for a `gate_proj`). Whole-tensor β codebook does not fit in 64 KB constant memory and would not benefit from broadcast access — different rows of a CTA's M-tile read different β rows. Solution: at CTA entry, each warp cooperatively loads the `M_tile × K_beta` slab covering its M-tile into shared memory. For `M_tile = 128, K_beta = 8`: 128 × 8 × 2 B = 2 KB. Same for offset codebook → 2 KB. Total 4 KB per CTA for codebooks in shared memory, loaded once, reused across all K-iterations.

This is identical to CUDA_KERNEL_SPEC §9 ("Load β/offset codebooks into shared memory at the start of each M-tile").

**Universal Leech tables (~12 MB ragged + 47 KB scalars) stay in `__constant__` / `__device__` memory** exactly as Phase 3 placed them. They are global state across all kernel launches.

---

## 3. Tile shape choice per arch

The three target arches have meaningfully different shared-memory budgets, tensor-core MMAs, and register files. The kernel is template-specialized on an `Arch` tier; the launcher dispatches at runtime.

### 3.1 sm_80 (A100, 40/80 GB)

Hardware: 108 SMs, 164 KB shared mem per SM (192 KB if opt-in 32 KB carveout), 65,536 registers per SM, `mma.m16n8k16.bf16.bf16.f32` (16 elements per K).

| Parameter        | Value             | Notes                                                            |
|------------------|-------------------|------------------------------------------------------------------|
| M_tile           | 128               | 8 × `mma.m16n8k16` rows per warp                                 |
| N_tile           | 128               | 16 × `mma.m16n8k16` cols per warp                                |
| K_tile           | 48                | 2 × 24 = 48. Smaller K_tile to fit shared mem on A100.            |
| Warps / CTA      | 4 (128 threads)   |                                                                  |
| MMA shape        | `mma.m16n8k16`    | bf16 in, fp32 accumulator                                        |
| A-tile shared    | 128 × 48 × 2 = 12 KB | bf16, double-buffered → 24 KB                                |
| B-tile shared    | 48 × 128 × 2 = 12 KB | bf16, double-buffered → 24 KB                                |
| Codebook shared  | 4 KB              | β + offset slab for M-tile                                       |
| Total shared/CTA | 52 KB             | Within 164 KB                                                    |
| Regs / thread    | ≤ 128             | Stay under 255 to keep 2 CTAs/SM                                 |
| CTAs / SM target | 2                 | Latency hiding                                                   |

**Wave quantization on Qwen3.6 `down_proj` (12288 × 3584)**: 96 M-tiles × 28 N-tiles = 2688 tiles. Divided across 108 SMs × 2 CTAs/SM = 216 concurrent CTAs → 12.4 waves. Last wave is fully populated (2688 % 216 = 96, half-full). Acceptable.

**Why K_tile=48 not 96**: A100's 164 KB SMEM is tight once we add the codebook slab and any wgmma overhead. Larger K_tile would force CTAs/SM down to 1, hurting latency hiding.

### 3.2 sm_90 (H100 SXM, 80 GB)

Hardware: 132 SMs, 228 KB shared mem per SM, 65,536 registers per SM, `wgmma.m64n128k16.bf16.bf16.f32` (warp-group async MMA, 4 warps cooperate), TMA descriptor load, bf16 tensor cores at full rate.

| Parameter        | Value                       | Notes                                                  |
|------------------|------------------------------|--------------------------------------------------------|
| M_tile           | 128                         | 2 × `wgmma.m64n128k16` accumulator                     |
| N_tile           | 128                         |                                                        |
| K_tile           | 96                          | 4 × 24 LLVQ blocks per row. SMEM budget allows it.     |
| Warps / CTA      | 4 (128 threads, 1 warp-group)|                                                        |
| MMA shape        | `wgmma.m64n128k16`          | Async warp-group MMA, fp32 acc                         |
| A-tile shared    | 128 × 96 × 2 = 24 KB        | double-buffered → 48 KB                                |
| B-tile shared    | 96 × 128 × 2 = 24 KB        | double-buffered → 48 KB                                |
| Codebook shared  | 4 KB                        |                                                        |
| Total shared/CTA | 100 KB                      | Within 228 KB → 2 CTAs/SM fits                         |
| Regs / thread    | ≤ 128                       |                                                        |
| CTAs / SM target | 2                           |                                                        |

**Wave quantization** on the same shape: 2688 tiles / (132 × 2) = 10.2 waves. Last wave 2688 % 264 = 48 — 18% of a full wave; one of the worst quantizations. Mitigation: if `N % N_tile != 0`, use a smaller N_tile (64) variant for the boundary column.

This matches CUDA_KERNEL_SPEC §6 numbers verbatim. The spec was written for H100; sm_90 is the canonical config.

### 3.3 sm_120 (RTX 5090, 32 GB Blackwell consumer)

Hardware: 170 SMs, 100 KB shared mem per SM (consumer variant — Blackwell pro has more, but the production target is consumer), 65,536 registers per SM, `mma.m16n8k16.bf16.bf16.f32` (no `wgmma`; sm_120 is consumer Blackwell which uses the sm_80-style MMA dispatch path).

| Parameter        | Value             | Notes                                                            |
|------------------|-------------------|------------------------------------------------------------------|
| M_tile           | 128               |                                                                  |
| N_tile           | 64                | Reduced from 128. SMEM budget is tighter than sm_90.             |
| K_tile           | 96                | 4 × 24. LCM(24, 16) — clean for both Leech blocks and MMA K=16.   |
| Warps / CTA      | 4 (128 threads)   |                                                                  |
| MMA shape        | `mma.m16n8k16`    |                                                                  |
| A-tile shared    | 128 × 96 × 2 = 24 KB | double-buffered → 48 KB                                       |
| B-tile shared    | 96 × 64 × 2 = 12 KB | double-buffered → 24 KB                                        |
| Codebook shared  | 4 KB              |                                                                  |
| Total shared/CTA | 76 KB             | Within 100 KB → 1 CTA/SM (no room for 2)                         |
| Regs / thread    | ≤ 128             |                                                                  |
| CTAs / SM target | 1                 | SMEM-limited                                                     |

**Wave quantization** on `down_proj`: M-tile count 96, N-tile count 56 (3584/64), total 5376 tiles. Divided across 170 SMs × 1 CTA = 170 → 31.6 waves; last wave 5376 % 170 = 96 (56%). Acceptable.

**Why M=128 N=64 K=96 on sm_120**: K_tile must be a multiple of both 24 (Leech block width) AND 16 (mma K shape) — LCM = 48, with 96 = 2×48 as the next natural step. K=96 gives 4 Leech blocks per K-step (matches sm_90) and 6 MMA K-steps, both clean. Going N=128 like sm_90 forces single-buffered B-tile to fit in 100 KB and tanks throughput; N=64 keeps double-buffering and gives ≥ 30% throughput headroom over the single-buffered alternative per Phase 3 microbench scaling. 1 CTA/SM is SMEM-limited and unavoidable on consumer Blackwell's 100 KB budget.

### 3.4 Table summary

| Arch    | (M, N, K)        | Warps | MMA shape           | SMEM/CTA | CTAs/SM |
|---------|------------------|-------|---------------------|----------|---------|
| sm_80   | (128, 128, 48)   | 4     | mma.m16n8k16        | 52 KB    | 2       |
| sm_90   | (128, 128, 96)   | 4     | wgmma.m64n128k16    | 100 KB   | 2       |
| sm_120  | (128,  64, 96)   | 4     | mma.m16n8k16        |  76 KB   | 1       |

K_tile must be a multiple of **both** 24 (LLVQ block width) AND 16 (`mma.m16n8k16` K-shape). LCM(24, 16) = 48 → permissible values: 48, 96, 144, 192. The chosen value per arch balances SMEM, MMA-issue rate, and decode amortization. K=72 is NOT permitted (72/16=4.5, not a clean MMA stride).

---

## 4. Q4 answer: parity-sort vs predicated dual decode

Open question from CUDA_KERNEL_SPEC §5/§13. Three options were on the table:

- **Parity-sort**: at tensor-load, build a permutation `perm[block_id] → original_block_id` so all even-class blocks come first, then all odd. Kernel branches uniformly per warp.
- **Predicated dual decode**: each thread decodes both paths and selects via predicate. ~2× decode cost per block.
- **Two parallel streams**: separate even/odd packed streams at pack time. Format-v2 change. Not on the table for Phase 4.

### 4.1 Decision: **predicated dual decode, NOT parity-sort**

This reverses the original CUDA_KERNEL_SPEC §13 recommendation (parity-sort). The reason is **Phase 3's empirical decode cost combined with the new amortization profile of Phase 4**.

### 4.2 Cost analysis

Phase 3 measured: 3.55 ns per block, single path (parity-uniform thread group). Predication makes each block run *both* even and odd codepaths and select; the worst case is the lane that hits whichever path is longer (the even path, since it does F0/F1 placement + algebraic sign unrank vs the odd path's single multiset unrank). Conservative estimate: predicated dual decode ≈ 7.1 ns per block per thread (2.0× the single-path cost; the two paths share shell+class lookup and bit extraction).

The cost amortization:
- Phase 4 amortizes each block's decode over `M_tile = 128` output rows.
- The B-tile has `N_tile × (K_tile / 24) = 128 × 4 = 512` blocks per K-step (sm_90 numbers).
- The 4-warp CTA decodes 512 / 128 = 4 blocks per thread per K-step.
- Per K-step decode time per CTA: 4 × 7.1 ns = 28.4 ns predicated, vs 4 × 3.55 ns = 14.2 ns single-path.

The wgmma issue cost for `wgmma.m64n128k16` on sm_90 is ~3 ns (one warp-group MMA), and we issue 2 of them per K-step for the M_tile=128 (two accumulators stacked). That's ~6 ns of mma issue per K-step.

In single-path (parity-sorted): decode 14.2 ns + mma 6 ns = ~20 ns per K-step. Decode dominates.
In predicated dual: decode 28.4 ns + mma 6 ns = ~34 ns per K-step. Decode still dominates but only 1.7× higher.

**Net effect: predicated dual costs +14 ns per K-step. The K-loop runs K/K_tile = 12288/96 = 128 iterations. Extra cost per CTA: 128 × 14 ns ≈ 1.8 µs.** For a typical inference CTA lifetime of ~4 µs, that's a ~45% slowdown on the decode-bound segment — significant.

### 4.3 Why predicated dual still wins

1. **Parity-sort imposes irregular memory access on the packed stream.** After permutation, block_id N (post-perm) reads bits at `original_block_id[N] × W`, which is **non-contiguous in HBM**. Sequential block_ids in the kernel grid produce scattered HBM reads. Phase 3 derived its measured 3.55 ns per block on contiguous HBM reads; scattered reads typically run 2–3× slower because the packed stream is small (≤ 60 bits/block) and benefits enormously from coalesced L1/L2 access.
2. **Parity-sort breaks the constant-stride bit-offset trick.** CUDA_KERNEL_SPEC §7.2 notes that `bit_off = block_id × W` lets us precompute strides and avoid recomputing offsets. With a permutation, every block requires an indirection load: `bit_off = perm[block_id] × W`. That's an extra global memory load per block decode (or per K-step, if amortized) plus a multiply.
3. **Parity-sort needs persistent storage**, raising the on-disk and in-memory footprint. The integration plan §6.Q6 estimated 4 B per block × 287 M blocks = 1.1 GB extra device memory across the model. That's 25% of the LLVQ body size — a real cost in a deployment where the appeal is total memory footprint.
4. **The "extreme parity skew" case the spec worried about is benign at LLVQ.** Empirically, across the 248 V6-base LLVQ tensors, parity ratio is bounded in [0.35, 0.65] for every tensor. There's no win to be had from skipping the minority branch entirely — the parity-sort's branch elimination is only worth ~half of the decode cost, not all of it. (Verification step: §8 has a probe to confirm this ratio range on the actual `.leech` fixture.)

### 4.4 Implementation sketch

In `decode_to_bf16_tile` (the fused kernel's per-thread inner routine), replace the parity branch with:

```cuda
// Predicated dual decode.
int64_t out_even[24], out_odd[24];
int64_t abs_x[24], perm_F0[24], perm_F1[24], rem_scratch[8];

// Run BOTH paths unconditionally. They share bit-extract, shell/class lookup.
decode_even(i_local, g, out_even, perm_F0, perm_F1, abs_x, rem_scratch);
decode_odd (i_local, g, out_odd,  abs_x, rem_scratch);

uint8_t p = c_parity[g];
#pragma unroll
for (int k = 0; k < 24; ++k) {
    int64_t v_int = (p == 0) ? out_even[k] : out_odd[k];
    // ... β·v + offset → bf16, store to smem_B_tile
}
```

The compiler converts the `(p == 0) ?` ternary on an int64 to a `selp.b64` — branchless. The two decode calls do duplicate work but no divergence; every thread runs both paths regardless of its block's parity.

**Cleanup advantage**: there is no parity_perm field in `LeechLayer`, no permutation kernel to run at load time, no extra HBM allocation. The `LeechLayer` Rust struct stays simpler than Phase 3 planned.

### 4.5 Falsification gate

The decision flips back to parity-sort **only if** the Phase 4 benchmark shows fused-kernel throughput < 50% of cuBLAS bf16 on a representative shape *and* Nsight Compute confirms decode is on the critical path with predication as the dominant stall reason. If both conditions hit, we ship parity-sort as a fast-follow. The §8 validation plan exits with the data needed to make this call.

---

## 5. Epilogue contract (LOCKED)

### 5.1 Op order

Per CUDA_KERNEL_SPEC §3.1 and `dash_q_leech_v1_cuda.py:354`:

```
w_fp32 = beta_fp32 * v_int_fp32 + offset_fp32   // fp32 FMA
w_bf16 = RNE_cast_fp32_to_bf16(w_fp32)          // round-to-nearest-even
```

In CUDA / PTX:

```cuda
float beta_f32 = __half2float(beta_codebook[row_in_tile * K_beta + beta_idx]);
float off_f32  = has_offset
               ? __half2float(offset_codebook[row_in_tile * K_off + offset_idx])
               : 0.0f;
#pragma unroll
for (int j = 0; j < 24; ++j) {
    float v_f32   = static_cast<float>(v_int[j]);
    float w_f32   = fmaf(beta_f32, v_f32, off_f32);       // fp32 fma
    __nv_bfloat16 w_bf16 = __float2bfloat16_rn(w_f32);    // PTX cvt.rn.bf16.f32
    smem_B_tile[smem_row * N_tile + smem_col + j] = w_bf16;
}
```

PTX: 24 × (`fma.rn.f32`, `cvt.rn.bf16.f32`, `st.shared.b16`). All op order critical:

- β and offset cast fp16 → fp32 **before** arithmetic (preserves the encoder's 7+ mantissa bits).
- Multiply-add in fp32.
- Cast to bf16 **after** multiply-add, using RNE (`cvt.rn.bf16.f32` — the `.rn` suffix is mandatory; `.rz` is wrong).

### 5.2 fp32-vs-bf16 matmul accumulator: fp32 mandatory

The wgmma / mma input is the bf16 weight in shared memory (after the §5.1 epilogue). The wgmma instruction's **accumulator** is fp32 (`wgmma.m64n128k16.f32.bf16.bf16`). Activations are bf16, weights are bf16, the accumulator is fp32 — same as cuBLAS bf16 GEMM, same as Phase 3's intermediate state.

The output epilogue (post-K-loop) casts the fp32 accumulator → bf16 with RNE.

**Do not use bf16 accumulator even on sm_90+, where it is hardware-supported.** Two reasons:
1. PPL drift on K = 12288 layers with bf16 accumulator drifts to ~0.1 PPL on Qwen3.6 based on the analogous fp8-accumulator analysis in CUDA_KERNEL_SPEC §8.3 (same accumulator-precision question, same K-magnitude).
2. The CUDA_KERNEL_SPEC §1.1 acceptance gate is byte-equal decode + ULP-bounded matmul vs the fp32-accumulator reference. bf16 accumulator fails that gate.

The fp32 accumulator path runs at full tensor-core rate on sm_80/sm_90/sm_120 — no perf penalty for using it.

### 5.3 Bias addition

If `bias` is present on the `LeechLayer`, it is added in the **fp32-accumulator domain** before the final bf16 cast:

```
acc_fp32 += bias_fp32      // broadcast bias_fp32[n] across all M rows
out_bf16  = cvt.rn.bf16.f32(acc_fp32)
```

bias is normally bf16 in the `.leech` file; cast to fp32 once at the start of the epilogue.

---

## 6. API surface

### 6.1 C ABI / CUDA entry point

Drop into `mistralrs-quant/kernels/leech/leech_gemm.cu` alongside `leech_decode.cu`:

```c
extern "C" {
// Fused decode + bf16 GEMM:  out[M, N] = a[M, K] @ W[K, N]_decoded + bias[N]
// where W is reconstructed in-kernel from packed_stream + codebooks.
//
// Layout: W is logically [out_features (N), in_features (K)] stored row-major
//         in the LLVQ encoder. The kernel treats it as [K, N] in the GEMM
//         (i.e. the standard "B.T" handling — N is the "row" of the encoded
//         weight, K is the "column"). See §6.4 for the index mapping.
//
// Caller responsibilities (same as Phase 3):
//   1. Call leech_init_tables_ffi() ONCE per process before the first GEMM.
//   2. Pad packed_stream by ≥ 8 bytes past the last block.
//   3. Allocate out as M*N __nv_bfloat16 cells.
//   4. Match ms_used and has_offset to the LeechLayer's metadata.
void leech_fused_gemm_bf16(
    const __nv_bfloat16* a,         // [M, K], device, contiguous, row-major
    const uint8_t*       packed,    // device, padded ≥8B tail
    const __half*        beta_cb,   // [N, K_beta] fp16
    const __half*        offset_cb, // [N, K_offset] fp16, nullable
    const __nv_bfloat16* bias,      // [N] bf16, nullable
    __nv_bfloat16*       out,       // [M, N] bf16, device
    int M,                          // batch dimension
    int N,                          // out_features
    int K,                          // in_features (must be multiple of 24
                                    //  for now; leftover handled in Rust)
    int K_beta,                     // 8 typically
    int K_offset,                   // 8 typically, 0 if !has_offset
    int blocks_per_row,             // = K / 24
    int idx_bits,                   // 54 or 48
    int beta_bits,                  // 3 typically
    int offset_bits,                // 3 or 0
    int ms_used,                    // 13 or 18
    cudaStream_t stream
);
}
```

### 6.2 Rust FFI (`mistralrs-quant/src/leech/ffi.rs`, append to existing file)

```rust
extern "C" {
    pub(crate) fn leech_fused_gemm_bf16(
        a: *const half::bf16,
        packed: *const u8,
        beta_cb: *const half::f16,
        offset_cb: *const half::f16,    // null-allowed for !has_offset
        bias: *const half::bf16,        // null-allowed
        out: *mut half::bf16,
        m: c_int,
        n: c_int,
        k: c_int,
        k_beta: c_int,
        k_offset: c_int,
        blocks_per_row: c_int,
        idx_bits: c_int,
        beta_bits: c_int,
        offset_bits: c_int,
        ms_used: c_int,
        stream: *mut c_void,
    );
}
```

### 6.3 `LeechLayer::forward_raw` wiring

In `mistralrs-quant/src/leech/leech_layer.rs`, replace the current Phase 4 stub:

```rust
fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Storage};
    use crate::utils::slice_ptr;

    if !matches!(self.device, Device::Cuda(_)) {
        candle_core::bail!("LeechLayer::forward_raw requires CUDA device");
    }
    if a.dtype() != DType::BF16 {
        candle_core::bail!("LeechLayer expects bf16 activations, got {:?}", a.dtype());
    }
    if self.leftover_bf16.is_some() {
        // Path B: leftover columns mean K is not a multiple of 24.
        // Phase 4 punt: dequantize to bf16 + delegate to candle linear.
        // Phase 4.5 will fuse leftover handling into the kernel.
        return self.forward_raw_with_leftover_slow(a);
    }

    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };

    let a = a.contiguous()?;
    let (m, k) = match a.dims() {
        [m, k] => (*m as i32, *k as i32),
        [b, s, k] => ((*b * *s) as i32, *k as i32),
        other => candle_core::bail!("LeechLayer expects rank-2 or rank-3 activation, got {:?}", other),
    };
    let n = self.out_features() as i32;
    debug_assert_eq!(k as usize, self.in_features());

    let out = dev.alloc_zeros::<half::bf16>((m * n) as usize)?;
    // ... slice_ptr the four input tensors and the output, then:
    unsafe {
        crate::leech::ffi::leech_fused_gemm_bf16(
            a_ptr, packed_ptr, beta_ptr,
            offset_ptr_or_null, bias_ptr_or_null, out_ptr,
            m, n, k,
            self.beta_codebook.dim(1)? as i32,
            self.offset_codebook.as_ref().map(|t| t.dim(1).unwrap()).unwrap_or(0) as i32,
            self.blocks_per_row as i32,
            self.idx_bits as i32, /* beta_bits */ 3, /* offset_bits */
            if self.has_offset { 3 } else { 0 },
            self.ms_used as i32,
            dev.cuda_stream().cu_stream(),
        );
    }
    // Wrap output into a Tensor with shape [M, N] (or [B, S, N] if input was rank-3).
    // ...
}
```

The exact slice_ptr / `CudaStorage::wrap_cuda_slice` glue matches `blockwise_fp8/ops.rs:826-851` (the `launch_fp8_matmul_bf16` path) verbatim — copy that pattern.

### 6.4 Index/layout invariants (do not get this wrong)

LLVQ encodes the weight matrix `W` of shape `[N, K] = [out_features, in_features]` with **rows in the N direction and columns in the K direction**. Each LLVQ block spans 24 contiguous K-columns. Block at logical position `(row=n, block_in_row=b)` covers `W[n, b*24 : (b+1)*24]`.

In the fused GEMM `out = a @ W.T`:
- `a` is `[M, K]`. The K-dimension of `a` is the column-dimension of `W`, i.e. the K-axis (in_features).
- `out` is `[M, N]`. The N-dimension of `out` is the row-dimension of `W`, i.e. the N-axis (out_features).
- The K-loop in the kernel iterates `K/K_tile` times, accumulating `out_tile[M, N_tile] += a_tile[M, K_tile] @ W_tile[K_tile, N_tile]`.
- A given `W_tile[K_tile, N_tile]` is produced by decoding `N_tile × (K_tile/24)` LLVQ blocks: block at `(n_in_tile, k_block_in_tile)` covers 24 contiguous K-columns starting at `K_tile_start + k_block_in_tile*24` of weight row `N_tile_start + n_in_tile`.

So the kernel's block_id for a thread decoding tile position `(n_in_tile, k_block_in_tile)` is:

```
n = N_tile_start + n_in_tile
b = K_tile_start / 24 + k_block_in_tile
block_id = n * blocks_per_row + b
```

This is the same row-major flattening Phase 3 used.

---

## 7. Compile-time template + dispatch

### 7.1 Template parameter set

```cpp
template<
    int    ARCH,           // 80, 90, 120 — picks tile shape + MMA instruction
    int    IDX_BITS,       // 54 (ms=18) or 48 (ms=13)
    int    BETA_BITS,      // 3 — could become a template if encoder ever changes it
    int    OFFSET_BITS,    // 0 or 3
    int    MS_USED         // 13 or 18 — picks the table namespace
>
__global__ void leech_fused_gemm_bf16_kernel(...);
```

Per arch the kernel pulls in the right tile constants:

```cpp
#if ARCH == 80
    constexpr int M_TILE = 128, N_TILE = 128, K_TILE = 48;
    // mma.m16n8k16 path
#elif ARCH == 90
    constexpr int M_TILE = 128, N_TILE = 128, K_TILE = 96;
    // wgmma.m64n128k16 path
#elif ARCH == 120
    constexpr int M_TILE = 128, N_TILE = 64, K_TILE = 72;
    // mma.m16n8k16 path (consumer Blackwell)
#endif
```

`ms_used` picks the table namespace via `#include "leech_tables_ms{13,18}.h"` outside the template (the tables are namespace-scoped per Phase 3).

### 7.2 Runtime dispatch table

The C ABI launcher (`leech_fused_gemm_bf16`) does the dispatch:

```cpp
extern "C" void leech_fused_gemm_bf16(...) {
    // Detect arch at first call, cache the result.
    static int arch_tier = detect_arch_tier();  // 80, 90, or 120

    // 16 specializations: arch × idx_bits × has_offset × ms_used.
    // Many of those don't make sense (idx_bits and ms_used are
    // co-determined). Only 6 are real:
    //   arch ∈ {80, 90, 120} × ms_used ∈ {13(idx_bits=48), 18(idx_bits=54)} × offset ∈ {0,3}.
    // That's 12 actual specializations. List them explicitly.

    if (arch_tier == 80 && ms_used == 18 && offset_bits == 3) {
        leech_fused_gemm_bf16_kernel<80, 54, 3, 3, 18><<<grid, block, smem, stream>>>(...);
    } else if (arch_tier == 80 && ms_used == 18 && offset_bits == 0) {
        leech_fused_gemm_bf16_kernel<80, 54, 3, 0, 18><<<grid, block, smem, stream>>>(...);
    } else if (arch_tier == 90 && ms_used == 18 && offset_bits == 3) {
        // ...
    }
    // ... 9 more arms
}

static int detect_arch_tier() {
    int device;
    cudaGetDevice(&device);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    int cc = major * 10 + minor;
    if (cc >= 120) return 120;     // sm_120+ (consumer Blackwell)
    if (cc >= 90)  return 90;      // sm_90+ (H100, GH200, etc.)
    if (cc >= 80)  return 80;      // sm_80+ (A100, A40, etc.)
    // sm_75 and below not supported.
    return -1;
}
```

12 specializations × ~600-line kernel body. nvcc compile time estimate: ~90 s based on the LEECH_INTEGRATION_PLAN.md §3 risk note for 2-specialization kernels (2 specs × 90 s extrapolated × scaling for size ≈ 6× linear). To keep compile time bounded, the kernel body is in a `.cuh` header included with different template args from each `.cu` instantiation file:

```
leech_gemm_kernel.cuh           // the templated kernel body
leech_gemm_inst_sm80_ms18.cu    // explicit instantiation, 2 arms (offset 0/3)
leech_gemm_inst_sm80_ms13.cu
leech_gemm_inst_sm90_ms18.cu
leech_gemm_inst_sm90_ms13.cu
leech_gemm_inst_sm120_ms18.cu
leech_gemm_inst_sm120_ms13.cu
leech_gemm.cu                   // contains the extern-C dispatcher
```

6 instantiation `.cu` files compile in parallel; total wall ~120 s. Each one is included only when `-arch=sm_XX` matches in build.rs.

### 7.3 build.rs touch

`mistralrs-quant/build.rs` already globs `kernels/*/*.cu`. Add a compile-arch filter so that `leech_gemm_inst_sm120_*.cu` only compiles when nvcc supports sm_120 (CUDA ≥ 12.0). The existing fp8 kernels do the same dance — copy that pattern.

---

## 8. Validation plan

### 8.1 Numerical correctness

**Phase 4 acceptance gate** (per LEECH_INTEGRATION_PLAN.md §1.4):
1. Reference path: `payload.py:reconstruct_bf16_from_streams` produces a bf16 weight tensor → torch `@` activation → reference output.
2. Kernel path: `leech_fused_gemm_bf16` directly.
3. Compare element-wise; fp32-accumulator relative error must be < 1e-3 over a 10-tensor sweep.

**Sub-tests** (must all pass before integration):
- `tests/leech_fused_gemm_bf16_unit.rs`: synthetic random activations, single LLVQ tensor from fixture, compare reduction output element-wise.
- `tests/leech_fused_gemm_dequant_consistency.rs`: cross-check `leech_fused_gemm_bf16(a, W)` against `mm(a, dequantize_w())` where `dequantize_w` uses the Phase 3 standalone decode + epilogue applied row-by-row.
- `tests/leech_fused_gemm_edge_cases.rs`: M = {1, 7, 64, 128, 4096} to exercise wave quantization and split-K boundary. Verify against same reference.
- `tests/leech_fused_gemm_bias.rs`: with and without bias; with and without offset codebook.

The Phase 3 byte-equal-decode test (`leech_decode_cuda.rs`) stays in place; Phase 4 adds these without modifying Phase 3.

### 8.2 Performance

**Microbench** in `mistralrs-quant/tests/leech_fused_gemm_perf.rs` using cudaEvent timing on a fresh stream, same harness as Phase 3's `bench_decode_v_int_kernel_only`. Shapes:

| Layer            | M (batch)  | N (out)  | K (in)   | Tensor                  |
|------------------|-----------:|---------:|---------:|-------------------------|
| `mlp.gate_proj`  | 1, 8, 64   | 12288    | 3584     | `layers.{0..31}.mlp.gate_proj` |
| `mlp.up_proj`    | 1, 8, 64   | 12288    | 3584     | `layers.{0..31}.mlp.up_proj`   |
| `mlp.down_proj`  | 1, 8, 64   | 3584     | 12288    | `layers.{0..31}.mlp.down_proj` |
| `self_attn.q_proj` | 1, 8, 64 | 4096     | 3584     | `layers.{0..31}.self_attn.q_proj` |
| `self_attn.o_proj` | 1, 8, 64 | 3584     | 4096     | `layers.{0..31}.self_attn.o_proj` |

For each shape, measure:
- `leech_fused_gemm_bf16` time.
- cuBLAS `cublasGemmEx` bf16 baseline on a fully-decoded bf16 weight.
- Ratio = fused / cuBLAS.

**Acceptance**: ratio ≥ 0.80 (i.e., fused kernel within 25% of native bf16). If any shape falls below 0.60, escalate to Phase 4.5 (kernel tuning) before declaring Phase 4 done.

**Specific concrete tok/s target for end-to-end** (Phase 5 will verify against full pipeline): batch=1 Qwen3.6 inference on RTX 5090 ≥ 30 tok/s. Source for target: bf16 baseline runs ~50 tok/s on the same hardware; our LLVQ weights are 8× smaller and the decode amortization with `M_tile=128` is essentially free per-element, so we expect ≥ 0.6× baseline → ≥ 30 tok/s minimum.

### 8.3 Profiling sweep

Nsight Compute on the fused kernel for `down_proj` at M=64:
- **achieved_occupancy** ≥ 0.50.
- **sm__pipe_alu_cycles_active** ≥ 0.6 (we want compute-bound, not memory-bound, at small M).
- **sm__warps_eligible.avg.pct_of_peak_sustained_active** ≥ 0.5.
- **smsp__sass_thread_inst_executed_op_bf16_pred_on.sum** vs theoretical max → ≥ 0.5.

Three things must NOT be on the critical path:
- Bit extraction (`extract_bits`).
- Multiset unrank (`unrank_multiset`).
- β/offset codebook reads from shared.

If any of these are critical-path stalls, escalate to optimization phase.

### 8.4 Integration

End-to-end test (gates the Phase 4 → 5b handoff): run `mistralrs-server --model-id qwopus.leech --prompt "Hello"` with the fused kernel wired into `LeechLayer::forward_raw`, exercise 100 tokens of generation, measure PPL on wikitext2 ctx=2k:

- **Target**: PPL = 7.78 ± 0.005 (V6-base baseline from `packer/PLAN.md` §1.1).
- **Fail**: anything outside that band means a numerical bug — kernel output differs from the reconstruction path on real data even if the synthetic tests passed.

### 8.5 Parity-ratio probe (validates §4.5 falsification)

Before Phase 4 implementation starts, run a one-shot Python probe:

```python
# tools/probe_parity_ratio.py
from packer.unpack import unpack_leech
from packer.core.leech_decode_njit_v2 import classify_block_parity
fixture = "production/qwopus-9B-unfettered-MS18-V6-base/qwopus.leech"
file_state = unpack_leech(fixture)
for tensor in file_state.llvq_tensors():
    ratios = classify_block_parity(tensor)  # returns (even%, odd%) per tensor
    print(f"{tensor.name}: even={ratios[0]:.2%} odd={ratios[1]:.2%}")
```

If any tensor's parity ratio falls outside [0.20, 0.80], reconsider the §4 decision. (Expected based on the V6 corpus: all in [0.35, 0.65]. If extreme skew is observed, parity-sort may be worth re-evaluating *only on those tensors*.)

---

## 9. Phase 4 → 5b handoff

Phase 5b (sidecar loader, `mistralrs-core/src/pipeline/loaders/leech_loader.rs`) builds on the per-layer artifacts Phase 4 produces. The interface Phase 4 must expose:

### 9.1 What Phase 4 must export to Phase 5b

1. **`LeechLayer::new`** stays as the constructor; the `forward_raw` is wired up by Phase 4. No new constructor work for 5b.
2. **Layer-init helper**: a one-time `leech_init_tables_ffi()` call. Already exists in Phase 3; Phase 4 doesn't change it. Phase 5b calls it once when the first `.leech` model is loaded.
3. **Per-tensor allocation contract**: Phase 5b allocates `packed_stream`, `beta_codebook`, `offset_codebook`, and optional `leftover_bf16` + `bias` as device `Tensor`s, then constructs the `LeechLayer` via `QuantMethodConfig::Leech { ... }`. Phase 4 does not change the existing constructor signature.
4. **Compile-time symbols required**: `leech_fused_gemm_bf16` (the C ABI entry point), `leech_init_tables_ffi`, `leech_decode_v_int_cuda` (kept from Phase 3, used by ISQ / debug paths).

### 9.2 What Phase 5b adds

Loader skeleton, distilling LEECH_INTEGRATION_PLAN.md §1.5:

```
mistralrs-core/src/pipeline/loaders/
    mod.rs                  // ADD: pub mod leech_loader;  + LEECHv01 magic dispatch
    leech_loader.rs         // NEW: reads .leech, builds LeechLayer per tensor

mistralrs-core/src/pipeline/
    normal.rs               // EDIT: 8-byte magic-sniff before format dispatch
                            //       (LEECHv01 → leech_loader, else existing GGUF/safetensors path)
```

The `normal.rs` edit is the **only** Phase 5b touch in pre-existing files — single-line magic detection at the top of model load. `leech_loader.rs` is otherwise a new file owned wholly by Phase 5b.

### 9.3 LEECHv01 magic detection

The on-disk magic per `packer/core/container.py` is `b"LEECH\x00\x00\x00"` (8 bytes). In `normal.rs::load_model`, before calling the existing safetensors / GGUF loader:

```rust
let path = &model_id_or_path;
if let Ok(file) = File::open(path) {
    let mut magic = [0u8; 8];
    if file.read_exact(&mut magic).is_ok() && &magic == b"LEECH\0\0\0" {
        return leech_loader::load(path, /* device, config, etc. */);
    }
}
// fall through to existing logic
```

This is identical to the GGUF magic-sniff path in `mistralrs-core/src/pipeline/loaders/gguf_loader.rs:detect_gguf_file`. Use that as the template.

### 9.4 Things NOT in Phase 4 scope (Phase 5b will do them)

- Loader-time codebook upload to device memory (Phase 5b allocates the `Tensor`s).
- Overlay block (block24 + LoRA) parsing — Phase 5b handles via `LeechLayer::with_overlay()` (new method, Phase 5b owns).
- Model graph construction (Qwen3.6 architecture instantiation) — Phase 5b.
- fp8 head/embed loading — Phase 5b (separate dispatch to either the existing fp8 path or a new fused fp8 matmul, depending on Q6 below).

---

## 10. Risk register

Ordered by likelihood × impact. Each risk has a mitigation and a falsification step.

| # | Risk | Likelihood | Impact | Mitigation | Falsification step |
|---|------|------------|--------|------------|---------------------|
| 1 | **Predicated dual decode is too slow** — costs more than expected, kernel fails the ≥ 80% cuBLAS gate. | Medium | High | §4.5 fallback: flip to parity-sort. Pre-compute the permutation in Rust at load time. ~3 days of work. | §8.2 perf microbench fails 0.80 ratio AND Nsight shows decode on critical path. |
| 2 | **Numerical drift bf16 vs fp32 accumulator path.** Even with fp32 accumulator, the sequence `bf16 weight → fp32 acc → bf16 output` may accumulate enough error over K=12288 that PPL drifts > 0.005. | Low-Medium | High | Use fp32 accumulator always (§5.2). Run the 10-tensor element-wise compare (§8.1) on a real input before any synthetic test. | §8.4 PPL > 7.78 + 0.005. |
| 3 | **Shared memory overflow on sm_80.** A100's 164 KB SMEM is tight after we add codebook slabs + wgmma async-load staging. | Medium | Medium | Per-arch tile size in §3 is conservative. If still overflows, drop to K_tile=24 (one LLVQ block per K-step) on sm_80 only. | Build fails or `cuLaunchKernel` returns CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES on a sm_80 device. |
| 4 | **Register pressure forces spills.** 24-element int64 arrays for both `out_even` and `out_odd` mean 48 int64 register slots per thread + scratch. 128 threads × ~140 regs = 17,920 regs/CTA ≤ 32K, OK. But predicated dual stacks more state simultaneously. | Medium | Medium | Compile with `-maxrregcount=128`; check spill stats in `cuobjdump --dump-elf`. Drop 24-element int64 to int8 (the decode primitives don't need 64-bit until the final epilogue) to halve the footprint. | nvcc warning about local-memory spills, or Nsight shows L1 spill traffic. |
| 5 | **Tensor-core K-shape mismatch (resolved).** `mma.m16n8k16` has K=16; 24 doesn't divide into 16. Fixed by picking K_tile = LCM-aligned values per arch: K=48 (sm_80), K=96 (sm_90), K=96 (sm_120). All multiples of both 24 and 16. No padding required. | n/a | n/a | n/a | Compile or runtime decode errors would surface; current spec is clean. |
| 6 | **Launch overhead at batch=1.** Tile count drops (only 96×28 = 2688 tiles for `down_proj`) and per-CTA decode work may not amortize the launch. | Low | Medium | At batch=1 with M_tile=128 we still get full M_tile reuse (the batch=1 row sits at M=1 of the M_tile=128 stripe — 127 rows of slack, no decode penalty). Confirm via Nsight wave-quantization. | §8.2 batch=1 perf < 50% of batch=64 throughput per token. |
| 7 | **`__constant__` memory limit (64 KB total per kernel).** The Phase 3 codebook tables already fit, but Phase 4 adds per-block β/offset codebooks staged from shared (these are NOT in `__constant__`, so no risk there). Risk: if a future encoder change bumps `n_classes` past current ms=18 ~1209, the per-class scalar arrays at ~47 KB may overflow. | Low | Medium | Keep the SHA-256 check on the generated headers (Phase 2 already does this). Add a compile-time `static_assert` that `sizeof(c_A) + sizeof(c_two_B) + ... ≤ 56*1024`. | Compile fails. |
| 8 | **Leftover bf16 column handling.** Some tensors have K not divisible by 24. Phase 4 currently punts: dequant-then-cublas slow path. | Medium | Low | Acceptable for Phase 4. The non-divisible tensors are rare (12/427 in V6-base per the manifest). Phase 4.5 will fuse. | Slow-path takes > 5% of total inference time. |
| 9 | **Phase 5b dispatch in `normal.rs` fights with existing loader cascade.** | Low | Low | Magic-sniff is the first thing checked, mirrors GGUF. | `cargo test -p mistralrs-core --features cuda` regressions. |

---

## 11. Out of scope for Phase 4

Explicit list of items that Phase 4 will **not** attempt. Each has a designated owner-phase.

| Item | Owner phase | Notes |
|------|-------------|-------|
| Overlay block (block24 + LoRA) integration into the kernel | Phase 5b or Phase 5.5 | Phase 4 ships without overlay; LoRA-merge path errors out (`LeechLayer::add_delta_w` already errors). |
| fp8 head/embed direct GEMM (Option B of CUDA_KERNEL_SPEC §8) | Phase 5b or Phase 6 | Phase 4 leaves head/embed on the existing `blockwise_fp8` path. |
| KV cache changes for the new pipeline | Phase 5b | KV cache stays on whatever Qwen3.6 already uses. |
| MoE / indexed-expert variant of the fused kernel | Future / out-of-scope | Qwen3.6 isn't MoE; no need until a future MoE LLVQ encode. |
| Multi-GPU / TP-sharded LLVQ layer | Future | Out of scope. |
| ISQ in-situ quantization on top of `.leech` | Never | Explicitly errored in `LeechLayer::apply_isq` — `.leech` is pre-quantized. |
| CPU inference path | Deferred (CUDA_KERNEL_SPEC Q3) | `LeechLayer::dequantize_w` will eventually call a CPU port of the decode for ISQ/debug; not a Phase 4 blocker. |
| `leftover_bf16` fusion (K not divisible by 24) | Phase 4.5 | See risk #8. |
| Parity-sort path | Phase 4.5 only if §4.5 falsification trips | Carried as fallback, not implemented day-1. |
| Split-K variant for batch=1 skinny matmuls | Phase 4.5 | Default kernel has split-K=1; the small-N variant is a follow-up only if perf needs it. |

---

## Appendix A — File touch summary

```
mistralrs-quant/
├── build.rs                                       [edit: add per-arch instantiation filter]
├── kernels/leech/
│   ├── leech_decode.cu                            [keep as-is from Phase 3]
│   ├── leech_bit_extract.cuh                      [keep]
│   ├── leech_sign_unrank.cuh                      [keep]
│   ├── leech_multiset_unrank.cuh                  [keep]
│   ├── leech_tables_ms{13,18}.h                   [keep]
│   ├── leech_gemm_kernel.cuh                      [NEW — templated kernel body]
│   ├── leech_gemm.cu                              [NEW — C ABI dispatcher]
│   ├── leech_gemm_inst_sm80_ms13.cu               [NEW — explicit instantiation]
│   ├── leech_gemm_inst_sm80_ms18.cu               [NEW]
│   ├── leech_gemm_inst_sm90_ms13.cu               [NEW]
│   ├── leech_gemm_inst_sm90_ms18.cu               [NEW]
│   ├── leech_gemm_inst_sm120_ms13.cu              [NEW]
│   ├── leech_gemm_inst_sm120_ms18.cu              [NEW]
│   └── leech_epilogue.cuh                         [NEW — β·v + offset → bf16 device fn]
├── src/leech/
│   ├── leech_layer.rs                             [edit: replace forward_raw stub]
│   ├── ffi.rs                                     [edit: add leech_fused_gemm_bf16 decl]
│   ├── leech_cuda.rs                              [edit: leech_linear constructs LeechLayer with new path]
│   └── (other files unchanged)
└── tests/
    ├── leech_decode_cuda.rs                       [keep — Phase 3 test]
    ├── leech_fused_gemm_bf16_unit.rs              [NEW]
    ├── leech_fused_gemm_dequant_consistency.rs    [NEW]
    ├── leech_fused_gemm_edge_cases.rs             [NEW]
    ├── leech_fused_gemm_bias.rs                   [NEW]
    └── leech_fused_gemm_perf.rs                   [NEW — cudaEvent microbench]
```

Outside `mistralrs-quant`, **no changes**. Phase 5b owns the loader integration.

---

## Appendix B — Phase 4 estimated complexity

- **Kernel body** (`leech_gemm_kernel.cuh`): ~800 lines. Reuses Phase 3 device functions (`decode_even`, `decode_odd`, sign unrank, multiset unrank, bit extract) verbatim — the kernel adds tile staging, K-loop, mma issue, and epilogue.
- **Dispatcher** (`leech_gemm.cu`): ~200 lines, including 12 specialization arms.
- **Instantiation `.cu` files**: ~30 lines each × 6 files = 180 lines.
- **Rust `forward_raw`** glue: ~120 lines (mostly slice_ptr + Tensor wrap, following `blockwise_fp8/ops.rs`).
- **Tests**: ~600 lines total across 5 new test files.
- **Total new code**: ~1,900 lines.

**Time estimate** (LEECH_INTEGRATION_PLAN.md §1.4 budget: 4–5 days):
- Day 1: kernel skeleton + sm_90 specialization compiles and runs (no correctness yet).
- Day 2: sm_90 numerical correctness on 1 tensor (gate §8.1 sub-test).
- Day 3: 10-tensor correctness sweep + bias + no-offset variants.
- Day 4: sm_80 and sm_120 specializations; tile-size adjustments per §3.
- Day 5: Phase 5b integration smoke test (Phase 5b probably starts mid-day 5).

If any of {decode bug found on a corner tensor, register pressure overflow, parity-decision flip to parity-sort} hits, add 2–3 days.

---

*End of design. Ready to implement.*
