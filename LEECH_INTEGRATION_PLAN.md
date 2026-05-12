# `.leech` → mistral.rs Integration Plan

**Branch**: `leech-quant` (in `vukasinjockovic/mistral.rs`)
**Upstream baseline**: `2d4ba4f1` (master, 2026-05-12)
**Source artefacts**: `/var/www/vibe-marketing/docs/antsquant/packer/` — feature-complete CPU prototype + 580-line CUDA design spec.

This plan is a fork-only build. We do **not** open a PR upstream. The fork lives, gets updated, and is consumed by our own deployment.

---

## 0. Design Decisions (locked)

| Choice | Decision | Rationale |
|---|---|---|
| Loader path | **Sidecar `.leech` loader** in `mistralrs-core/src/pipeline/loaders/` | One file = one model. Mirrors `gguf_loader.rs`. The shim-into-safetensors alternative bloats UQFF and obscures the container contract. |
| Static tables | **`constexpr` arrays in `leech_tables_ms18.h`** generated once, checked into `mistralrs-quant/kernels/leech/` | No `__constant__` upload at runtime, no `.leech`-file readback for tables, max const-cache locality. Header is generated from `build_flat_tables(ms_max=18)` via a Python tool. |
| Op order in kernel | `fp32 fma` then `cvt.rn.bf16.f32` | Encoder bake order (`packer/CUDA_KERNEL_SPEC.md` §3.1). Reversing fails the byte-exact roundtrip. |
| Parity dispatch | Parity-sort permutation at tensor-load (§5 of spec) | Cheap, robust, predictable. Predicated-dual-decode kept as fallback for tensors with extreme parity skew. |
| fp8 head/embed | Option B — direct `wgmma.f32.e4m3.e4m3` | Halves HBM traffic on the 2 GB head+embed. fp8→bf16 fallback path stays available for debug. |
| ms_used compile-time? | **Template-specialized for `ms_used ∈ {13, 18}`** | Lets `idx_bits` be a `constexpr` in the hot path. Both V1 (ms=13) and V6 (ms=18) bundles supported. |

---

## 1. Phase Roadmap

Six phases, all on `leech-quant`. Each phase has a verification gate; no phase starts until the previous one's gate is green.

### Phase 1 — Container reader in Rust (2–3 days)
- **Goal**: Parse `.leech` end-to-end, byte-equal to `packer/unpack.py`, with CRC32 verification.
- **Files** (new):
  - `mistralrs-leech/Cargo.toml` — new crate inside the workspace
  - `mistralrs-leech/src/lib.rs` — re-exports
  - `mistralrs-leech/src/container.rs` — MAGIC + 64 B HEADER + manifest JSON + 128 B TOC
  - `mistralrs-leech/src/payload.rs` — LLVQ payload header + β/offset codebook bytes + packed stream + leftover
  - `mistralrs-leech/src/overlay.rs` — OVERLAY_BLOCK parser (block24 + lora pieces)
  - `mistralrs-leech/src/fp8.rs` — fp8_e4m3 byte → bf16 (matches `packer/core/fp8_decode.py`)
  - `mistralrs-leech/src/error.rs`
  - `mistralrs-leech/tests/parse_roundtrip.rs` — fixture parse against the 4.05 GB qwopus.leech
- **Workspace edit**: add `mistralrs-leech` to root `Cargo.toml` `[workspace] members`.
- **Acceptance gate**:
  1. Parses the local `production/qwopus-9B-unfettered-MS18-V6-base/qwopus.leech` reference fixture (or wherever it lands).
  2. CRC32 verifies (0xdd0682e3 for the V6-base build).
  3. Enumerates all 427 TOC entries with correct (offset, length, role).
  4. Reads the manifest JSON and confirms 248 LLVQ + 2 fp8 + 177 bf16 = 427 tensors.
  5. Manifest SHA-256 digests match the file on disk.

### Phase 2 — Universal table codegen ✅ DONE (commit pending)
- **Goal**: Generate `leech_tables_ms18.h` and `leech_tables_ms13.h` as `constexpr` C++ headers. Bake the universal Leech tables into the kernel TU at compile time.
- **Finding (important — pre-existing kernel spec was wrong)**: `valid_signs_flat`
  is **666 MB** at ms=13 and **2.9 GB** at ms=18 — the pre-enumerated even-class
  sign table cannot be baked into a header. It is omitted from the generated
  headers. Phase 3's CUDA decoder must replace the lookup with an **algorithmic
  sign unrank** derived from the per-class `nz_distinct_desc` / `f0_counts` /
  `f1_counts` / `parity` metadata. The paper's §3.3 step 4 already does this
  for odd classes (XOR with codeword); even classes need the same treatment.
- **Actual baked totals**:
  | ms_max | n_classes | baked bytes | header bytes | omitted (signs) |
  |---|---|---|---|---|
  | 13 | 383 | 3.7 MB | 9 MB | 666 MB |
  | 18 | 1209 | 12 MB | 29 MB | 2.9 GB |
- **Dtype narrowing**: `codewords_flat` was narrowed `int64 → uint32` (Golay
  codewords are 24-bit). Halves the dominant array.
- **Storage strategy for Phase 3**: per-class scalars (A, two_B, parity, etc.,
  ~10–30 KB) → `__constant__`. The medium-sized ragged arrays (codewords_flat
  at ~12 MB) → `__device__` initialized from the constexpr header. `nvcc` handles
  the binary-bake automatically.
- **Files** (new):
  - `tools/gen_leech_tables.py` (in antsquant repo, NOT mistral.rs) — runs `build_flat_tables(ms_max=M)` and emits a `.h` file
  - `mistralrs-quant/kernels/leech/leech_tables_ms13.h` (committed; generated artefact)
  - `mistralrs-quant/kernels/leech/leech_tables_ms18.h` (committed; generated artefact)
  - `mistralrs-quant/kernels/leech/leech_tables_sha.h` (committed; SHA-256 of each generated header for runtime sanity)
- **Header contents** (per `ms_max`):
  ```cpp
  // Auto-generated from packer/core/codebook_tables.py — DO NOT EDIT
  namespace leech_ms18 {
    static constexpr uint64_t N_cumulative[20] = { 0, 1, ... };
    static constexpr int32_t  shell_class_start[19] = { ... };
    static constexpr int32_t  shell_class_count[19] = { ... };
    static constexpr uint64_t A[N_CLASSES] = { ... };
    static constexpr uint64_t two_B[N_CLASSES] = { ... };
    static constexpr uint64_t orbit_F1[N_CLASSES] = { ... };
    static constexpr uint8_t  parity[N_CLASSES] = { ... };
    static constexpr uint64_t class_cum_offset[N_CLASSES] = { ... };
    // ragged arrays + their prefix-sum offsets:
    static constexpr uint64_t codewords_flat[CW_TOTAL] = { ... };
    static constexpr int32_t  codewords_ofs[N_CLASSES + 1] = { ... };
    // ... valid_signs / f0 / f1 / multiset / nz_distinct / their _ofs ...
    static constexpr int N_CLASSES = ...;
    static constexpr int IDX_BITS  = 54;  // ms=18 → 54; ms=13 → 41
  }
  ```
- **Codegen invariants**:
  1. Output bit-equal across runs (deterministic — `build_flat_tables` is pure).
  2. Embedded SHA-256 of the source FlatTables in a comment header.
  3. Total static-data size ≤ 64 KB per `ms_max` (must fit constant cache).
- **Acceptance gate**:
  1. `mistralrs-quant/kernels/leech/leech_tables_ms18.h` compiles in a 5-line test TU.
  2. Python sanity script loads both `build_flat_tables(ms_max=18)` and parses the header, asserts `np.array_equal` on every member.
  3. Total array bytes printed; must be ≤ 64 KB.

### Phase 3 — `mistralrs-quant/src/leech/` skeleton (2 days)
- **Goal**: A working `LeechLayer` that decodes one tensor on GPU (no GEMM yet) and matches the Python v2 decoder byte-for-byte.
- **Files** (new):
  - `mistralrs-quant/src/leech/mod.rs` — feature-flag switch (`cpu` vs `cuda`)
  - `mistralrs-quant/src/leech/ffi.rs` — `extern "C"` declarations
  - `mistralrs-quant/src/leech/leech_cuda.rs` — `LeechLayer`, `impl QuantMethod`, `pub fn leech_linear(...)`
  - `mistralrs-quant/src/leech/leech_cpu.rs` — stub that bails with "build with `--features cuda` for `.leech`"
  - `mistralrs-quant/kernels/leech/leech_decode.cu` — decode-only kernel (per CUDA_KERNEL_SPEC §3)
  - `mistralrs-quant/kernels/leech/leech_bit_extract.cuh` — branchless bit extraction (per spec §7)
- **`LeechLayer` shape**:
  ```rust
  pub struct LeechLayer {
      packed_stream: CudaSlice<u8>,      // body bits
      beta_codebook: CudaSlice<f16>,      // [R, K_beta]
      offset_codebook: Option<CudaSlice<f16>>,
      parity_perm: CudaSlice<u32>,        // built at load time, see spec §5
      r_count: usize,
      b_count: usize,
      idx_bits: u8,
      beta_bits: u8,
      offset_bits: u8,
      ms_used: u8,                        // 13 or 18 — selects kernel specialization
      out_features: usize,
      in_features: usize,
      leftover: Option<CudaSlice<bf16>>,  // verbatim bf16 trailing columns
  }
  ```
- **`extern "C"` interface**:
  ```c
  // decode-only — Phase 3 acceptance target
  void leech_decode_v_int(
      const uint8_t* packed,   const half* beta_cb, const half* offset_cb,
      const uint32_t* parity_perm,
      int8_t* out_v_int,                  // [R, B, 24]
      int R, int B, int idx_bits, int beta_bits, int offset_bits, int ms_used,
      cudaStream_t stream
  );
  ```
- **Workspace edits to `mistralrs-quant/src/lib.rs`**:
  ```
  + line ~22:  mod leech;
  +            pub use leech::{LeechLayer, leech_linear};
  + line ~297: add `Leech { ms_used: u8, has_offset: bool, idx_bits: u8 }` to QuantizedConfig
  + line ~369: add `m == "leech"` arm in custom deserializer
  + line ~424: add Leech variant to QuantMethodConfig
  + line ~853: add `Leech = 7,` to QuantizedSerdeType + TryFrom arm
  + line ~1078: add `QuantizedConfig::Leech { .. } => leech_linear(...)` to linear_no_bias
  + line ~1143: same arm in linear
  + line ~1198: same arm in linear_b
  ```
- **Acceptance gate**:
  1. `cargo build -p mistralrs-quant --features cuda` succeeds.
  2. New `mistralrs-quant/tests/leech_decode_cuda.rs` test loads one LLVQ tensor from the fixture, runs `leech_decode_v_int`, compares `int8[R, B, 24]` byte-for-byte to the output of `packer/tests/test_decode_v2_exhaustive.py` on the same tensor.
  3. Throughput target: ≥ 100 G blocks/s on H100 (well below fused-kernel target — just a milestone).

### Phase 4 — Fuse decode + bf16 GEMM (4–5 days)
- **Goal**: One kernel produces a bf16 output tile without materializing decoded weights to global memory.
- **Files**:
  - `mistralrs-quant/kernels/leech/leech_gemm.cu` — fused dequant + `wgmma` GEMM (new)
  - `mistralrs-quant/kernels/leech/leech_epilogue.cuh` — β·v + offset → bf16 (new)
  - `mistralrs-quant/src/leech/ffi.rs` — add `leech_fused_gemm(...)` decl
  - `mistralrs-quant/src/leech/leech_cuda.rs` — `impl QuantMethod::forward_raw` calls fused kernel
- **Tile sizes** (from CUDA_KERNEL_SPEC §6): M=N=128, K=96, 4 warps/CTA, SMEM 48 KB/CTA.
- **`extern "C"` interface**:
  ```c
  void leech_fused_gemm(
      const __nv_bfloat16* a,             // [M, K] activations
      const uint8_t*       packed,
      const half*          beta_cb,
      const half*          offset_cb,
      const uint32_t*      parity_perm,
      __nv_bfloat16*       out,           // [M, N]
      int M, int N, int K,
      int R, int B, int idx_bits, int beta_bits, int offset_bits, int ms_used,
      cudaStream_t stream
  );
  ```
- **Verification harness**:
  1. Reference path: `packer/core/payload.py:reconstruct_bf16_from_streams` → numpy `@` activation → output.
  2. CUDA path: `leech_fused_gemm` directly.
  3. Compare element-wise; fp32-accumulator relative error must be < 1e-3 over a 10-tensor sweep.
- **Acceptance gate**:
  1. End-to-end matmul matches reference within 1e-3 relative error.
  2. Nsight Compute: warp-divergence stall < 5%, achieved occupancy ≥ 50%.
  3. Perf: ≥ 80% of native bf16 cuBLAS GEMM on a `down_proj` shape (M=64 N=4096 K=12288).

### Phase 5 — `mistralrs-core` sidecar loader (3 days)
- **Goal**: `mistralrs-server --model-id /path/to/qwopus.leech` loads, runs inference, produces coherent text.
- **Files** (new):
  - `mistralrs-core/src/pipeline/loaders/leech_loader.rs` — sidecar loader analogous to `gguf_loader.rs`
  - `mistralrs-core/src/pipeline/leech_pipeline.rs` (if needed; otherwise extend existing)
- **Workspace edits** (`mistralrs-core/src/pipeline/loaders/mod.rs`):
  - Add `pub mod leech_loader;`
  - Add file-magic detection: read first 8 bytes, if `b"LEECH\x00\x00\x00"` (per `packer/core/container.py`) → dispatch to `leech_loader`.
- **Loader responsibilities**:
  1. Open `.leech`, parse via `mistralrs-leech` crate.
  2. Build the model graph (Qwen-style hybrid: 32 layers, mix of linear_attn + self_attn) — read `config.json` from inside the manifest JSON (it's embedded per `packer/core/provenance.py`).
  3. For each LLVQ tensor: build `LeechLayer` and feed into `ShardedVarBuilder` as an `Arc<dyn QuantMethod>`.
  4. For each fp8 tensor (head, embed): mount as fp8 weight (Option B path from spec §8).
  5. For each bf16 tensor: mount as `UnquantLinear`.
  6. Apply overlay block: if present, load `block24` + `lora` from the OVERLAY_BLOCK and patch them into the graph.
- **Acceptance gate**:
  1. `cargo run -p mistralrs-server -- --model-id ./qwopus.leech --prompt "hello"` produces non-garbage text.
  2. PPL on wikitext2 ctx=2k matches V6-base (7.7897 ± 0.005, per `packer/PLAN.md` §1.1).
  3. Token throughput ≥ 60 tok/s on H100 at batch=1 (sanity floor; real target measured in Phase 6).

### Phase 6 — Profile + tune (3 days)
- **Goal**: Production-grade perf. Sweep tile sizes, validate parity-sort vs predicated-dual, measure fp8 head/embed savings.
- **Workstreams**:
  1. Nsight Compute pass over the fused kernel. Identify hot stages.
  2. Tile sweep: `M_tile ∈ {64, 128}`, `K_tile ∈ {48, 72, 96, 120}`. Pick the Pareto-best.
  3. Parity strategy benchmark: run both parity-sort and predicated-dual on 10 representative tensors; pick per-tensor or globally based on data.
  4. fp8 head/embed: A/B Option A (decode-once-to-bf16) vs Option B (direct fp8 GEMM). Likely +15–20% throughput on small-batch inference.
  5. Cold-load time profile: parity_perm build, codebook upload, header parse.
- **Acceptance gate**:
  1. `mistralrs-bench` numbers vs equivalent bf16 baseline: ≥ 2× tok/s improvement.
  2. PPL unchanged from Phase 5 (no perf-regression that broke correctness).
  3. Decision matrix documented in `mistralrs-quant/src/leech/PERF_NOTES.md`.

---

## 2. File-Touch Summary (every file we add or modify)

```
mistral.rs/
├── Cargo.toml                                  [edit: + mistralrs-leech member]
├── LEECH_INTEGRATION_PLAN.md                   [new — this file]
├── mistralrs-leech/                            [NEW CRATE — Phase 1]
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── container.rs
│   │   ├── payload.rs
│   │   ├── overlay.rs
│   │   ├── fp8.rs
│   │   └── error.rs
│   └── tests/
│       └── parse_roundtrip.rs
│
├── mistralrs-quant/
│   ├── src/
│   │   ├── lib.rs                              [edit ×7: enum + dispatch arms]
│   │   └── leech/                              [NEW MODULE — Phase 3]
│   │       ├── mod.rs
│   │       ├── ffi.rs
│   │       ├── leech_cuda.rs
│   │       └── leech_cpu.rs
│   └── kernels/
│       └── leech/                              [NEW KERNEL DIR — Phase 2 + 3 + 4]
│           ├── leech_tables_ms13.h             [generated, committed]
│           ├── leech_tables_ms18.h             [generated, committed]
│           ├── leech_tables_sha.h
│           ├── leech_bit_extract.cuh
│           ├── leech_epilogue.cuh
│           ├── leech_decode.cu                 [Phase 3]
│           └── leech_gemm.cu                   [Phase 4]
│
└── mistralrs-core/
    └── src/pipeline/
        ├── loaders/
        │   ├── mod.rs                          [edit: + leech_loader, + magic detect]
        │   └── leech_loader.rs                 [NEW — Phase 5]
        └── leech_pipeline.rs                   [NEW if needed — Phase 5]
```

Also in `antsquant` repo (not in mistral.rs):
```
tools/
└── gen_leech_tables.py                         [NEW — Phase 2 codegen tool]
```

---

## 3. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Upstream mistral.rs evolves fast; rebases get painful. | Keep `leech-quant` rebased weekly on `upstream/master`. Touch as few existing files as possible — most additions are new files, only `mistralrs-quant/src/lib.rs` and `mistralrs-core/src/pipeline/loaders/mod.rs` see edits. |
| Kernel correctness regression after a refactor. | `mistralrs-quant/tests/leech_decode_cuda.rs` runs against the same 250 M-block reference from `packer/tests/test_decode_v2_exhaustive.py`. Phase 3 gate is byte-equal — no fudge. |
| Tile sizes don't generalize from `down_proj` (12288) to small attn projections (e.g. 256). | Phase 6 explicitly sweeps M/K tiles. Fallback: a small-N variant of the fused kernel selected by dispatch based on `N < 1024`. |
| ms=13 and ms=18 specialization explodes compile time. | Both are `template<int ms_used>` instantiations of one kernel body. nvcc compile time for 2 specializations of a ~600-line kernel ≈ 90 s, acceptable. |
| Constant-cache pressure from large `__constant__` tables. | Tables are `constexpr` (literal pool), not `__constant__`. nvcc places them in the constant bank automatically; verify via `cuobjdump --dump-elf`. |
| fp8 wgmma path (`f32.e4m3.e4m3`) only on sm_90+. | `#if __CUDA_ARCH__ >= 900` guard; fall back to fp8→bf16 conversion + bf16 wgmma on older arches. We only target H100/H200 for v1 anyway. |
| Overlay block (block24 + lora) integration is fiddly. | Implement as a separate post-load patch step in Phase 5; if it bites, ship Phase 5 without it and add overlay support as Phase 5.5. |

---

## 4. Test Fixtures

| Fixture | Location | Purpose |
|---|---|---|
| `qwopus.leech` (4.05 GB) | `/var/www/vibe-marketing/docs/antsquant/production/qwopus-9B-unfettered-MS18-V6-base/` (or wherever; needs symlink) | End-to-end correctness, perf bench. |
| `v_ref.npy` per tensor (~250 MB total) | `/var/www/vibe-marketing/docs/antsquant/refs/` (to be generated, Phase 3) | Byte-equal decode reference, dumped from `packer/core/leech_decode_njit_v2.py`. |
| Wikitext-2 raw ctx=2k | already on disk under `production/` | PPL acceptance gate Phase 5. |

---

## 5. Branch Hygiene

- Branch: `leech-quant`, off `master`.
- Push frequency: end of each phase (6 pushes total).
- Per-phase commit message format: `leech: phase N — <one-line summary>`.
- Tag at each green gate: `leech-phase-N`.
- Never merge to `master`. The fork's `master` stays a clean mirror of `upstream/master` for easy rebases.

---

## 6. Open questions deferred to bring-up

These are flagged in `packer/CUDA_KERNEL_SPEC.md` §13 as Q1–Q8. Most resolve during Phase 4 or 6 once we have a kernel to measure:

- **Q1** Language: confirmed CUDA C kernels + Rust glue.
- **Q2** Trait shape: confirmed extending `QuantMethod` with a `Leech` arm in the existing enum.
- **Q3** CPU inference path: deferred. `mistralrs-leech` crate parses on CPU but doesn't decode; if needed, port `leech_decode_njit_v2.py` to Rust later.
- **Q4** Parity strategy: Phase 6 picks.
- **Q5** `idx_bits` compile-time: confirmed `template<int ms_used>` specialization for ms ∈ {13, 18}.
- **Q6** Parity permutation storage: build at tensor-load, hold in HBM next to weight. ~1 GB total ≈ 0.05% over LLVQ body — acceptable; we don't have the 4 B/block-budget cost the spec flagged because parity_perm is uint32 indexed by block_id, not per-element.
- **Q7** Accumulator precision: fp32, locked.
- **Q8** Test vehicle pre-integration: Phase 3 standalone test gives us this already.

---

## 7. Definition of Done

The fork is "done" when:
1. `cargo build -p mistralrs-server --features cuda --release` succeeds on this branch.
2. `mistralrs-server --model-id ./qwopus.leech --prompt "Hello"` produces text matching V6-base PPL ± 0.005.
3. Tokens/sec on H100 batch=1 ≥ 2× the bf16 baseline of the same model.
4. The 250 M-block byte-equal decode test passes against `packer/tests/test_decode_v2_exhaustive.py` reference.
5. `LEECH_INTEGRATION_PLAN.md` is updated with the final perf numbers in §1 Phase 6.
