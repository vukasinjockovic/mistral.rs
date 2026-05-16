//! Q24-tANS v4 fused GEMV correctness + bench.
//!
//! v4 splits each tile's FSE chain into K parallel sub-streams (one thread
//! per sub-stream) and warp-reduces across K lanes before the atomicAdd.
//!
//! This test exercises:
//!   - K=1: must produce output bit-equal (within bf16 ULP) to the v0 kernel,
//!     validating the v3→v4 compat shim and the lane-reduce fast path at K=1.
//!   - K=2,4,8,16,32: when the artifact is v4-native (encoder used
//!     --num-streams ≥ 2), exercise the K-parallel path and re-check
//!     correctness against the CPU v_int reference.
//!
//! Env vars (same as the v0 test):
//!   LEECHQ24_ARTIFACT    — path to a .leech file (v3 or v4)
//!   LEECHQ24_V_REFS      — directory with `<tensor>.v_int.bin`
//!   LEECHQ24_TENSOR      — tensor name (default: layers.0.mlp.up_proj.weight)
//!   LEECHQ24_BENCH       — when "1"/"true", run warmup + iters loop
//!   LEECHQ24_BENCH_ITERS — iteration count (default 200)
//!   LEECHQ24_V4_K        — comma-separated K values to test (default "1")
//!                          For v3 artifacts only K=1 is meaningful via the
//!                          compat shim; for v4 K≥2 artifacts the artifact
//!                          MUST have been encoded with the matching K.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use half::bf16;
use mistralrs_leech_q24::{LeechQ24File, Role};
use mistralrs_quant::leech_q24::{
    compute_substream_bit_offsets, compute_tile_bit_offsets, init_tables,
    leech_q24_gemv_bf16, leech_q24_gemv_bf16_v4,
};

fn env_path(var: &str) -> Option<PathBuf> {
    let p = std::env::var(var).ok()?;
    let pb = PathBuf::from(p);
    if pb.exists() {
        Some(pb)
    } else {
        None
    }
}

fn unpack_3bit(packed: &[u8], i: usize) -> u8 {
    let bit_pos = i * 3;
    let bi = bit_pos >> 3;
    let bo = (bit_pos & 7) as u32;
    let b0 = packed[bi] as u32;
    let b1 = packed.get(bi + 1).copied().unwrap_or(0) as u32;
    ((((b0 | (b1 << 8)) >> bo) & 0x7) as u8)
}

/// Compute the CPU host reference y[r] = Σ_k weight[r, k] * a_act[k].
fn host_reference(
    n_blocks: usize, b_blocks: usize, r_rows: usize, k_total: usize,
    v_ref_bytes: &[u8],
    beta_idx_packed: &[u8],
    offset_idx_packed: &[u8],
    beta_lloyd_host: &[f32],
    offset_lloyd_host: &[f32],
    a_host: &[bf16],
    k_beta: usize, k_offset: usize, has_offset: bool,
) -> Vec<bf16> {
    let mut y_ref_f32 = vec![0.0f32; r_rows];
    for blk in 0..n_blocks {
        let row = blk / b_blocks;
        let col_block = blk - row * b_blocks;
        let beta_idx = unpack_3bit(beta_idx_packed, blk) as usize;
        let beta_val = beta_lloyd_host[row * k_beta + beta_idx];
        let offset_val = if has_offset {
            let oi = unpack_3bit(offset_idx_packed, blk) as usize;
            offset_lloyd_host[row * k_offset + oi]
        } else {
            0.0
        };
        let k_base = col_block * 24;
        for j in 0..24 {
            let v_int = v_ref_bytes[blk * 24 + j] as i8 as f32;
            let w_val = beta_val * v_int + offset_val;
            let a_f32 = a_host[k_base + j].to_f32();
            y_ref_f32[row] += w_val * a_f32;
        }
    }
    let _ = k_total; // silence unused
    y_ref_f32.iter().copied().map(bf16::from_f32).collect()
}

#[test]
fn gemv_v4_K1_matches_v0() {
    // K=1 compat: the v4 kernel with num_streams=1 should produce output
    // bit-equal to v0 on the same artifact (via the compat shim for v3 files,
    // or directly for v4 K=1 files).
    let Some(leech_path) = env_path("LEECHQ24_ARTIFACT") else {
        eprintln!("LEECHQ24_ARTIFACT not set — skipping");
        return;
    };
    let Some(refs_dir) = env_path("LEECHQ24_V_REFS") else {
        eprintln!("LEECHQ24_V_REFS not set — skipping");
        return;
    };
    let tensor_name = std::env::var("LEECHQ24_TENSOR").unwrap_or_else(|_| {
        "model.language_model.layers.0.mlp.up_proj.weight".to_owned()
    });
    let ref_path = refs_dir.join(format!("{}.v_int.bin", tensor_name));
    if !ref_path.is_file() {
        eprintln!("no v_int reference at {} — skipping", ref_path.display());
        return;
    }

    let leech = LeechQ24File::open(&leech_path).expect("open .leech");
    let (toc_idx, entry) = leech
        .find_tensor(&tensor_name)
        .unwrap_or_else(|| panic!("tensor {tensor_name:?} not in TOC"));
    assert_eq!(entry.role, Role::LlvqTans);
    let payload = leech.llvq_tans_payload(toc_idx).expect("payload");

    let n_blocks = entry.n_blocks as u32;
    let n_tiles = entry.n_tiles as u32;
    let tile_size = entry.tile_size as i32;
    let symbol_set_id = entry.symbol_set_id;
    let cb = leech
        .codebook_set(symbol_set_id)
        .unwrap_or_else(|| panic!("codebook sid {symbol_set_id} missing"));
    let w_offset = cb.w_offset as i32;
    let r = entry.r;
    let b = entry.b;
    let k_total = (b as usize) * 24;
    let k_beta = entry.k_beta;
    let k_offset = entry.k_offset;
    let has_offset = k_offset > 0;
    let file_num_streams = payload.num_streams as i32;

    eprintln!(
        "[v4 K=1] tensor {} R={} B={} K_total={} n_blocks={} n_tiles={} tile_size={} file_num_streams={} (testing K=1)",
        tensor_name, r, b, k_total, n_blocks, n_tiles, tile_size, file_num_streams
    );

    // Only test K=1 here regardless of file_num_streams. For K>=2 we test the
    // full sweep in gemv_v4_K_sweep below (requires v4-encoded artifact at
    // each K).
    let test_k: i32 = 1;
    // The compat shim ensures num_streams=1 even on v3 files, so substream_*
    // aliases tile_* and substream_bit_offsets == compute_tile_bit_offsets.
    if file_num_streams != 1 {
        eprintln!(
            "[v4 K=1] artifact has num_streams={} != 1; skipping (use a v3 or v4-K=1 file)",
            file_num_streams
        );
        return;
    }

    // ── Decode tables init ────────────────────────────────────────────
    let dt_bytes = leech
        .decode_tables_bytes(symbol_set_id)
        .expect("decode tables");
    let dt_host: Vec<u32> = dt_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    init_tables(&dt_host, symbol_set_id).expect("init_tables");

    // ── Host data ─────────────────────────────────────────────────────
    let states_host: Vec<u16> = payload
        .tile_states
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let nb_totals: Vec<u16> = payload
        .tile_nb_totals
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let bit_offsets_tile = compute_tile_bit_offsets(&nb_totals);
    // For K=1, substream_states / substream_nb_totals alias the tile arrays,
    // and substream_bit_offsets == tile_bit_offsets.
    let sub_states_host: Vec<u16> = payload
        .substream_states
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let sub_nb_totals: Vec<u16> = payload
        .substream_nb_totals
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    // Sanity check the alias at K=1.
    assert_eq!(sub_states_host, states_host);
    assert_eq!(sub_nb_totals, nb_totals);
    let sub_bit_offsets = compute_substream_bit_offsets(&nb_totals, &sub_nb_totals, test_k as usize);
    assert_eq!(sub_bit_offsets.len(), bit_offsets_tile.len());
    for i in 0..bit_offsets_tile.len() {
        assert_eq!(sub_bit_offsets[i], bit_offsets_tile[i],
                   "K=1 substream_bit_offsets must equal tile_bit_offsets");
    }

    let bs_words: Vec<u64> = payload
        .tile_bitstream
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    let beta_lloyd_host: Vec<f32> = payload
        .beta_lloyd
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect();
    let offset_lloyd_host: Vec<f32> = if has_offset {
        payload
            .offset_lloyd
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    } else {
        Vec::new()
    };

    let v_ref_bytes = std::fs::read(&ref_path).expect("read v_int.bin");
    assert_eq!(v_ref_bytes.len(), (n_blocks as usize) * 24);

    let a_host: Vec<bf16> = (0..k_total)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(2654435761) >> 16) as i32 as f32;
            let f = (x.sin() * 0.5).clamp(-1.0, 1.0);
            bf16::from_f32(f)
        })
        .collect();

    let beta_idx_packed = payload.beta_idx_packed;
    let offset_idx_packed = payload.offset_idx_packed;
    let y_ref_bf16 = host_reference(
        n_blocks as usize, b as usize, r as usize, k_total,
        &v_ref_bytes, beta_idx_packed, offset_idx_packed,
        &beta_lloyd_host, &offset_lloyd_host, &a_host,
        k_beta as usize, k_offset as usize, has_offset,
    );

    // ── GPU run ──────────────────────────────────────────────────────
    use candle_core::backend::BackendDevice;
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::Device;

    let device = Device::new_cuda(0).expect("Device::Cuda");
    let cuda = device.as_cuda_device().expect("cuda device");

    let mut packed_padded = Vec::with_capacity(payload.buckets_packed.len() + 4);
    packed_padded.extend_from_slice(payload.buckets_packed);
    packed_padded.extend_from_slice(&[0u8; 4]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(packed_padded.len()).expect("alloc packed") };
    cuda.memcpy_htod(&packed_padded, &mut d_packed).expect("htod packed");

    let mut d_sub_states = unsafe { cuda.alloc::<u16>(sub_states_host.len()).expect("alloc sub_states") };
    cuda.memcpy_htod(&sub_states_host, &mut d_sub_states).expect("htod sub_states");
    let mut d_sub_nb = unsafe { cuda.alloc::<u16>(sub_nb_totals.len()).expect("alloc sub_nb") };
    cuda.memcpy_htod(&sub_nb_totals, &mut d_sub_nb).expect("htod sub_nb");
    let mut d_bs = unsafe { cuda.alloc::<u64>(bs_words.len()).expect("alloc bs") };
    cuda.memcpy_htod(&bs_words, &mut d_bs).expect("htod bs");
    let mut d_subofs = unsafe { cuda.alloc::<u64>(sub_bit_offsets.len()).expect("alloc subofs") };
    cuda.memcpy_htod(&sub_bit_offsets, &mut d_subofs).expect("htod subofs");

    let mut bp = Vec::with_capacity(beta_idx_packed.len() + 4);
    bp.extend_from_slice(beta_idx_packed);
    bp.extend_from_slice(&[0u8; 4]);
    let mut d_bidx = unsafe { cuda.alloc::<u8>(bp.len()).expect("alloc bidx") };
    cuda.memcpy_htod(&bp, &mut d_bidx).expect("htod bidx");

    let (mut d_oidx, oidx_ptr_const) = if has_offset {
        let mut op = Vec::with_capacity(offset_idx_packed.len() + 4);
        op.extend_from_slice(offset_idx_packed);
        op.extend_from_slice(&[0u8; 4]);
        let mut d = unsafe { cuda.alloc::<u8>(op.len()).expect("alloc oidx") };
        cuda.memcpy_htod(&op, &mut d).expect("htod oidx");
        let ptr = d.device_ptr(d.stream()).0;
        (Some(d), ptr as *const u8)
    } else {
        (None, std::ptr::null::<u8>())
    };

    let mut d_beta_lloyd = unsafe {
        cuda.alloc::<f32>(beta_lloyd_host.len()).expect("alloc beta lloyd")
    };
    cuda.memcpy_htod(&beta_lloyd_host, &mut d_beta_lloyd).expect("htod beta lloyd");

    let (mut d_offset_lloyd, off_lloyd_ptr_const) = if has_offset {
        let mut d = unsafe {
            cuda.alloc::<f32>(offset_lloyd_host.len()).expect("alloc offset lloyd")
        };
        cuda.memcpy_htod(&offset_lloyd_host, &mut d).expect("htod offset lloyd");
        let ptr = d.device_ptr(d.stream()).0;
        (Some(d), ptr as *const f32)
    } else {
        (None, std::ptr::null::<f32>())
    };

    let a_u16: Vec<u16> = a_host.iter().map(|x| x.to_bits()).collect();
    let mut d_a = unsafe { cuda.alloc::<u16>(k_total).expect("alloc a") };
    cuda.memcpy_htod(&a_u16, &mut d_a).expect("htod a");

    let d_out = unsafe { cuda.alloc::<u16>(r as usize).expect("alloc out") };
    let mut d_acc = unsafe { cuda.alloc::<f32>(r as usize).expect("alloc acc") };
    let zeros = vec![0.0f32; r as usize];
    cuda.memcpy_htod(&zeros, &mut d_acc).expect("htod zero");

    let packed_ptr = d_packed.device_ptr(d_packed.stream()).0 as *const u8;
    let sub_states_ptr = d_sub_states.device_ptr(d_sub_states.stream()).0 as *const u16;
    let sub_nb_ptr = d_sub_nb.device_ptr(d_sub_nb.stream()).0 as *const u16;
    let bs_ptr = d_bs.device_ptr(d_bs.stream()).0 as *const u64;
    let subofs_ptr = d_subofs.device_ptr(d_subofs.stream()).0 as *const u64;
    let bidx_ptr = d_bidx.device_ptr(d_bidx.stream()).0 as *const u8;
    let beta_lloyd_ptr = d_beta_lloyd.device_ptr(d_beta_lloyd.stream()).0 as *const f32;
    let a_ptr = d_a.device_ptr(d_a.stream()).0 as *const c_void;
    let out_ptr = d_out.device_ptr(d_out.stream()).0 as *mut c_void;
    let acc_ptr = d_acc.device_ptr(d_acc.stream()).0 as *mut f32;

    let run_once = || unsafe {
        leech_q24_gemv_bf16_v4(
            a_ptr,
            packed_ptr,
            sub_states_ptr, sub_nb_ptr,
            bs_ptr, subofs_ptr,
            bidx_ptr, oidx_ptr_const,
            beta_lloyd_ptr, off_lloyd_ptr_const,
            acc_ptr, out_ptr,
            r, b, n_blocks, n_tiles,
            k_beta, k_offset,
            w_offset, tile_size, test_k,
            has_offset,
            std::ptr::null_mut::<c_void>(),
        )
        .expect("gemv_v4 launch");
    };

    run_once();
    cuda.synchronize().expect("sync");

    let mut host_out_u16 = vec![0u16; r as usize];
    cuda.memcpy_dtoh(&d_out, &mut host_out_u16).expect("dtoh");
    let host_out: Vec<bf16> = host_out_u16.iter().copied().map(bf16::from_bits).collect();

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut n_bad = 0usize;
    for (i, (g, r_ref)) in host_out.iter().zip(y_ref_bf16.iter()).enumerate() {
        let gf = g.to_f32();
        let rf = r_ref.to_f32();
        let diff = (gf - rf).abs();
        let rel = if rf.abs() > 1e-6 { diff / rf.abs() } else { diff };
        if diff > max_abs_diff { max_abs_diff = diff; }
        if rel > max_rel_diff { max_rel_diff = rel; }
        if rel > 0.05 && diff > 0.01 {
            if n_bad < 8 {
                eprintln!("  diff @{i}: gpu={gf} ref={rf} rel={rel}");
            }
            n_bad += 1;
        }
    }
    eprintln!(
        "[v4 K=1] max_abs_diff={:.6e} max_rel_diff={:.6e} bad={}/{}",
        max_abs_diff, max_rel_diff, n_bad, host_out.len()
    );
    assert_eq!(n_bad, 0, "{n_bad} mismatched outputs above tolerance");

    // ── Cross-check: v4 K=1 vs v0 on same artifact ───────────────────
    // Run v0 with the legacy (tile-) bit offsets and compare bf16 outputs.
    let tile_bit_offsets = compute_tile_bit_offsets(&nb_totals);
    let mut d_states_v0 = unsafe { cuda.alloc::<u16>(states_host.len()).expect("alloc states_v0") };
    cuda.memcpy_htod(&states_host, &mut d_states_v0).expect("htod states_v0");
    let mut d_nb_v0 = unsafe { cuda.alloc::<u16>(nb_totals.len()).expect("alloc nb_v0") };
    cuda.memcpy_htod(&nb_totals, &mut d_nb_v0).expect("htod nb_v0");
    let mut d_tilofs = unsafe { cuda.alloc::<u64>(tile_bit_offsets.len()).expect("alloc tilofs") };
    cuda.memcpy_htod(&tile_bit_offsets, &mut d_tilofs).expect("htod tilofs");

    let d_out_v0 = unsafe { cuda.alloc::<u16>(r as usize).expect("alloc out_v0") };
    let mut d_acc_v0 = unsafe { cuda.alloc::<f32>(r as usize).expect("alloc acc_v0") };
    cuda.memcpy_htod(&zeros, &mut d_acc_v0).expect("htod zero v0");
    let states_v0_ptr = d_states_v0.device_ptr(d_states_v0.stream()).0 as *const u16;
    let nb_v0_ptr = d_nb_v0.device_ptr(d_nb_v0.stream()).0 as *const u16;
    let tilofs_ptr = d_tilofs.device_ptr(d_tilofs.stream()).0 as *const u64;
    let out_v0_ptr = d_out_v0.device_ptr(d_out_v0.stream()).0 as *mut c_void;
    let acc_v0_ptr = d_acc_v0.device_ptr(d_acc_v0.stream()).0 as *mut f32;

    unsafe {
        leech_q24_gemv_bf16(
            a_ptr, packed_ptr,
            states_v0_ptr, nb_v0_ptr,
            bs_ptr, tilofs_ptr,
            bidx_ptr, oidx_ptr_const,
            beta_lloyd_ptr, off_lloyd_ptr_const,
            acc_v0_ptr, out_v0_ptr,
            r, b, n_blocks, n_tiles,
            k_beta, k_offset, w_offset, tile_size,
            has_offset,
            std::ptr::null_mut::<c_void>(),
        ).expect("v0 launch");
    }
    cuda.synchronize().expect("sync v0");
    let mut host_out_v0 = vec![0u16; r as usize];
    cuda.memcpy_dtoh(&d_out_v0, &mut host_out_v0).expect("dtoh v0");
    let mut max_diff_v4_v0: f32 = 0.0;
    let mut n_mismatch = 0usize;
    for (i, (a, b)) in host_out.iter().zip(host_out_v0.iter().copied().map(bf16::from_bits)).enumerate() {
        let af = a.to_f32();
        let bf = b.to_f32();
        let d = (af - bf).abs();
        if d > max_diff_v4_v0 { max_diff_v4_v0 = d; }
        if d > 1e-3 * (1.0 + bf.abs()) {
            if n_mismatch < 8 {
                eprintln!("  v4 vs v0 @{i}: v4={af} v0={bf} d={d}");
            }
            n_mismatch += 1;
        }
    }
    eprintln!(
        "[v4 K=1] vs v0 max_diff={:.6e} n_mismatch={}/{}",
        max_diff_v4_v0, n_mismatch, host_out.len()
    );
    assert_eq!(n_mismatch, 0, "v4 K=1 must match v0 on the same artifact");

    eprintln!("PASS: v4 K=1 GEMV matches CPU reference + v0 (max_rel={:.4e})", max_rel_diff);

    // ── Optional bench ───────────────────────────────────────────────
    if std::env::var("LEECHQ24_BENCH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        let iters: u32 = std::env::var("LEECHQ24_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);

        for _ in 0..16 { run_once(); }
        cuda.synchronize().expect("sync");

        let t0 = std::time::Instant::now();
        for _ in 0..iters { run_once(); }
        cuda.synchronize().expect("sync");
        let elapsed = t0.elapsed();
        let per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
        let flops_per_call = (n_blocks as f64) * 24.0 * 4.0;
        let gflops = flops_per_call * iters as f64 / elapsed.as_secs_f64() / 1e9;
        eprintln!(
            "BENCH leech_q24_gemv_bf16_v4 K=1 (R={}, B={}, n_blocks={}, M=1):",
            r, b, n_blocks
        );
        eprintln!("    iters:       {}", iters);
        eprintln!("    per-call:    {:.1} µs", per_call_us);
        eprintln!("    GFLOPs:      {:.1}", gflops);
        eprintln!("    baseline T={}: 1414.5 µs (T=4) / 2271 µs (T=32)", tile_size);
    }

    drop(d_packed); drop(d_sub_states); drop(d_sub_nb); drop(d_bs); drop(d_subofs);
    drop(d_bidx); let _ = d_oidx.take();
    drop(d_beta_lloyd); let _ = d_offset_lloyd.take();
    drop(d_a); drop(d_out); drop(d_acc);
    drop(d_states_v0); drop(d_nb_v0); drop(d_tilofs); drop(d_out_v0); drop(d_acc_v0);
}

/// Tests the v4 kernel at K ≥ 2 against the CPU v_int reference. The artifact
/// MUST have been encoded with `--num-streams = file_num_streams`; we read
/// payload.num_streams and use that as the value of K for the kernel launch.
///
/// This test SKIPS for v3 / v4-K=1 artifacts (where K=1 is already covered
/// by gemv_v4_K1_matches_v0 above).
#[test]
fn gemv_v4_native_K_matches_reference() {
    let Some(leech_path) = env_path("LEECHQ24_ARTIFACT") else {
        eprintln!("LEECHQ24_ARTIFACT not set — skipping");
        return;
    };
    let Some(refs_dir) = env_path("LEECHQ24_V_REFS") else {
        eprintln!("LEECHQ24_V_REFS not set — skipping");
        return;
    };
    let tensor_name = std::env::var("LEECHQ24_TENSOR").unwrap_or_else(|_| {
        "model.language_model.layers.0.mlp.up_proj.weight".to_owned()
    });
    let ref_path = refs_dir.join(format!("{}.v_int.bin", tensor_name));
    if !ref_path.is_file() {
        eprintln!("no v_int reference at {} — skipping", ref_path.display());
        return;
    }

    let leech = LeechQ24File::open(&leech_path).expect("open .leech");
    let (toc_idx, entry) = leech
        .find_tensor(&tensor_name)
        .unwrap_or_else(|| panic!("tensor {tensor_name:?} not in TOC"));
    let payload = leech.llvq_tans_payload(toc_idx).expect("payload");
    let test_k = payload.num_streams as i32;
    if test_k < 2 {
        eprintln!(
            "[v4 K-native] artifact has num_streams={}; skipping (use a v4 K>=2 artifact)",
            test_k
        );
        return;
    }

    eprintln!(
        "[v4 K-native] tensor {} R={} B={} n_blocks={} n_tiles={} tile_size={} K={}",
        tensor_name, entry.r, entry.b, entry.n_blocks, entry.n_tiles,
        entry.tile_size, test_k
    );

    // Re-implements the same fixture pattern as the K=1 test.
    let n_blocks = entry.n_blocks as u32;
    let n_tiles = entry.n_tiles as u32;
    let tile_size = entry.tile_size as i32;
    let symbol_set_id = entry.symbol_set_id;
    let cb = leech.codebook_set(symbol_set_id).expect("codebook");
    let w_offset = cb.w_offset as i32;
    let r = entry.r;
    let b = entry.b;
    let k_total = (b as usize) * 24;
    let k_beta = entry.k_beta;
    let k_offset = entry.k_offset;
    let has_offset = k_offset > 0;

    let dt_bytes = leech.decode_tables_bytes(symbol_set_id).expect("decode tables");
    let dt_host: Vec<u32> = dt_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    init_tables(&dt_host, symbol_set_id).expect("init_tables");

    let sub_states_host: Vec<u16> = payload.substream_states.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let sub_nb_totals: Vec<u16> = payload.substream_nb_totals.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let nb_totals: Vec<u16> = payload.tile_nb_totals.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let sub_bit_offsets = compute_substream_bit_offsets(&nb_totals, &sub_nb_totals, test_k as usize);

    let bs_words: Vec<u64> = payload.tile_bitstream.chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    let beta_lloyd_host: Vec<f32> = payload.beta_lloyd.chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
    let offset_lloyd_host: Vec<f32> = if has_offset {
        payload.offset_lloyd.chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect()
    } else { Vec::new() };

    let v_ref_bytes = std::fs::read(&ref_path).expect("read v_int.bin");
    assert_eq!(v_ref_bytes.len(), (n_blocks as usize) * 24);

    let a_host: Vec<bf16> = (0..k_total)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(2654435761) >> 16) as i32 as f32;
            let f = (x.sin() * 0.5).clamp(-1.0, 1.0);
            bf16::from_f32(f)
        })
        .collect();

    let y_ref_bf16 = host_reference(
        n_blocks as usize, b as usize, r as usize, k_total,
        &v_ref_bytes, payload.beta_idx_packed, payload.offset_idx_packed,
        &beta_lloyd_host, &offset_lloyd_host, &a_host,
        k_beta as usize, k_offset as usize, has_offset,
    );

    use candle_core::backend::BackendDevice;
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::Device;
    let device = Device::new_cuda(0).expect("Device::Cuda");
    let cuda = device.as_cuda_device().expect("cuda device");

    let mut packed_padded = Vec::with_capacity(payload.buckets_packed.len() + 4);
    packed_padded.extend_from_slice(payload.buckets_packed);
    packed_padded.extend_from_slice(&[0u8; 4]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(packed_padded.len()).unwrap() };
    cuda.memcpy_htod(&packed_padded, &mut d_packed).unwrap();
    let mut d_sub_states = unsafe { cuda.alloc::<u16>(sub_states_host.len()).unwrap() };
    cuda.memcpy_htod(&sub_states_host, &mut d_sub_states).unwrap();
    let mut d_sub_nb = unsafe { cuda.alloc::<u16>(sub_nb_totals.len()).unwrap() };
    cuda.memcpy_htod(&sub_nb_totals, &mut d_sub_nb).unwrap();
    let mut d_bs = unsafe { cuda.alloc::<u64>(bs_words.len()).unwrap() };
    cuda.memcpy_htod(&bs_words, &mut d_bs).unwrap();
    let mut d_subofs = unsafe { cuda.alloc::<u64>(sub_bit_offsets.len()).unwrap() };
    cuda.memcpy_htod(&sub_bit_offsets, &mut d_subofs).unwrap();

    let mut bp = Vec::with_capacity(payload.beta_idx_packed.len() + 4);
    bp.extend_from_slice(payload.beta_idx_packed);
    bp.extend_from_slice(&[0u8; 4]);
    let mut d_bidx = unsafe { cuda.alloc::<u8>(bp.len()).unwrap() };
    cuda.memcpy_htod(&bp, &mut d_bidx).unwrap();

    let (mut d_oidx, oidx_ptr_const) = if has_offset {
        let mut op = Vec::with_capacity(payload.offset_idx_packed.len() + 4);
        op.extend_from_slice(payload.offset_idx_packed);
        op.extend_from_slice(&[0u8; 4]);
        let mut d = unsafe { cuda.alloc::<u8>(op.len()).unwrap() };
        cuda.memcpy_htod(&op, &mut d).unwrap();
        let ptr = d.device_ptr(d.stream()).0;
        (Some(d), ptr as *const u8)
    } else {
        (None, std::ptr::null::<u8>())
    };

    let mut d_beta_lloyd = unsafe { cuda.alloc::<f32>(beta_lloyd_host.len()).unwrap() };
    cuda.memcpy_htod(&beta_lloyd_host, &mut d_beta_lloyd).unwrap();
    let (mut d_offset_lloyd, off_lloyd_ptr_const) = if has_offset {
        let mut d = unsafe { cuda.alloc::<f32>(offset_lloyd_host.len()).unwrap() };
        cuda.memcpy_htod(&offset_lloyd_host, &mut d).unwrap();
        let ptr = d.device_ptr(d.stream()).0;
        (Some(d), ptr as *const f32)
    } else {
        (None, std::ptr::null::<f32>())
    };

    let a_u16: Vec<u16> = a_host.iter().map(|x| x.to_bits()).collect();
    let mut d_a = unsafe { cuda.alloc::<u16>(k_total).unwrap() };
    cuda.memcpy_htod(&a_u16, &mut d_a).unwrap();

    let d_out = unsafe { cuda.alloc::<u16>(r as usize).unwrap() };
    let mut d_acc = unsafe { cuda.alloc::<f32>(r as usize).unwrap() };
    let zeros = vec![0.0f32; r as usize];
    cuda.memcpy_htod(&zeros, &mut d_acc).unwrap();

    let packed_ptr = d_packed.device_ptr(d_packed.stream()).0 as *const u8;
    let sub_states_ptr = d_sub_states.device_ptr(d_sub_states.stream()).0 as *const u16;
    let sub_nb_ptr = d_sub_nb.device_ptr(d_sub_nb.stream()).0 as *const u16;
    let bs_ptr = d_bs.device_ptr(d_bs.stream()).0 as *const u64;
    let subofs_ptr = d_subofs.device_ptr(d_subofs.stream()).0 as *const u64;
    let bidx_ptr = d_bidx.device_ptr(d_bidx.stream()).0 as *const u8;
    let beta_lloyd_ptr = d_beta_lloyd.device_ptr(d_beta_lloyd.stream()).0 as *const f32;
    let a_ptr = d_a.device_ptr(d_a.stream()).0 as *const c_void;
    let out_ptr = d_out.device_ptr(d_out.stream()).0 as *mut c_void;
    let acc_ptr = d_acc.device_ptr(d_acc.stream()).0 as *mut f32;

    let run_once = || unsafe {
        leech_q24_gemv_bf16_v4(
            a_ptr, packed_ptr,
            sub_states_ptr, sub_nb_ptr, bs_ptr, subofs_ptr,
            bidx_ptr, oidx_ptr_const,
            beta_lloyd_ptr, off_lloyd_ptr_const,
            acc_ptr, out_ptr,
            r, b, n_blocks, n_tiles, k_beta, k_offset,
            w_offset, tile_size, test_k, has_offset,
            std::ptr::null_mut::<c_void>(),
        ).expect("gemv_v4 launch")
    };

    run_once();
    cuda.synchronize().expect("sync");

    let mut host_out_u16 = vec![0u16; r as usize];
    cuda.memcpy_dtoh(&d_out, &mut host_out_u16).expect("dtoh");
    let host_out: Vec<bf16> = host_out_u16.iter().copied().map(bf16::from_bits).collect();

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut n_bad = 0usize;
    for (i, (g, r_ref)) in host_out.iter().zip(y_ref_bf16.iter()).enumerate() {
        let gf = g.to_f32();
        let rf = r_ref.to_f32();
        let diff = (gf - rf).abs();
        let rel = if rf.abs() > 1e-6 { diff / rf.abs() } else { diff };
        if diff > max_abs_diff { max_abs_diff = diff; }
        if rel > max_rel_diff { max_rel_diff = rel; }
        if rel > 0.05 && diff > 0.01 {
            if n_bad < 8 {
                eprintln!("  diff @{i}: gpu={gf} ref={rf} rel={rel}");
            }
            n_bad += 1;
        }
    }
    eprintln!(
        "[v4 K={}] max_abs_diff={:.6e} max_rel_diff={:.6e} bad={}/{}",
        test_k, max_abs_diff, max_rel_diff, n_bad, host_out.len()
    );
    assert_eq!(n_bad, 0, "[v4 K={test_k}] {n_bad} mismatched outputs");

    if std::env::var("LEECHQ24_BENCH").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false) {
        let iters: u32 = std::env::var("LEECHQ24_BENCH_ITERS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(200);
        for _ in 0..16 { run_once(); }
        cuda.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..iters { run_once(); }
        cuda.synchronize().expect("sync");
        let elapsed = t0.elapsed();
        let per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
        let flops = (n_blocks as f64) * 24.0 * 4.0 * iters as f64;
        let gflops = flops / elapsed.as_secs_f64() / 1e9;
        eprintln!(
            "BENCH v4 K={} T={} (R={}, B={}, n_blocks={}):",
            test_k, tile_size, r, b, n_blocks
        );
        eprintln!("    iters:       {}", iters);
        eprintln!("    per-call:    {:.1} µs", per_call_us);
        eprintln!("    GFLOPs:      {:.1}", gflops);
        eprintln!("    speedup vs T=4 K=1 baseline (1414.5 µs): {:.2}×", 1414.5 / per_call_us);
    }

    drop(d_packed); drop(d_sub_states); drop(d_sub_nb); drop(d_bs); drop(d_subofs);
    drop(d_bidx); let _ = d_oidx.take();
    drop(d_beta_lloyd); let _ = d_offset_lloyd.take();
    drop(d_a); drop(d_out); drop(d_acc);
}
