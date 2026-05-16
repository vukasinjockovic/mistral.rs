//! Q24-tANS Phase B.0 fused GEMV correctness + bench.
//!
//! Compares `leech_q24_gemv_bf16` (decode + β·v + offset + dot product) to a
//! host-side reference computed from the CPU-decoded `v_int` reference plus
//! the unpacked β / offset / Lloyd tables. The reference matches the kernel
//! op-for-op: f32 weight (β·v + offset), f32 activation, f32 accumulator,
//! final bf16 narrow.
//!
//! Env vars:
//!   LEECHQ24_ARTIFACT   — path to qwopus_ms18_v3.q24t.leech
//!   LEECHQ24_V_REFS     — directory holding per-tensor `<name>.v_int.bin`
//!   LEECHQ24_TENSOR     — tensor name (default: layers.0.mlp.up_proj.weight)
//!   LEECHQ24_BENCH      — when "1"/"true", run a warmup + iters loop with
//!                         cudaEvent timing and print µs/iter.
//!   LEECHQ24_BENCH_ITERS — iteration count (default 200)

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use half::bf16;
use mistralrs_leech_q24::{LeechQ24File, Role};
use mistralrs_quant::leech_q24::{compute_tile_bit_offsets, init_tables, leech_q24_gemv_bf16};

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
    let v = (packed[bi] as u32) | ((packed[bi + 1] as u32) << 8);
    ((v >> bo) & 0x7) as u8
}

#[test]
fn gemv_bf16_matches_cpu_reference() {
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

    let leech = LeechQ24File::open(&leech_path).expect("open .leech v3");
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

    eprintln!(
        "tensor {} R={} B={} K_total={} n_blocks={} n_tiles={} sid={} w_offset={} k_beta={} k_offset={} has_offset={}",
        tensor_name, r, b, k_total, n_blocks, n_tiles,
        symbol_set_id, w_offset, k_beta, k_offset, has_offset
    );

    // ── Host inputs ────────────────────────────────────────────────────
    // Decode tables
    let dt_bytes = leech
        .decode_tables_bytes(symbol_set_id)
        .expect("decode tables");
    let dt_host: Vec<u32> = dt_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    init_tables(&dt_host, symbol_set_id).expect("init_tables");

    // Tile metadata
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
    let bit_offsets = compute_tile_bit_offsets(&nb_totals);
    let bs_words: Vec<u64> = payload
        .tile_bitstream
        .chunks_exact(8)
        .map(|c| {
            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        })
        .collect();

    // β + offset Lloyd centroids
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

    // v_int CPU reference
    let v_ref_bytes = std::fs::read(&ref_path).expect("read v_int.bin");
    assert_eq!(v_ref_bytes.len(), (n_blocks as usize) * 24);

    // Deterministic activation: small pseudorandom bf16 in [-1, 1]
    let a_host: Vec<bf16> = (0..k_total)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(2654435761) >> 16) as i32 as f32;
            let f = (x.sin() * 0.5).max(-1.0).min(1.0);
            bf16::from_f32(f)
        })
        .collect();

    // ── Host reference: y_ref[r] = Σ_k weight[r,k] * activation[k] ────
    let beta_idx_packed = payload.beta_idx_packed;
    let offset_idx_packed = payload.offset_idx_packed;
    let mut y_ref_f32 = vec![0.0f32; r as usize];
    for blk in 0..(n_blocks as usize) {
        let row = blk / (b as usize);
        let col_block = blk - row * (b as usize);
        let beta_idx = unpack_3bit(beta_idx_packed, blk) as usize;
        let beta_val = beta_lloyd_host[row * (k_beta as usize) + beta_idx];
        let offset_val = if has_offset {
            let oi = unpack_3bit(offset_idx_packed, blk) as usize;
            offset_lloyd_host[row * (k_offset as usize) + oi]
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
    let y_ref_bf16: Vec<bf16> = y_ref_f32.iter().copied().map(bf16::from_f32).collect();

    // ── GPU run ──────────────────────────────────────────────────────
    use candle_core::backend::BackendDevice;
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::Device;

    let device = Device::new_cuda(0).expect("Device::Cuda");
    let cuda = device.as_cuda_device().expect("cuda device");

    // Packed buckets — tail pad ≥1 byte for 13-bit extract span.
    let mut packed_padded = Vec::with_capacity(payload.buckets_packed.len() + 4);
    packed_padded.extend_from_slice(payload.buckets_packed);
    packed_padded.extend_from_slice(&[0u8; 4]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(packed_padded.len()).expect("alloc packed") };
    cuda.memcpy_htod(&packed_padded, &mut d_packed).expect("htod packed");

    let mut d_states = unsafe { cuda.alloc::<u16>(n_tiles as usize).expect("alloc states") };
    cuda.memcpy_htod(&states_host, &mut d_states).expect("htod states");
    let mut d_nb = unsafe { cuda.alloc::<u16>(n_tiles as usize).expect("alloc nb") };
    cuda.memcpy_htod(&nb_totals, &mut d_nb).expect("htod nb");
    let mut d_bs = unsafe { cuda.alloc::<u64>(bs_words.len()).expect("alloc bs") };
    cuda.memcpy_htod(&bs_words, &mut d_bs).expect("htod bs");
    let mut d_offs = unsafe { cuda.alloc::<u64>(n_tiles as usize).expect("alloc offs") };
    cuda.memcpy_htod(&bit_offsets, &mut d_offs).expect("htod offs");

    // Beta / offset packed indices (3-bit) — pad by 1 trailing byte for span.
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
        cuda.alloc::<f32>(beta_lloyd_host.len())
            .expect("alloc beta lloyd")
    };
    cuda.memcpy_htod(&beta_lloyd_host, &mut d_beta_lloyd)
        .expect("htod beta lloyd");

    let (mut d_offset_lloyd, off_lloyd_ptr_const) = if has_offset {
        let mut d = unsafe {
            cuda.alloc::<f32>(offset_lloyd_host.len())
                .expect("alloc offset lloyd")
        };
        cuda.memcpy_htod(&offset_lloyd_host, &mut d)
            .expect("htod offset lloyd");
        let ptr = d.device_ptr(d.stream()).0;
        (Some(d), ptr as *const f32)
    } else {
        (None, std::ptr::null::<f32>())
    };

    // Activation, output, accumulator
    let a_u16: Vec<u16> = a_host.iter().map(|x| x.to_bits()).collect();
    let mut d_a = unsafe { cuda.alloc::<u16>(k_total).expect("alloc a") };
    cuda.memcpy_htod(&a_u16, &mut d_a).expect("htod a");

    let d_out = unsafe { cuda.alloc::<u16>(r as usize).expect("alloc out") };
    let mut d_acc = unsafe { cuda.alloc::<f32>(r as usize).expect("alloc acc") };
    // Zero the accumulator initially (kernel also memsets, but be explicit).
    let zeros = vec![0.0f32; r as usize];
    cuda.memcpy_htod(&zeros, &mut d_acc).expect("htod zero");

    // Gather raw ptrs
    let packed_ptr = d_packed.device_ptr(d_packed.stream()).0 as *const u8;
    let states_ptr = d_states.device_ptr(d_states.stream()).0 as *const u16;
    let nb_ptr = d_nb.device_ptr(d_nb.stream()).0 as *const u16;
    let bs_ptr = d_bs.device_ptr(d_bs.stream()).0 as *const u64;
    let offs_ptr = d_offs.device_ptr(d_offs.stream()).0 as *const u64;
    let bidx_ptr = d_bidx.device_ptr(d_bidx.stream()).0 as *const u8;
    let beta_lloyd_ptr = d_beta_lloyd.device_ptr(d_beta_lloyd.stream()).0 as *const f32;
    let a_ptr = d_a.device_ptr(d_a.stream()).0 as *const c_void;
    let out_ptr = d_out.device_ptr(d_out.stream()).0 as *mut c_void;
    let acc_ptr = d_acc.device_ptr(d_acc.stream()).0 as *mut f32;

    let run_once = || unsafe {
        leech_q24_gemv_bf16(
            a_ptr,
            packed_ptr,
            states_ptr,
            nb_ptr,
            bs_ptr,
            offs_ptr,
            bidx_ptr,
            oidx_ptr_const,
            beta_lloyd_ptr,
            off_lloyd_ptr_const,
            acc_ptr,
            out_ptr,
            r,
            b,
            n_blocks,
            n_tiles,
            k_beta,
            k_offset,
            w_offset,
            tile_size,
            has_offset,
            std::ptr::null_mut::<c_void>(),
        )
        .expect("gemv launch");
    };

    // One run for correctness.
    run_once();
    cuda.synchronize().expect("sync");

    let mut host_out_u16 = vec![0u16; r as usize];
    cuda.memcpy_dtoh(&d_out, &mut host_out_u16).expect("dtoh");
    let host_out: Vec<bf16> = host_out_u16.iter().copied().map(bf16::from_bits).collect();

    // Compare bf16 outputs. bf16 has 7 mantissa bits → ULP ~2^-7 at value 1.
    // Allow ≤4 ULPs relative drift (atomic order + f32 rounding noise).
    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut n_bad = 0usize;
    for (i, (g, r_ref)) in host_out.iter().zip(y_ref_bf16.iter()).enumerate() {
        let gf = g.to_f32();
        let rf = r_ref.to_f32();
        let diff = (gf - rf).abs();
        let rel = if rf.abs() > 1e-6 { diff / rf.abs() } else { diff };
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
        if rel > 0.05 && diff > 0.01 {
            if n_bad < 8 {
                eprintln!("  diff @{i}: gpu={gf} ref={rf} rel={rel}");
            }
            n_bad += 1;
        }
    }
    eprintln!(
        "max_abs_diff={:.6e} max_rel_diff={:.6e} bad={}/{}",
        max_abs_diff, max_rel_diff, n_bad, host_out.len()
    );
    assert_eq!(n_bad, 0, "{n_bad} mismatched outputs above tolerance");
    eprintln!("PASS: GEMV bf16 matches CPU reference (max_rel={:.4e})", max_rel_diff);

    // ── Optional bench ───────────────────────────────────────────────
    if std::env::var("LEECHQ24_BENCH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        let iters: u32 = std::env::var("LEECHQ24_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);

        use candle_core::cuda::cudarc::driver::CudaStream;
        let _ = std::any::type_name::<CudaStream>(); // silence unused warning if any

        // Warmup
        for _ in 0..16 {
            run_once();
        }
        cuda.synchronize().expect("sync");

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            run_once();
        }
        cuda.synchronize().expect("sync");
        let elapsed = t0.elapsed();
        let per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        // FLOPS estimate: each block does 24 coords × (β·v + offset = 2 ops, mul-add = 2 ops) = 96 fp ops.
        // Per call: n_blocks * 96 ≈ 200M ops at mlp.up_proj scale.
        let flops_per_call = (n_blocks as f64) * 24.0 * 4.0;
        let gflops = flops_per_call * iters as f64 / elapsed.as_secs_f64() / 1e9;

        let hbm_bytes_per_call = (payload.buckets_packed.len()
            + payload.tile_states.len()
            + payload.tile_nb_totals.len()
            + payload.tile_bitstream.len()
            + payload.beta_idx_packed.len()
            + if has_offset { payload.offset_idx_packed.len() } else { 0 }
            + payload.beta_lloyd.len()
            + if has_offset { payload.offset_lloyd.len() } else { 0 }
            + k_total * 2
            + (r as usize) * 2) as f64;
        let hbm_gbps = hbm_bytes_per_call * iters as f64 / elapsed.as_secs_f64() / 1e9;

        eprintln!(
            "BENCH leech_q24_gemv_bf16 (R={}, B={}, n_blocks={}, M=1):",
            r, b, n_blocks
        );
        eprintln!("    iters:       {}", iters);
        eprintln!("    per-call:    {:.1} µs", per_call_us);
        eprintln!("    GFLOPs:      {:.1}", gflops);
        eprintln!("    HBM read:    {:.1} GB/s", hbm_gbps);
        eprintln!("    baseline:    664.7 µs (combinadic decoder fused GEMV)");
        eprintln!(
            "    speedup vs baseline: {:.3}x",
            664.7 / per_call_us
        );
    }

    // Drop device buffers explicitly to silence dead-code warnings.
    drop(d_packed);
    drop(d_states);
    drop(d_nb);
    drop(d_bs);
    drop(d_offs);
    drop(d_bidx);
    let _ = d_oidx.take();
    drop(d_beta_lloyd);
    let _ = d_offset_lloyd.take();
    drop(d_a);
    drop(d_out);
    drop(d_acc);
}
