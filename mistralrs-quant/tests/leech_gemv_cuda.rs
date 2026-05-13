//! Phase 4.0b acceptance: leech_gemv_bf16 must produce the same output as
//! (dequantize via leech_decode_bf16) → standard bf16 GEMM.
//!
//! Tolerance: small fp32-accumulator order differences may cause ULP-level
//! drift, so we check element-wise max-abs-diff < threshold.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::Device;
use half::bf16;
use mistralrs_leech::{LeechFile, Role};
use mistralrs_quant::leech::{leech_compute_block_parity, leech_decode_bf16, leech_gemv_bf16};

fn fixture_path() -> Option<PathBuf> {
    let p = std::env::var("LEECH_FIXTURE").unwrap_or_else(|_| "/tmp/qwopus.leech".to_owned());
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

fn pick_tensor<'a>(leech: &'a LeechFile, name: &str) -> usize {
    leech
        .toc()
        .iter()
        .position(|e| e.name == name)
        .expect("tensor not in TOC")
}

/// Reference: dequantize weight via leech_decode_bf16, then bf16 GEMV on host.
/// host-only — predictable, no second kernel involved in the reference path.
fn gemv_ref(weight_bf16: &[bf16], a: &[bf16], m: usize, n: usize, k: usize) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; m * n];
    for mi in 0..m {
        let a_row = &a[mi * k..(mi + 1) * k];
        for ni in 0..n {
            let w_row = &weight_bf16[ni * k..(ni + 1) * k];
            let mut acc: f32 = 0.0;
            for j in 0..k {
                let w_f = bf16::from_bits(w_row[j].to_bits()).to_f32();
                let a_f = bf16::from_bits(a_row[j].to_bits()).to_f32();
                acc += a_f * w_f;
            }
            out[mi * n + ni] = bf16::from_f32(acc);
        }
    }
    out
}

#[test]
fn gemv_bf16_matches_dequant_matmul() {
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE missing — skipping");
        return;
    };
    let leech = LeechFile::open(&leech_path).expect("open .leech");

    // Small tensor — layers.0.linear_attn.in_proj_a: R=32, B=170 → K_total=4080.
    let target = "model.language_model.layers.0.linear_attn.in_proj_a.weight";
    let toc_idx = pick_tensor(&leech, target);
    let entry = &leech.toc()[toc_idx];
    assert_eq!(entry.role, Role::Llvq);
    let payload = leech.parse_llvq_payload(toc_idx).expect("parse");

    let n_rows = payload.r as usize;
    let b_blocks = payload.b as usize;
    let k_total = b_blocks * 24;
    let m: usize = 4; // small batch to verify both M=1 and M>1 paths

    eprintln!(
        "tensor: {}  N_rows={} K_total={} M(batch)={}",
        target, n_rows, k_total, m
    );

    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    // Upload weight inputs.
    let mut padded = Vec::with_capacity(payload.packed_stream.len() + 8);
    padded.extend_from_slice(payload.packed_stream);
    padded.extend_from_slice(&[0u8; 8]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(padded.len()).expect("alloc packed") };
    cuda.memcpy_htod(&padded, &mut d_packed).expect("htod packed");

    let beta_host = payload.beta_codebook_f16();
    let bbits: Vec<u16> = beta_host.iter().map(|x| x.to_bits()).collect();
    let mut d_beta = unsafe { cuda.alloc::<u16>(bbits.len()).expect("alloc beta") };
    cuda.memcpy_htod(&bbits, &mut d_beta).expect("htod beta");

    let offset_host = payload.offset_codebook_f16();
    let d_offset = if payload.has_offset {
        let obits: Vec<u16> = offset_host.iter().map(|x| x.to_bits()).collect();
        let mut d = unsafe { cuda.alloc::<u16>(obits.len()).expect("alloc offset") };
        cuda.memcpy_htod(&obits, &mut d).expect("htod offset");
        Some(d)
    } else {
        None
    };

    // Generate deterministic bf16 activations.
    let mut a_host: Vec<bf16> = Vec::with_capacity(m * k_total);
    let mut seed: u64 = 0xC0FFEE_BEEF_5678_u64;
    for _ in 0..(m * k_total) {
        // xorshift64 deterministic PRNG
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Map to [-1, 1) range via 24 LSBs
        let scaled = ((seed & 0xFF_FFFF) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        a_host.push(bf16::from_f32(scaled));
    }
    let a_bits: Vec<u16> = a_host.iter().map(|x| x.to_bits()).collect();
    let mut d_a = unsafe { cuda.alloc::<u16>(a_bits.len()).expect("alloc a") };
    cuda.memcpy_htod(&a_bits, &mut d_a).expect("htod a");

    // ── Path 1: dequantize → host GEMV reference ─────────────────────────
    let d_w = cuda
        .alloc_zeros::<bf16>(n_rows * k_total)
        .expect("alloc d_w");
    let stream = cuda.cuda_stream();
    let (packed_ptr, _g_p) = d_packed.device_ptr(&stream);
    let (beta_ptr, _g_b) = d_beta.device_ptr(&stream);
    let offset_ptr_u64: u64 = match d_offset.as_ref() {
        Some(d) => {
            let (p, _) = d.device_ptr(&stream);
            p
        }
        None => 0,
    };
    let (w_ptr, _g_w) = d_w.device_ptr(&stream);
    let stream_raw = stream.cu_stream() as *mut c_void;
    unsafe {
        leech_decode_bf16(
            packed_ptr as *const u8,
            beta_ptr as *const c_void,
            offset_ptr_u64 as *const c_void,
            w_ptr as *mut c_void,
            n_rows as u32,
            b_blocks as u32,
            payload.k_beta as u32,
            payload.k_offset as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("decode_bf16");
    }
    device.synchronize().expect("sync");
    let mut w_host: Vec<bf16> = vec![bf16::ZERO; n_rows * k_total];
    cuda.memcpy_dtoh(&d_w, &mut w_host).expect("dtoh w");
    let ref_out = gemv_ref(&w_host, &a_host, m, n_rows, k_total);

    // ── Path 2: leech_gemv_bf16 fused kernel ─────────────────────────────
    let d_y = cuda.alloc_zeros::<bf16>(m * n_rows).expect("alloc y");
    let (a_ptr, _g_a) = d_a.device_ptr(&stream);
    let (y_ptr, _g_y) = d_y.device_ptr(&stream);
    unsafe {
        leech_gemv_bf16(
            a_ptr as *const c_void,
            packed_ptr as *const u8,
            beta_ptr as *const c_void,
            offset_ptr_u64 as *const c_void,
            std::ptr::null::<c_void>(),
            y_ptr as *mut c_void,
            m as u32,
            n_rows as u32,
            b_blocks as u32,
            payload.k_beta as u32,
            payload.k_offset as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("gemv");
    }
    device.synchronize().expect("sync");
    let mut got_out: Vec<bf16> = vec![bf16::ZERO; m * n_rows];
    cuda.memcpy_dtoh(&d_y, &mut got_out).expect("dtoh y");

    // ── Compare ──────────────────────────────────────────────────────────
    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for i in 0..(m * n_rows) {
        let g = got_out[i].to_f32();
        let r = ref_out[i].to_f32();
        let abs_diff = (g - r).abs();
        let rel_diff = if r.abs() > 1e-3 {
            abs_diff / r.abs()
        } else {
            abs_diff
        };
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
        if first_bad.is_none() && rel_diff > 0.05 && abs_diff > 0.01 {
            first_bad = Some((i, g, r));
        }
    }
    eprintln!(
        "max_abs_diff = {:.6e}   max_rel_diff = {:.6e}",
        max_abs_diff, max_rel_diff
    );
    if let Some((i, g, r)) = first_bad {
        eprintln!("First > 5% rel diff at index {i}: got {g}  ref {r}");
    }
    // Tolerance: bf16 GEMV accumulating ~4080 fp32 products has expected
    // ULP-level drift due to non-associativity. Allow up to 1% relative on
    // outputs > 1e-3, with max abs diff capped at output range scale.
    assert!(
        max_rel_diff < 0.01 || max_abs_diff < 0.05,
        "gemv output drifted: max_rel={max_rel_diff} max_abs={max_abs_diff}"
    );
}

/// Attack vector #1: parity-sort. Build a parity-sorted permutation and verify
/// that leech_gemv_bf16 with parity_perm produces the same output (within
/// fp32-accumulator order tolerance) as the unpermuted call.
#[test]
fn gemv_bf16_parity_perm_matches_no_perm() {
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE not set / missing — skipping");
        return;
    };
    let leech = LeechFile::open(&leech_path).expect("open .leech");
    let toc_idx = pick_tensor(&leech, "model.language_model.layers.0.linear_attn.in_proj_a.weight");
    let entry = &leech.toc()[toc_idx];
    assert_eq!(entry.role, Role::Llvq);
    let payload = leech.parse_llvq_payload(toc_idx).expect("parse");

    let n_rows = payload.r as usize;
    let b_blocks = payload.b as usize;
    let k_total = b_blocks * 24;
    let m: usize = 4;

    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    let mut padded = Vec::with_capacity(payload.packed_stream.len() + 8);
    padded.extend_from_slice(payload.packed_stream);
    padded.extend_from_slice(&[0u8; 8]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(padded.len()).expect("alloc") };
    cuda.memcpy_htod(&padded, &mut d_packed).expect("htod");

    let beta_host = payload.beta_codebook_f16();
    let bbits: Vec<u16> = beta_host.iter().map(|x| x.to_bits()).collect();
    let mut d_beta = unsafe { cuda.alloc::<u16>(bbits.len()).expect("alloc") };
    cuda.memcpy_htod(&bbits, &mut d_beta).expect("htod");

    let offset_host = payload.offset_codebook_f16();
    let d_offset = if payload.has_offset {
        let obits: Vec<u16> = offset_host.iter().map(|x| x.to_bits()).collect();
        let mut d = unsafe { cuda.alloc::<u16>(obits.len()).expect("alloc") };
        cuda.memcpy_htod(&obits, &mut d).expect("htod");
        Some(d)
    } else {
        None
    };

    let mut a_host: Vec<bf16> = Vec::with_capacity(m * k_total);
    let mut seed: u64 = 0xABCDEF_1234_5678_u64;
    for _ in 0..(m * k_total) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let scaled = ((seed & 0xFF_FFFF) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        a_host.push(bf16::from_f32(scaled));
    }
    let a_bits: Vec<u16> = a_host.iter().map(|x| x.to_bits()).collect();
    let mut d_a = unsafe { cuda.alloc::<u16>(a_bits.len()).expect("alloc a") };
    cuda.memcpy_htod(&a_bits, &mut d_a).expect("htod a");

    let stream = cuda.cuda_stream();
    let (packed_ptr, _g_p) = d_packed.device_ptr(&stream);
    let (beta_ptr, _g_b) = d_beta.device_ptr(&stream);
    let offset_ptr_u64: u64 = match d_offset.as_ref() {
        Some(d) => {
            let (p, _) = d.device_ptr(&stream);
            p
        }
        None => 0,
    };
    let (a_ptr, _g_a) = d_a.device_ptr(&stream);
    let stream_raw = stream.cu_stream() as *mut c_void;

    // ── Path A: no parity_perm
    let d_y_no = cuda.alloc_zeros::<bf16>(m * n_rows).expect("alloc y_no");
    let (y_no_ptr, _g_yn) = d_y_no.device_ptr(&stream);
    unsafe {
        leech_gemv_bf16(
            a_ptr as *const c_void,
            packed_ptr as *const u8,
            beta_ptr as *const c_void,
            offset_ptr_u64 as *const c_void,
            std::ptr::null::<c_void>(),
            y_no_ptr as *mut c_void,
            m as u32,
            n_rows as u32,
            b_blocks as u32,
            payload.k_beta as u32,
            payload.k_offset as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("gemv no perm");
    }
    device.synchronize().expect("sync");

    // ── Path B: build parity perm, then GEMV
    let n_blocks = n_rows * b_blocks;
    let d_parity = cuda.alloc_zeros::<u8>(n_blocks).expect("alloc parity");
    let (par_ptr_u64, _par_guard) = d_parity.device_ptr(&stream);
    unsafe {
        leech_compute_block_parity(
            packed_ptr as *const u8,
            par_ptr_u64 as *mut u8,
            n_blocks as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("parity");
    }
    device.synchronize().expect("sync parity");
    drop(_par_guard);
    let mut parity_host: Vec<u8> = vec![0u8; n_blocks];
    cuda.memcpy_dtoh(&d_parity, &mut parity_host).expect("dtoh parity");

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
    // Verify it's a valid permutation per row.
    for r in 0..n_rows {
        let row_perm = &perm_host[r * b_blocks..(r + 1) * b_blocks];
        let mut seen = vec![false; b_blocks];
        for &k in row_perm {
            assert!((k as usize) < b_blocks, "perm entry out of range");
            assert!(!seen[k as usize], "duplicate index in row {r}");
            seen[k as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "missing index in row {r}");
    }

    let mut d_perm = unsafe { cuda.alloc::<u16>(n_blocks).expect("alloc perm") };
    cuda.memcpy_htod(&perm_host, &mut d_perm).expect("htod perm");
    let (perm_ptr_u64, _perm_guard) = d_perm.device_ptr(&stream);

    let d_y_p = cuda.alloc_zeros::<bf16>(m * n_rows).expect("alloc y_p");
    let (y_p_ptr, _g_yp) = d_y_p.device_ptr(&stream);
    unsafe {
        leech_gemv_bf16(
            a_ptr as *const c_void,
            packed_ptr as *const u8,
            beta_ptr as *const c_void,
            offset_ptr_u64 as *const c_void,
            perm_ptr_u64 as *const c_void,
            y_p_ptr as *mut c_void,
            m as u32,
            n_rows as u32,
            b_blocks as u32,
            payload.k_beta as u32,
            payload.k_offset as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("gemv with perm");
    }
    device.synchronize().expect("sync");
    drop(_perm_guard);

    // Compare bf16 outputs.
    let mut out_no: Vec<bf16> = vec![bf16::ZERO; m * n_rows];
    let mut out_p: Vec<bf16> = vec![bf16::ZERO; m * n_rows];
    cuda.memcpy_dtoh(&d_y_no, &mut out_no).expect("dtoh no");
    cuda.memcpy_dtoh(&d_y_p, &mut out_p).expect("dtoh p");

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut n_exact: usize = 0;
    for i in 0..(m * n_rows) {
        let no = out_no[i].to_f32();
        let p = out_p[i].to_f32();
        if out_no[i].to_bits() == out_p[i].to_bits() {
            n_exact += 1;
        }
        let abs_diff = (no - p).abs();
        let rel_diff = if no.abs() > 1e-3 { abs_diff / no.abs() } else { abs_diff };
        if abs_diff > max_abs_diff { max_abs_diff = abs_diff; }
        if rel_diff > max_rel_diff { max_rel_diff = rel_diff; }
    }
    eprintln!(
        "parity_perm vs no perm:  {}/{} bit-exact   max_abs_diff = {:.3e}   max_rel_diff = {:.3e}",
        n_exact, m * n_rows, max_abs_diff, max_rel_diff
    );
    assert!(
        max_rel_diff < 0.02 || max_abs_diff < 0.05,
        "parity_perm drifted too far from no perm: max_rel={max_rel_diff} max_abs={max_abs_diff}"
    );
}

/// Microbench batch=1 GEMV on a large MLP-style tensor.
#[test]
fn bench_gemv_bf16_batch1() {
    if std::env::var("LEECH_BENCH_GEMV").is_err() {
        eprintln!("LEECH_BENCH_GEMV not set — skipping");
        return;
    }
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE missing — skipping");
        return;
    };
    let leech = LeechFile::open(&leech_path).expect("open .leech");
    let target = "model.language_model.layers.0.mlp.up_proj.weight";
    let toc_idx = pick_tensor(&leech, target);
    let entry = &leech.toc()[toc_idx];
    let payload = leech.parse_llvq_payload(toc_idx).expect("parse");

    let n_rows = payload.r as usize;
    let b_blocks = payload.b as usize;
    let k_total = b_blocks * 24;
    let m: usize = 1;

    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    let mut padded = Vec::with_capacity(payload.packed_stream.len() + 8);
    padded.extend_from_slice(payload.packed_stream);
    padded.extend_from_slice(&[0u8; 8]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(padded.len()).expect("alloc") };
    cuda.memcpy_htod(&padded, &mut d_packed).expect("htod");

    let beta_host = payload.beta_codebook_f16();
    let bbits: Vec<u16> = beta_host.iter().map(|x| x.to_bits()).collect();
    let mut d_beta = unsafe { cuda.alloc::<u16>(bbits.len()).expect("alloc") };
    cuda.memcpy_htod(&bbits, &mut d_beta).expect("htod");

    let offset_host = payload.offset_codebook_f16();
    let d_offset = if payload.has_offset {
        let obits: Vec<u16> = offset_host.iter().map(|x| x.to_bits()).collect();
        let mut d = unsafe { cuda.alloc::<u16>(obits.len()).expect("alloc") };
        cuda.memcpy_htod(&obits, &mut d).expect("htod");
        Some(d)
    } else {
        None
    };

    let mut a_bits = vec![0u16; m * k_total];
    let mut seed: u64 = 0xDEADBEEF_CAFEBABEu64;
    for v in a_bits.iter_mut() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let f = ((seed & 0xFF_FFFF) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        *v = bf16::from_f32(f).to_bits();
    }
    let mut d_a = unsafe { cuda.alloc::<u16>(a_bits.len()).expect("alloc") };
    cuda.memcpy_htod(&a_bits, &mut d_a).expect("htod a");

    let d_y = cuda.alloc_zeros::<bf16>(m * n_rows).expect("alloc y");

    let stream = cuda.cuda_stream();
    let (packed_ptr, _g_p) = d_packed.device_ptr(&stream);
    let (beta_ptr, _g_b) = d_beta.device_ptr(&stream);
    let offset_ptr_u64: u64 = match d_offset.as_ref() {
        Some(d) => {
            let (p, _) = d.device_ptr(&stream);
            p
        }
        None => 0,
    };
    let (a_ptr, _g_a) = d_a.device_ptr(&stream);
    let (y_ptr, _g_y) = d_y.device_ptr(&stream);
    let stream_raw = stream.cu_stream() as *mut c_void;

    let warmup = 10;
    let iters = 200;
    for _ in 0..warmup {
        unsafe {
            leech_gemv_bf16(
                a_ptr as *const c_void,
                packed_ptr as *const u8,
                beta_ptr as *const c_void,
                offset_ptr_u64 as *const c_void,
                std::ptr::null::<c_void>(),
                y_ptr as *mut c_void,
                m as u32,
                n_rows as u32,
                b_blocks as u32,
                payload.k_beta as u32,
                payload.k_offset as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
    }
    device.synchronize().expect("sync");

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            leech_gemv_bf16(
                a_ptr as *const c_void,
                packed_ptr as *const u8,
                beta_ptr as *const c_void,
                offset_ptr_u64 as *const c_void,
                std::ptr::null::<c_void>(),
                y_ptr as *mut c_void,
                m as u32,
                n_rows as u32,
                b_blocks as u32,
                payload.k_beta as u32,
                payload.k_offset as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
    }
    device.synchronize().expect("sync");
    let elapsed = t0.elapsed();

    let per_call_us_noperm = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let flops_per_iter = 2.0 * m as f64 * n_rows as f64 * k_total as f64; // 2*M*N*K mult-adds
    let tflops_noperm = flops_per_iter * iters as f64 / elapsed.as_secs_f64() / 1e12;
    // HBM bytes read (lower bound): packed_stream + small codebooks + activations
    let packed_bytes = payload.packed_stream.len() as f64;
    let act_bytes = (m * k_total * 2) as f64;
    let hbm_bytes_per_iter = packed_bytes + act_bytes;
    let hbm_gbps_noperm = hbm_bytes_per_iter * iters as f64 / elapsed.as_secs_f64() / 1e9;

    // ── Path B: build parity perm + run GEMV with parity_perm (attack vector #1)
    let n_blocks = n_rows * b_blocks;
    let d_parity = cuda.alloc_zeros::<u8>(n_blocks).expect("alloc parity");
    let (par_ptr_u64, _par_guard) = d_parity.device_ptr(&stream);
    unsafe {
        leech_compute_block_parity(
            packed_ptr as *const u8,
            par_ptr_u64 as *mut u8,
            n_blocks as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("parity");
    }
    device.synchronize().expect("sync parity");
    drop(_par_guard);
    let mut parity_host: Vec<u8> = vec![0u8; n_blocks];
    cuda.memcpy_dtoh(&d_parity, &mut parity_host).expect("dtoh parity");

    let mut perm_host: Vec<u16> = Vec::with_capacity(n_blocks);
    let mut n_even = 0usize;
    let mut n_odd = 0usize;
    for r in 0..n_rows {
        let row_par = &parity_host[r * b_blocks..(r + 1) * b_blocks];
        for (k, &p) in row_par.iter().enumerate() {
            if p == 0 {
                perm_host.push(k as u16);
                n_even += 1;
            }
        }
        for (k, &p) in row_par.iter().enumerate() {
            if p == 1 {
                perm_host.push(k as u16);
                n_odd += 1;
            }
        }
    }
    let mut d_perm = unsafe { cuda.alloc::<u16>(n_blocks).expect("alloc perm") };
    cuda.memcpy_htod(&perm_host, &mut d_perm).expect("htod perm");
    let (perm_ptr_u64, _perm_guard) = d_perm.device_ptr(&stream);

    // Warmup with parity perm
    for _ in 0..warmup {
        unsafe {
            leech_gemv_bf16(
                a_ptr as *const c_void,
                packed_ptr as *const u8,
                beta_ptr as *const c_void,
                offset_ptr_u64 as *const c_void,
                perm_ptr_u64 as *const c_void,
                y_ptr as *mut c_void,
                m as u32,
                n_rows as u32,
                b_blocks as u32,
                payload.k_beta as u32,
                payload.k_offset as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
    }
    device.synchronize().expect("sync");

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            leech_gemv_bf16(
                a_ptr as *const c_void,
                packed_ptr as *const u8,
                beta_ptr as *const c_void,
                offset_ptr_u64 as *const c_void,
                perm_ptr_u64 as *const c_void,
                y_ptr as *mut c_void,
                m as u32,
                n_rows as u32,
                b_blocks as u32,
                payload.k_beta as u32,
                payload.k_offset as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
    }
    device.synchronize().expect("sync");
    let elapsed_perm = t0.elapsed();
    drop(_perm_guard);

    let per_call_us_perm = elapsed_perm.as_secs_f64() * 1_000_000.0 / iters as f64;
    let tflops_perm = flops_per_iter * iters as f64 / elapsed_perm.as_secs_f64() / 1e12;
    let hbm_gbps_perm = hbm_bytes_per_iter * iters as f64 / elapsed_perm.as_secs_f64() / 1e9;
    let speedup = per_call_us_noperm / per_call_us_perm;
    let even_pct = 100.0 * n_even as f64 / n_blocks as f64;
    let odd_pct = 100.0 * n_odd as f64 / n_blocks as f64;

    eprintln!();
    eprintln!("=== BENCH: leech_gemv_bf16 (Phase 4.0b)  M={m} ===");
    eprintln!("  tensor:        {}", entry.name);
    eprintln!("  M={}  N={}  K={}", m, n_rows, k_total);
    eprintln!();
    eprintln!("  no parity_perm:");
    eprintln!("    per-call:    {:.1} µs", per_call_us_noperm);
    eprintln!("    TFLOPS:      {:.3}", tflops_noperm);
    eprintln!("    HBM read:    {:.3} GB/s", hbm_gbps_noperm);
    eprintln!();
    eprintln!("  with parity_perm (attack vector #1):");
    eprintln!("    parity mix:  {:.1}% even / {:.1}% odd", even_pct, odd_pct);
    eprintln!("    per-call:    {:.1} µs", per_call_us_perm);
    eprintln!("    TFLOPS:      {:.3}", tflops_perm);
    eprintln!("    HBM read:    {:.3} GB/s", hbm_gbps_perm);
    eprintln!();
    eprintln!("  SPEEDUP: {:.3}x  ({:.1} µs → {:.1} µs)",
             speedup, per_call_us_noperm, per_call_us_perm);
    eprintln!("  vs RTX 5090 HBM peak ~1.7 TB/s: {:.2}% (perm) vs {:.2}% (no perm)",
             100.0 * hbm_gbps_perm * 1e9 / 1.7e12,
             100.0 * hbm_gbps_noperm * 1e9 / 1.7e12);
}
