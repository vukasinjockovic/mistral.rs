//! Phase 4.0a acceptance gate: leech_decode_bf16 (fused decode + β·v + offset)
//! must produce bf16 weights exactly equal (bit-identical) to the reference
//! constructed from:
//!   1. The already-validated Phase 3 decode_v_int kernel → int8 v_int
//!   2. Per-block β/offset lookup from packed_stream + beta/offset codebooks
//!   3. The LOCKED epilogue: bf16 = RNE(fp32(β * v_int + offset))
//!
//! Both paths use the SAME op order, so any difference is a bug.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::Device;
use half::{bf16, f16};
use mistralrs_leech::{LeechFile, Role};
use mistralrs_quant::leech::{leech_decode_bf16, leech_decode_v_int};

fn fixture_path() -> Option<PathBuf> {
    let p = std::env::var("LEECH_FIXTURE").unwrap_or_else(|_| "/tmp/qwopus.leech".to_owned());
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

/// Read W ≤ 60 bits from `packed` starting at `bit_off`. Matches the
/// MSB-first-within-byte layout used by the CUDA bit_extract.
fn extract_bits(packed: &[u8], bit_off: u64, w: u32) -> u64 {
    let byte_off = (bit_off / 8) as usize;
    let bit_in_byte = (bit_off % 8) as u32;
    // We need (bit_in_byte + W) bits from byte_off onward. Load up to 16 bytes
    // and pack into a u128 MSB-first.
    let mut acc: u128 = 0;
    for i in 0..16usize {
        let b = if byte_off + i < packed.len() {
            packed[byte_off + i] as u128
        } else {
            0u128
        };
        acc = (acc << 8) | b;
    }
    let shift = 128u32 - bit_in_byte - w;
    let mask: u64 = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    ((acc >> shift) as u64) & mask
}

/// Decompose one block's W bits into (i_global, beta_idx, offset_idx).
/// Layout LSB→MSB: [offset_idx (OFFSET_BITS) | beta_idx (BETA_BITS) | i_global (IDX_BITS)]
fn unpack_block(
    packed: &[u8],
    block_id: u64,
    idx_bits: u32,
    has_offset: bool,
) -> (u64, u32, u32) {
    let offset_bits: u32 = if has_offset { 3 } else { 0 };
    let beta_bits: u32 = 3;
    let w = idx_bits + beta_bits + offset_bits;
    let bit_off = block_id * (w as u64);
    let mut val = extract_bits(packed, bit_off, w);

    let offset_idx = if has_offset {
        let v = (val & ((1u64 << offset_bits) - 1)) as u32;
        val >>= offset_bits;
        v
    } else {
        0
    };
    let beta_idx = (val & ((1u64 << beta_bits) - 1)) as u32;
    val >>= beta_bits;
    let i_global = val & if idx_bits == 64 { u64::MAX } else { (1u64 << idx_bits) - 1 };
    (i_global, beta_idx, offset_idx)
}

/// Same LOCKED epilogue as the kernel: bf16 = RNE(fp32(β * v_int + offset)).
fn epilogue_bf16(v_int: i8, beta: f32, offset: f32) -> bf16 {
    let v_fp32 = beta * (v_int as f32) + offset;
    bf16::from_f32(v_fp32)
}

/// Validate ONE tensor end-to-end. Returns (n_compared, n_mismatch, first_bad).
fn check_one_tensor(
    leech: &LeechFile,
    toc_idx: usize,
    device: &Device,
    cuda: &candle_core::CudaDevice,
) -> (usize, usize, Option<(usize, usize, usize, bf16, bf16)>) {
    let entry = &leech.toc()[toc_idx];
    assert_eq!(entry.role, Role::Llvq);
    let payload = leech.parse_llvq_payload(toc_idx).expect("parse payload");

    let r = payload.r as usize;
    let b_blocks = payload.b as usize;
    let k_beta = payload.k_beta as usize;
    let k_offset = payload.k_offset as usize;
    let n_blocks = r * b_blocks;
    let n_done = b_blocks * 24;
    let n_weight = r * n_done;

    // Parse codebooks to host fp16.
    let beta_host = payload.beta_codebook_f16(); // [R * K_beta]
    let offset_host = payload.offset_codebook_f16(); // [R * K_offset] or empty

    // Upload packed_stream (tail-padded by 8 B) to device.
    let mut packed_padded = Vec::with_capacity(payload.packed_stream.len() + 8);
    packed_padded.extend_from_slice(payload.packed_stream);
    packed_padded.extend_from_slice(&[0u8; 8]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(packed_padded.len()).expect("alloc packed") };
    cuda.memcpy_htod(&packed_padded, &mut d_packed).expect("htod packed");

    // Upload beta codebook (fp16) to device.
    let beta_bytes: Vec<u16> = beta_host.iter().map(|x| x.to_bits()).collect();
    let mut d_beta = unsafe { cuda.alloc::<u16>(beta_bytes.len()).expect("alloc beta") };
    cuda.memcpy_htod(&beta_bytes, &mut d_beta).expect("htod beta");

    // Upload offset codebook (fp16) if present.
    let d_offset = if payload.has_offset {
        let off_bits: Vec<u16> = offset_host.iter().map(|x| x.to_bits()).collect();
        let mut d = unsafe { cuda.alloc::<u16>(off_bits.len()).expect("alloc offset") };
        cuda.memcpy_htod(&off_bits, &mut d).expect("htod offset");
        Some(d)
    } else {
        None
    };

    // Allocate output bf16 buffer.
    let d_out = cuda.alloc_zeros::<bf16>(n_weight).expect("alloc d_out");

    let stream = cuda.cuda_stream();
    let (packed_ptr, _g1) = d_packed.device_ptr(&stream);
    let (beta_ptr, _g2) = d_beta.device_ptr(&stream);
    let offset_ptr_u64: u64 = match d_offset.as_ref() {
        Some(d) => {
            let (p, _) = d.device_ptr(&stream);
            p
        }
        None => 0,
    };
    let (out_ptr, _g4) = d_out.device_ptr(&stream);
    let stream_raw = stream.cu_stream() as *mut c_void;

    unsafe {
        leech_decode_bf16(
            packed_ptr as *const u8,
            beta_ptr as *const c_void,
            offset_ptr_u64 as *const c_void,
            out_ptr as *mut c_void,
            r as u32,
            b_blocks as u32,
            k_beta as u32,
            k_offset as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("kernel call");
    }
    device.synchronize().expect("sync");

    // Download kernel result.
    let mut host_kernel: Vec<bf16> = vec![bf16::ZERO; n_weight];
    cuda.memcpy_dtoh(&d_out, &mut host_kernel).expect("dtoh");

    // === Build reference via Phase 3 decode_v_int + per-block epilogue. ===
    // 1. Decode v_int via Phase 3 kernel.
    let mut d_vint = cuda.alloc_zeros::<i8>(n_blocks * 24).expect("alloc v_int");
    unsafe {
        let (v_ptr, _vg) = d_vint.device_ptr(&stream);
        let v_slice = std::slice::from_raw_parts_mut(v_ptr as *mut i8, n_blocks * 24);
        let packed_slice = std::slice::from_raw_parts(packed_ptr as *const u8, packed_padded.len());
        leech_decode_v_int(
            packed_slice,
            v_slice,
            n_blocks as u32,
            payload.idx_bits as u32,
            payload.has_offset,
            stream_raw,
        )
        .expect("decode_v_int");
    }
    device.synchronize().expect("sync v_int");
    let mut host_vint: Vec<i8> = vec![0; n_blocks * 24];
    cuda.memcpy_dtoh(&d_vint, &mut host_vint).expect("dtoh v_int");

    // 2. Apply per-block epilogue on host using the same op order as the kernel.
    let mut mismatches: usize = 0;
    let mut first_mismatch: Option<(usize, usize, usize, bf16, bf16)> = None;

    for row in 0..r {
        for blk in 0..b_blocks {
            let block_id = (row * b_blocks + blk) as u64;
            let (_i_global, beta_idx, offset_idx) =
                unpack_block(&packed_padded, block_id, payload.idx_bits as u32, payload.has_offset);
            let beta = beta_host[row * k_beta + beta_idx as usize].to_f32();
            let offset = if payload.has_offset {
                offset_host[row * k_offset + offset_idx as usize].to_f32()
            } else {
                0.0f32
            };
            for k in 0..24 {
                let v_int = host_vint[(block_id as usize) * 24 + k];
                let v_ref = epilogue_bf16(v_int, beta, offset);
                let out_idx = row * n_done + blk * 24 + k;
                if host_kernel[out_idx].to_bits() != v_ref.to_bits() {
                    mismatches += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((row, blk, k, host_kernel[out_idx], v_ref));
                    }
                }
            }
        }
    }
    (n_weight, mismatches, first_mismatch)
}

#[test]
fn decode_bf16_matches_reference() {
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE not set / missing — skipping");
        return;
    };
    let leech = LeechFile::open(&leech_path).expect("open .leech");
    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    // Default: 8 representative tensors. LEECH_DEQUANT_EXHAUSTIVE=1 → all LLVQ.
    let exhaustive = std::env::var("LEECH_DEQUANT_EXHAUSTIVE").is_ok();

    let llvq_indices: Vec<usize> = leech
        .toc()
        .iter()
        .enumerate()
        .filter(|(_, e)| e.role == Role::Llvq)
        .map(|(i, _)| i)
        .collect();

    let take_n = if exhaustive { llvq_indices.len() } else { 8 };
    let mut total_elems: usize = 0;
    let mut total_fail: usize = 0;
    let mut first_global_fail: Option<(String, usize, usize, usize, bf16, bf16)> = None;

    for (i, &toc_idx) in llvq_indices.iter().take(take_n).enumerate() {
        let entry_name = leech.toc()[toc_idx].name.clone();
        let (n_elems, n_fail, first_bad) = check_one_tensor(&leech, toc_idx, &device, cuda);
        total_elems += n_elems;
        total_fail += n_fail;
        if first_global_fail.is_none() {
            if let Some((row, blk, k, got, want)) = first_bad {
                first_global_fail = Some((entry_name.clone(), row, blk, k, got, want));
            }
        }
        if n_fail == 0 {
            if i % 16 == 0 || i + 1 == take_n {
                eprintln!("  [{:>3}/{}] OK   {} ({} elements)", i + 1, take_n, entry_name, n_elems);
            }
        } else {
            eprintln!(
                "  [{:>3}/{}] FAIL {} ({}/{} bf16 cells mismatch)",
                i + 1, take_n, entry_name, n_fail, n_elems
            );
        }
    }

    eprintln!(
        "Tested {} tensors / {} bf16 elements: {} OK, {} mismatches",
        take_n,
        total_elems,
        if total_fail == 0 { take_n } else { take_n - 1 },
        total_fail
    );
    if let Some((name, row, blk, k, got, want)) = first_global_fail {
        eprintln!(
            "  First global mismatch in {} at (row={row} blk={blk} k={k}): got {} (0x{:04x})  want {} (0x{:04x})",
            name,
            got.to_f32(),
            got.to_bits(),
            want.to_f32(),
            want.to_bits()
        );
    }
    assert_eq!(total_fail, 0, "bf16 output drift");
}

/// Microbench the fused decode+epilogue kernel on one large representative
/// tensor (mlp.up_proj, ~50M bf16 elements). Skips unless LEECH_BENCH_DEQUANT=1.
#[test]
fn bench_decode_bf16_kernel() {
    if std::env::var("LEECH_BENCH_DEQUANT").is_err() {
        eprintln!("LEECH_BENCH_DEQUANT not set — skipping bench");
        return;
    }
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE missing — skipping");
        return;
    };

    let leech = LeechFile::open(&leech_path).expect("open .leech");
    let target = "model.language_model.layers.0.mlp.up_proj.weight";
    let toc_idx = leech
        .toc()
        .iter()
        .position(|e| e.name == target)
        .expect("target tensor in TOC");
    let entry = &leech.toc()[toc_idx];
    let payload = leech.parse_llvq_payload(toc_idx).expect("parse payload");

    let r = payload.r as usize;
    let b_blocks = payload.b as usize;
    let k_beta = payload.k_beta as usize;
    let k_offset = payload.k_offset as usize;
    let n_blocks = r * b_blocks;
    let n_weight = r * b_blocks * 24;

    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    let mut packed_padded = Vec::with_capacity(payload.packed_stream.len() + 8);
    packed_padded.extend_from_slice(payload.packed_stream);
    packed_padded.extend_from_slice(&[0u8; 8]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(packed_padded.len()).expect("alloc") };
    cuda.memcpy_htod(&packed_padded, &mut d_packed).expect("htod packed");

    let beta_host = payload.beta_codebook_f16();
    let beta_bits: Vec<u16> = beta_host.iter().map(|x| x.to_bits()).collect();
    let mut d_beta = unsafe { cuda.alloc::<u16>(beta_bits.len()).expect("alloc beta") };
    cuda.memcpy_htod(&beta_bits, &mut d_beta).expect("htod beta");

    let offset_host = payload.offset_codebook_f16();
    let d_offset = if payload.has_offset {
        let off_bits: Vec<u16> = offset_host.iter().map(|x| x.to_bits()).collect();
        let mut d = unsafe { cuda.alloc::<u16>(off_bits.len()).expect("alloc offset") };
        cuda.memcpy_htod(&off_bits, &mut d).expect("htod offset");
        Some(d)
    } else {
        None
    };

    let d_out = cuda.alloc_zeros::<bf16>(n_weight).expect("alloc d_out");

    let stream = cuda.cuda_stream();
    let (packed_ptr, _g1) = d_packed.device_ptr(&stream);
    let (beta_ptr, _g2) = d_beta.device_ptr(&stream);
    let offset_ptr_u64: u64 = match d_offset.as_ref() {
        Some(d) => {
            let (p, _) = d.device_ptr(&stream);
            p
        }
        None => 0,
    };
    let (out_ptr, _g4) = d_out.device_ptr(&stream);
    let stream_raw = stream.cu_stream() as *mut c_void;

    let warmup = 10;
    let iters = 200;

    // Warmup
    for _ in 0..warmup {
        unsafe {
            leech_decode_bf16(
                packed_ptr as *const u8,
                beta_ptr as *const c_void,
                offset_ptr_u64 as *const c_void,
                out_ptr as *mut c_void,
                r as u32,
                b_blocks as u32,
                k_beta as u32,
                k_offset as u32,
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
            leech_decode_bf16(
                packed_ptr as *const u8,
                beta_ptr as *const c_void,
                offset_ptr_u64 as *const c_void,
                out_ptr as *mut c_void,
                r as u32,
                b_blocks as u32,
                k_beta as u32,
                k_offset as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
    }
    device.synchronize().expect("sync");
    let elapsed = t0.elapsed();

    let per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let blocks_per_sec = (n_blocks as f64 * iters as f64) / elapsed.as_secs_f64();
    let bytes_out_per_sec = (n_weight as f64 * 2.0 * iters as f64) / elapsed.as_secs_f64();
    let bytes_in_per_sec = (payload.packed_stream.len() as f64 * iters as f64) / elapsed.as_secs_f64();
    eprintln!();
    eprintln!("=== BENCH: leech_decode_bf16 (Phase 4.0a) ===");
    eprintln!("  tensor:        {}", entry.name);
    eprintln!("  R={}  B={}  blocks={}  idx_bits={}  has_offset={}", r, b_blocks, n_blocks, payload.idx_bits, payload.has_offset);
    eprintln!("  per-call:      {:.1} µs", per_call_us);
    eprintln!("  blocks/sec:    {:.3e}", blocks_per_sec);
    eprintln!("  bf16 out:      {:.3} GB/s", bytes_out_per_sec / 1e9);
    eprintln!("  packed in:     {:.3} GB/s", bytes_in_per_sec / 1e9);
    eprintln!("  vs RTX 5090 HBM peak (~1.7 TB/s output): {:.2}%",
              100.0 * bytes_out_per_sec / 1.7e12);
}
