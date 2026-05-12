//! Phase 3 acceptance gate: GPU leech_decode_v_int_cuda must produce
//! int8[R, B, 24] bit-identical to the v2 CPU reference (proven at antsquant
//! commit 7a465fd against the 287M-block exhaustive test).
//!
//! Skipped at compile time on non-CUDA builds. Skipped at runtime if either
//! the .leech fixture or the reference dump are missing — set:
//!
//!   LEECH_FIXTURE=/path/to/qwopus.leech
//!   LEECH_REFS=/path/to/leech_refs        # produced by:
//!     #   python antsquant/tools/dump_v_int_references.py \
//!     #       --leech /path/to/qwopus.leech --out /path/to/leech_refs
//!
//! Defaults to /tmp/qwopus.leech and /tmp/leech_refs (matches the dev box).

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::Device;
use mistralrs_leech::{LeechFile, Role};
use mistralrs_quant::leech::leech_decode_v_int;

#[derive(Debug, serde::Deserialize)]
struct ManifestTensor {
    rank: usize,
    name: String,
    file: String,
    #[serde(rename = "R")]
    r: u32,
    #[serde(rename = "B")]
    b: u32,
    n_blocks: u64,
    idx_bits: u32,
    has_offset: bool,
    byte_size: usize,
    sha256: String,
}

#[derive(Debug, serde::Deserialize)]
struct Manifest {
    n_tensors: usize,
    total_blocks: u64,
    tensors: Vec<ManifestTensor>,
}

fn fixture_path() -> Option<PathBuf> {
    let p = std::env::var("LEECH_FIXTURE").unwrap_or_else(|_| "/tmp/qwopus.leech".to_owned());
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

fn refs_path() -> Option<PathBuf> {
    let p = std::env::var("LEECH_REFS").unwrap_or_else(|_| "/tmp/leech_refs".to_owned());
    let pb = PathBuf::from(p);
    if pb.join("manifest.json").exists() {
        Some(pb)
    } else {
        None
    }
}

fn load_manifest(dir: &Path) -> Manifest {
    let mf_path = dir.join("manifest.json");
    let bytes = fs::read(&mf_path).expect("read manifest.json");
    serde_json::from_slice(&bytes).expect("parse manifest.json")
}

/// Verify the CUDA decoder is bit-identical to the CPU reference across a
/// sweep of tensors. Default sweep = 16 representative tensors picked from
/// the .leech bundle, covering linear_attn/self_attn/mlp shapes. Setting
/// `LEECH_EXHAUSTIVE=1` runs all 248 LLVQ tensors.
#[test]
fn decode_v_int_matches_cpu_reference() {
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE not set / missing — skipping");
        return;
    };
    let Some(refs_dir) = refs_path() else {
        eprintln!("LEECH_REFS not set / missing — skipping");
        eprintln!("Produce references with:");
        eprintln!(
            "  python antsquant/tools/dump_v_int_references.py \\\n    --leech {} --out /tmp/leech_refs",
            leech_path.display()
        );
        return;
    };

    let leech = LeechFile::open(&leech_path).expect("open .leech");
    let manifest = load_manifest(&refs_dir);
    assert!(manifest.n_tensors > 0, "manifest empty");

    // Map TOC name -> TOC index for fast lookup.
    let toc_index_by_name: std::collections::HashMap<&str, usize> = leech
        .toc()
        .iter()
        .enumerate()
        .map(|(i, e)| (e.name.as_str(), i))
        .collect();

    // Acquire a CUDA device. If this fails, surface as a hard error since the
    // test is feature-gated on cuda — the user opted in.
    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    let exhaustive = std::env::var("LEECH_EXHAUSTIVE").is_ok();
    let max_tensors = if exhaustive { manifest.tensors.len() } else { 16 };

    let mut total_blocks: u64 = 0;
    let mut n_ok = 0usize;
    let mut n_fail = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (rank, mt) in manifest.tensors.iter().take(max_tensors).enumerate() {
        let &toc_idx = toc_index_by_name
            .get(mt.name.as_str())
            .unwrap_or_else(|| panic!("tensor {:?} missing from .leech TOC", mt.name));

        let entry = &leech.toc()[toc_idx];
        assert_eq!(entry.role, Role::Llvq, "expected LLVQ role for {}", entry.name);

        let payload = leech
            .parse_llvq_payload(toc_idx)
            .unwrap_or_else(|e| panic!("parse_llvq_payload({}): {e}", entry.name));
        let n_blocks = payload.r as u64 * payload.b as u64;
        assert_eq!(n_blocks, mt.n_blocks, "n_blocks mismatch for {}", mt.name);
        assert_eq!(payload.idx_bits as u32, mt.idx_bits);
        assert_eq!(payload.has_offset, mt.has_offset);
        total_blocks += n_blocks;

        // Reference bytes — raw int8[R, B, 24].
        let ref_path = refs_dir.join(&mt.file);
        let ref_bytes = fs::read(&ref_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", ref_path.display()));
        assert_eq!(ref_bytes.len(), mt.byte_size);

        // Copy packed_stream to device with 8 B tail-pad.
        let mut padded = Vec::with_capacity(payload.packed_stream.len() + 8);
        padded.extend_from_slice(payload.packed_stream);
        padded.extend_from_slice(&[0u8; 8]);
        let mut d_packed = unsafe {
            cuda.alloc::<u8>(padded.len()).expect("alloc d_packed")
        };
        cuda.memcpy_htod(&padded, &mut d_packed)
            .expect("htod_copy packed_stream");

        // Allocate output.
        let out_len = (n_blocks as usize) * 24;
        let d_out = cuda
            .alloc_zeros::<i8>(out_len)
            .expect("alloc_zeros out_v_int");

        // Cast device pointers and call the kernel.
        let stream = cuda.cuda_stream();
        unsafe {
            use candle_core::cuda::cudarc::driver::DevicePtr;
            let (packed_ptr, _packed_guard) = d_packed.device_ptr(&stream);
            let (out_ptr, _out_guard) = d_out.device_ptr(&stream);
            let stream_raw = stream.cu_stream() as *mut c_void;
            // Build slices for the safe wrapper.
            let packed_slice = std::slice::from_raw_parts(
                packed_ptr as *const u8,
                padded_packed_len(&payload, &mt),
            );
            let out_slice = std::slice::from_raw_parts_mut(out_ptr as *mut i8, out_len);
            leech_decode_v_int(
                packed_slice,
                out_slice,
                n_blocks as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("leech_decode_v_int");
        }
        device.synchronize().expect("cuda synchronize");

        // Copy result back and diff.
        let mut host_out: Vec<i8> = vec![0i8; out_len];
        cuda.memcpy_dtoh(&d_out, &mut host_out)
            .expect("memcpy_dtoh");
        let ref_i8: &[i8] = unsafe {
            std::slice::from_raw_parts(ref_bytes.as_ptr() as *const i8, ref_bytes.len())
        };

        if host_out.as_slice() == ref_i8 {
            n_ok += 1;
            if rank % 4 == 0 || rank == max_tensors - 1 {
                eprintln!(
                    "  [{:>3}/{}] OK   {} (R={} B={} blocks={})",
                    rank + 1, max_tensors, mt.name, payload.r, payload.b, n_blocks
                );
            }
        } else {
            n_fail += 1;
            let n_diff: usize = host_out
                .iter()
                .zip(ref_i8.iter())
                .filter(|(a, b)| a != b)
                .count();
            let first_diff = host_out
                .iter()
                .zip(ref_i8.iter())
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .unwrap();
            let (i, got, want) = first_diff;
            let blk = i / 24;
            let r = blk / payload.b as usize;
            let b = blk % payload.b as usize;
            let k = i % 24;
            let msg = format!(
                "  [{:>3}/{}] FAIL {} — {} of {} differ; first at (r={}, b={}, k={}) got {} want {}",
                rank + 1, max_tensors, mt.name, n_diff, out_len, r, b, k, got, want
            );
            eprintln!("{msg}");
            failures.push(msg);
        }
    }

    eprintln!();
    eprintln!(
        "Tested {} tensors / {} blocks: {} OK, {} FAIL",
        n_ok + n_fail, total_blocks, n_ok, n_fail
    );

    assert!(n_fail == 0, "{} tensor(s) failed CUDA decode:\n{}", n_fail, failures.join("\n"));
}

/// Helper: byte length of the packed_stream slice we copy to device, including
/// the 8B tail-pad. The slice we hand the kernel must be at least this big.
fn padded_packed_len(payload: &mistralrs_leech::LlvqPayload<'_>, _mt: &ManifestTensor) -> usize {
    payload.packed_stream.len() + 8
}

/// Microbench: measure kernel-only execution time on one representative tensor.
/// Setup (htod, alloc) happens once; we time N kernel iterations under a single
/// sync. Skips unless LEECH_BENCH=1 is set.
#[test]
fn bench_decode_v_int_kernel_only() {
    if std::env::var("LEECH_BENCH").is_err() {
        eprintln!("LEECH_BENCH not set — skipping microbench");
        return;
    }
    let Some(leech_path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE missing — skipping");
        return;
    };

    let leech = LeechFile::open(&leech_path).expect("open .leech");
    // Pick a large MLP tensor — high block count, exercises IDX_BITS=54 path.
    let target = "model.language_model.layers.0.mlp.up_proj.weight";
    let toc_idx = leech
        .toc()
        .iter()
        .position(|e| e.name == target)
        .expect("target tensor in TOC");
    let entry = &leech.toc()[toc_idx];
    let payload = leech.parse_llvq_payload(toc_idx).expect("parse_llvq_payload");
    let n_blocks = payload.r as u64 * payload.b as u64;

    let device = Device::new_cuda(0).expect("acquire CUDA device 0");
    let cuda = device.as_cuda_device().expect("Device::Cuda");

    let mut padded = Vec::with_capacity(payload.packed_stream.len() + 8);
    padded.extend_from_slice(payload.packed_stream);
    padded.extend_from_slice(&[0u8; 8]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(padded.len()).expect("alloc") };
    cuda.memcpy_htod(&padded, &mut d_packed).expect("htod");
    let out_len = (n_blocks as usize) * 24;
    let d_out = cuda.alloc_zeros::<i8>(out_len).expect("alloc_zeros");

    let warmup = std::env::var("LEECH_BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let iters = std::env::var("LEECH_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(200);

    let stream = cuda.cuda_stream();
    let packed_len = payload.packed_stream.len() + 8;

    unsafe {
        use candle_core::cuda::cudarc::driver::DevicePtr;
        let (packed_ptr, _g1) = d_packed.device_ptr(&stream);
        let (out_ptr, _g2) = d_out.device_ptr(&stream);
        let stream_raw = stream.cu_stream() as *mut c_void;
        let packed_slice = std::slice::from_raw_parts(packed_ptr as *const u8, packed_len);
        let out_slice = std::slice::from_raw_parts_mut(out_ptr as *mut i8, out_len);

        // Warmup.
        for _ in 0..warmup {
            leech_decode_v_int(
                packed_slice,
                out_slice,
                n_blocks as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
        device.synchronize().expect("sync after warmup");

        // Timed loop.
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            leech_decode_v_int(
                packed_slice,
                out_slice,
                n_blocks as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream_raw,
            )
            .expect("kernel");
        }
        device.synchronize().expect("sync after bench");
        let elapsed = t0.elapsed();

        let per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
        let blocks_per_sec = (n_blocks as f64 * iters as f64) / elapsed.as_secs_f64();
        let elements_per_sec = blocks_per_sec * 24.0;
        let bytes_in_per_sec = (packed_len as f64 * iters as f64) / elapsed.as_secs_f64();
        let bytes_out_per_sec = (out_len as f64 * iters as f64) / elapsed.as_secs_f64();
        eprintln!();
        eprintln!("=== BENCH: leech_decode_v_int kernel-only ===");
        eprintln!("  tensor:        {}", entry.name);
        eprintln!("  R={}  B={}  blocks={}  idx_bits={}  has_offset={}",
            payload.r, payload.b, n_blocks, payload.idx_bits, payload.has_offset);
        eprintln!("  warmup:        {} calls", warmup);
        eprintln!("  iters:         {} calls", iters);
        eprintln!("  per-call:      {:.1} µs", per_call_us);
        eprintln!("  blocks/sec:    {:.3e}", blocks_per_sec);
        eprintln!("  int8 elt/sec:  {:.3e}", elements_per_sec);
        eprintln!("  packed in B/s: {:.3} GB/s", bytes_in_per_sec / 1e9);
        eprintln!("  out B/s:       {:.3} GB/s", bytes_out_per_sec / 1e9);
    }
}
