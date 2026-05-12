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
        let d_packed = cuda.htod_copy(padded).expect("htod_copy packed_stream");

        // Allocate output.
        let out_len = (n_blocks as usize) * 24;
        let d_out = cuda
            .alloc_zeros::<i8>(out_len)
            .expect("alloc_zeros out_v_int");

        // Cast device pointers and call the kernel.
        unsafe {
            use candle_core::cuda::cudarc::driver::DevicePtr;
            let packed_ptr = *d_packed.device_ptr() as *const u8;
            let out_ptr = *d_out.device_ptr() as *mut i8;
            let stream = cuda.cu_stream() as *mut c_void;
            // Build slices for the safe wrapper.
            let packed_slice =
                std::slice::from_raw_parts(packed_ptr, padded_packed_len(&payload, &mt));
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, out_len);
            leech_decode_v_int(
                packed_slice,
                out_slice,
                n_blocks as u32,
                payload.idx_bits as u32,
                payload.has_offset,
                stream,
            )
            .expect("leech_decode_v_int");
        }
        cuda.synchronize().expect("cuda synchronize");

        // Copy result back and diff.
        let host_out: Vec<i8> = cuda.dtoh_sync_copy(&d_out).expect("dtoh_sync_copy");
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
