//! Q24-tANS Phase A correctness gate: GPU `leech_q24_decode_v_int_cuda` must
//! produce int8[N, 24] bit-identical to the Python CPU reference produced by
//! `production/packer/q24_tans/unpack.py` for the same tensor.
//!
//! Skipped at compile time on non-CUDA builds. Skipped at runtime if either
//! the .leech v3 artifact or the per-tensor `v_best_int` reference is missing.
//!
//! Env vars:
//!   LEECHQ24_ARTIFACT   — path to qwopus_ms18_v3.q24t.leech (canonical on pod: /root/...)
//!   LEECHQ24_V_REFS     — directory holding per-tensor `<name>.v_int.bin`
//!                         dumps; the v_int.bin file is the raw int8[N, 24]
//!                         that the CUDA kernel should reproduce.
//!   LEECHQ24_TENSOR     — optional override of which tensor to exercise.
//!                         Default: "model.language_model.layers.0.linear_attn.in_proj_a.weight"
//!                         (R=32, B=170, n_blocks=5440, n_tiles=170 — small enough
//!                         to bring up the kernel without TB-scale memory churn).

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use mistralrs_leech_q24::{LeechQ24File, Role};
use mistralrs_quant::leech_q24::{compute_tile_bit_offsets, init_tables, leech_q24_decode_v_int};

fn artifact_path() -> Option<PathBuf> {
    let p = std::env::var("LEECHQ24_ARTIFACT").ok()?;
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

fn refs_dir() -> Option<PathBuf> {
    let p = std::env::var("LEECHQ24_V_REFS").ok()?;
    let pb = PathBuf::from(p);
    pb.is_dir().then_some(pb)
}

#[test]
fn decode_v_int_matches_cpu_reference() {
    let Some(leech_path) = artifact_path() else {
        eprintln!("LEECHQ24_ARTIFACT not set — skipping");
        return;
    };
    let leech = LeechQ24File::open(&leech_path).expect("open .leech v3");

    let tensor_name = std::env::var("LEECHQ24_TENSOR").unwrap_or_else(|_| {
        "model.language_model.layers.0.linear_attn.in_proj_a.weight".to_owned()
    });
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

    eprintln!(
        "tensor {} n_blocks={} n_tiles={} tile_size={} sid={} w_offset={}",
        tensor_name, n_blocks, n_tiles, tile_size, symbol_set_id, w_offset
    );

    // Decode tables (host).
    let dt_bytes = leech
        .decode_tables_bytes(symbol_set_id)
        .expect("decode tables");
    assert_eq!(dt_bytes.len(), 4 * 1024 * 4, "expected 16 KB decode tables");
    let dt_host: Vec<u32> = dt_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    init_tables(&dt_host, symbol_set_id).expect("init_tables");

    // Host-side tile_bit_offsets prefix sum.
    let nb_totals: Vec<u16> = payload
        .tile_nb_totals
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(nb_totals.len(), n_tiles as usize);
    let bit_offsets = compute_tile_bit_offsets(&nb_totals);

    // Locate (and skip if absent) the per-tensor v_int reference.
    let v_int_ref = match refs_dir() {
        Some(d) => {
            let p = d.join(format!("{}.v_int.bin", tensor_name));
            if !p.is_file() {
                eprintln!("no reference at {} — skipping correctness check", p.display());
                None
            } else {
                Some(std::fs::read(&p).expect("read v_int.bin"))
            }
        }
        None => {
            eprintln!("LEECHQ24_V_REFS not set — running kernel for smoke only");
            None
        }
    };

    // Upload to device.
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::{Device, Storage};
    let device = Device::new_cuda(0).expect("Device::Cuda");
    let cuda = device.as_cuda_device().expect("cuda device");

    // packed_buckets — add ≥1 byte tail pad so the 3-byte read at the last bucket is safe.
    let mut packed_padded = Vec::with_capacity(payload.buckets_packed.len() + 4);
    packed_padded.extend_from_slice(payload.buckets_packed);
    packed_padded.extend_from_slice(&[0u8; 4]);
    let mut d_packed = unsafe { cuda.alloc::<u8>(packed_padded.len()).expect("alloc packed") };
    cuda.memcpy_htod(&packed_padded, &mut d_packed).expect("htod packed");

    // tile_states (u16) — bytes already u16-aligned in payload.
    let mut d_states = unsafe { cuda.alloc::<u16>(n_tiles as usize).expect("alloc states") };
    let states_host: Vec<u16> = payload
        .tile_states
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    cuda.memcpy_htod(&states_host, &mut d_states).expect("htod states");

    let mut d_nb = unsafe { cuda.alloc::<u16>(n_tiles as usize).expect("alloc nb") };
    cuda.memcpy_htod(&nb_totals, &mut d_nb).expect("htod nb");

    // tile_bitstream (u64).
    let words: Vec<u64> = payload
        .tile_bitstream
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    let mut d_bs = unsafe { cuda.alloc::<u64>(words.len()).expect("alloc bs") };
    cuda.memcpy_htod(&words, &mut d_bs).expect("htod bs");

    let mut d_offs = unsafe { cuda.alloc::<u64>(n_tiles as usize).expect("alloc offs") };
    cuda.memcpy_htod(&bit_offsets, &mut d_offs).expect("htod offs");

    let out_elems = (n_blocks as usize) * 24;
    let mut d_out = unsafe { cuda.alloc::<i8>(out_elems).expect("alloc out") };

    // Launch.
    let (packed_ptr, _) = d_packed.device_ptr(d_packed.stream());
    let (states_ptr, _) = d_states.device_ptr(d_states.stream());
    let (nb_ptr, _) = d_nb.device_ptr(d_nb.stream());
    let (bs_ptr, _) = d_bs.device_ptr(d_bs.stream());
    let (offs_ptr, _) = d_offs.device_ptr(d_offs.stream());
    let (out_ptr, _) = d_out.device_ptr(d_out.stream());

    unsafe {
        leech_q24_decode_v_int(
            packed_ptr as *const u8,
            states_ptr as *const u16,
            nb_ptr as *const u16,
            bs_ptr as *const u64,
            offs_ptr as *const u64,
            out_ptr as *mut i8,
            n_blocks,
            n_tiles,
            w_offset,
            tile_size,
            std::ptr::null_mut::<c_void>(),
        )
        .expect("decode launch");
    }
    cuda.synchronize().expect("sync");

    // Read back.
    let mut host_out = vec![0i8; out_elems];
    cuda.memcpy_dtoh(&d_out, &mut host_out).expect("dtoh out");

    if let Some(ref_bytes) = v_int_ref {
        assert_eq!(
            ref_bytes.len(),
            host_out.len(),
            "ref byte count {} != GPU output count {}",
            ref_bytes.len(),
            host_out.len()
        );
        let ref_i8: &[i8] = unsafe {
            std::slice::from_raw_parts(ref_bytes.as_ptr() as *const i8, ref_bytes.len())
        };
        let mut n_diff = 0usize;
        for (i, (g, r)) in host_out.iter().zip(ref_i8.iter()).enumerate() {
            if g != r {
                if n_diff < 8 {
                    eprintln!("  diff @{i}: gpu={g} ref={r}");
                }
                n_diff += 1;
            }
        }
        assert_eq!(n_diff, 0, "{n_diff} mismatched coords");
        eprintln!("PASS: bit-equal on {} coords", host_out.len());
    } else {
        // No reference — just sanity-bound the output range.
        let mut bad = 0usize;
        for v in &host_out {
            if !(-32..=32).contains(v) {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} v_int values out of [-32, 32]");
        eprintln!("SMOKE OK: {} coords decoded, all in [-32, 32]", host_out.len());
    }
}
