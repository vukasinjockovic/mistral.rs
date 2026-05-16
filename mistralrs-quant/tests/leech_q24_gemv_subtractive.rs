//! Q24-tANS Phase B.0 GEMV — **subtractive** (wall-time-only) profile.
//!
//! Runs the v0 baseline kernel and 6 diagnostic variants on the same
//! artifact / activation / iteration count. Each variant disables ONE
//! per-coord operation; output values are INCORRECT but wall time reveals
//! which stage is on the critical path.
//!
//! NOTE: **no correctness check** on variants. They are known to produce
//! wrong output. The baseline (v0) is run first for reference timing, but
//! is also not validated here (the validating test lives in
//! `leech_q24_gemv_cuda.rs` and runs separately).
//!
//! ## Measured master matrix (RTX 5090, mlp.up_proj L0, 200 iters)
//!
//! Master subtractive Δ table (µs saved by disabling stage at each T):
//!
//! ```text
//! Variant       | T=4      | T=8      | T=16     | T=32
//! --------------|----------|----------|----------|----------
//! baseline      | 1414.5   | 1758.7   | 1646.3   | 2271.4
//! V_NO_PAT      |  +138.6  |  -126.7  |   -24.5  |   -37.7
//! V_NO_DECODE   | -1026.7  | -1171.5  |  -918.0  | -1933.0
//! V_NO_BITS     |   -47.0  |  -484.2  |  -651.1  | -1363.5
//! V_NO_AACT     |  +142.7  |  -116.3  |    -9.0  |    -6.1
//! V_NO_ATOMIC   |    +6.0  |   +48.5  |    +3.3  |    -2.1
//! V_NO_STATE    | -1338.5  | -1675.3  | -1572.3  | -2199.5
//! ```
//!
//! Dominant bottleneck at every T: V_NO_STATE (FSE state-chain serialization).
//! Breaking the chain drops wall to ~75 µs floor at every T (kernel launch +
//! memset + finalize). v0's 1414 µs envelope at T=4 has 18.6× headroom.
//!
//! T=8 anomaly explained: V_NO_BITS Δ jumps from -47 µs (T=4) to -484 µs (T=8).
//! At T=8 the per-tile bitstream LDG window crosses one L1 line boundary; the
//! L1→L2 step function adds ~440 µs to the T=8 baseline despite shorter
//! per-thread chain length than T=4.
//!
//! Output: a single ASCII table:
//!
//! ```text
//! SUBTRACTIVE PROFILE (T=<T>, <tensor>, <iters> iters)
//!
//! Variant                | Wall µs | Δ vs baseline | % of baseline
//! -----------------------|---------|---------------|--------------
//! baseline (full kernel) | 1409    | —             | 100.0%
//! V_NO_PAT               | ...     | ...           | ...
//! V_NO_DECODE            | ...     | ...           | ...
//! ...
//! ```
//!
//! Env vars (same as the other GEMV tests):
//!   LEECHQ24_ARTIFACT       — path to a Q24-tANS .leech artifact
//!   LEECHQ24_V_REFS         — directory (used only to skip when missing)
//!   LEECHQ24_TENSOR         — tensor name (default mlp.up_proj layer 0)
//!   LEECHQ24_BENCH_ITERS    — iters per variant (default 200)

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use half::bf16;
use mistralrs_leech_q24::{LeechQ24File, Role};
use mistralrs_quant::leech_q24::{
    compute_tile_bit_offsets, init_tables, leech_q24_gemv_bf16,
    leech_q24_gemv_bf16_subtractive, SubtractiveVariant,
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

#[test]
fn gemv_bf16_subtractive_profile() {
    let Some(leech_path) = env_path("LEECHQ24_ARTIFACT") else {
        eprintln!("LEECHQ24_ARTIFACT not set — skipping");
        return;
    };
    // Refs dir not used (no correctness check) but presence indicates the
    // standard env is wired up.
    let _refs_dir = env_path("LEECHQ24_V_REFS");

    let tensor_name = std::env::var("LEECHQ24_TENSOR").unwrap_or_else(|_| {
        "model.language_model.layers.0.mlp.up_proj.weight".to_owned()
    });
    let iters: u32 = std::env::var("LEECHQ24_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

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
        "SUBTRACTIVE tensor {} R={} B={} K_total={} n_blocks={} n_tiles={} tile_size={} sid={} w_offset={} k_beta={} k_offset={} has_offset={} iters={}",
        tensor_name, r, b, k_total, n_blocks, n_tiles,
        tile_size, symbol_set_id, w_offset, k_beta, k_offset, has_offset, iters
    );

    // ── Host inputs ────────────────────────────────────────────────────
    let dt_bytes = leech
        .decode_tables_bytes(symbol_set_id)
        .expect("decode tables");
    let dt_host: Vec<u32> = dt_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    init_tables(&dt_host, symbol_set_id).expect("init_tables");

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

    // Deterministic activation: small pseudorandom bf16 in [-1, 1]
    let a_host: Vec<bf16> = (0..k_total)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(2654435761) >> 16) as i32 as f32;
            let f = (x.sin() * 0.5).max(-1.0).min(1.0);
            bf16::from_f32(f)
        })
        .collect();

    let beta_idx_packed = payload.beta_idx_packed;
    let offset_idx_packed = payload.offset_idx_packed;

    // ── GPU upload ────────────────────────────────────────────────────
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

    let mut d_states = unsafe { cuda.alloc::<u16>(n_tiles as usize).expect("alloc states") };
    cuda.memcpy_htod(&states_host, &mut d_states).expect("htod states");
    let mut d_nb = unsafe { cuda.alloc::<u16>(n_tiles as usize).expect("alloc nb") };
    cuda.memcpy_htod(&nb_totals, &mut d_nb).expect("htod nb");
    let mut d_bs = unsafe { cuda.alloc::<u64>(bs_words.len()).expect("alloc bs") };
    cuda.memcpy_htod(&bs_words, &mut d_bs).expect("htod bs");
    let mut d_offs = unsafe { cuda.alloc::<u64>(n_tiles as usize).expect("alloc offs") };
    cuda.memcpy_htod(&bit_offsets, &mut d_offs).expect("htod offs");

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

    let a_u16: Vec<u16> = a_host.iter().map(|x| x.to_bits()).collect();
    let mut d_a = unsafe { cuda.alloc::<u16>(k_total).expect("alloc a") };
    cuda.memcpy_htod(&a_u16, &mut d_a).expect("htod a");

    let d_out = unsafe { cuda.alloc::<u16>(r as usize).expect("alloc out") };
    let mut d_acc = unsafe { cuda.alloc::<f32>(r as usize).expect("alloc acc") };
    let zeros = vec![0.0f32; r as usize];
    cuda.memcpy_htod(&zeros, &mut d_acc).expect("htod zero");

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

    // ── Bench one kernel ──────────────────────────────────────────────
    let bench_baseline = || {
        // Warmup
        for _ in 0..16 {
            unsafe {
                leech_q24_gemv_bf16(
                    a_ptr, packed_ptr, states_ptr, nb_ptr, bs_ptr, offs_ptr,
                    bidx_ptr, oidx_ptr_const, beta_lloyd_ptr, off_lloyd_ptr_const,
                    acc_ptr, out_ptr, r, b, n_blocks, n_tiles, k_beta, k_offset,
                    w_offset, tile_size, has_offset, std::ptr::null_mut::<c_void>(),
                )
                .expect("baseline launch");
            }
        }
        cuda.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                leech_q24_gemv_bf16(
                    a_ptr, packed_ptr, states_ptr, nb_ptr, bs_ptr, offs_ptr,
                    bidx_ptr, oidx_ptr_const, beta_lloyd_ptr, off_lloyd_ptr_const,
                    acc_ptr, out_ptr, r, b, n_blocks, n_tiles, k_beta, k_offset,
                    w_offset, tile_size, has_offset, std::ptr::null_mut::<c_void>(),
                )
                .expect("baseline launch");
            }
        }
        cuda.synchronize().expect("sync");
        t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
    };
    let bench_variant = |v: SubtractiveVariant| {
        // Warmup
        for _ in 0..16 {
            unsafe {
                leech_q24_gemv_bf16_subtractive(
                    v, a_ptr, packed_ptr, states_ptr, nb_ptr, bs_ptr, offs_ptr,
                    bidx_ptr, oidx_ptr_const, beta_lloyd_ptr, off_lloyd_ptr_const,
                    acc_ptr, out_ptr, r, b, n_blocks, n_tiles, k_beta, k_offset,
                    w_offset, tile_size, has_offset, std::ptr::null_mut::<c_void>(),
                )
                .expect("variant launch");
            }
        }
        cuda.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                leech_q24_gemv_bf16_subtractive(
                    v, a_ptr, packed_ptr, states_ptr, nb_ptr, bs_ptr, offs_ptr,
                    bidx_ptr, oidx_ptr_const, beta_lloyd_ptr, off_lloyd_ptr_const,
                    acc_ptr, out_ptr, r, b, n_blocks, n_tiles, k_beta, k_offset,
                    w_offset, tile_size, has_offset, std::ptr::null_mut::<c_void>(),
                )
                .expect("variant launch");
            }
        }
        cuda.synchronize().expect("sync");
        t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
    };

    let baseline_us = bench_baseline();
    let pat_us = bench_variant(SubtractiveVariant::NoPat);
    let dec_us = bench_variant(SubtractiveVariant::NoDecode);
    let bits_us = bench_variant(SubtractiveVariant::NoBits);
    let aact_us = bench_variant(SubtractiveVariant::NoAact);
    let atom_us = bench_variant(SubtractiveVariant::NoAtomic);
    let stat_us = bench_variant(SubtractiveVariant::NoState);

    eprintln!();
    eprintln!(
        "SUBTRACTIVE PROFILE (T={}, {}, {} iters)",
        tile_size, tensor_name, iters
    );
    eprintln!();
    eprintln!(
        "{:<23}| {:>9} | {:>13} | {:>13}",
        "Variant", "Wall µs", "Δ vs baseline", "% of baseline"
    );
    eprintln!("{}|{}|{}|{}", "-".repeat(23), "-".repeat(11), "-".repeat(15), "-".repeat(15));
    eprintln!(
        "{:<23}| {:>9.1} | {:>13} | {:>13}",
        "baseline (full kernel)", baseline_us, "—", "100.0%"
    );
    let row = |name: &str, us: f64| {
        let delta = us - baseline_us;
        let pct = us / baseline_us * 100.0;
        eprintln!(
            "{:<23}| {:>9.1} | {:>+13.1} | {:>12.1}%",
            name, us, delta, pct
        );
    };
    row("V_NO_PAT", pat_us);
    row("V_NO_DECODE", dec_us);
    row("V_NO_BITS", bits_us);
    row("V_NO_AACT", aact_us);
    row("V_NO_ATOMIC", atom_us);
    row("V_NO_STATE", stat_us);
    eprintln!();
    eprintln!("(Variants intentionally produce wrong output; this is a wall-time-only diagnostic.)");

    // Machine-readable single line for easy table-stitching across T values.
    eprintln!(
        "SUBTRACTIVE_CSV T={} baseline={:.1} no_pat={:.1} no_decode={:.1} no_bits={:.1} no_aact={:.1} no_atomic={:.1} no_state={:.1}",
        tile_size, baseline_us, pat_us, dec_us, bits_us, aact_us, atom_us, stat_us
    );

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
