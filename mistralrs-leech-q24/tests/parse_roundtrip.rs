//! End-to-end smoke tests against `production/qwopus_ms18_v3.q24t.leech`.
//!
//! These tests open the actual ~4.5 GB artifact via mmap and exercise the
//! parser surface — header, manifest, codebooks, TOC, and one tensor's
//! LLVQ_TANS payload. CRC verification is opt-in via the env var
//! `LEECHQ24_VERIFY_CRC=1` (it touches the whole file and is slow).
//!
//! All tests are guarded on the artifact's presence: they print and skip
//! when it is missing, so the suite still passes on machines that don't
//! have the artifact synced.

use std::path::PathBuf;

use mistralrs_leech_q24::{
    bucket_unpack, DtypeTag, LeechQ24File, OpenOptions, Role, FORMAT_VERSION, HEADER_SIZE, MAGIC,
};

fn artifact_path() -> Option<PathBuf> {
    // Allow override; otherwise probe the canonical antsquant path.
    if let Ok(p) = std::env::var("LEECHQ24_ARTIFACT") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let candidates = [
        "/var/www/vibe-marketing/docs/antsquant/production/qwopus_ms18_v3.q24t.leech",
        // On the runpod the file lands in $HOME.
        "/root/qwopus_ms18_v3.q24t.leech",
    ];
    for p in candidates {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    None
}

fn open_artifact() -> Option<LeechQ24File> {
    let path = artifact_path()?;
    println!("opening artifact at {}", path.display());
    let verify = std::env::var("LEECHQ24_VERIFY_CRC")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    Some(
        LeechQ24File::open_with(
            &path,
            OpenOptions {
                verify_crc: verify,
                validate_toc: true,
            },
        )
        .expect("failed to open .leech v3 artifact"),
    )
}

#[test]
fn artifact_present_or_skip() {
    match artifact_path() {
        Some(p) => println!("OK artifact at {}", p.display()),
        None => println!("SKIP: no artifact on this machine"),
    }
}

#[test]
fn magic_constant_is_leechq24() {
    assert_eq!(&MAGIC, b"LEECHQ24");
}

#[test]
fn header_layout_constants() {
    assert_eq!(HEADER_SIZE, 256);
    assert_eq!(FORMAT_VERSION, 3);
}

#[test]
fn header_parses() {
    let Some(f) = open_artifact() else { return };
    let h = f.header();
    assert_eq!(h.format_version, FORMAT_VERSION);
    assert_eq!(h.header_size, HEADER_SIZE as u32);
    assert!(h.toc_entry_count > 0);
    assert!(h.manifest_size > 0);
    assert!(h.codebook_size > 0);
    assert!(h.payload_size > 0);
    assert!(h.tile_size_default >= 1);
    println!(
        "header: fmt={} tile_size_default={} toc_count={} n_llvq={} n_fp8={} n_bf16={} n_overlay={} total_blocks={}",
        h.format_version,
        h.tile_size_default,
        h.toc_entry_count,
        h.n_llvq_tensors,
        h.n_fp8_tensors,
        h.n_bf16_tensors,
        h.n_overlay_tensors,
        h.total_blocks,
    );
}

#[test]
fn manifest_is_valid_json() {
    let Some(f) = open_artifact() else { return };
    let v = f.manifest_value().expect("manifest is not JSON");
    assert!(v.is_object(), "manifest top-level is not a JSON object");
}

#[test]
fn codebooks_parse_and_count_matches_header() {
    let Some(f) = open_artifact() else { return };
    let cbs = f.codebooks();
    let h = f.header();
    assert_eq!(
        cbs.len() as u32,
        h.n_codebook_sets,
        "parsed codebook count != header.n_codebook_sets"
    );
    for cb in cbs {
        assert_eq!(cb.n_codebooks as usize, mistralrs_leech_q24::N_CODEBOOKS);
        assert_eq!(cb.table_log, mistralrs_leech_q24::TABLE_LOG);
        assert!(cb.w_max >= cb.w_min, "w_min..w_max range invalid");
        // n_symbols == w_max - w_min + 1
        assert_eq!(cb.n_symbols as i64, (cb.w_max - cb.w_min + 1) as i64);
        println!(
            "cb sid={} S={} w in [{},{}] tables_bytes={}",
            cb.symbol_set_id, cb.n_symbols, cb.w_min, cb.w_max, cb.decode_tables_bytes,
        );
    }
}

#[test]
fn toc_role_counts_match_header() {
    let Some(f) = open_artifact() else { return };
    let h = f.header();
    let mut n_llvq = 0u32;
    let mut n_fp8 = 0u32;
    let mut n_bf16 = 0u32;
    let mut n_overlay = 0u32;
    for e in f.toc() {
        match e.role {
            Role::LlvqTans => n_llvq += 1,
            Role::Fp8Passthrough => n_fp8 += 1,
            Role::Bf16Passthrough => n_bf16 += 1,
            Role::Fp16Overlay => n_overlay += 1,
        }
    }
    assert_eq!(n_llvq, h.n_llvq_tensors, "LLVQ count");
    assert_eq!(n_fp8, h.n_fp8_tensors, "FP8 count");
    assert_eq!(n_bf16, h.n_bf16_tensors, "BF16 count");
    assert_eq!(n_overlay, h.n_overlay_tensors, "OVERLAY count");
}

#[test]
fn total_blocks_sum_matches_header() {
    let Some(f) = open_artifact() else { return };
    let h = f.header();
    let sum: u64 = f
        .toc()
        .iter()
        .filter(|e| e.is_llvq_tans())
        .map(|e| e.n_blocks)
        .sum();
    assert_eq!(sum, h.total_blocks, "Σ n_blocks (LLVQ) != header.total_blocks");
}

#[test]
fn first_llvq_payload_parses_and_decodes_one_bucket() {
    let Some(f) = open_artifact() else { return };
    let (idx, entry) = f
        .toc()
        .iter()
        .enumerate()
        .find(|(_, e)| e.is_llvq_tans())
        .expect("no LLVQ_TANS tensor in file");
    println!(
        "exercising tensor {}: {} shape={:?} R={} B={} n_blocks={} n_tiles={} sid={} k_beta={} k_offset={}",
        idx, entry.name, entry.shape, entry.r, entry.b, entry.n_blocks,
        entry.n_tiles, entry.symbol_set_id, entry.k_beta, entry.k_offset,
    );
    assert_eq!(entry.dtype_tag, DtypeTag::Packed);
    let p = f.llvq_tans_payload(idx).expect("payload parse");

    // Section sizes look sane.
    let expected_packed = bucket_unpack::packed_buckets_byte_count(p.n_blocks as usize);
    assert!(
        p.buckets_packed.len() >= expected_packed,
        "buckets_packed.len()={} < expected {}",
        p.buckets_packed.len(),
        expected_packed,
    );
    assert_eq!(p.tile_states.len(), (p.n_tiles * 2) as usize);
    assert_eq!(p.tile_nb_totals.len(), (p.n_tiles * 2) as usize);
    assert_eq!(p.tile_bitstream.len(), (p.tile_bitstream_words * 8) as usize);
    assert!(!p.beta_lloyd.is_empty());

    // β centroids look like sensible f32s (no NaNs, finite).
    for k in 0..p.k_beta.min(4) {
        let v = p.beta_lloyd_at(0, k).expect("beta_lloyd_at");
        assert!(v.is_finite(), "beta_lloyd[0,{}] = {:?} is non-finite", k, v);
    }

    // Decode 8 buckets via the CPU reference and confirm they're in [0, 8192).
    let n_probe = 8usize.min(p.n_blocks as usize);
    let mut buf = vec![0u16; n_probe];
    bucket_unpack::unpack_buckets_13bit(p.buckets_packed, n_probe, &mut buf).expect("unpack");
    for (i, &b) in buf.iter().enumerate() {
        // Sentinels are stored out-of-band; raw stream never contains 0xFFFF
        // for the on-disk indices (they were masked to 0 before packing).
        assert!(
            b < 8192,
            "bucket {} = {} out of 13-bit range",
            i, b
        );
    }
    println!("first {} buckets: {:?}", n_probe, buf);
}

#[test]
fn first_fp8_passthrough_byte_count_matches_dtype() {
    let Some(f) = open_artifact() else { return };
    let (idx, entry) = match f
        .toc()
        .iter()
        .enumerate()
        .find(|(_, e)| e.role == Role::Fp8Passthrough)
    {
        Some(x) => x,
        None => {
            println!("SKIP: no FP8_PASSTHROUGH in file");
            return;
        }
    };
    let raw = f.passthrough_bytes(idx).expect("passthrough bytes");
    let elem: usize = entry.shape.iter().map(|&d| d as usize).product();
    let expected = elem
        .checked_mul(entry.dtype_tag.elem_size().unwrap_or(1))
        .unwrap();
    assert_eq!(
        raw.len() as u64,
        entry.passthrough_size,
        "TOC.passthrough_size does not match slice length"
    );
    assert_eq!(
        raw.len(),
        expected,
        "FP8 raw byte count does not match (Π shape × elem_size)"
    );
    println!(
        "fp8 tensor {} shape={:?} bytes={}",
        entry.name,
        entry.shape,
        raw.len()
    );
}

#[test]
#[ignore = "slow: scans 4.5 GB. Run with `--ignored` or LEECHQ24_VERIFY_CRC=1."]
fn crc32_roundtrips() {
    let path = match artifact_path() {
        Some(p) => p,
        None => {
            println!("SKIP: no artifact");
            return;
        }
    };
    let f = LeechQ24File::open_with(
        &path,
        OpenOptions {
            verify_crc: true,
            validate_toc: true,
        },
    )
    .expect("open with crc");
    let stored = f.stored_crc();
    let computed = f.compute_crc();
    assert_eq!(stored, computed, "CRC32 mismatch");
    println!("CRC32 matches: {:#010x}", stored);
}
