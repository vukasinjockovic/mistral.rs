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
    bucket_unpack, DtypeTag, LeechQ24File, OpenOptions, Role, FORMAT_VERSION,
    FORMAT_VERSION_V3_COMPAT, HEADER_SIZE, MAGIC,
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
    // v1.1 of the plan bumped FORMAT_VERSION 3 → 4; v3 files are still
    // accepted via the compat shim (see container::TocEntry::unpack).
    assert_eq!(FORMAT_VERSION, 4);
    assert_eq!(FORMAT_VERSION_V3_COMPAT, 3);
}

#[test]
fn header_parses() {
    let Some(f) = open_artifact() else { return };
    let h = f.header();
    // The canonical on-disk artifact is the v3 file; the v4-aware loader
    // reads it via the compat shim, which preserves header.format_version
    // verbatim (the shim only rewrites TOC entries).
    assert!(
        h.format_version == FORMAT_VERSION
            || h.format_version == FORMAT_VERSION_V3_COMPAT,
        "header.format_version {} not in {{{FORMAT_VERSION}, {FORMAT_VERSION_V3_COMPAT}}}",
        h.format_version,
    );
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
fn v3_compat_shim_forces_num_streams_one() {
    // On a v3 artifact, the v4 loader's compat shim must force num_streams=1
    // and alias substream_states_offset / substream_nb_totals_offset to the
    // v3 tile_*_offset values (so downstream CUDA can treat v3 == v4 K=1).
    let Some(f) = open_artifact() else { return };
    let h = f.header();
    if h.format_version != FORMAT_VERSION_V3_COMPAT {
        println!("SKIP: artifact is not v3 (got fmt={})", h.format_version);
        return;
    }
    let mut checked = 0usize;
    for e in f.toc() {
        if !e.is_llvq_tans() {
            continue;
        }
        assert_eq!(
            e.num_streams, 1,
            "v3 compat shim must force num_streams=1, got {} for {:?}",
            e.num_streams, e.name
        );
        assert_eq!(
            e.substream_states_offset, e.tile_states_offset,
            "v3 substream_states_offset must alias tile_states_offset"
        );
        assert_eq!(
            e.substream_nb_totals_offset, e.tile_nb_totals_offset,
            "v3 substream_nb_totals_offset must alias tile_nb_totals_offset"
        );
        checked += 1;
        if checked >= 5 {
            break;
        }
    }
    assert!(checked > 0, "no LLVQ tensors found to validate");
    println!("v3 compat shim verified on {} LLVQ tensors", checked);
}

#[test]
fn v3_compat_payload_aliases_tile_arrays() {
    // The LlvqTansPayload substream_states / substream_nb_totals slices must
    // be byte-equal to tile_states / tile_nb_totals for a v3 artifact.
    let Some(f) = open_artifact() else { return };
    if f.header().format_version != FORMAT_VERSION_V3_COMPAT {
        println!("SKIP: artifact is not v3");
        return;
    }
    let (idx, _) = f
        .toc()
        .iter()
        .enumerate()
        .find(|(_, e)| e.is_llvq_tans())
        .expect("no LLVQ tensor");
    let p = f.llvq_tans_payload(idx).expect("payload");
    assert_eq!(p.num_streams, 1);
    assert_eq!(p.substream_states.len(), p.tile_states.len());
    assert_eq!(p.substream_nb_totals.len(), p.tile_nb_totals.len());
    assert_eq!(
        p.substream_states, p.tile_states,
        "v3 substream_states must byte-equal tile_states"
    );
    assert_eq!(
        p.substream_nb_totals, p.tile_nb_totals,
        "v3 substream_nb_totals must byte-equal tile_nb_totals"
    );
    println!(
        "v3 compat payload alias verified ({} tiles)",
        p.n_tiles
    );
}

#[test]
fn synthetic_v4_toc_entry_parses() {
    // Construct a synthetic v4 TOC entry byte-buffer with num_streams=4
    // and verify the parser reads the new fields.
    use mistralrs_leech_q24::{TocEntry, TOC_ENTRY_SIZE};

    let mut buf = vec![0u8; TOC_ENTRY_SIZE];
    // name: "synthetic_v4"
    let name = b"synthetic_v4";
    buf[..name.len()].copy_from_slice(name);
    // role=LlvqTans (0)
    buf[200] = 0;
    // dtype_tag=Packed (5)
    buf[201] = 5;
    // rank=2
    buf[202..204].copy_from_slice(&2u16.to_le_bytes());
    // shape[0]=64, shape[1]=24
    buf[204..208].copy_from_slice(&64u32.to_le_bytes());
    buf[208..212].copy_from_slice(&24u32.to_le_bytes());
    // flags=0 @236
    // symbol_set_id=0 @240
    // tile_size=32 @244
    buf[244..248].copy_from_slice(&32u32.to_le_bytes());
    // k_beta=8 @248
    buf[248..252].copy_from_slice(&8u32.to_le_bytes());
    // k_offset=0 @252
    // R=2 @256
    buf[256..260].copy_from_slice(&2u32.to_le_bytes());
    // B=32 @260
    buf[260..264].copy_from_slice(&32u32.to_le_bytes());
    // num_streams=4 @264..268
    buf[264..268].copy_from_slice(&4u32.to_le_bytes());
    // n_blocks=64 @268
    buf[268..276].copy_from_slice(&64u64.to_le_bytes());
    // n_tiles=2 @276
    buf[276..284].copy_from_slice(&2u64.to_le_bytes());
    // buckets_offset=0x1000 @284
    buf[284..292].copy_from_slice(&0x1000u64.to_le_bytes());
    // tile_states_offset=0x2000 @316
    buf[316..324].copy_from_slice(&0x2000u64.to_le_bytes());
    // tile_nb_totals_offset=0x2100 @324
    buf[324..332].copy_from_slice(&0x2100u64.to_le_bytes());
    // tile_bitstream_offset=0x3000 @332
    buf[332..340].copy_from_slice(&0x3000u64.to_le_bytes());
    // substream_states_offset=0x4000 @428
    buf[428..436].copy_from_slice(&0x4000u64.to_le_bytes());
    // substream_nb_totals_offset=0x4100 @436
    buf[436..444].copy_from_slice(&0x4100u64.to_le_bytes());

    let entry = TocEntry::unpack(&buf, 0, FORMAT_VERSION).expect("parse v4 synthetic");
    assert_eq!(entry.name, "synthetic_v4");
    assert_eq!(entry.num_streams, 4);
    assert_eq!(entry.tile_size, 32);
    assert_eq!(entry.substream_states_offset, 0x4000);
    assert_eq!(entry.substream_nb_totals_offset, 0x4100);
    assert_eq!(entry.tile_states_offset, 0x2000);
    assert_eq!(entry.tile_nb_totals_offset, 0x2100);

    // Now rebuild with num_streams=24 (NOT a power of two ≤ 32) → must error.
    let mut bad = buf.clone();
    bad[264..268].copy_from_slice(&24u32.to_le_bytes());
    let err = TocEntry::unpack(&bad, 0, FORMAT_VERSION);
    assert!(err.is_err(), "K=24 must be rejected");

    // num_streams=8 but tile_size=4 (K does not divide T) → must error.
    let mut bad2 = buf.clone();
    bad2[264..268].copy_from_slice(&8u32.to_le_bytes());
    bad2[244..248].copy_from_slice(&4u32.to_le_bytes());
    let err2 = TocEntry::unpack(&bad2, 0, FORMAT_VERSION);
    assert!(err2.is_err(), "K=8, T=4 (K not dividing T) must be rejected");

    // Same buffer but read as v3-compat: num_streams must be forced to 1
    // and substream_*_offset must alias tile_*_offset.
    let entry_v3 =
        TocEntry::unpack(&buf, 0, FORMAT_VERSION_V3_COMPAT).expect("parse v3 compat");
    assert_eq!(entry_v3.num_streams, 1);
    assert_eq!(
        entry_v3.substream_states_offset, entry_v3.tile_states_offset,
        "v3 compat shim must alias substream_states_offset to tile_states_offset"
    );
    assert_eq!(
        entry_v3.substream_nb_totals_offset, entry_v3.tile_nb_totals_offset,
        "v3 compat shim must alias substream_nb_totals_offset to tile_nb_totals_offset"
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
