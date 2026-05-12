//! Integration test for the `.leech` container reader.
//!
//! Requires a real fixture file. Default path `/tmp/qwopus.leech` (the 4.05 GB
//! V6-base bundle); override with `LEECH_FIXTURE=/path/to/file.leech`.
//!
//! When the fixture is absent the test is skipped (printed and treated as
//! pass) so CI without the fixture doesn't fail.

use mistralrs_leech::{DtypeTag, LeechFile, OpenOptions, Role};
use std::path::PathBuf;

fn fixture_path() -> Option<PathBuf> {
    let p = std::env::var("LEECH_FIXTURE").unwrap_or_else(|_| "/tmp/qwopus.leech".to_owned());
    let pb = PathBuf::from(p);
    if pb.exists() {
        Some(pb)
    } else {
        None
    }
}

#[test]
fn parse_qwopus_fixture_basics() {
    let Some(path) = fixture_path() else {
        eprintln!("LEECH_FIXTURE not set and /tmp/qwopus.leech missing — skipping");
        return;
    };

    let f = LeechFile::open(&path).expect("open .leech");

    // Header sanity.
    let h = f.header();
    assert_eq!(h.format_version, 1);
    assert!(h.toc_entry_count > 0);
    assert!(h.payload_offset >= h.toc_offset + h.toc_entry_count * 128);

    // Manifest sanity.
    let m = f.manifest();
    assert_eq!(m.format_version, 1);
    assert_eq!(m.structural_schema_version, "LLVQ_NIEMEIER_G24_v1");

    // 248 LLVQ + 2 fp8 + 177 bf16 = 427 tensors for V6-base.
    println!("tensor_count = {}", f.tensor_count());
    let mut role_counts = [0usize; 4];
    for entry in f.toc() {
        role_counts[entry.role as usize] += 1;
    }
    println!(
        "roles: LLVQ={}, fp8={}, bf16={}, overlay_role={}",
        role_counts[Role::Llvq as usize],
        role_counts[Role::Fp8E4m3 as usize],
        role_counts[Role::Bf16 as usize],
        role_counts[Role::Fp16Overlay as usize],
    );
    // Hard floors based on V6-base bundle.
    assert!(role_counts[Role::Llvq as usize] >= 200);
    assert!(role_counts[Role::Bf16 as usize] >= 100);
}

#[test]
fn parse_one_llvq_payload() {
    let Some(path) = fixture_path() else {
        eprintln!("fixture missing — skipping");
        return;
    };
    let f = LeechFile::open(&path).expect("open .leech");

    // Find the first LLVQ tensor and parse its payload header.
    let (idx, entry) = f
        .toc()
        .iter()
        .enumerate()
        .find(|(_, e)| e.role == Role::Llvq)
        .expect("at least one LLVQ tensor");

    assert_eq!(entry.dtype_tag, DtypeTag::Packed);
    let p = f.parse_llvq_payload(idx).expect("parse payload");
    println!(
        "{}: R={} B={} ms_used={} idx_bits={} K_beta={} K_offset={} has_offset={} leftover_kind={}",
        entry.name,
        p.r,
        p.b,
        p.ms_used,
        p.idx_bits,
        p.k_beta,
        p.k_offset,
        p.has_offset,
        p.leftover_kind,
    );

    // ms_used must be ≤ 18 for V6.
    assert!(p.ms_used <= 18, "ms_used out of expected range: {}", p.ms_used);
    // idx_bits must match per-block bit width arithmetic.
    let expected_bits =
        p.idx_bits as u32 + 3 + if p.has_offset { 3 } else { 0 };
    assert_eq!(p.per_block_bits, expected_bits);

    // β codebook length cross-check.
    let beta = p.beta_codebook_f16();
    assert_eq!(beta.len(), p.r as usize * p.k_beta as usize);

    // Packed stream length cross-check.
    let total_bits = p.r as usize * p.b as usize * p.per_block_bits as usize;
    let expected_stream = (total_bits + 7) / 8;
    assert_eq!(p.packed_stream.len(), expected_stream);
}

#[test]
fn crc_verifies() {
    let Some(path) = fixture_path() else {
        eprintln!("fixture missing — skipping");
        return;
    };
    // Full-file CRC. ~4 GB, takes a few seconds.
    let _f = LeechFile::open_with(
        &path,
        OpenOptions {
            verify_crc: true,
            require_schema: true,
        },
    )
    .expect("CRC must verify");
}

#[test]
fn overlay_present_for_ft_bundle() {
    let Some(path) = fixture_path() else {
        eprintln!("fixture missing — skipping");
        return;
    };
    let f = LeechFile::open(&path).expect("open .leech");
    if f.header().has_overlays() {
        let overlays = f.overlays().expect("parse overlays");
        println!("overlay_count = {}", overlays.len());
        for o in &overlays {
            println!("  kind={:?} name={:?} payload_bytes={}", o.kind, o.name, o.payload.len());
        }
        // V6b has block24 + lora = 2 overlays.
        assert!(!overlays.is_empty());
    } else {
        eprintln!("no overlays in fixture; skipping overlay assertions");
    }
}
