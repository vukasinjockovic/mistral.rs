//! Per-symbol-set FSE codebook blob parser.
//!
//! Layout (one entry per symbol-set, packed contiguously starting at
//! `Header::codebook_offset`):
//!
//!   CB_HEADER (32 B)            symbol_set_id, n_codebooks (=4),
//!                                n_symbols, table_log (=10),
//!                                w_offset, w_min, w_max, has_encoder_tables
//!   decode_tables               n_codebooks × M × u32   (M = 1 << table_log)
//!
//! After all entries the blob is padded to 64-byte alignment by the writer.
//! Each decode-table entry packs (sym, nb_bits, base) into the u32:
//!   bits  0..8   symbol index in [0, n_symbols)
//!   bits  8..16  nb_bits to read on this transition
//!   bits 16..32  base state (offset within the table, masked by M-1)
//!
//! Symbol → w mapping:  w = (sym as i32) - w_offset.

use crate::error::{LeechQ24Error, Result};
use byteorder::{ByteOrder, LittleEndian as LE};

pub const CB_HEADER_SIZE: usize = 32;
pub const N_CODEBOOKS: usize = 4;
pub const TABLE_LOG: u32 = 10;
pub const M_TABLE: usize = 1 << TABLE_LOG;

#[derive(Debug, Clone)]
pub struct CodebookSet {
    pub symbol_set_id: u32,
    pub n_codebooks: u32,
    pub n_symbols: u32,
    pub table_log: u32,
    pub w_offset: u32,
    pub w_min: i32,
    pub w_max: i32,
    pub has_encoder_tables: bool,
    /// Byte offset of `decode_tables` inside the file.
    pub decode_tables_offset: u64,
    /// Length of `decode_tables` in bytes (n_codebooks × M × 4).
    pub decode_tables_bytes: u64,
}

impl CodebookSet {
    /// Number of bytes consumed in the blob (header + decode tables).
    pub fn total_size(&self) -> u64 {
        CB_HEADER_SIZE as u64 + self.decode_tables_bytes
    }

    /// Look up one decode-table entry packed as (sym, nb_bits, base).
    /// `cb_idx` ∈ [0, N_CODEBOOKS), `state` ∈ [0, M_TABLE).
    pub fn decode_entry(
        &self,
        decode_blob: &[u8],
        cb_idx: usize,
        state: u32,
    ) -> Result<DecodeEntry> {
        if cb_idx >= self.n_codebooks as usize {
            return Err(LeechQ24Error::BadCodebookHeader {
                offset: self.decode_tables_offset,
                reason: format!(
                    "cb_idx {} ≥ n_codebooks {}",
                    cb_idx, self.n_codebooks
                ),
            });
        }
        if (state as usize) >= M_TABLE {
            return Err(LeechQ24Error::BadCodebookHeader {
                offset: self.decode_tables_offset,
                reason: format!("state {} ≥ M {}", state, M_TABLE),
            });
        }
        let off = (cb_idx * M_TABLE + state as usize) * 4;
        if off + 4 > decode_blob.len() {
            return Err(LeechQ24Error::Truncated {
                offset: self.decode_tables_offset + off as u64,
                needed: 4,
                len: decode_blob.len() as u64,
            });
        }
        let raw = LE::read_u32(&decode_blob[off..off + 4]);
        Ok(DecodeEntry::from_raw(raw))
    }
}

/// Decoded view of one packed `u32` table entry.
#[derive(Debug, Clone, Copy)]
pub struct DecodeEntry {
    pub sym: u8,
    pub nb_bits: u8,
    pub base: u16,
}

impl DecodeEntry {
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        DecodeEntry {
            sym: (raw & 0xFF) as u8,
            nb_bits: ((raw >> 8) & 0xFF) as u8,
            base: ((raw >> 16) & 0xFFFF) as u16,
        }
    }
}

/// Parse the codebook blob into a list of `CodebookSet` entries.
///
/// `blob` must be the file bytes; `blob_offset` is the absolute file offset
/// of the codebook blob (so `decode_tables_offset` in each set is absolute).
/// `blob_len` is the length of the blob from `Header::codebook_size`.
pub fn parse_codebook_blob(
    blob: &[u8],
    blob_offset: u64,
    blob_len: u64,
) -> Result<Vec<CodebookSet>> {
    let end = blob_offset
        .checked_add(blob_len)
        .ok_or(LeechQ24Error::BadCodebookHeader {
            offset: blob_offset,
            reason: "blob_offset + blob_len overflow".to_owned(),
        })? as usize;
    let start = blob_offset as usize;
    if end > blob.len() {
        return Err(LeechQ24Error::Truncated {
            offset: blob_offset,
            needed: blob_len as usize,
            len: blob.len() as u64,
        });
    }
    let mut sets = Vec::new();
    let mut pos = start;
    while pos + CB_HEADER_SIZE <= end {
        let h = &blob[pos..pos + CB_HEADER_SIZE];
        // The writer pads the blob to ALIGN with zero bytes — treat a zero
        // header as end-of-list.
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let symbol_set_id = LE::read_u32(&h[0..4]);
        let n_codebooks = LE::read_u32(&h[4..8]);
        let n_symbols = LE::read_u32(&h[8..12]);
        let table_log = LE::read_u32(&h[12..16]);
        let w_offset = LE::read_u32(&h[16..20]);
        let w_min = LE::read_i32(&h[20..24]);
        let w_max = LE::read_i32(&h[24..28]);
        let has_encoder_tables = LE::read_u32(&h[28..32]) != 0;

        if n_codebooks as usize != N_CODEBOOKS {
            return Err(LeechQ24Error::BadCodebookHeader {
                offset: pos as u64,
                reason: format!(
                    "n_codebooks {} != {} (only 4-codebook tANS supported)",
                    n_codebooks, N_CODEBOOKS
                ),
            });
        }
        if table_log != TABLE_LOG {
            return Err(LeechQ24Error::BadCodebookHeader {
                offset: pos as u64,
                reason: format!(
                    "table_log {} != {} (TABLE_LOG hardcoded for now)",
                    table_log, TABLE_LOG
                ),
            });
        }

        let m = 1u64 << table_log;
        let decode_tables_bytes = (n_codebooks as u64) * m * 4;
        let decode_tables_offset = (pos + CB_HEADER_SIZE) as u64;
        let next_pos = pos + CB_HEADER_SIZE + decode_tables_bytes as usize;
        if next_pos > end {
            return Err(LeechQ24Error::Truncated {
                offset: decode_tables_offset,
                needed: decode_tables_bytes as usize,
                len: blob.len() as u64,
            });
        }
        sets.push(CodebookSet {
            symbol_set_id,
            n_codebooks,
            n_symbols,
            table_log,
            w_offset,
            w_min,
            w_max,
            has_encoder_tables,
            decode_tables_offset,
            decode_tables_bytes,
        });
        pos = next_pos;
    }
    Ok(sets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_entry_unpack() {
        // sym=7, nb_bits=3, base=0x1234
        let raw = 0x1234_03_07u32;
        let e = DecodeEntry::from_raw(raw);
        assert_eq!(e.sym, 0x07);
        assert_eq!(e.nb_bits, 0x03);
        assert_eq!(e.base, 0x1234);
    }
}
