//! OVERLAY_BLOCK parser — mirror of `packer/core/overlay.py`.
//!
//! Layout (all little-endian):
//!     +0  u32 overlay_count
//!     +4  u32 total_size                (entire block, including this header)
//!     +8  u64 reserved=0
//!     [entries: variable-length]
//!         u8  kind (0=block24, 1=lora)
//!         u8  reserved=0
//!         u16 name_len
//!         char[name_len] name (utf-8)
//!         pad to 4-byte alignment from block start
//!         u64 payload_offset (from OVERLAY_BLOCK start)
//!         u64 payload_size
//!     [payloads concatenated]

use crate::error::{LeechError, Result};
use byteorder::{ByteOrder, LittleEndian as LE};

pub const OVERLAY_HEADER_SIZE: usize = 16;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayKind {
    Block24 = 0,
    Lora = 1,
}

impl OverlayKind {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(OverlayKind::Block24),
            1 => Ok(OverlayKind::Lora),
            other => Err(LeechError::BadOverlay(format!("unknown overlay kind {other}"))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OverlayEntry<'a> {
    pub kind: OverlayKind,
    pub name: String,
    pub payload: &'a [u8],
}

pub fn parse_overlay_block(block: &[u8]) -> Result<Vec<OverlayEntry<'_>>> {
    if block.is_empty() {
        return Ok(Vec::new());
    }
    if block.len() < OVERLAY_HEADER_SIZE {
        return Err(LeechError::BadOverlay(format!(
            "overlay block too short: {} < {OVERLAY_HEADER_SIZE}",
            block.len()
        )));
    }
    let overlay_count = LE::read_u32(&block[0..4]) as usize;
    let total_size = LE::read_u32(&block[4..8]) as usize;
    // bytes 8..16 reserved.
    if total_size != block.len() {
        return Err(LeechError::BadOverlay(format!(
            "overlay total_size {total_size} != block length {}",
            block.len()
        )));
    }

    let mut entries = Vec::with_capacity(overlay_count);
    let mut cur = OVERLAY_HEADER_SIZE;
    for i in 0..overlay_count {
        if cur + 4 > block.len() {
            return Err(LeechError::BadOverlay(format!(
                "truncated overlay entry header at idx {i}"
            )));
        }
        let kind = OverlayKind::from_u8(block[cur])?;
        // block[cur + 1] is reserved.
        let name_len = LE::read_u16(&block[cur + 2..cur + 4]) as usize;
        cur += 4;
        if cur + name_len > block.len() {
            return Err(LeechError::BadOverlay(format!(
                "overlay name truncated at idx {i}"
            )));
        }
        let name = std::str::from_utf8(&block[cur..cur + name_len])
            .map_err(LeechError::from)?
            .to_owned();
        cur += name_len;
        let pad = (4 - (cur % 4)) % 4;
        cur += pad;
        if cur + 16 > block.len() {
            return Err(LeechError::BadOverlay(format!(
                "overlay payload header truncated at idx {i}"
            )));
        }
        let payload_offset = LE::read_u64(&block[cur..cur + 8]) as usize;
        let payload_size = LE::read_u64(&block[cur + 8..cur + 16]) as usize;
        cur += 16;
        let end = payload_offset
            .checked_add(payload_size)
            .ok_or_else(|| LeechError::BadOverlay(format!("payload offset overflow at idx {i}")))?;
        if end > block.len() {
            return Err(LeechError::BadOverlay(format!(
                "overlay payload {i} extends past block: {end} > {}",
                block.len()
            )));
        }
        entries.push(OverlayEntry {
            kind,
            name,
            payload: &block[payload_offset..end],
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_block_is_empty() {
        assert!(parse_overlay_block(&[]).unwrap().is_empty());
    }

    #[test]
    fn truncated_header_fails() {
        let bad = [0u8; 8];
        assert!(parse_overlay_block(&bad).is_err());
    }

    /// Synthesize a minimal overlay block with one entry and verify it roundtrips.
    #[test]
    fn synthetic_single_entry() {
        let name = "block24.safetensors";
        let payload = b"\x01\x02\x03\x04\x05";
        // entry header: 4 B + name_len = 4 + 19 = 23, pad to 24, +16 = 40.
        let entry_header_unpadded = 4 + name.len();
        let pad = (4 - (entry_header_unpadded % 4)) % 4; // = 1
        let entry_header_total = entry_header_unpadded + pad + 16; // 40
        let payload_offset = OVERLAY_HEADER_SIZE + entry_header_total; // 56
        let total = payload_offset + payload.len();

        let mut block = vec![0u8; total];
        LE::write_u32(&mut block[0..4], 1); // overlay_count
        LE::write_u32(&mut block[4..8], total as u32);
        // bytes 8..16 reserved 0.

        let mut cur = OVERLAY_HEADER_SIZE;
        block[cur] = OverlayKind::Block24 as u8;
        // block[cur+1] reserved
        LE::write_u16(&mut block[cur + 2..cur + 4], name.len() as u16);
        cur += 4;
        block[cur..cur + name.len()].copy_from_slice(name.as_bytes());
        cur += name.len();
        cur += pad;
        LE::write_u64(&mut block[cur..cur + 8], payload_offset as u64);
        LE::write_u64(&mut block[cur + 8..cur + 16], payload.len() as u64);
        cur += 16;
        assert_eq!(cur, payload_offset);
        block[payload_offset..payload_offset + payload.len()].copy_from_slice(payload);

        let parsed = parse_overlay_block(&block).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].kind, OverlayKind::Block24);
        assert_eq!(parsed[0].name, name);
        assert_eq!(parsed[0].payload, payload);
    }
}
