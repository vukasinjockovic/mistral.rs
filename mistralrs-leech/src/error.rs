use thiserror::Error;

#[derive(Debug, Error)]
pub enum LeechError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic: expected {:?}, got {got:x?}", crate::container::MAGIC)]
    BadMagic { got: [u8; 8] },

    #[error("unsupported format_version {0} (this build supports v1 only)")]
    UnsupportedVersion(u32),

    #[error("structural_schema_version mismatch: got {got:?}, expected {expected:?}")]
    SchemaMismatch { got: String, expected: String },

    #[error("manifest json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("utf-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("CRC32 mismatch: file says {file:#010x}, computed {computed:#010x}")]
    CrcMismatch { file: u32, computed: u32 },

    #[error("invalid TOC entry at index {idx}: {reason}")]
    BadTocEntry { idx: usize, reason: String },

    #[error("invalid payload header for tensor {name:?}: {reason}")]
    BadPayloadHeader { name: String, reason: String },

    #[error("truncated read: needed {needed} bytes at offset {offset}, file is {len} bytes")]
    Truncated {
        offset: u64,
        needed: usize,
        len: u64,
    },

    #[error("invalid overlay block: {0}")]
    BadOverlay(String),
}

pub type Result<T> = std::result::Result<T, LeechError>;
