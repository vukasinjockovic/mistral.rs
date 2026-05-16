use thiserror::Error;

#[derive(Debug, Error)]
pub enum LeechQ24Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic: expected {:?}, got {got:x?}", crate::container::MAGIC)]
    BadMagic { got: [u8; 8] },

    #[error("unsupported format_version {0} (this build supports v3 only)")]
    UnsupportedVersion(u32),

    #[error("manifest json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("utf-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("CRC32 mismatch: file says {file:#010x}, computed {computed:#010x}")]
    CrcMismatch { file: u32, computed: u32 },

    #[error("invalid TOC entry {idx}: {reason}")]
    BadTocEntry { idx: usize, reason: String },

    #[error("invalid codebook header at offset {offset}: {reason}")]
    BadCodebookHeader { offset: u64, reason: String },

    #[error("truncated read: needed {needed} bytes at offset {offset}, file is {len} bytes")]
    Truncated {
        offset: u64,
        needed: usize,
        len: u64,
    },

    #[error("not an LLVQ_TANS tensor: {name:?} has role {role}")]
    NotLlvqTans { name: String, role: u8 },

    #[error("bucket unpack OOB: index {idx} ≥ n_blocks {n}")]
    BucketUnpackOob { idx: usize, n: usize },
}

pub type Result<T> = std::result::Result<T, LeechQ24Error>;
