use alloc::string::String;
use core::fmt;

use gguf_quants::dtype::UnknownDtype;

pub type Result<T> = core::result::Result<T, GgufError>;

/// Every way a GGUF header can fail to be one.
///
/// A parser reads numbers out of a file it has not verified and then allocates
/// according to them, so most of these are about refusing to do that: a length
/// that does not fit the machine, a count larger than the bytes that could hold
/// it, a product that overflows. None of it is memory-unsafe in Rust, and all
/// of it is the difference between an error and a panic or an out-of-memory.
///
/// Without `std` there is no `Io` variant, because without `std` this reader is
/// handed bytes and never opens anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// Only with the `std` feature, which is the only place a file is opened.
    #[cfg(feature = "std")]
    Io(String),
    BadMagic(u32),
    UnsupportedVersion(u32, &'static [u32]),
    UnknownValueType(u32),
    UnknownGgmlType(u32),
    UnknownDtype(UnknownDtype),
    BadUtf8,
    MissingKey(String),
    TypeMismatch {
        key: String,
        expected: &'static str,
        actual: &'static str,
    },
    Truncated {
        offset: u64,
        needed: u64,
    },
    NestedArray(String),
    TooManyDims {
        name: String,
        n_dims: u32,
    },
    NotBlockAligned {
        name: String,
        block: usize,
        numel: u64,
    },
    /// A count or length past what this parser will accept. `limit` is either
    /// a `ParseLimits` field or the bytes actually available, whichever bound
    /// it broke.
    TooLarge {
        what: &'static str,
        value: u64,
        limit: u64,
    },
    /// The same metadata key or tensor name twice. Which one wins would be a
    /// choice, and a file that makes a reader choose is malformed.
    Duplicate {
        what: &'static str,
        name: String,
    },
    /// `general.alignment` that is zero, or not a power of two.
    BadAlignment(u64),
    /// Arithmetic on values from the file left the range they must stay in.
    Overflow(&'static str),
    /// A length that does not fit this machine's `usize`. Real on `wasm32`,
    /// where `usize` is 32 bits and a GGUF length is 64.
    TooLargeForMachine(u64),
}

impl GgufError {
    /// Whether reading more of the file could change this answer.
    ///
    /// A header's size depends on how large the embedded tokenizer vocabulary
    /// is, which is not known before reading it, so a header read starts small
    /// and doubles. Without this, that loop cannot tell a short read from a
    /// file that will never be a header, and a wrong magic number costs the
    /// whole 64 MiB of doubling before it is refused.
    ///
    /// Only truncation says "read more". A count past `ParseLimits`, a bad
    /// magic, an unknown version or a duplicate name are all final: more bytes
    /// will not make them acceptable.
    pub fn needs_more_bytes(&self) -> bool {
        matches!(self, Self::Truncated { .. })
    }
}

impl From<UnknownDtype> for GgufError {
    fn from(e: UnknownDtype) -> Self {
        Self::UnknownDtype(e)
    }
}
impl From<core::str::Utf8Error> for GgufError {
    fn from(_: core::str::Utf8Error) -> Self {
        Self::BadUtf8
    }
}
impl From<alloc::string::FromUtf8Error> for GgufError {
    fn from(_: alloc::string::FromUtf8Error) -> Self {
        Self::BadUtf8
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic(m) => write!(f, "not a GGUF file: bad magic 0x{m:08x}"),
            Self::UnsupportedVersion(v, ok) => {
                write!(f, "unsupported GGUF version {v} (supported: {ok:?})")
            }
            Self::UnknownValueType(t) => write!(f, "unknown ValueType tag {t}"),
            Self::UnknownGgmlType(t) => write!(f, "unknown ggml dtype tag {t}"),
            Self::UnknownDtype(e) => write!(f, "{e}"),
            Self::BadUtf8 => write!(f, "string was not valid UTF-8"),
            Self::MissingKey(k) => write!(f, "expected metadata key `{k}` was not found"),
            Self::TypeMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "metadata key `{key}` had wrong type: expected {expected}, got {actual}"
            ),
            Self::Truncated { offset, needed } => write!(
                f,
                "file truncated: needed {needed} more bytes at offset {offset}"
            ),
            Self::NestedArray(k) => {
                write!(f, "nested arrays of arrays are not supported (key `{k}`)")
            }
            Self::TooManyDims { name, n_dims } => write!(
                f,
                "tensor `{name}` has unsupported {n_dims} dimensions (max 4)"
            ),
            Self::NotBlockAligned { name, block, numel } => write!(
                f,
                "tensor `{name}` length not a multiple of block size {block} (numel = {numel})"
            ),
            Self::TooLarge { what, value, limit } => {
                write!(f, "{what} is {value}, past the limit of {limit}")
            }
            Self::Duplicate { what, name } => write!(f, "duplicate {what}: `{name}`"),
            Self::BadAlignment(a) => write!(
                f,
                "general.alignment is {a}; it must be a power of two above zero"
            ),
            Self::Overflow(what) => write!(f, "{what} overflowed"),
            Self::TooLargeForMachine(v) => {
                write!(f, "length {v} does not fit this machine's pointer width")
            }
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(alloc::format!("{e}"))
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GgufError {}
