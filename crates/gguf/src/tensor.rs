//! Per-tensor metadata. The dtype tag itself lives in `ggml-quants` because
//! quantization owns the layout semantics; we just re-export here.

use alloc::string::String;
use alloc::vec::Vec;

pub use gguf_quants::GgmlType;

/// Per-tensor metadata, parsed from the GGUF tensor info table.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,

    /// Logical shape, *as stored* in GGUF. Note GGUF dimensions are stored
    /// fastest-varying-first (the mirror of NumPy / PyTorch convention).
    pub shape: Vec<u64>,

    pub dtype: GgmlType,

    /// Offset *within the tensor data section* (not the file). Caller adds
    /// `GgufFile::tensor_data_start` to get the absolute file offset.
    pub offset: u64,
}

impl TensorInfo {
    /// How many values this tensor holds.
    ///
    /// Saturating, not wrapping. A `TensorInfo` the parser produced can never
    /// reach the ceiling -- the product is checked there, against
    /// `ParseLimits::max_tensor_elements` -- but one built by hand could, and
    /// a shape from an untrusted file must not turn into a small plausible
    /// number by wrapping.
    pub fn numel(&self) -> u64 {
        self.shape
            .iter()
            .fold(1u64, |total, d| total.saturating_mul(*d))
    }

    /// How many bytes it occupies, in whatever format it is stored in.
    /// Saturating, for the same reason.
    pub fn nbytes(&self) -> u64 {
        let block = (self.dtype.block_size() as u64).max(1);
        (self.numel() / block).saturating_mul(self.dtype.type_size() as u64)
    }
}
