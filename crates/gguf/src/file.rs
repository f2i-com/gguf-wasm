//! Opening a GGUF file natively, and reading tensors out of it by range.
//!
//! Available with the `std` feature. Without it this crate parses a header it
//! is handed and never touches a filesystem, which is what a browser needs;
//! with it you get the same reader over an actual file.
//!
//! The distinction that matters is the same in both places: **the header is
//! not the file**. A quantized checkpoint is routinely several gigabytes, so
//! nothing here loads one. The header is parsed from the first few megabytes,
//! and every tensor after that is a byte range read on demand.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use gguf_quants::GgmlType;

use crate::error::{GgufError, Result};
use crate::reader::GgufHeader;
use crate::tensor::TensorInfo;

/// The first read, and the largest the header is allowed to be. A header's
/// size depends on how large the tokenizer vocabulary embedded in its metadata
/// is, which is not known before reading it, so the read doubles until it
/// parses.
const FIRST: usize = 1 << 20;
const LARGEST: usize = 64 << 20;

/// A GGUF file on disk: its header in memory, its tensors still on the disk.
#[derive(Debug)]
pub struct GgufReader {
    file: File,
    header: GgufHeader,
    size: u64,
}

impl GgufReader {
    /// Open a file and parse its header, reading only as much as that takes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        // Saturating rather than checked, and that is not a truncation bug:
        // `LARGEST` caps this at 64 MiB anyway, so a file too large to measure
        // in a `usize` just means "read the first chunk". Every comparison
        // against the real length stays in `u64` below.
        let whole = usize::try_from(size).unwrap_or(usize::MAX);
        let mut want = FIRST.min(whole).max(1);
        loop {
            let mut head = alloc_bytes(want);
            file.seek(SeekFrom::Start(0))?;
            read_exactly(&mut file, &mut head)?;
            match GgufHeader::from_bytes(&head) {
                Ok(header) => return Ok(Self { file, header, size }),
                Err(error) => {
                    // Only a short read is worth another go. Anything else --
                    // a bad magic, a count past the limits -- is what this
                    // file says, and reading more of it will not say otherwise.
                    if !error.needs_more_bytes() || want as u64 >= size || want >= LARGEST {
                        return Err(error);
                    }
                    want = (want * 4).min(LARGEST).min(whole);
                }
            }
        }
    }

    /// The parsed header: metadata, tensor names, shapes and offsets.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// How large the file is, which is not how large the header is.
    pub fn size(&self) -> u64 {
        self.size
    }

    fn info(&self, name: &str) -> Result<TensorInfo> {
        self.header
            .tensor_by_name(name)
            .cloned()
            .ok_or_else(|| GgufError::MissingKey(name.into()))
    }

    /// A tensor's bytes, exactly as the file stores them.
    ///
    /// For a quantized tensor these are its blocks, undecoded. That is the
    /// interesting form: 144 bytes per 256 values for Q4_K, against 1,024 once
    /// expanded, and a runtime that can read blocks should never see the
    /// expansion.
    pub fn tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let range = self.header.range_of(name)?;
        self.range(range.offset, range.bytes)
    }

    /// A tensor decoded to float32.
    pub fn tensor_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        // The bytes come first on purpose: `range` has already bounded them by
        // the file's own length, so by the time this reserves for the decoded
        // form the count it is reserving for is one the file could back.
        let bytes = self.tensor_bytes(name)?;
        let numel = info.numel();
        let mut out =
            vec![0.0f32; usize::try_from(numel).map_err(|_| GgufError::TooLargeForMachine(numel))?];
        gguf_quants::dequantize(info.dtype, &bytes, &mut out)
            .map_err(|e| GgufError::MissingKey(format!("{name}: {e}")))?;
        Ok(out)
    }

    /// Whole rows of a matrix, without touching the rest of it.
    ///
    /// Every block format packs a whole number of blocks into each row, so a
    /// row is individually addressable. For an embedding table that is the
    /// difference between a few kilobytes and a gigabyte, and it is why a
    /// vocabulary-sized tensor is usable at all.
    pub fn rows_f32(&mut self, name: &str, rows: &[u64]) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        // GGUF stores a shape fastest-varying first, so a matrix is
        // [width, rows] here and its width is the first dimension.
        let (width, count) = match info.shape.as_slice() {
            [width, count] => (*width, *count),
            _ => {
                return Err(GgufError::TooManyDims {
                    name: name.into(),
                    n_dims: info.shape.len() as u32,
                })
            }
        };
        let block = info.dtype.block_size() as u64;
        if width % block != 0 {
            return Err(GgufError::NotBlockAligned {
                name: name.into(),
                block: block as usize,
                numel: width,
            });
        }
        let row_bytes = (width / block) * info.dtype.type_size() as u64;
        let start = self.header.range_of(name)?.offset;
        // Both of these are numbers out of the file, so neither is cast: a
        // width past this machine's `usize` is a refusal, and a row count
        // times that width is checked rather than wrapped.
        let width_usize =
            usize::try_from(width).map_err(|_| GgufError::TooLargeForMachine(width))?;
        let capacity = rows
            .len()
            .checked_mul(width_usize)
            .ok_or(GgufError::Overflow("row output size"))?;
        let mut out = Vec::with_capacity(capacity);
        for &row in rows {
            if row >= count {
                return Err(GgufError::Truncated {
                    offset: row,
                    needed: count,
                });
            }
            let at = row
                .checked_mul(row_bytes)
                .and_then(|o| start.checked_add(o))
                .ok_or(GgufError::Overflow("row offset"))?;
            let bytes = self.range(at, row_bytes)?;
            let mut decoded = vec![0.0f32; width_usize];
            gguf_quants::dequantize(info.dtype, &bytes, &mut decoded)
                .map_err(|e| GgufError::MissingKey(format!("{name}: {e}")))?;
            out.extend_from_slice(&decoded);
        }
        Ok(out)
    }

    /// One tensor's dtype, for a caller deciding whether to decode it.
    pub fn dtype(&self, name: &str) -> Result<GgmlType> {
        Ok(self.info(name)?.dtype)
    }

    fn range(&mut self, offset: u64, length: u64) -> Result<Vec<u8>> {
        // Checked, and bounded by the file, before anything is reserved: both
        // numbers reach here from a header, and a header comes from the file.
        match offset.checked_add(length) {
            Some(end) if end <= self.size => {}
            _ => {
                return Err(GgufError::Truncated {
                    offset,
                    needed: length,
                })
            }
        }
        let mut bytes = alloc_bytes(
            usize::try_from(length).map_err(|_| GgufError::TooLargeForMachine(length))?,
        );
        self.file.seek(SeekFrom::Start(offset))?;
        read_exactly(&mut self.file, &mut bytes)?;
        Ok(bytes)
    }
}

fn alloc_bytes(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

fn read_exactly(file: &mut File, into: &mut [u8]) -> Result<()> {
    file.read_exact(into).map_err(GgufError::from)
}
