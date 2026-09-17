//! GGUF header parsing: typed metadata, plus the shape, format and byte range
//! of every tensor.
//!
//! **A header is not a file.** What comes out of here describes where a tensor
//! lives; it does not contain one, and it does not keep the bytes it was parsed
//! from. That is deliberate: a checkpoint is routinely several gigabytes and a
//! caller reads tensor bodies by range, from a `File`, a `Blob`, a ranged
//! request or a peer. The `std` feature's [`crate::GgufReader`] is where an
//! actual file gets read.
//!
//! ## Parsing something you did not write
//!
//! Every number below comes out of a file this crate has no reason to trust,
//! and several of them decide how much memory to allocate. So:
//!
//!   * lengths are `u64` in the format and `usize` on the machine, and on
//!     `wasm32` that is a narrowing -- checked, never cast;
//!   * a count is refused if the remaining bytes could not hold that many
//!     elements even at their minimum size, which bounds every allocation by
//!     the file's own length before a single byte is reserved;
//!   * additions and products that could overflow are checked;
//!   * [`ParseLimits`] bounds the rest, and a caller that knows its inputs can
//!     raise or lower it.
//!
//! None of this is about Rust memory safety, which is not in question. It is
//! about a malformed or hostile file producing an error rather than a panic, a
//! silently truncated length, or an allocation that takes the process down.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::error::{GgufError, Result};
use crate::tensor::{GgmlType, TensorInfo};
use crate::value::{Array, Value, ValueType};
use crate::{DEFAULT_ALIGNMENT, GGUF_MAGIC, SUPPORTED_VERSIONS};

/// What a parser will accept from a file it has not verified.
///
/// The defaults are far above any real checkpoint and far below anything that
/// would hurt: a production GGUF has tens of metadata entries, hundreds to
/// thousands of tensors, and arrays whose largest member is a vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLimits {
    /// Metadata key/value pairs. Real files carry tens.
    pub max_metadata_entries: u64,
    /// Tensors. A 400B model has a few thousand.
    pub max_tensors: u64,
    /// Elements in one metadata array. A vocabulary is the large one; 256k
    /// vocabularies exist and this leaves room above them.
    pub max_array_len: u64,
    /// Bytes in one string. A chat template is the long one.
    pub max_string_bytes: u64,
    /// Dimensions per tensor. GGUF itself allows four.
    pub max_dimensions: u32,
    /// Elements in one tensor, which bounds its byte length too.
    pub max_tensor_elements: u64,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_metadata_entries: 1 << 16,
            max_tensors: 1 << 20,
            max_array_len: 1 << 24,
            max_string_bytes: 1 << 24,
            max_dimensions: 4,
            max_tensor_elements: 1 << 44,
        }
    }
}

/// Where a tensor's bytes are, in the file the header came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRange {
    /// Absolute offset from the start of the file.
    pub offset: u64,
    /// How many bytes the tensor occupies, in whatever format it is stored in.
    pub bytes: u64,
}

impl TensorRange {
    /// The end offset, which the parser has already checked does not overflow.
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.bytes)
    }
}

/// A parsed GGUF header. Cheap to clone (`Arc` internally).
///
/// It holds no file bytes. Ask it where a tensor is with [`Self::tensor_range`]
/// and read that range from wherever the file actually is.
#[derive(Clone, Debug)]
pub struct GgufHeader {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    version: u32,
    metadata: BTreeMap<String, Value>,
    tensors: Vec<TensorInfo>,
    tensors_by_name: BTreeMap<String, usize>,
    tensor_data_start: u64,
    alignment: u64,
    header_bytes: u64,
}

impl GgufHeader {
    /// Parse a header. `bytes` need only reach the end of the tensor-info
    /// table; nothing after that is read, and nothing is retained.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_limits(bytes, ParseLimits::default())
    }

    /// As [`Self::from_bytes`], with limits of your own.
    pub fn from_bytes_with_limits(bytes: &[u8], limits: ParseLimits) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(parse(bytes, limits)?),
        })
    }

    pub fn version(&self) -> u32 {
        self.inner.version
    }
    pub fn alignment(&self) -> u64 {
        self.inner.alignment
    }

    /// Where the tensor data section begins, which is also how far into the
    /// file the header reaches once padded.
    pub fn tensor_data_start(&self) -> u64 {
        self.inner.tensor_data_start
    }

    /// How many bytes the header itself occupied, before alignment padding.
    pub fn header_bytes(&self) -> u64 {
        self.inner.header_bytes
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.inner.metadata
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.inner.tensors
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.inner
            .tensors_by_name
            .get(name)
            .map(|&i| &self.inner.tensors[i])
    }

    /// Where a tensor's bytes are in the file.
    ///
    /// This is the whole of what a header can tell you about a body. Reading
    /// that range is the caller's job, because only the caller knows whether
    /// the file is a `File`, a `Blob`, an HTTP resource or a peer.
    pub fn tensor_range(&self, tensor: &TensorInfo) -> TensorRange {
        TensorRange {
            // Both checked during parsing, so neither can overflow here.
            offset: self.inner.tensor_data_start.saturating_add(tensor.offset),
            bytes: tensor.nbytes(),
        }
    }

    /// [`Self::tensor_range`] by name.
    pub fn range_of(&self, name: &str) -> Result<TensorRange> {
        let tensor = self
            .tensor_by_name(name)
            .ok_or_else(|| GgufError::MissingKey(name.to_string()))?;
        Ok(self.tensor_range(tensor))
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.inner
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        let v = self.get(key)?;
        v.as_u64().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "integer",
            actual: v.type_str(),
        })
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        let v = self.get(key)?;
        v.as_f32().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "float",
            actual: v.type_str(),
        })
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        let v = self.get(key)?;
        v.as_str().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "string",
            actual: v.type_str(),
        })
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let v = self.get(key)?;
        v.as_bool().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "bool",
            actual: v.type_str(),
        })
    }

    /// Convenience: `general.architecture`.
    pub fn architecture(&self) -> Result<&str> {
        self.get_str("general.architecture")
    }
}

// ----- parsing -------------------------------------------------------------

/// A `u64` from the file, as a `usize` on this machine -- or an error. On
/// `wasm32` this is a real narrowing and the difference between a refusal and
/// a silently truncated length.
#[inline]
fn as_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| GgufError::TooLargeForMachine(value))
}

#[inline]
fn at_most(what: &'static str, value: u64, limit: u64) -> Result<u64> {
    if value > limit {
        Err(GgufError::TooLarge { what, value, limit })
    } else {
        Ok(value)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: ParseLimits,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], limits: ParseLimits) -> Self {
        Self {
            bytes,
            pos: 0,
            limits,
        }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    #[inline]
    fn ensure(&self, n: usize) -> Result<()> {
        // Checked: `pos + n` on a length read out of the file is exactly where
        // an unchecked add would wrap and turn a truncated file into a read.
        match self.pos.checked_add(n) {
            Some(end) if end <= self.bytes.len() => Ok(()),
            _ => Err(GgufError::Truncated {
                offset: self.pos as u64,
                needed: (n as u64).saturating_sub(self.remaining() as u64),
            }),
        }
    }

    #[inline]
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        self.ensure(n)?;
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    #[inline]
    fn read_i8(&mut self) -> Result<i8> {
        self.read_u8().map(|b| b as i8)
    }
    #[inline]
    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    #[inline]
    fn read_i16(&mut self) -> Result<i16> {
        self.read_u16().map(|v| v as i16)
    }
    #[inline]
    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    #[inline]
    fn read_i32(&mut self) -> Result<i32> {
        self.read_u32().map(|v| v as i32)
    }
    #[inline]
    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    #[inline]
    fn read_i64(&mut self) -> Result<i64> {
        self.read_u64().map(|v| v as i64)
    }
    #[inline]
    fn read_f32(&mut self) -> Result<f32> {
        self.read_u32().map(f32::from_bits)
    }
    #[inline]
    fn read_f64(&mut self) -> Result<f64> {
        self.read_u64().map(f64::from_bits)
    }
    #[inline]
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = at_most(
            "string length",
            self.read_u64()?,
            self.limits.max_string_bytes,
        )?;
        let len = as_usize(len)?;
        // `take` bounds-checks before anything is copied, so a length larger
        // than the file cannot reserve for itself.
        Ok(String::from_utf8(self.take(len)?.to_vec())?)
    }

    /// How many elements of `size` bytes each could possibly remain. Reserving
    /// past this is reserving for bytes that are not there, which is how a
    /// four-byte length field turns into a four-gigabyte allocation.
    fn capacity_for(&self, count: u64, size: usize) -> Result<usize> {
        let possible = (self.remaining() / size.max(1)) as u64;
        at_most(
            "array length",
            count,
            possible.min(self.limits.max_array_len),
        )?;
        as_usize(count)
    }

    fn read_value(&mut self, ty: ValueType, key: &str) -> Result<Value> {
        Ok(match ty {
            ValueType::U8 => Value::U8(self.read_u8()?),
            ValueType::I8 => Value::I8(self.read_i8()?),
            ValueType::U16 => Value::U16(self.read_u16()?),
            ValueType::I16 => Value::I16(self.read_i16()?),
            ValueType::U32 => Value::U32(self.read_u32()?),
            ValueType::I32 => Value::I32(self.read_i32()?),
            ValueType::F32 => Value::F32(self.read_f32()?),
            ValueType::Bool => Value::Bool(self.read_bool()?),
            ValueType::String => Value::String(self.read_string()?),
            ValueType::U64 => Value::U64(self.read_u64()?),
            ValueType::I64 => Value::I64(self.read_i64()?),
            ValueType::F64 => Value::F64(self.read_f64()?),
            ValueType::Array => Value::Array(self.read_array(key)?),
        })
    }

    fn read_array(&mut self, key: &str) -> Result<Array> {
        let elem_ty = ValueType::from_u32(self.read_u32()?)?;
        let count = self.read_u64()?;

        macro_rules! collect {
            ($variant:ident, $reader:ident, $size:expr) => {{
                let len = self.capacity_for(count, $size)?;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(self.$reader()?);
                }
                Array::$variant(out)
            }};
        }

        Ok(match elem_ty {
            ValueType::U8 => collect!(U8, read_u8, 1),
            ValueType::I8 => collect!(I8, read_i8, 1),
            ValueType::U16 => collect!(U16, read_u16, 2),
            ValueType::I16 => collect!(I16, read_i16, 2),
            ValueType::U32 => collect!(U32, read_u32, 4),
            ValueType::I32 => collect!(I32, read_i32, 4),
            ValueType::F32 => collect!(F32, read_f32, 4),
            ValueType::Bool => collect!(Bool, read_bool, 1),
            ValueType::U64 => collect!(U64, read_u64, 8),
            ValueType::I64 => collect!(I64, read_i64, 8),
            ValueType::F64 => collect!(F64, read_f64, 8),
            // A string is at least its own eight-byte length prefix.
            ValueType::String => collect!(String, read_string, 8),
            ValueType::Array => return Err(GgufError::NestedArray(key.to_string())),
        })
    }
}

fn parse(bytes: &[u8], limits: ParseLimits) -> Result<Inner> {
    let mut c = Cursor::new(bytes, limits);

    let magic = c.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic(magic));
    }

    let version = c.read_u32()?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion(version, SUPPORTED_VERSIONS));
    }

    let tensor_count = at_most("tensor count", c.read_u64()?, limits.max_tensors)?;
    let kv_count = at_most(
        "metadata entry count",
        c.read_u64()?,
        limits.max_metadata_entries,
    )?;

    // Metadata. A key is at least its own length prefix plus a type tag, so
    // twelve bytes bounds how many could possibly be here.
    at_most(
        "metadata entry count",
        kv_count,
        (c.remaining() / 12) as u64,
    )?;
    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = c.read_string()?;
        let ty = ValueType::from_u32(c.read_u32()?)?;
        let value = c.read_value(ty, &key)?;
        // Which of two identical keys wins would be a choice, and a file that
        // makes a reader choose is malformed.
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufError::Duplicate {
                what: "metadata key",
                name: key,
            });
        }
    }

    let alignment = match metadata.get("general.alignment") {
        Some(value) => value.as_u64().unwrap_or(DEFAULT_ALIGNMENT),
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::BadAlignment(alignment));
    }

    // Tensor table. The smallest possible entry is a length prefix, an empty
    // name, a dimension count, a dtype and an offset: twenty bytes.
    at_most("tensor count", tensor_count, (c.remaining() / 20) as u64)?;
    let mut tensors = Vec::with_capacity(as_usize(tensor_count)?);
    let mut tensors_by_name = BTreeMap::new();
    for index in 0..tensor_count {
        let name = c.read_string()?;
        let n_dims = c.read_u32()?;
        if n_dims > limits.max_dimensions {
            return Err(GgufError::TooManyDims { name, n_dims });
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(c.read_u64()?);
        }
        let dtype = GgmlType::from_u32(c.read_u32()?)?;
        let offset = c.read_u64()?;

        // A shape is a product of file values, so it is exactly where an
        // unchecked multiply wraps to something plausible and small.
        let mut numel: u64 = 1;
        for dimension in &shape {
            numel = numel
                .checked_mul(*dimension)
                .ok_or(GgufError::Overflow("tensor element count"))?;
        }
        at_most("tensor element count", numel, limits.max_tensor_elements)?;

        let block = dtype.block_size();
        if block > 1 && !numel.is_multiple_of(block as u64) {
            return Err(GgufError::NotBlockAligned { name, block, numel });
        }
        // And the byte length, and where it ends, so `tensor_range` cannot.
        let nbytes = (numel / block.max(1) as u64)
            .checked_mul(dtype.type_size() as u64)
            .ok_or(GgufError::Overflow("tensor byte length"))?;
        offset
            .checked_add(nbytes)
            .ok_or(GgufError::Overflow("tensor end offset"))?;

        if tensors_by_name
            .insert(name.clone(), as_usize(index)?)
            .is_some()
        {
            return Err(GgufError::Duplicate {
                what: "tensor name",
                name,
            });
        }
        tensors.push(TensorInfo {
            name,
            shape,
            dtype,
            offset,
        });
    }

    let header_bytes = c.pos as u64;
    let tensor_data_start =
        align_up(header_bytes, alignment).ok_or(GgufError::Overflow("tensor data start"))?;
    // Every tensor's end is relative to that start, and it must stay a number.
    for tensor in &tensors {
        tensor_data_start
            .checked_add(tensor.offset)
            .and_then(|start| start.checked_add(tensor.nbytes()))
            .ok_or(GgufError::Overflow("tensor end offset"))?;
    }

    Ok(Inner {
        version,
        metadata,
        tensors,
        tensors_by_name,
        tensor_data_start,
        alignment,
        header_bytes,
    })
}

#[inline]
fn align_up(value: u64, align: u64) -> Option<u64> {
    if align == 0 {
        return Some(value);
    }
    match value % align {
        0 => Some(value),
        r => value.checked_add(align - r),
    }
}

// ----- minimal write-side for round-trip tests -----------------------------

/// Writes a GGUF file from already-parsed pieces. Intended primarily for
/// round-trip testing; production writers will likely want a streaming variant.
#[doc(hidden)]
pub fn write_to_vec(
    metadata: &BTreeMap<String, Value>,
    tensors: &[(TensorInfo, Vec<u8>)],
    alignment: u64,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (key, value) in metadata {
        write_string(&mut out, key);
        write_value(&mut out, value);
    }

    // First pass: tensor offsets relative to tensor_data_start.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for (info, data) in tensors {
        offsets.push(cursor);
        let nbytes = info.nbytes();
        debug_assert_eq!(
            nbytes as usize,
            data.len(),
            "tensor `{}` data length mismatch",
            info.name
        );
        cursor += nbytes;
        // Tensor data is *not* internally re-aligned per tensor in GGUF; only
        // the start of the data section is aligned.
    }

    for ((info, _), &offset) in tensors.iter().zip(offsets.iter()) {
        write_string(&mut out, &info.name);
        out.extend_from_slice(&(info.shape.len() as u32).to_le_bytes());
        for dimension in &info.shape {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&(info.dtype as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    let aligned = align_up(out.len() as u64, alignment).ok_or(GgufError::Overflow("alignment"))?;
    out.resize(aligned as usize, 0);

    for (_, data) in tensors {
        out.extend_from_slice(data);
    }

    Ok(out)
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    let tag = match v {
        Value::U8(_) => ValueType::U8,
        Value::I8(_) => ValueType::I8,
        Value::U16(_) => ValueType::U16,
        Value::I16(_) => ValueType::I16,
        Value::U32(_) => ValueType::U32,
        Value::I32(_) => ValueType::I32,
        Value::F32(_) => ValueType::F32,
        Value::Bool(_) => ValueType::Bool,
        Value::String(_) => ValueType::String,
        Value::Array(_) => ValueType::Array,
        Value::U64(_) => ValueType::U64,
        Value::I64(_) => ValueType::I64,
        Value::F64(_) => ValueType::F64,
    };
    out.extend_from_slice(&(tag as u32).to_le_bytes());

    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => write_string(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(a) => write_array(out, a),
    }
}

fn write_array(out: &mut Vec<u8>, a: &Array) {
    out.extend_from_slice(&(a.element_type() as u32).to_le_bytes());
    out.extend_from_slice(&(a.len() as u64).to_le_bytes());
    match a {
        Array::U8(v) => out.extend_from_slice(v),
        Array::I8(v) => out.extend_from_slice(bytemuck_i8(v)),
        Array::U16(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::I16(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::U32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::I32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::F32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::Bool(v) => {
            for x in v {
                out.push(*x as u8);
            }
        }
        Array::String(v) => {
            for s in v {
                write_string(out, s);
            }
        }
        Array::U64(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::I64(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Array::F64(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

#[inline]
fn bytemuck_i8(v: &[i8]) -> &[u8] {
    // Safety: i8 and u8 have the same layout.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

#[cfg(test)]
mod tests;
