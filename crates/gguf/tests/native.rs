//! Reading a real checkpoint from disk.
//!
//! A GGUF file is not redistributed here -- they are gigabytes and they belong
//! to whoever trained them -- so this skips without one:
//!
//!     GGUF_MODEL=path/to/model.gguf cargo test -p gguf --features std
#![cfg(feature = "std")]

use gguf::{GgufReader, Value};

fn model() -> Option<GgufReader> {
    let path = std::env::var("GGUF_MODEL").ok()?;
    Some(GgufReader::open(path).expect("GGUF_MODEL did not open"))
}

#[test]
fn the_header_is_a_small_read_of_a_large_file() {
    let Some(reader) = model() else { return };
    let header = reader.header();
    assert!(!header.tensors().is_empty(), "no tensors");
    assert!(
        header.metadata().contains_key("general.architecture"),
        "a checkpoint names its architecture"
    );
    // The header parsed without the file being loaded: it is read by doubling
    // from a megabyte, and a checkpoint is very much larger than that.
    assert!(reader.size() > 0);
}

#[test]
fn a_tensor_reads_as_its_own_bytes_and_as_floats() {
    let Some(mut reader) = model() else { return };
    let small = reader
        .header()
        .tensors()
        .iter()
        .filter(|t| t.numel() > 1 && t.numel() < (1 << 16))
        .min_by_key(|t| t.numel())
        .expect("no small tensor")
        .clone();

    let packed = reader.tensor_bytes(&small.name).expect("bytes");
    assert_eq!(
        packed.len() as u64,
        small.nbytes(),
        "packed length is the file's own"
    );

    let decoded = reader.tensor_f32(&small.name).expect("floats");
    assert_eq!(decoded.len() as u64, small.numel());
    assert!(
        decoded.iter().all(|v| v.is_finite()),
        "a decoded weight is not finite"
    );

    // The point of handing over packed bytes: for a quantized tensor there are
    // far fewer of them than the values they stand for.
    if small.dtype.block_size() > 1 {
        assert!(
            packed.len() < decoded.len() * 4,
            "{:?} packed should be smaller than float32",
            small.dtype
        );
    }
}

#[test]
fn a_row_costs_a_row() {
    let Some(mut reader) = model() else { return };
    let matrix = reader
        .header()
        .tensors()
        .iter()
        .filter(|t| t.shape.len() == 2)
        .max_by_key(|t| t.numel())
        .expect("no matrix")
        .clone();
    let width = matrix.shape[0];
    let rows = matrix.shape[1];

    let gathered = reader
        .rows_f32(&matrix.name, &[0, 1, rows - 1])
        .expect("rows");
    assert_eq!(gathered.len() as u64, 3 * width);
    assert!(gathered.iter().all(|v| v.is_finite()));

    // Reading the same row twice gives the same values.
    let again = reader.rows_f32(&matrix.name, &[1]).expect("row again");
    assert_eq!(&again[..], &gathered[width as usize..2 * width as usize]);
}

#[test]
fn a_row_past_the_end_is_refused() {
    let Some(mut reader) = model() else { return };
    let matrix = reader
        .header()
        .tensors()
        .iter()
        .find(|t| t.shape.len() == 2)
        .expect("no matrix")
        .clone();
    assert!(
        reader.rows_f32(&matrix.name, &[matrix.shape[1]]).is_err(),
        "a row past the end is an error, not a read of whatever follows"
    );
}

#[test]
fn the_tokenizer_metadata_is_reachable() {
    let Some(reader) = model() else { return };
    let header = reader.header();
    if let Some(Value::Array(array)) = header.metadata().get("tokenizer.ggml.tokens") {
        assert!(!array.is_empty(), "a vocabulary with no entries");
    }
}
