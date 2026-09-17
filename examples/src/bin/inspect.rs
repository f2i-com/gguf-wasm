//! What is in this file?
//!
//!     cargo run -p gguf-examples --bin inspect -- model.gguf
//!
//! Prints the architecture, the hyperparameters and a summary of the tensors,
//! having read only the header -- which on a 6.8 GB checkpoint is a few
//! megabytes.
use std::collections::BTreeMap;
use std::env;

use gguf::{GgufReader, Value};

fn main() {
    let path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: inspect <model.gguf>");
            std::process::exit(2);
        }
    };
    let reader = match GgufReader::open(&path) {
        Ok(reader) => reader,
        Err(error) => {
            eprintln!("{path}: {error}");
            std::process::exit(1);
        }
    };
    let header = reader.header();

    println!("{path}");
    println!(
        "  {:.2} GB on disk, GGUF v{}",
        reader.size() as f64 / 1e9,
        header.version()
    );

    // A long array is a vocabulary; say how long rather than printing it.
    println!("\nmetadata");
    for (key, value) in header.metadata() {
        let shown = match value {
            Value::Array(array) => format!("[{} entries]", array.len()),
            Value::String(s) if s.len() > 60 => format!("{:?}...", &s[..60]),
            other => format!("{other}"),
        };
        println!("  {key:<44} {shown}");
    }

    let mut by_dtype: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut total = 0u64;
    for tensor in header.tensors() {
        let entry = by_dtype.entry(format!("{:?}", tensor.dtype)).or_default();
        entry.0 += 1;
        entry.1 += tensor.nbytes();
        total += tensor.numel();
    }
    println!(
        "\n{} tensors, {:.2}B parameters",
        header.tensors().len(),
        total as f64 / 1e9
    );
    for (dtype, (count, bytes)) in &by_dtype {
        println!(
            "  {dtype:<8} {count:>4} tensors  {:>8.1} MiB",
            *bytes as f64 / 1048576.0
        );
    }

    // The biggest is usually the embedding table, and usually the reason
    // reading a row at a time matters.
    if let Some(biggest) = header.tensors().iter().max_by_key(|t| t.numel()) {
        println!(
            "\nlargest: {} {:?} {:?}, {:.1} MiB",
            biggest.name,
            biggest.shape,
            biggest.dtype,
            biggest.nbytes() as f64 / 1048576.0
        );
    }
}
