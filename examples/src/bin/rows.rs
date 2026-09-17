//! Read a few rows of a tensor without decoding the rest of it.
//!
//!     cargo run -p gguf-examples --bin rows -- model.gguf token_embd.weight 0 1 2
//!
//! This is the distinctive thing about reading GGUF this way. Every block
//! format packs a whole number of blocks into each row, so a row is
//! individually addressable: for a 151,936-entry embedding table that is the
//! difference between a few kilobytes and a gigabyte, and it is why a
//! vocabulary-sized tensor is usable at all.
use std::env;
use std::time::Instant;

use gguf::GgufReader;

fn main() {
    let mut args = env::args().skip(1);
    let (path, name) = match (args.next(), args.next()) {
        (Some(path), Some(name)) => (path, name),
        _ => {
            eprintln!("usage: rows <model.gguf> <tensor> [row ...]");
            std::process::exit(2);
        }
    };
    let rows: Vec<u64> = args.filter_map(|a| a.parse().ok()).collect();
    let rows = if rows.is_empty() { vec![0, 1, 2] } else { rows };

    let mut reader = GgufReader::open(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let info = reader
        .header()
        .tensor_by_name(&name)
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("no tensor named {name}");
            std::process::exit(1)
        });
    println!(
        "{name} {:?} {:?}, {:.1} MiB in the file",
        info.shape,
        info.dtype,
        info.nbytes() as f64 / 1048576.0
    );

    let started = Instant::now();
    let values = reader.rows_f32(&name, &rows).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let elapsed = started.elapsed();

    let width = values.len() / rows.len();
    println!(
        "\n{} rows of {width} in {:.1} ms -- {:.1} KiB decoded, not {:.1} MiB",
        rows.len(),
        elapsed.as_secs_f64() * 1000.0,
        (values.len() * 4) as f64 / 1024.0,
        info.nbytes() as f64 / 1048576.0
    );
    for (i, row) in rows.iter().enumerate() {
        let shown = 6.min(width);
        println!(
            "  row {row:<6} {:?} ...",
            &values[i * width..i * width + shown]
        );
    }
}
