//! Tokenize with the vocabulary the file itself carries.
//!
//!     cargo run -p gguf-examples --bin tokenize -- model.gguf "Hello world"
//!
//! The pre-tokenizer matters more than it looks: Qwen2 splits digits one at a
//! time where Llama-3 takes three and GPT-2 takes a run. A GGUF file records
//! which one it was trained with, and using the wrong one gives token ids that
//! are individually valid and collectively wrong.
use std::env;

use gguf::{Array, GgufHeader, GgufReader, Value};
use gguf_tokenizer::{Pattern, Tokenizer};

fn strings(header: &GgufHeader, key: &str) -> Vec<String> {
    match header.metadata().get(key) {
        Some(Value::Array(Array::String(items))) => items.clone(),
        _ => Vec::new(),
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let (path, text) = match (args.next(), args.next()) {
        (Some(path), Some(text)) => (path, text),
        _ => {
            eprintln!("usage: tokenize <model.gguf> <text>");
            std::process::exit(2);
        }
    };
    let reader = GgufReader::open(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let header = reader.header();

    let named = match header.metadata().get("tokenizer.ggml.pre") {
        Some(Value::String(name)) => name.clone(),
        _ => String::from("default"),
    };
    let pattern = Pattern::from_name(&named).unwrap_or_else(|| {
        eprintln!("tokenizer.ggml.pre is {named:?}, which this build does not scan");
        std::process::exit(1)
    });

    let tokens = strings(header, "tokenizer.ggml.tokens");
    let merges = strings(header, "tokenizer.ggml.merges");
    if tokens.is_empty() || merges.is_empty() {
        eprintln!("this checkpoint carries no byte-level BPE vocabulary");
        std::process::exit(1);
    }
    // Control and user-defined tokens are matched literally, so a chat marker
    // stays one token instead of six bytes the model has never seen together.
    let specials = match header.metadata().get("tokenizer.ggml.token_type") {
        Some(Value::Array(Array::I32(kinds))) => kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == 3 || **kind == 4)
            .map(|(id, _)| id as u32)
            .collect(),
        _ => Vec::new(),
    };

    let tokenizer = Tokenizer::new(tokens, merges, pattern, specials, None);
    println!(
        "vocabulary {} entries, pre-tokenizer {named:?}",
        tokenizer.vocab_size()
    );

    let ids = tokenizer.encode(&text).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    println!("\n{text:?}");
    println!("  ids    {ids:?}");
    let pieces: Vec<&str> = ids.iter().filter_map(|id| tokenizer.token(*id)).collect();
    println!("  pieces {}", pieces.join("|"));
    println!("  back   {:?}", tokenizer.decode(&ids));
}
