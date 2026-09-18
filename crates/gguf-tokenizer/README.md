# gguf-tokenizer

Byte-level BPE over a GGUF file's own vocabulary. Part of
[gguf-wasm](https://github.com/f2i-com/gguf-wasm).

```toml
gguf-tokenizer = { package = "f2i-gguf-tokenizer", version = "0.0.5" }
```

The pre-tokenizer is per vocabulary and it matters: Qwen2 splits digits **one at
a time** where Llama-3 takes them three at a time and GPT-2 takes a run with its
leading space. Using the wrong one produces token ids that are individually
valid and collectively wrong. A GGUF file records which it was trained with, in
`tokenizer.ggml.pre`, and a name this build does not scan is refused rather than
guessed.

```rust
let tokenizer = Tokenizer::new(tokens, merges, Pattern::Qwen2, specials, None);
tokenizer.encode("The capital of France is")?;   // [785, 6722, 315, 9625, 374]
tokenizer.decode(&ids);
```

The patterns are regular expressions upstream and hand-scanned here: a regex
engine with Unicode classes is a large dependency to put in a browser for six
alternatives, and `core` already knows what a letter is.

Licensed under Apache-2.0 or MIT, at your option.
