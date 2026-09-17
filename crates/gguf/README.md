# gguf

GGUF container reading: metadata, tensor shapes, and the byte range of every
tensor. Part of [gguf-wasm](https://github.com/f2i-com/gguf-wasm).

```toml
gguf = { package = "f2i-gguf", version = "0.0.2", features = ["std"] }
```

**A header is not a file.** What this returns describes where a tensor lives;
it does not contain one, and it does not keep the bytes it was parsed from. A
checkpoint is routinely several gigabytes, so the header is parsed from the
first few megabytes and bodies are read by range.

```rust
use gguf::GgufReader;                       // `std` feature

let mut model = GgufReader::open("model.gguf")?;
let packed = model.tensor_bytes("blk.0.attn_q.weight")?;   // the file's own bytes
let values = model.tensor_f32("blk.0.attn_q.weight")?;     // decoded
let row = model.rows_f32("token_embd.weight", &[12095])?;  // one row, kilobytes
```

Without `std` there is no filesystem: hand `GgufHeader::from_bytes` a header you
read yourself, and ask it for `tensor_range` to know what to read next. That is
what a browser and an embedded target need.

Parsing is hardened against files that are not trying to be read: lengths are
checked rather than cast, counts are refused when the remaining bytes could not
hold them, products are checked, and `ParseLimits` bounds the rest.

Licensed under Apache-2.0 or MIT, at your option.
