# gguf-wasm

The `wasm-bindgen` surface for [gguf-wasm](https://github.com/f2i-com/gguf-wasm):
GGUF reading, block dequantization and tokenization in a browser.

This crate is built, not depended on. See the repository README for how, or take
a built module from a
[release](https://github.com/f2i-com/gguf-wasm/releases).

```js
const header = new GgufHeader(headerBytes);
header.metadata();                          // JSON
header.tensors();                           // JSON: name, shape, dtype, range
header.row_range('token_embd.weight', 12095, 1);
dequantize('Q4_K', bytes, elements);

const tokenizer = header.tokenizer();       // built inside, stays inside
tokenizer.encode('Hello world');
```

Licensed under Apache-2.0 or MIT, at your option.
