# gguf-quants

The ggml block-quantization formats, decoded exactly. Part of
[gguf-wasm](https://github.com/f2i-com/gguf-wasm).

```toml
gguf-quants = { package = "f2i-gguf-quants", version = "0.0.1" }
```

F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
IQ4_XS.

```rust
let mut out = vec![0.0f32; elements];
gguf_quants::dequantize(GgmlType::Q4_K, &packed, &mut out)?;
```

Decoding is integer arithmetic and one or two half-precision scales, so there is
nothing to round differently and a decoder is either right or wrong. Every
intermediate here rounds to float32 where ggml's does -- including the sub-block
scale and the product before the minimum comes off. Doing that arithmetic in
double and rounding once at the end agrees most of the time and differs in the
last bit the rest of it.

Nothing here allocates: every routine decodes into a slice its caller already
owns.

Licensed under Apache-2.0 or MIT, at your option.
