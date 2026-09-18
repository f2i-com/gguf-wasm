#!/usr/bin/env python3
"""Regenerate the golden block vectors in crates/gguf-quants/tests/golden/.

    pip install gguf numpy
    python scripts/golden-vectors.py                    # synthetic formats only
    python scripts/golden-vectors.py E:/models/*.gguf   # and real K-quant blocks

## Why these exist

`gguf-quants` is checked against ggml by reading ggml. That catches a decoder
written from a misremembered layout; it does not catch one where the layout was
misread the same way twice. So these vectors come from a *different*
implementation: `gguf-py`, the llama.cpp project's own GGUF library, written in
numpy by other people from the same specification.

Block decoding is integer arithmetic and one or two half-precision scales.
There is nothing to round differently, so the comparison is equality rather
than a tolerance, and a single differing bit is a wrong decoder.

## Where a vector comes from

Two sources, because they catch different things:

* **synthesised** -- gguf-py quantizes pseudo-random values and we both decode
  the result. Available only for the formats gguf-py can quantize, which is the
  non-K ones. The input is deliberately wide-ranged so scales land all over.
* **real** -- blocks lifted out of an actual checkpoint. This is the only way
  to reach K-quants, since gguf-py decodes those but does not produce them, and
  it is worth having anyway: real weights cluster in ways random data does not.

A format nothing here can supply is listed as missing rather than skipped
quietly. Q2_K, Q3_K, IQ4_NL and IQ4_XS need a checkpoint that uses them.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
from gguf import GGUFReader, quants
from gguf.constants import GGMLQuantizationType as Q

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE / "crates" / "gguf-quants" / "tests" / "golden"

# Every format this crate decodes.
ALL = ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
       "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS"]
BLOCKS = 8          # per format; enough for sub-block structure to matter
ROWS, WIDTH = 8, 256


def synthesise(name):
    """gguf-py quantizes, then both sides decode what it produced."""
    rng = np.random.default_rng(0x9E3779B9 ^ sum(name.encode()))
    # A wide range on purpose: block scales are per-block, so values that vary
    # in magnitude between blocks exercise the scale path rather than one
    # comfortable exponent.
    data = (rng.standard_normal((ROWS, WIDTH)).astype(np.float32)
            * np.float32(10.0) ** rng.integers(-3, 3, (ROWS, 1)).astype(np.float32))
    packed = quants.quantize(data, getattr(Q, name))
    decoded = np.asarray(quants.dequantize(packed, getattr(Q, name)), np.float32)
    return packed.tobytes(), decoded.ravel(), "synthesised by gguf-py"


def from_model(name, paths):
    """Blocks out of a real checkpoint, for the formats gguf-py cannot make."""
    per_block = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210,
                 "IQ4_NL": 18, "IQ4_XS": 136}[name]
    values = 32 if name == "IQ4_NL" else 256
    for path in paths:
        reader = GGUFReader(str(path))
        for tensor in reader.tensors:
            if tensor.tensor_type.name != name:
                continue
            raw = tensor.data.tobytes()[: BLOCKS * per_block]
            if len(raw) < BLOCKS * per_block:
                continue
            decoded = np.asarray(
                quants.dequantize(tensor.data, tensor.tensor_type), np.float32
            ).ravel()[: BLOCKS * values]
            digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
            return raw, decoded, f"{pathlib.Path(path).name} {tensor.name} (sha256 {digest[:16]})"
    return None


def main():
    models = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, missing = {}, []

    for name in ALL:
        made = None
        try:
            made = synthesise(name)
        except NotImplementedError:
            pass
        if made is None and models:
            made = from_model(name, models)
        if made is None:
            missing.append(name)
            print(f"  --   {name:7} no source: needs a checkpoint that uses it")
            continue

        blocks, decoded, provenance = made
        (OUT / f"{name}.blocks").write_bytes(blocks)
        (OUT / f"{name}.f32").write_bytes(decoded.astype("<f4").tobytes())
        manifest[name] = {"values": int(decoded.size), "bytes": len(blocks),
                          "from": provenance}
        print(f"  ok   {name:7} {len(blocks):6} bytes -> {decoded.size:5} values   {provenance}")

    (OUT / "manifest.json").write_text(
        json.dumps({"generator": "scripts/golden-vectors.py",
                    "reference": "gguf-py (the llama.cpp project's Python GGUF library)",
                    "vectors": manifest,
                    "missing": missing}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"\n{len(manifest)} of {len(ALL)} formats covered"
          + (f"; missing {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
