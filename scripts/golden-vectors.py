#!/usr/bin/env python3
"""Regenerate the golden block vectors in crates/gguf-quants/tests/golden/.

    pip install gguf numpy
    python scripts/golden-vectors.py

## Why these exist

`gguf-quants` is checked against ggml by reading ggml. That catches a layout
misremembered; it does not catch one misread the same way twice. Every other
test here compares a packed path with a decoded path, and both use *this*
decoder -- so a decoder wrong in some corner agrees with itself perfectly.

So these vectors are decoded by `gguf-py`, the llama.cpp project's own GGUF
library: numpy, written by other people, from the same specification.

Block decoding is integer arithmetic and one or two half-precision scales.
Nothing in it can round two ways, so the comparison is equality rather than a
tolerance, and a single differing bit is a wrong decoder.

## Where the blocks come from, and why not from a model

Earlier versions of this lifted K-quant blocks out of real checkpoints, because
gguf-py decodes those formats without being able to produce them. That worked
and was a licensing mistake: those `.blocks` files were third-party model
weights, however few, sitting in a repository offered under MIT/Apache-2.0.

They are synthesised now, and nothing here comes from a model.

The trick is that a decoder does not need a *quantizer*. Every bit pattern is a
legal block -- the quants are fixed-width indices and the scales are f16, so
there is no encoding to get wrong and nothing to reject. So the blocks are
pseudo-random bytes, with one correction: the f16 scale fields are overwritten
with finite values, because a random f16 is NaN or infinity about one time in
128 and a NaN's payload is not something two implementations must agree on.

Random bytes are also better coverage than real weights. A checkpoint's blocks
cluster: quant indices near the middle of the range, scales within a few
exponents. Random ones sit at the ends, which is where an off-by-one in a shift
or a sign extension actually shows.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
from gguf import quants
from gguf.constants import GGMLQuantizationType as Q

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE / "crates" / "gguf-quants" / "tests" / "golden"

# Per format: packed bytes per block, values per block, and where the f16
# scales live. Offsets are from crates/gguf-quants/src/<format>.rs, which is
# the same layout ggml-quants.c defines.
LAYOUT = {
    "Q4_0":   (18,  32,  [0, 2]),
    "Q4_1":   (20,  32,  [0, 2]),
    "Q5_0":   (22,  32,  [0]),
    "Q5_1":   (24,  32,  [0, 2]),
    "Q8_0":   (34,  32,  [0]),
    "Q2_K":   (84,  256, [80, 82]),
    "Q3_K":   (110, 256, [108]),
    "Q4_K":   (144, 256, [0, 2]),
    "Q5_K":   (176, 256, [0, 2]),
    "Q6_K":   (210, 256, [208]),
    "IQ4_NL": (18,  32,  [0]),
    "IQ4_XS": (136, 256, [0]),
}
PLAIN = ["F32", "F16", "BF16"]      # values, not blocks
BLOCKS = 16                          # per format


def finite_f16(rng, count):
    """f16 scales that are real numbers, over a wide range of exponents.

    A scale is what multiplies a block's quants, so it wants to span orders of
    magnitude -- that is what separates a decoder that reads the exponent
    correctly from one that happens to work for scales near one.
    """
    exponent = rng.integers(-8, 8, count).astype(np.float32)
    mantissa = rng.uniform(-2.0, 2.0, count).astype(np.float32)
    return (mantissa * np.float32(2.0) ** exponent).astype(np.float16)


def synth_blocks(name, rng):
    per_block, per_values, scale_at = LAYOUT[name]
    raw = bytearray(rng.integers(0, 256, BLOCKS * per_block, dtype=np.uint8).tobytes())
    scales = finite_f16(rng, BLOCKS * len(scale_at))
    k = 0
    for block in range(BLOCKS):
        base = block * per_block
        for offset in scale_at:
            raw[base + offset: base + offset + 2] = scales[k].tobytes()
            k += 1
    return bytes(raw), BLOCKS * per_values


def synth_plain(name, rng):
    count = BLOCKS * 256
    values = (rng.standard_normal(count).astype(np.float32)
              * np.float32(2.0) ** rng.integers(-6, 6, count).astype(np.float32))
    packed = quants.quantize(values.reshape(BLOCKS, 256), getattr(Q, name))
    return packed.tobytes(), count


def reference_revision():
    """Which gguf-py decoded these, so the evidence names its source."""
    try:
        import gguf
        version = subprocess.run(
            [sys.executable, "-m", "pip", "show", "gguf"],
            capture_output=True, text=True, check=False).stdout
        for line in version.splitlines():
            if line.startswith("Version:"):
                return f"gguf-py {line.split(':', 1)[1].strip()}"
        return f"gguf-py (from {pathlib.Path(gguf.__file__).parent})"
    except Exception:
        return "gguf-py (version unknown)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = reference_revision()
    manifest, missing = {}, []

    for name in PLAIN + list(LAYOUT):
        rng = np.random.default_rng(0x9E3779B9 ^ int.from_bytes(name.encode(), "little"))
        try:
            blocks, count = (synth_plain if name in PLAIN else synth_blocks)(name, rng)
            decoded = np.asarray(
                quants.dequantize(np.frombuffer(blocks, dtype=np.uint8), getattr(Q, name)),
                dtype=np.float32).ravel()[:count]
        except NotImplementedError:
            missing.append(name)
            print(f"  --   {name:7} gguf-py cannot decode it")
            continue

        if not np.all(np.isfinite(decoded)):
            bad = int((~np.isfinite(decoded)).sum())
            print(f"  !!   {name:7} {bad} non-finite values; scale masking is wrong")
            missing.append(name)
            continue

        (OUT / f"{name}.blocks").write_bytes(blocks)
        (OUT / f"{name}.f32").write_bytes(decoded.astype("<f4").tobytes())
        manifest[name] = {
            "values": int(decoded.size),
            "bytes": len(blocks),
            "blocks_sha256": hashlib.sha256(blocks).hexdigest(),
            "values_sha256": hashlib.sha256(decoded.astype("<f4").tobytes()).hexdigest(),
            "from": "synthesised: pseudo-random blocks, finite f16 scales",
        }
        print(f"  ok   {name:7} {len(blocks):6} bytes -> {decoded.size:5} values"
              f"   [{decoded.min():+.4g}, {decoded.max():+.4g}]")

    (OUT / "manifest.json").write_text(
        json.dumps({
            "generator": "scripts/golden-vectors.py",
            "reference": reference,
            "reference_project": "https://github.com/ggml-org/llama.cpp/tree/master/gguf-py",
            "provenance": "Fully synthetic. No model weights are included here, "
                          "so nothing in this directory carries a third-party "
                          "model licence.",
            "vectors": manifest,
            "missing": missing,
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n{len(manifest)} of {len(PLAIN) + len(LAYOUT)} formats covered"
          + (f"; missing {', '.join(missing)}" if missing else "")
          + f"\nreference: {reference}")


if __name__ == "__main__":
    main()
