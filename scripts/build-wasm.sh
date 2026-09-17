#!/usr/bin/env sh
# Build the WebAssembly module, for a browser and for Node.
#
#   sh scripts/build-wasm.sh          # into pkg/ and pkg-node/
#   OUT=dist sh scripts/build-wasm.sh # somewhere else
#
# Two targets from one crate, because they need different glue: `web` is an ES
# module a page imports and `fetch`es its `.wasm` beside, `nodejs` is CommonJS
# a test suite can `require`. The `.wasm` is identical in both.
#
# The wasm-bindgen CLI version must match the `wasm-bindgen` dependency the
# crate pins, or the generated glue will not match the module. That is checked
# here rather than discovered at run time.
set -eu
cd "$(dirname "$0")/.."
OUT="${OUT:-.}"
CRATE=crates/gguf-wasm
WASM=target/wasm32-unknown-unknown/release/gguf_wasm.wasm

want=$(grep -o '"=0\.2\.[0-9]*"' "$CRATE/Cargo.toml" | tr -d '"=')
have=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}' || true)
if [ -z "$have" ]; then
  echo "wasm-bindgen CLI not found: cargo install wasm-bindgen-cli --version $want" >&2
  exit 1
fi
[ "$want" = "$have" ] || { echo "wasm-bindgen CLI is $have, the crate pins $want" >&2; exit 1; }

cargo build --release --locked --target wasm32-unknown-unknown -p f2i-gguf-wasm

for target in web nodejs; do
  case "$target" in
    web) dir="$OUT/pkg" ;;
    nodejs) dir="$OUT/pkg-node" ;;
  esac
  rm -rf "$dir"
  wasm-bindgen --target "$target" --out-dir "$dir" "$WASM"
  # A CommonJS build under a package that may declare "type": "module" needs to
  # say so for its own directory, which is what lets a host `require` it.
  [ "$target" = nodejs ] && echo '{"type":"commonjs"}' > "$dir/package.json"
  echo "$dir: $(wc -c < "$dir/gguf_wasm_bg.wasm") bytes"
done
