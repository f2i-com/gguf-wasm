// Does the built module load, parse and decode?
//
//   sh scripts/build-wasm.sh && node scripts/smoke.mjs
//
// A module that compiles and does not load is not a module, and the failures
// that matter here -- a wasm-bindgen glue version that does not match the
// binary, an export that changed name, a decoder that returns nothing -- only
// show when something actually calls it.
//
// It writes its own GGUF, so no checkpoint is needed and this runs anywhere.
import {createRequire} from 'node:module';
import {existsSync} from 'node:fs';

const require = createRequire(import.meta.url);
const MODULE = new URL('../pkg-node/gguf_wasm.js', import.meta.url);
if (!existsSync(MODULE)) {
  console.error('build it first: sh scripts/build-wasm.sh');
  process.exit(1);
}
const gguf = require(MODULE.pathname.replace(/^\/([A-Za-z]:)/, '$1'));

// ---- a GGUF, written here so the test owns both sides ----------------------

const parts = [];
const push = (...bytes) => parts.push(Uint8Array.from(bytes));
const u32 = v => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); parts.push(b); };
const u64 = v => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); parts.push(b); };
const f32 = v => { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); parts.push(b); };
const str = s => { const b = new TextEncoder().encode(s); u64(b.length); parts.push(b); };

const WIDTH = 8, ROWS = 8, VALUES = WIDTH * ROWS;   // one F32 matrix

u32(0x46554747);                         // "GGUF"
u32(3);                                  // version
u64(1);                                  // one tensor
u64(2);                                  // two metadata entries
str('general.architecture'); u32(8); str('smoke');
str('general.alignment'); u32(4); u32(32);
// Rank two, so a row is addressable: GGUF writes the fastest-varying
// dimension first, so this is [width, rows].
str('weight'); u32(2); u64(WIDTH); u64(ROWS); u32(0); u64(0);

let header = Uint8Array.from(parts.flatMap(p => [...p]));
const start = Math.ceil(header.length / 32) * 32;
const file = new Uint8Array(start + VALUES * 4);
file.set(header);
const view = new DataView(file.buffer);
for (let i = 0; i < VALUES; i++) view.setFloat32(start + i * 4, i * 0.5, true);

// ---- and the module reads it ----------------------------------------------

let failures = 0;
const check = (what, ok, detail = '') => {
  if (ok) { console.log(`  ok   ${what}`); }
  else { failures++; console.log(`  FAIL ${what}${detail ? ': ' + detail : ''}`); }
};

const parsed = new gguf.GgufHeader(file);
check('the header parses', true);
check('version is reported', parsed.version() === 3, String(parsed.version()));

const metadata = JSON.parse(parsed.metadata());
check('metadata reads back', metadata['general.architecture'] === 'smoke',
  JSON.stringify(metadata['general.architecture']));

const tensors = JSON.parse(parsed.tensors());
check('one tensor is listed', tensors.length === 1, String(tensors.length));
const [tensor] = tensors;
check('its dtype is F32', tensor.dtype === 'F32', tensor.dtype);
check('its range is where the data was put',
  tensor.offset === start && tensor.bytes === VALUES * 4,
  `${tensor.offset}/${tensor.bytes} against ${start}/${VALUES * 4}`);

const bytes = file.subarray(tensor.offset, tensor.offset + tensor.bytes);
const decoded = gguf.dequantize('F32', bytes, tensor.elements);
check('it decodes to the values that were written',
  decoded.length === VALUES && decoded[0] === 0 && decoded[VALUES - 1] === (VALUES - 1) * 0.5,
  `${decoded.length} values, last ${decoded[VALUES - 1]}`);

const row = JSON.parse(parsed.row_range('weight', 1, 1));
check('a row range lands on that row',
  row.offset === start + WIDTH * 4 && row.bytes === WIDTH * 4 && row.elements === WIDTH,
  JSON.stringify(row));
const rowValues = gguf.dequantize(row.dtype, file.subarray(row.offset, row.offset + row.bytes), row.elements);
check('and reads that row back', rowValues[0] === WIDTH * 0.5, String(rowValues[0]));

check('the supported formats are listed', gguf.supported_dtypes().includes('Q4_K'));

// A file that is not one must be refused, not accepted quietly.
let refused = false;
try { new gguf.GgufHeader(new Uint8Array(64)); } catch { refused = true; }
check('a file that is not a GGUF is refused', refused);

console.log(failures ? `\n${failures} failed` : '\nthe module loads and works');
process.exit(failures ? 1 : 0);
