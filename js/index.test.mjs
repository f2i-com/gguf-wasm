// Reading one object over many requests.
//
// The failure that matters here is not a request that fails. It is a request
// that *succeeds* against a different object than the last one did -- a header
// parsed from version A and a weight fetched from version B, both 206, both
// the right length, and a model that is quietly wrong. These check that such a
// read is an error instead.
//
//   node --test js/index.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import {fromBlob, fromURL, fromFileHandle, openGGUF} from './index.mjs';

/** A server that honours Range, with whatever identity it is told to have. */
function server({body, etag = '"v1"', lastModified = null, ignoreRange = false, onRequest = () => {}}) {
  const seen = [];
  const fetcher = async (url, {headers = {}} = {}) => {
    seen.push(headers);
    onRequest(headers, seen.length);
    const current = server.identity ?? etag;
    if (ignoreRange) {
      return response(200, body, {etag: current, 'last-modified': lastModified});
    }
    const match = /^bytes=(\d+)-(\d+)$/.exec(headers.Range ?? '');
    if (!match) return response(200, body, {etag: current});
    const [, from, to] = match.map(Number);
    const slice = body.subarray(from, Math.min(to + 1, body.length));
    return response(206, slice, {
      etag: current,
      'last-modified': lastModified,
      'content-range': `bytes ${from}-${from + slice.length - 1}/${body.length}`,
    });
  };
  return {fetcher, seen};
}

function response(status, bytes, headers) {
  const map = new Map(Object.entries(headers).filter(([, v]) => v != null).map(([k, v]) => [k.toLowerCase(), v]));
  return {
    status,
    headers: {get: name => map.get(name.toLowerCase()) ?? null},
    async arrayBuffer() { return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength); },
  };
}

const body = Uint8Array.from({length: 512}, (_, i) => i & 0xff);

test('a blob is read by slice and never whole', async () => {
  let asked = null;
  const blob = {size: body.length, slice(from, to) { asked = [from, to]; return {
    async arrayBuffer() { return body.subarray(from, to).buffer.slice(from, to); }}; }};
  const source = fromBlob(blob);
  assert.equal(source.size(), 512);
  await source.read(16, 32);
  assert.deepEqual(asked, [16, 48], 'only the slice asked for');
});

test('a URL is opened with a one-byte range, not a HEAD', async () => {
  const {fetcher, seen} = server({body});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  assert.equal(seen.length, 1);
  assert.equal(seen[0].Range, 'bytes=0-0', 'the probe is a range request');
  // The total came from Content-Range, which is proof rather than a promise.
  assert.equal(source.size(), 512);
  assert.equal(source.identity(), '"v1"');
});

test('every read afterwards carries If-Range', async () => {
  const {fetcher, seen} = server({body});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  const bytes = await source.read(64, 16);
  assert.deepEqual([...bytes], [...body.subarray(64, 80)]);
  assert.equal(seen.at(-1)['If-Range'], '"v1"', 'the pinned identity goes with the read');
});

test('an object that changed is refused, not read', async () => {
  // The server answers a conditional range with the whole entity, which is
  // what a server does when the validator no longer matches.
  const {fetcher} = server({body, ignoreRange: false});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  server.identity = '"v2"';
  try {
    await assert.rejects(source.read(64, 16), /changed identity/);
  } finally { delete server.identity; }
});

test('a whole-entity answer to a range request is refused', async () => {
  const {fetcher} = server({body});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  // A server that has decided to ignore Range from here on.
  const changed = fromURL('https://example/model.gguf', {
    fetch: async () => response(200, body, {etag: '"v1"'}),
  });
  await assert.rejects(changed.prepare(), /did not honour|whole entity/);
});

test('a server that ignores Range at all is refused at open', async () => {
  const {fetcher} = server({body, ignoreRange: true});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await assert.rejects(source.prepare(), /whole entity|did not honour/);
});

test('a range that is not the range asked for is refused', async () => {
  const fetcher = async () => response(206, body.subarray(0, 16), {
    etag: '"v1"',
    // Says it is a different part of the file than was requested.
    'content-range': 'bytes 999-1014/512',
  });
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await assert.rejects(source.prepare(), /returned bytes 999-1014/);
});

test('a total length that changed is refused', async () => {
  let total = 512;
  const fetcher = async (_url, {headers}) => {
    const [, from, to] = /^bytes=(\d+)-(\d+)$/.exec(headers.Range).map(Number);
    const slice = body.subarray(from, to + 1);
    return response(206, slice, {
      etag: '"v1"',
      'content-range': `bytes ${from}-${from + slice.length - 1}/${total}`,
    });
  };
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  total = 1024;                       // the object grew underneath us
  await assert.rejects(source.read(0, 8), /is now 1024 bytes, was 512/);
});

test('a response with no Content-Range is refused', async () => {
  const fetcher = async () => response(206, body.subarray(0, 1), {etag: '"v1"'});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await assert.rejects(source.prepare(), /no usable Content-Range/);
});

test('a short body is refused even when the headers agree', async () => {
  const fetcher = async (_url, {headers}) => {
    const [, from, to] = /^bytes=(\d+)-(\d+)$/.exec(headers.Range).map(Number);
    // Headers claim the full range; the body is one byte short.
    return response(206, body.subarray(from, to), {
      etag: '"v1"',
      'content-range': `bytes ${from}-${to}/512`,
    });
  };
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  await assert.rejects(source.read(0, 16), /returned 15 of 16 bytes/);
});

test('Last-Modified stands in when there is no ETag', async () => {
  const {fetcher, seen} = server({body, etag: null, lastModified: 'Wed, 17 Sep 2026 00:00:00 GMT'});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  assert.equal(source.identity(), 'Wed, 17 Sep 2026 00:00:00 GMT');
  await source.read(0, 4);
  assert.equal(seen.at(-1)['If-Range'], 'Wed, 17 Sep 2026 00:00:00 GMT');
});

test('a file handle reads exactly what it is asked for', async () => {
  const handle = {async read(out, at, length, position) {
    out.set(body.subarray(position, position + length), at);
    return {bytesRead: length};
  }};
  const source = fromFileHandle(handle, body.length);
  assert.deepEqual([...await source.read(8, 4)], [...body.subarray(8, 12)]);
});

test('a server offering no validator at all is refused unless allowed', async () => {
  // Neither ETag nor Last-Modified: nothing to pin reads to, so a replacement
  // of the same length between two reads would go unnoticed. That is the one
  // failure this source exists to prevent, so it is not quietly downgraded.
  const {fetcher} = server({body, etag: null, lastModified: null});
  await assert.rejects(
    fromURL('https://example/model.gguf', {fetch: fetcher}).prepare(),
    /neither a strong ETag nor Last-Modified/);

  const anyway = fromURL('https://example/model.gguf', {fetch: fetcher, allowUnvalidated: true});
  await anyway.prepare();
  assert.equal(anyway.identity(), null, 'and it admits it pinned nothing');
  assert.deepEqual([...await anyway.read(0, 4)], [...body.subarray(0, 4)]);
});

// ---- ranges out of a header the caller did not write ------------------------

/** A stand-in for the wasm module, answering with whatever tensor table a test
 *  wants. Only what openGGUF actually touches is implemented. */
function moduleListing(tensors) {
  const first = tensors[0];
  // stored is [width, count]; a row is `width` values of four bytes.
  const width = first.shape?.[0] ?? 0;
  return {
    header_needs_more_bytes: () => false,
    dequantize: (dtype, bytes, elements) => new Float32Array(elements),
    GgufHeader: class {
      constructor() {}
      version() { return 3; }
      metadata() { return JSON.stringify({'general.architecture': 'test'}); }
      tensors() { return JSON.stringify(tensors); }
      row_range(name, start) {
        return JSON.stringify({
          dtype: first.dtype, elements: width,
          offset: first.offset + start * width * 4, bytes: width * 4,
        });
      }
    },
  };
}

const oneTensor = extra => [{
  name: 'blk.0.weight', dtype: 'f32', shape: [4, 4],
  offset: 64, bytes: 64, elements: 16, readable: true, ...extra,
}];

test('a tensor pointing past the end of the file is refused at open', async () => {
  // Blob.slice clamps rather than throwing, so without this check the read
  // comes back short and a decoder sees a truncated tensor as a valid one.
  const source = fromBlob(new Blob([new Uint8Array(256)]));
  await assert.rejects(
    openGGUF(source, {module: moduleListing(oneTensor({offset: 200, bytes: 100}))}),
    /past the end of a 256-byte file/);
});

test('a length JavaScript cannot represent exactly is refused at open', async () => {
  // A GGUF length is a u64 and a JSON number is a double: past 2^53 the value
  // arrives near what the file said rather than equal to it.
  const source = fromBlob(new Blob([new Uint8Array(256)]));
  for (const field of ['offset', 'bytes', 'elements']) {
    await assert.rejects(
      openGGUF(source, {module: moduleListing(oneTensor({[field]: 2 ** 53 + 1}))}),
      new RegExp(`${field} is .*not a byte count this can represent exactly`),
      `${field} past 2^53`);
  }
  await assert.rejects(
    openGGUF(source, {module: moduleListing(oneTensor({offset: -1}))}),
    /offset is -1/);
});

test('a header that is not one fails without reading the whole file', async () => {
  // The doubling loop must tell a short read from a file that will never
  // parse, or opening the wrong file costs the full 64 MiB schedule.
  let reads = 0;
  const source = {
    size: () => 128 << 20,
    async read(offset, length) { reads++; return new Uint8Array(length); },
  };
  const module = {
    header_needs_more_bytes: () => false,
    GgufHeader: class { constructor() { throw new Error('not a GGUF file: bad magic'); } },
  };
  await assert.rejects(openGGUF(source, {module}), /bad magic/);
  assert.equal(reads, 1, 'it stopped after the first read');
});

test('a valid tensor table opens and reads', async () => {
  const bytes = new Uint8Array(256);
  bytes.set([1, 2, 3, 4], 64);
  const source = fromBlob(new Blob([bytes]));
  const model = await openGGUF(source, {module: moduleListing(oneTensor())});
  assert.equal(model.architecture, 'test');
  assert.deepEqual(model.tensors.get('blk.0.weight').shape, [4, 4]);
  assert.deepEqual([...(await model.bytes('blk.0.weight')).subarray(0, 4)], [1, 2, 3, 4]);
});

test('a row index the caller made up is refused, not silently converted', async () => {
  // A row index crosses into the module as a u32, and JavaScript will hand
  // 2^32+1, -1 or 1.5 to that conversion without complaint -- each of which
  // reads some other row and returns it as though it were the one asked for.
  const source = fromBlob(new Blob([new Uint8Array(256)]));
  // stored [width, count] = [4, 4], so four rows of four.
  const model = await openGGUF(source, {module: moduleListing(oneTensor())});
  for (const bad of [4, 5, -1, 1.5, 2 ** 32 + 1, NaN, '2']) {
    await assert.rejects(
      model.rows('blk.0.weight', [bad]),
      /is not one of its 4 rows/,
      `row ${String(bad)}`);
  }
  await assert.rejects(model.rows('blk.0.weight', 3), /takes a list of row indices/);
  // And a real one still works.
  assert.equal((await model.rows('blk.0.weight', [0, 1])).length, 8);
});

test('a weak ETag is not used as an If-Range validator', async () => {
  // If-Range needs a strong validator. A weak one promises only semantic
  // equivalence, which is exactly the guarantee that does not hold when the
  // thing being joined is byte ranges -- and a server must ignore it, so
  // sending it would look like a guarantee while being none.
  const {fetcher, seen} = server({
    body, etag: 'W/"weak"', lastModified: 'Wed, 17 Sep 2026 00:00:00 GMT'});
  const source = fromURL('https://example/model.gguf', {fetch: fetcher});
  await source.prepare();
  assert.equal(source.identity(), 'Wed, 17 Sep 2026 00:00:00 GMT',
    'it fell back to Last-Modified rather than pinning the weak tag');
  await source.read(0, 4);
  assert.equal(seen.at(-1)['If-Range'], 'Wed, 17 Sep 2026 00:00:00 GMT');

  // And with nothing else to fall back to, a weak ETag alone is no validator.
  const {fetcher: weakOnly} = server({body, etag: 'W/"weak"', lastModified: null});
  await assert.rejects(
    fromURL('https://example/model.gguf', {fetch: weakOnly}).prepare(),
    /neither a strong ETag nor Last-Modified/);
});

test('a Content-Range past 2^53 is not a usable Content-Range', async () => {
  // Decimal digits from a header. Number() takes as many as it is given and
  // returns something close to, but not equal to, what was sent -- while the
  // entire point of reading this header is comparing it with what was asked.
  const huge = `bytes 0-0/${Number.MAX_SAFE_INTEGER + 2}`;
  const fetcher = async () => response(206, body.subarray(0, 1), {
    etag: '"v1"', 'content-range': huge});
  await assert.rejects(
    fromURL('https://example/model.gguf', {fetch: fetcher}).prepare(),
    /no usable Content-Range/);
});

test('what the module cannot address says so, rather than truncating', async () => {
  // dequantize() and row_range() take u32, because they produce values inside
  // wasm32 memory. wasm-bindgen would truncate a larger number, so a row index
  // of 2**32 + 5 would read row 5 and return it as though it were the one
  // asked for. bytes() has no such ceiling and is the way round it.
  const source = fromBlob(new Blob([new Uint8Array(256)]));
  const huge = 2 ** 32 + 5;

  const wide = await openGGUF(source, {
    module: moduleListing(oneTensor({shape: [4, huge], elements: 16})),
  });
  await assert.rejects(wide.rows('blk.0.weight', [huge - 1]),
    /past the 4294967295 this module addresses/);

  const many = await openGGUF(source, {
    module: moduleListing(oneTensor({elements: 2 ** 33})),
  });
  await assert.rejects(many.floats('blk.0.weight'),
    /past the 4294967295 this module can decode in one call/);
});
