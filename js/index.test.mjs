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

import {fromBlob, fromURL, fromFileHandle} from './index.mjs';

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
