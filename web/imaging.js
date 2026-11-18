/*
 * imaging.js — turn any image File into the model's 224×224×3 input, in the browser,
 * bit-for-bit like the Python pipeline (src/cell_classifier/preprocessing.py):
 *
 *   RGB  →  Pillow NEAREST resize to 224×224  →  /255  →  Float32 NHWC
 *
 * Pillow's NEAREST picks source pixel  floor(acc)  where acc starts at scale/2 and is
 * incremented by scale = srcSize/224 in double precision (verified identical over 316
 * sizes). Because only 224×224 = 50,176 source pixels are needed, we never decode the
 * whole image when we don't have to:
 *
 *   • Uncompressed TIFF  → read just the needed rows' bytes from the file.
 *   • Tiled/stripped compressed TIFF (JPEG, LZW, Deflate, PackBits) → decode only the
 *     tiles/strips that contain a sampled pixel. Gigapixel whole-slide TIFFs work
 *     without ever holding the full image in memory.
 *   • PNG/JPEG/BMP/WebP/GIF → browser decoder, then copy only the 224 needed rows.
 *
 * Exposes:  window.CellImaging.toModelInput(file, onProgress) → {data: Float32Array, width, height, kind}
 */
(() => {
  "use strict";
  const SIZE = 224;

  /** Pillow NEAREST source indices for one axis (double-precision accumulation, like Geometry.c). */
  function nearestIndices(n, out = SIZE) {
    const a = n / out, idx = new Int32Array(out);
    let xo = a * 0.5;
    for (let i = 0; i < out; i++) { idx[i] = Math.min(n - 1, Math.trunc(xo)); xo += a; }
    return idx;
  }

  class UnsupportedImageError extends Error {}

  // ---------------------------------------------------------------------------------------------
  // TIFF
  // ---------------------------------------------------------------------------------------------
  const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8 };

  async function readBytes(file, offset, length) {
    return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
  }

  async function parseTiff(file) {
    const head = await readBytes(file, 0, 16);
    const order = String.fromCharCode(head[0], head[1]);
    if (order !== "II" && order !== "MM") throw new UnsupportedImageError("Not a TIFF file.");
    const le = order === "II";
    let dv = new DataView(head.buffer);
    const magic = dv.getUint16(2, le);
    const big = magic === 43;
    if (magic !== 42 && !big) throw new UnsupportedImageError("Not a TIFF file.");
    const u64 = (d, o) => Number(d.getBigUint64(o, le));
    const ifdOffset = big ? u64(dv, 8) : dv.getUint32(4, le);

    // IFD0 = full-resolution page (same as Pillow, which uses frame 0)
    const cntBytes = await readBytes(file, ifdOffset, big ? 8 : 2);
    const count = big ? u64(new DataView(cntBytes.buffer), 0) : new DataView(cntBytes.buffer).getUint16(0, le);
    const entrySize = big ? 20 : 12;
    const ifd = await readBytes(file, ifdOffset + (big ? 8 : 2), count * entrySize);
    dv = new DataView(ifd.buffer);

    const tags = {};
    for (let i = 0; i < count; i++) {
      const p = i * entrySize;
      const tag = dv.getUint16(p, le), type = dv.getUint16(p + 2, le);
      const n = big ? u64(dv, p + 4) : dv.getUint32(p + 4, le);
      const tsize = TYPE_SIZE[type] || 1, total = n * tsize, inline = big ? 8 : 4;
      let vdv, voff;
      if (total <= inline) { vdv = dv; voff = p + (big ? 12 : 8); }
      else {
        const off = big ? u64(dv, p + 12) : dv.getUint32(p + 8, le);
        const bytes = await readBytes(file, off, total);
        vdv = new DataView(bytes.buffer); voff = 0;
      }
      if (type === 7 || type === 1 || type === 2) {   // UNDEFINED / BYTE / ASCII → raw bytes
        tags[tag] = new Uint8Array(vdv.buffer, vdv.byteOffset + voff, total).slice();
        continue;
      }
      const vals = new Array(n);
      for (let k = 0; k < n; k++) {
        const o = voff + k * tsize;
        vals[k] = type === 3 ? vdv.getUint16(o, le) : type === 4 ? vdv.getUint32(o, le)
          : type === 16 ? u64(vdv, o) : type === 8 ? vdv.getInt16(o, le) : type === 9 ? vdv.getInt32(o, le)
          : vdv.getUint32(o, le);
      }
      tags[tag] = vals;
    }
    const one = (t, d) => (tags[t] ? tags[t][0] : d);
    const info = {
      width: one(256), height: one(257),
      bits: tags[258] || [1], compression: one(259, 1), photometric: one(262, 2),
      spp: one(277, 1), planar: one(284, 1), predictor: one(317, 1),
      rowsPerStrip: one(278, 2 ** 32 - 1),
      tileW: one(322), tileH: one(323),
      offsets: tags[324] || tags[273], counts: tags[325] || tags[279],
      jpegTables: tags[347] || null, le,
    };
    info.tiled = !!info.tileW;
    return info;
  }

  // --- decompressors ----------------------------------------------------------------------------
  function packBits(src, expected) {
    const out = new Uint8Array(expected); let i = 0, o = 0;
    while (i < src.length && o < expected) {
      const n = (src[i++] << 24) >> 24;
      if (n >= 0) { out.set(src.subarray(i, i + n + 1), o); i += n + 1; o += n + 1; }
      else if (n !== -128) { out.fill(src[i++], o, o - n + 1); o += -n + 1; }
    }
    return out;
  }

  function lzw(src, expected) {           // TIFF LZW (MSB-first, early change)
    const out = new Uint8Array(expected);
    let o = 0, bitPos = 0, codeLen = 9;
    let dict = [], prev = null;
    const reset = () => { dict = []; for (let i = 0; i < 256; i++) dict[i] = [i]; dict[256] = dict[257] = null; codeLen = 9; };
    const read = () => {
      let v = 0;
      for (let k = 0; k < codeLen; k++) {
        const byte = src[(bitPos >> 3)]; if (byte === undefined) return 257;
        v = (v << 1) | ((byte >> (7 - (bitPos & 7))) & 1); bitPos++;
      }
      return v;
    };
    reset();
    while (o < expected) {
      const code = read();
      if (code === 257) break;
      if (code === 256) { reset(); prev = null; continue; }
      let entry;
      if (code < dict.length && dict[code]) entry = dict[code];
      else if (prev) entry = prev.concat([prev[0]]);
      else throw new UnsupportedImageError("Corrupt LZW data.");
      for (const b of entry) { if (o < expected) out[o++] = b; }
      if (prev) dict.push(prev.concat([entry[0]]));
      prev = entry;
      if (dict.length + 1 >= (1 << codeLen) && codeLen < 12) codeLen++;
    }
    return out;
  }

  async function inflate(src) {
    const stream = new Blob([src]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  function undoPredictor(buf, width, rows, spp) {   // horizontal differencing, 8-bit
    const rowLen = width * spp;
    for (let r = 0; r < rows; r++) {
      const base = r * rowLen;
      for (let i = spp; i < rowLen; i++) buf[base + i] = (buf[base + i] + buf[base + i - spp]) & 255;
    }
  }

  let jpegCanvas = null;
  async function decodeJpegChunk(tables, data, w, h) {
    let bytes = data;
    if (tables && tables.length > 4) {            // abbreviated stream: splice JPEGTables in
      bytes = new Uint8Array(tables.length - 2 + data.length - 2);
      bytes.set(tables.subarray(0, tables.length - 2), 0);
      bytes.set(data.subarray(2), tables.length - 2);
    }
    const bmp = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }),
      { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    if (!jpegCanvas || jpegCanvas.width < w || jpegCanvas.height < h) {
      jpegCanvas = makeCanvas(Math.max(w, 256), Math.max(h, 256));
    }
    const ctx = jpegCanvas.getContext("2d", { willReadFrequently: true });
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    return { rgba: ctx.getImageData(0, 0, w, h).data, stride: w * 4, spp: 4 };
  }

  function makeCanvas(w, h) {
    if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
    const c = document.createElement("canvas"); c.width = w; c.height = h; return c;
  }

  /** Decode one strip/tile → {rgba|bytes, stride, spp}. */
  async function decodeChunk(file, t, index, chunkW, chunkRows) {
    const raw = await readBytes(file, t.offsets[index], t.counts[index]);
    if (t.compression === 7 || t.compression === 6) return decodeJpegChunk(t.jpegTables, raw, chunkW, chunkRows);
    const expected = chunkW * chunkRows * t.spp;
    let buf;
    switch (t.compression) {
      case 1: buf = raw; break;
      case 5: buf = lzw(raw, expected); break;
      case 8: case 32946: buf = await inflate(raw); break;
      case 32773: buf = packBits(raw, expected); break;
      default: throw new UnsupportedImageError(`TIFF compression ${t.compression} is not supported in the browser. Export the slide as PNG or JPEG.`);
    }
    if (t.predictor === 2) undoPredictor(buf, chunkW, chunkRows, t.spp);
    return { bytes: buf, stride: chunkW * t.spp, spp: t.spp };
  }

  async function sampleTiff(file, onProgress) {
    const t = await parseTiff(file);
    const W = t.width, H = t.height;
    if (!W || !H) throw new UnsupportedImageError("TIFF has no image size.");
    if (t.planar !== 1 && t.spp > 1) throw new UnsupportedImageError("Planar TIFFs are not supported. Export as PNG or JPEG.");
    if (t.bits.some((b) => b !== 8)) throw new UnsupportedImageError(`Only 8-bit TIFFs are supported (this one is ${t.bits[0]}-bit). Export as PNG or JPEG.`);
    const isJpeg = t.compression === 7 || t.compression === 6;
    if (!isJpeg && !([1, 2].includes(t.photometric))) {
      throw new UnsupportedImageError("Only RGB or grayscale TIFFs are supported. Export as PNG or JPEG.");
    }

    const xs = nearestIndices(W), ys = nearestIndices(H);
    const out = new Uint8Array(SIZE * SIZE * 3);
    const gray = !isJpeg && t.spp < 3;

    const put = (oy, ox, src, i) => {   // src = bytes or rgba
      const o = (oy * SIZE + ox) * 3;
      if (gray) { out[o] = out[o + 1] = out[o + 2] = src[i]; }
      else { out[o] = src[i]; out[o + 1] = src[i + 1]; out[o + 2] = src[i + 2]; }
    };

    // Group the 224×224 sample points by the chunk (tile or strip) that contains them.
    const cw = t.tiled ? t.tileW : W;
    const ch = t.tiled ? t.tileH : Math.min(t.rowsPerStrip, H);
    const across = Math.ceil(W / cw);
    const jobs = new Map();  // chunkIndex → [[oy, ox, cx, cy]...]
    for (let oy = 0; oy < SIZE; oy++) {
      const y = ys[oy], cyIdx = Math.floor(y / ch);
      for (let ox = 0; ox < SIZE; ox++) {
        const x = xs[ox], cxIdx = Math.floor(x / cw);
        const idx = cyIdx * across + cxIdx;
        if (!jobs.has(idx)) jobs.set(idx, []);
        jobs.get(idx).push([oy, ox, x - cxIdx * cw, y - cyIdx * ch]);
      }
    }

    // Fast path: uncompressed strips → read only the needed row bytes.
    if (t.compression === 1 && !t.tiled) {
      const rowBytes = W * t.spp;
      for (let oy = 0; oy < SIZE; oy++) {
        const y = ys[oy], s = Math.floor(y / ch);
        const row = await readBytes(file, t.offsets[s] + (y - s * ch) * rowBytes, rowBytes);
        for (let ox = 0; ox < SIZE; ox++) put(oy, ox, row, xs[ox] * t.spp);
        if (oy % 16 === 0) onProgress?.(oy / SIZE);
      }
      return { pixels: out, width: W, height: H, kind: `TIFF (uncompressed, ${W}×${H})` };
    }

    const entries = [...jobs.entries()];
    let done = 0;
    const concurrency = isJpeg ? 6 : 2;
    const worker = async () => {
      while (entries.length) {
        const [idx, pts] = entries.shift();
        const rows = t.tiled ? ch : Math.min(ch, H - Math.floor(idx / across) * ch);
        const chunk = await decodeChunk(file, t, idx, cw, rows);
        const src = chunk.rgba || chunk.bytes;
        for (const [oy, ox, cx, cy] of pts) put(oy, ox, src, cy * chunk.stride + cx * chunk.spp);
        if (++done % 64 === 0) onProgress?.(done / jobs.size);
      }
    };
    await Promise.all(Array.from({ length: concurrency }, worker));
    const what = { 1: "uncompressed", 5: "LZW", 6: "JPEG", 7: "JPEG", 8: "Deflate", 32946: "Deflate", 32773: "PackBits" }[t.compression];
    return { pixels: out, width: W, height: H,
             kind: `TIFF (${what}, ${t.tiled ? "tiled" : "strips"}, ${W.toLocaleString()}×${H.toLocaleString()}, ${jobs.size.toLocaleString()} ${t.tiled ? "tiles" : "strips"} decoded)` };
  }

  // ---------------------------------------------------------------------------------------------
  // Everything else the browser can decode (PNG, JPEG, WebP, BMP, GIF, AVIF…)
  // ---------------------------------------------------------------------------------------------
  async function sampleBrowserImage(file) {
    let bmp;
    try {
      bmp = await createImageBitmap(file, { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" });
    } catch {
      throw new UnsupportedImageError("This browser can't decode that image. Use JPG, PNG or TIF (iPhone HEIC: set Camera → Formats → Most Compatible).");
    }
    const W = bmp.width, H = bmp.height;
    const xs = nearestIndices(W), ys = nearestIndices(H);
    const out = new Uint8Array(SIZE * SIZE * 3);
    const CHUNK = 8192;                                   // stay under every browser's canvas limits
    const canvas = makeCanvas(Math.min(W, CHUNK), 1);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    for (let oy = 0; oy < SIZE; oy++) {
      for (let x0 = 0; x0 < W; x0 += CHUNK) {
        const w = Math.min(CHUNK, W - x0);
        ctx.clearRect(0, 0, canvas.width, 1);
        ctx.drawImage(bmp, x0, ys[oy], w, 1, 0, 0, w, 1);  // copy exactly one source row segment
        const row = ctx.getImageData(0, 0, w, 1).data;
        for (let ox = 0; ox < SIZE; ox++) {
          const x = xs[ox];
          if (x < x0 || x >= x0 + w) continue;
          const i = (x - x0) * 4, o = (oy * SIZE + ox) * 3;
          out[o] = row[i]; out[o + 1] = row[i + 1]; out[o + 2] = row[i + 2];
        }
      }
    }
    bmp.close();
    return { pixels: out, width: W, height: H, kind: `${(file.type || "image").replace("image/", "").toUpperCase()} ${W}×${H}` };
  }

  async function isTiff(file) {
    const h = await readBytes(file, 0, 4);
    return (h[0] === 0x49 && h[1] === 0x49 && (h[2] === 42 || h[2] === 43) && h[3] === 0) ||
           (h[0] === 0x4d && h[1] === 0x4d && h[2] === 0 && (h[3] === 42 || h[3] === 43));
  }

  async function toModelInput(file, onProgress) {
    const r = (await isTiff(file)) ? await sampleTiff(file, onProgress) : await sampleBrowserImage(file);
    const data = new Float32Array(SIZE * SIZE * 3);
    for (let i = 0; i < data.length; i++) data[i] = r.pixels[i] / 255;
    return { data, width: r.width, height: r.height, kind: r.kind, pixels: r.pixels };
  }

  window.CellImaging = { toModelInput, nearestIndices, UnsupportedImageError, SIZE };
})();
