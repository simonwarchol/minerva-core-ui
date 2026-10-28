/** HDF5 filter pipeline decoding (deflate, shuffle, fletcher32, LZ4 plugin). */

export interface Hdf5Filter {
  id: number;
  name: string;
  clientData: number[];
  optional: boolean;
}

export const FILTER_NAMES: Record<number, string> = {
  1: "deflate",
  2: "shuffle",
  3: "fletcher32",
  4: "szip",
  5: "nbit",
  6: "scaleoffset",
  32001: "blosc",
  32004: "lz4",
  32008: "bitshuffle",
  32015: "zstd",
};

/**
 * Undo a chunk's filter pipeline. Filters are applied on write in pipeline
 * order, so they are reversed here; bit i of `filterMask` set means filter i
 * was skipped for this chunk.
 */
export async function decodeChunk(
  stored: Uint8Array,
  filters: Hdf5Filter[],
  filterMask: number,
  elementSize: number,
): Promise<Uint8Array> {
  let buf = stored;
  for (let i = filters.length - 1; i >= 0; i--) {
    if (filterMask & (1 << i)) continue;
    const f = filters[i]!;
    switch (f.id) {
      case 1:
        buf = await inflate(buf);
        break;
      case 2:
        buf = unshuffle(buf, f.clientData[0] || elementSize);
        break;
      case 3:
        buf = buf.subarray(0, Math.max(0, buf.length - 4));
        break;
      case 32004:
        buf = lz4HdfDecode(buf);
        break;
      default:
        throw new Error(
          `HDF5: unsupported filter ${f.id} (${f.name || FILTER_NAMES[f.id] || "unknown"})`,
        );
    }
  }
  return buf;
}

/** zlib-wrapped deflate (HDF5 filter 1), via the native DecompressionStream. */
async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as Uint8Array<ArrayBuffer>])
    .stream()
    .pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Byte shuffle (HDF5 filter 2): bytes are grouped by significance on write. */
function unshuffle(src: Uint8Array, elementSize: number): Uint8Array {
  if (elementSize <= 1) return src;
  const n = Math.floor(src.length / elementSize);
  const out = new Uint8Array(src.length);
  for (let b = 0; b < elementSize; b++) {
    const plane = b * n;
    for (let i = 0, o = b; i < n; i++, o += elementSize) {
      out[o] = src[plane + i]!;
    }
  }
  // Trailing bytes that do not form a whole element are stored unshuffled.
  const tail = n * elementSize;
  if (tail < src.length) out.set(src.subarray(tail), tail);
  return out;
}

/**
 * HDF5 LZ4 plugin (filter 32004) framing: 8-byte BE original size, 4-byte BE
 * block size, then per block a 4-byte BE compressed size and the LZ4 block
 * (stored raw when compressed size == block size).
 */
function lz4HdfDecode(src: Uint8Array): Uint8Array {
  const view = new DataView(src.buffer, src.byteOffset, src.byteLength);
  if (src.length < 12) throw new Error("HDF5 LZ4: truncated header");
  const origSize = view.getUint32(0) * 2 ** 32 + view.getUint32(4);
  let blockSize = view.getUint32(8);
  if (blockSize === 0) blockSize = origSize;
  const out = new Uint8Array(origSize);
  let ip = 12;
  let op = 0;
  while (op < origSize) {
    const bs = Math.min(blockSize, origSize - op);
    if (ip + 4 > src.length) throw new Error("HDF5 LZ4: truncated block");
    const cs = view.getUint32(ip);
    ip += 4;
    if (cs === bs) {
      out.set(src.subarray(ip, ip + cs), op);
    } else {
      const n = lz4DecodeBlock(src.subarray(ip, ip + cs), out, op, bs);
      if (n !== bs) {
        throw new Error(
          `HDF5 LZ4: block decoded to ${n} bytes, expected ${bs}`,
        );
      }
    }
    ip += cs;
    op += bs;
  }
  return out;
}

/** Raw LZ4 block decoder. Writes into dst at dstPos; returns bytes written. */
function lz4DecodeBlock(
  src: Uint8Array,
  dst: Uint8Array,
  dstPos: number,
  maxLen: number,
): number {
  let ip = 0;
  let op = dstPos;
  const end = dstPos + maxLen;
  while (ip < src.length) {
    const token = src[ip++]!;
    let lit = token >>> 4;
    if (lit === 15) {
      let b: number;
      do {
        b = src[ip++]!;
        lit += b;
      } while (b === 255);
    }
    if (op + lit > end || ip + lit > src.length) {
      throw new Error("HDF5 LZ4: corrupt block (literal overrun)");
    }
    dst.set(src.subarray(ip, ip + lit), op);
    ip += lit;
    op += lit;
    if (ip >= src.length) break; // last sequence has literals only
    const offset = src[ip]! | (src[ip + 1]! << 8);
    ip += 2;
    let len = token & 15;
    if (len === 15) {
      let b: number;
      do {
        b = src[ip++]!;
        len += b;
      } while (b === 255);
    }
    len += 4;
    let m = op - offset;
    if (offset === 0 || m < dstPos || op + len > end) {
      throw new Error("HDF5 LZ4: corrupt block (bad match)");
    }
    if (offset >= len) {
      dst.copyWithin(op, m, m + len);
      op += len;
    } else {
      // Overlapping match: byte-wise copy replicates the pattern.
      for (let k = 0; k < len; k++) dst[op++] = dst[m++]!;
    }
  }
  return op - dstPos;
}
