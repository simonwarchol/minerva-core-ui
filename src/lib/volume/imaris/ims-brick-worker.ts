/**
 * Imaris worker. Two jobs:
 *
 * - **extract** — this worker *owns* a chunk (chosen by address hash on the
 *   main thread): it inflates it once, keeps it in an LRU as normalised
 *   float16, and returns just the sub-region a brick needs. Kiln bricks carry
 *   a 1-voxel ghost border, so most bricks touch neighbouring chunks only for
 *   a sliver; owning chunks means each chunk is decoded once instead of once
 *   per neighbouring brick.
 * - **assemble** — gather the extracted pieces into the source region and
 *   box-resample it into one 66³ brick (float16 bits or 8-bit unorm).
 */

import { decodeChunk, type Hdf5Filter } from "./hdf5/filters";
import { PHYSICAL } from "./ims-layout";
import { ByteLru } from "./util/byte-lru";
import {
  float32ToFloat16Array,
  float32ToFloat16Bits,
  getFloat16ToFloat32Lut,
} from "./util/float16";

/** Sent once, before any other request. */
export interface ImsInitRequest {
  type: "init";
  /** Byte budget of this worker's decoded-chunk cache */
  chunkBudget: number;
}

export interface ImsExtractRequest {
  type: "extract";
  id: number;
  address: number;
  /** Compressed bytes, or null if the main thread believes we have it cached */
  bytes: Uint8Array | null;
  filters: Hdf5Filter[];
  filterMask: number;
  dtype: { class: "int" | "uint" | "float"; size: number };
  /** chunk shape [x, y, z] */
  chunkShape: [number, number, number];
  normLo: number;
  normHi: number;
  /** local [x0, y0, z0, x1, y1, z1) inside the chunk */
  region: [number, number, number, number, number, number];
}

export interface ImsExtractResponse {
  type: "extract";
  id: number;
  /** float16 bits, x fastest */
  data?: ArrayBuffer;
  /** the chunk was not cached and no bytes were sent */
  missing?: boolean;
  error?: string;
}

export interface ImsPiece {
  /** absolute source coords of the piece origin */
  x0: number;
  y0: number;
  z0: number;
  sx: number;
  sy: number;
  sz: number;
  data: Uint16Array;
}

export interface ImsAssembleRequest {
  type: "assemble";
  id: number;
  pieces: ImsPiece[];
  /** source box [lo, hi) in source voxels, [x, y, z] */
  lo: [number, number, number];
  hi: [number, number, number];
  /** per-axis [start, end) source ranges for each of the 66 output samples (absolute) */
  rx: Int32Array;
  ry: Int32Array;
  rz: Int32Array;
  /**
   * Source boxes [x0, y0, z0, x1, y1, z1) of ghost-border chunks that were
   * not fetched: filled by clamping into the brick interior
   */
  fills?: Array<[number, number, number, number, number, number]>;
  /** Output encoding: float16 bits (r16float atlas) or 8-bit unorm */
  format: "f16" | "u8";
}

export interface ImsBrickResponse {
  type: "brick";
  id: number;
  data?: ArrayBuffer;
  min: number;
  max: number;
  avg: number;
  assemblyMs: number;
  error?: string;
}

export type ImsWorkerRequest =
  | ImsInitRequest
  | ImsExtractRequest
  | ImsAssembleRequest;

const f16ToF32 = getFloat16ToFloat32Lut();

// --- owned-chunk cache (normalised float16) ------------------------------------

let cache = new ByteLru<number, Uint16Array>(
  64 * 1024 * 1024,
  (v) => v.byteLength,
);
const decoding = new Map<number, Promise<Uint16Array>>();

// Reused float32 scratch. Fresh multi-MB buffers per brick are mmapped and
// zero-filled by the OS, and those page faults serialise across workers:
// assembly ran ~6x slower with 8 workers busy than with one.
const scratchPool: Float32Array[] = [];
function scratch(slot: number, n: number): Float32Array {
  let s = scratchPool[slot];
  if (!s || s.length < n) {
    s = new Float32Array(Math.ceil(n * 1.25));
    scratchPool[slot] = s;
  }
  return s.subarray(0, n);
}

const NativeFloat16Array = (
  globalThis as unknown as {
    Float16Array?: new (
      n: number,
    ) => ArrayLike<number> & {
      [i: number]: number;
      buffer: ArrayBuffer;
    };
  }
).Float16Array;

/** Raw value (8/16-bit bit pattern) → normalised float16 bits, per dtype and range. */
const luts = new Map<string, Uint16Array>();

function lutFor(
  dtype: ImsExtractRequest["dtype"],
  lo: number,
  hi: number,
): Uint16Array {
  const key = `${dtype.class}${dtype.size}:${lo}:${hi}`;
  let lut = luts.get(key);
  if (!lut) {
    const n = dtype.size === 1 ? 256 : 65536;
    const signed = dtype.class === "int";
    const inv = hi > lo ? 1 / (hi - lo) : 0;
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = ((signed && i >= 32768 ? i - 65536 : i) - lo) * inv;
      f[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    lut = float32ToFloat16Array(f);
    if (luts.size >= 16) luts.clear(); // a handful of channels per file
    luts.set(key, lut);
  }
  return lut;
}

/** Decoded chunk bytes → values normalised to [0, 1], as float16 bits. */
function normalise(
  raw: Uint8Array,
  dtype: ImsExtractRequest["dtype"],
  lo: number,
  hi: number,
): Uint16Array {
  if (dtype.class !== "float") {
    // 8/16-bit: one table lookup per voxel (int16 indexed by its bit pattern).
    const lut = lutFor(dtype, lo, hi);
    const n = raw.byteLength / dtype.size;
    const src =
      dtype.size === 1
        ? raw
        : raw.byteOffset % 2 === 0
          ? new Uint16Array(raw.buffer, raw.byteOffset, n)
          : new Uint16Array(raw.slice().buffer);
    const out = new Uint16Array(n);
    for (let i = 0; i < n; i++) out[i] = lut[src[i]!]!;
    return out;
  }
  const d =
    raw.byteOffset % 4 === 0
      ? new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4)
      : new Float32Array(raw.slice().buffer);
  const inv = hi > lo ? 1 / (hi - lo) : 0;
  const n = d.length;
  if (NativeFloat16Array) {
    const h = new NativeFloat16Array(n);
    for (let i = 0; i < n; i++) {
      const v = (d[i]! - lo) * inv;
      h[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return new Uint16Array(h.buffer);
  }
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const v = (d[i]! - lo) * inv;
    out[i] = float32ToFloat16Bits(v < 0 ? 0 : v > 1 ? 1 : v);
  }
  return out;
}

async function handleExtract(
  req: ImsExtractRequest,
): Promise<ImsExtractResponse> {
  let chunk = cache.get(req.address);
  if (!chunk) {
    const pending = decoding.get(req.address);
    if (pending) {
      chunk = await pending;
    } else if (req.bytes) {
      const p = decodeChunk(
        req.bytes,
        req.filters,
        req.filterMask,
        req.dtype.size,
      ).then((raw) => {
        const v = normalise(raw, req.dtype, req.normLo, req.normHi);
        cache.set(req.address, v);
        return v;
      });
      decoding.set(req.address, p);
      try {
        chunk = await p;
      } finally {
        decoding.delete(req.address);
      }
    } else {
      return { type: "extract", id: req.id, missing: true };
    }
  }
  const [csx, csy] = req.chunkShape;
  const [x0, y0, z0, x1, y1, z1] = req.region;
  const sx = x1 - x0;
  const sy = y1 - y0;
  const out = new Uint16Array(sx * sy * (z1 - z0));
  let o = 0;
  for (let z = z0; z < z1; z++) {
    for (let y = y0; y < y1; y++) {
      const base = (z * csy + y) * csx;
      out.set(chunk.subarray(base + x0, base + x1), o);
      o += sx;
    }
  }
  return { type: "extract", id: req.id, data: out.buffer };
}

/** Pieces tile the region exactly when their volumes sum to it (they never overlap). */
function coversRegion(pieces: ImsPiece[], n: number): boolean {
  let v = 0;
  for (const p of pieces) v += p.sx * p.sy * p.sz;
  return v === n;
}

/** Copy each `fills` voxel from the nearest voxel of the brick interior. */
function fillClamped(
  src: Float32Array | Uint16Array,
  req: ImsAssembleRequest,
): void {
  if (!req.fills) return;
  const [lx, ly, lz] = req.lo;
  const sx = req.hi[0] - lx;
  const sy = req.hi[1] - ly;
  const inner = (PHYSICAL - 2) * 2 + 1;
  const clamp = (v: number, r: Int32Array) =>
    Math.min(r[inner]! - 1, Math.max(r[2]!, v));
  for (const [x0, y0, z0, x1, y1, z1] of req.fills) {
    for (let z = z0; z < z1; z++) {
      const cz = clamp(z, req.rz);
      for (let y = y0; y < y1; y++) {
        const cy = clamp(y, req.ry);
        for (let x = x0; x < x1; x++) {
          src[((z - lz) * sy + (y - ly)) * sx + (x - lx)] =
            src[((cz - lz) * sy + (cy - ly)) * sx + (clamp(x, req.rx) - lx)]!;
        }
      }
    }
  }
}

/** Every output sample covers exactly one source voxel (no resampling). */
function unitRanges(r: Int32Array): boolean {
  for (let i = 0; i < PHYSICAL; i++)
    if (r[i * 2 + 1]! - r[i * 2]! !== 1) return false;
  return true;
}

let scratch16 = new Uint16Array(0);

/**
 * The source level already is the brick's level: gather float16 bits
 * straight into the 66³ brick (no float32 round trip or box filter).
 */
function assembleUnresampled(req: ImsAssembleRequest): {
  data: Uint16Array | Uint8Array;
  min: number;
  max: number;
  avg: number;
} {
  const [lx, ly, lz] = req.lo;
  const sx = req.hi[0] - lx;
  const sy = req.hi[1] - ly;
  const sz = req.hi[2] - lz;
  const n = sx * sy * sz;
  if (scratch16.length < n) scratch16 = new Uint16Array(Math.ceil(n * 1.25));
  const src = scratch16.subarray(0, n);
  if (!coversRegion(req.pieces, n)) src.fill(0); // unallocated chunks = 0.0
  for (const p of req.pieces) {
    for (let z = 0; z < p.sz; z++)
      for (let y = 0; y < p.sy; y++) {
        const si = (z * p.sy + y) * p.sx;
        const di = ((p.z0 + z - lz) * sy + (p.y0 + y - ly)) * sx + (p.x0 - lx);
        src.set(p.data.subarray(si, si + p.sx), di);
      }
  }
  fillClamped(src, req);
  const P = PHYSICAL;
  const u8 = req.format === "u8";
  const data = u8 ? new Uint8Array(P * P * P) : new Uint16Array(P * P * P);
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  let sum = 0;
  let o = 0;
  for (let kz = 0; kz < P; kz++) {
    const zo = (req.rz[kz * 2]! - lz) * sy;
    for (let j = 0; j < P; j++) {
      const row = (zo + req.ry[j * 2]! - ly) * sx - lx;
      for (let i = 0; i < P; i++) {
        const bits = src[row + req.rx[i * 2]!]!;
        const v = f16ToF32[bits]!;
        data[o++] = u8 ? Math.round(v * 255) : bits;
        const q = v * 65535;
        if (q < min) min = q;
        if (q > max) max = q;
        sum += q;
      }
    }
  }
  return { data, min, max, avg: sum / (P * P * P) };
}

function assemble(req: ImsAssembleRequest): {
  data: Uint16Array | Uint8Array;
  min: number;
  max: number;
  avg: number;
} {
  if (unitRanges(req.rx) && unitRanges(req.ry) && unitRanges(req.rz)) {
    return assembleUnresampled(req);
  }
  const [lx, ly, lz] = req.lo;
  const sx = req.hi[0] - lx;
  const sy = req.hi[1] - ly;
  const sz = req.hi[2] - lz;

  // 1. Normalised source region (unallocated chunks stay 0)
  const src = scratch(0, sx * sy * sz);
  if (!coversRegion(req.pieces, sx * sy * sz)) src.fill(0);
  for (const p of req.pieces) {
    let si = 0;
    for (let z = 0; z < p.sz; z++) {
      for (let y = 0; y < p.sy; y++) {
        let di = ((p.z0 + z - lz) * sy + (p.y0 + y - ly)) * sx + (p.x0 - lx);
        for (let x = 0; x < p.sx; x++) src[di++] = f16ToF32[p.data[si++]!]!;
      }
    }
  }
  fillClamped(src, req);

  // 2. Separable box resample to 66³ (x, then y, then z)
  const P = PHYSICAL;
  const tx = scratch(1, P * sy * sz);
  for (let z = 0; z < sz; z++)
    for (let y = 0; y < sy; y++) {
      const row = (z * sy + y) * sx;
      const orow = (z * sy + y) * P;
      for (let i = 0; i < P; i++) {
        const a = req.rx[i * 2]! - lx;
        const b = req.rx[i * 2 + 1]! - lx;
        let s = 0;
        for (let k = a; k < b; k++) s += src[row + k]!;
        tx[orow + i] = s / (b - a);
      }
    }
  const ty = scratch(2, P * P * sz);
  for (let z = 0; z < sz; z++)
    for (let j = 0; j < P; j++) {
      const a = req.ry[j * 2]! - ly;
      const b = req.ry[j * 2 + 1]! - ly;
      const orow = (z * P + j) * P;
      const w = 1 / (b - a);
      for (let i = 0; i < P; i++) {
        let s = 0;
        for (let k = a; k < b; k++) s += tx[(z * sy + k) * P + i]!;
        ty[orow + i] = s * w;
      }
    }
  const flat = scratch(3, P * P * P);
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  let sum = 0;
  for (let kz = 0; kz < P; kz++) {
    const a = req.rz[kz * 2]! - lz;
    const b = req.rz[kz * 2 + 1]! - lz;
    const w = 1 / (b - a);
    for (let j = 0; j < P; j++) {
      const orow = (kz * P + j) * P;
      for (let i = 0; i < P; i++) {
        let s = 0;
        for (let k = a; k < b; k++) s += ty[(k * P + j) * P + i]!;
        const v = s * w;
        flat[orow + i] = v;
        const q = v * 65535;
        if (q < min) min = q;
        if (q > max) max = q;
        sum += q;
      }
    }
  }
  let data: Uint16Array | Uint8Array;
  if (req.format === "u8") {
    data = new Uint8Array(flat.length);
    for (let i = 0; i < flat.length; i++) data[i] = Math.round(flat[i]! * 255);
  } else {
    data = float32ToFloat16Array(flat);
  }
  return { data, min, max, avg: sum / (P * P * P) };
}

const post = postMessage as (m: unknown, t?: Transferable[]) => void;

self.onmessage = (e: MessageEvent<ImsWorkerRequest>) => {
  const msg = e.data;
  if (msg.type === "init") {
    cache = new ByteLru(msg.chunkBudget, (v) => v.byteLength);
  } else if (msg.type === "extract") {
    handleExtract(msg)
      .then((r) => post(r, r.data ? [r.data] : []))
      .catch((err) =>
        post({
          type: "extract",
          id: msg.id,
          error: err instanceof Error ? err.message : String(err),
        } satisfies ImsExtractResponse),
      );
  } else if (msg.type === "assemble") {
    const t0 = performance.now();
    try {
      const r = assemble(msg);
      const resp: ImsBrickResponse = {
        type: "brick",
        id: msg.id,
        data: r.data.buffer as ArrayBuffer,
        min: r.min,
        max: r.max,
        avg: r.avg,
        assemblyMs: performance.now() - t0,
      };
      post(resp, [r.data.buffer as ArrayBuffer]);
    } catch (err) {
      post({
        type: "brick",
        id: msg.id,
        min: 0,
        max: 0,
        avg: 0,
        assemblyMs: 0,
        error: err instanceof Error ? err.message : String(err),
      } satisfies ImsBrickResponse);
    }
  }
};
