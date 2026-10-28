/** Little-endian binary cursor used by all HDF5 structure parsers. */

export class Cursor {
  readonly view: DataView;
  pos: number;

  constructor(
    readonly buf: Uint8Array,
    pos = 0,
    /** Superblock "Size of Offsets" (bytes per file address). */
    readonly sizeOfOffsets = 8,
    /** Superblock "Size of Lengths". */
    readonly sizeOfLengths = 8,
  ) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.pos = pos;
  }

  get remaining(): number {
    return this.buf.length - this.pos;
  }

  /** A cursor over a sub-range, sharing the size parameters. */
  sub(start: number, length: number): Cursor {
    return new Cursor(
      this.buf.subarray(start, start + length),
      0,
      this.sizeOfOffsets,
      this.sizeOfLengths,
    );
  }

  private need(n: number): void {
    if (this.pos + n > this.buf.length) {
      throw new Error(
        `HDF5: truncated structure (need ${n} bytes at ${this.pos}, have ${this.buf.length})`,
      );
    }
  }

  u8(): number {
    this.need(1);
    return this.view.getUint8(this.pos++);
  }

  u16(): number {
    this.need(2);
    const v = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return v;
  }

  u32(): number {
    this.need(4);
    const v = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }

  /** Unsigned little-endian integer of `n` bytes (n <= 8; >2^53 loses precision). */
  uN(n: number): number {
    this.need(n);
    let v = 0;
    let mul = 1;
    for (let i = 0; i < n; i++) {
      v += this.view.getUint8(this.pos + i) * mul;
      mul *= 256;
    }
    this.pos += n;
    return v;
  }

  u64(): number {
    return this.uN(8);
  }

  /** File address; `null` for the "undefined address" (all bits set). */
  offset(): number | null {
    const n = this.sizeOfOffsets;
    this.need(n);
    for (let i = 0; i < n; i++) {
      if (this.view.getUint8(this.pos + i) !== 0xff) return this.uN(n);
    }
    this.pos += n;
    return null;
  }

  length(): number {
    return this.uN(this.sizeOfLengths);
  }

  /** Length field where all bits set means "undefined" (e.g. unlimited). */
  lengthOrUndefined(): number | null {
    const n = this.sizeOfLengths;
    this.need(n);
    for (let i = 0; i < n; i++) {
      if (this.view.getUint8(this.pos + i) !== 0xff) return this.uN(n);
    }
    this.pos += n;
    return null;
  }

  bytes(n: number): Uint8Array {
    this.need(n);
    const b = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return b;
  }

  skip(n: number): void {
    this.need(n);
    this.pos += n;
  }

  /** ASCII string of `n` bytes (signatures). */
  ascii(n: number): string {
    return String.fromCharCode(...this.bytes(n));
  }

  /** Advance to the next multiple of `a`, measured from `origin`. */
  align(a: number, origin = 0): void {
    const rel = this.pos - origin;
    this.pos = origin + Math.ceil(rel / a) * a;
  }
}

const utf8 = new TextDecoder("utf-8");

/** Decode a NUL-terminated / NUL-padded string. */
export function decodeCString(b: Uint8Array): string {
  const z = b.indexOf(0);
  return utf8.decode(z >= 0 ? b.subarray(0, z) : b);
}

export function decodeUtf8(b: Uint8Array): string {
  return utf8.decode(b);
}

/** floor(log2(v)) for v >= 1 (works beyond 2^32). */
export function log2Floor(v: number): number {
  let r = 0;
  while (v >= 2) {
    v = Math.floor(v / 2);
    r++;
  }
  return r;
}

/** Bytes needed to encode `v` the way H5VM_limit_enc_size does. */
export function limitEncSize(v: number): number {
  return Math.floor(log2Floor(Math.max(1, v)) / 8) + 1;
}

export function product(dims: readonly number[]): number {
  let p = 1;
  for (const d of dims) p *= d;
  return p;
}
