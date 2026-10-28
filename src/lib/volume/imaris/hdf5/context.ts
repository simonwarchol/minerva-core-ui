/** Shared per-file state: metadata source, superblock sizes, object header reader. */

import { Cursor } from "./cursor";
import { MSG, type RawMessage } from "./messages";
import type { RangeSource } from "./source";

/** Bytes fetched up front for an object header of unknown size. */
const OBJECT_HEADER_GUESS = 1024;

export class Hdf5Context {
  constructor(
    /** Page-cached source for metadata. */
    readonly meta: RangeSource,
    /** Base address that all file addresses are relative to. */
    readonly base: number,
    readonly sizeOfOffsets: number,
    readonly sizeOfLengths: number,
    /** B-tree K values (superblock v0/v1, else library defaults); read-size hints. */
    readonly k: { groupLeaf: number; groupInternal: number; chunk: number },
  ) {}

  /** Metadata read of exactly `length` bytes at file address `addr`. */
  async read(addr: number, length: number): Promise<Uint8Array> {
    const buf = await this.meta.read(this.base + addr, length);
    if (buf.length < length) {
      throw new Error(
        `HDF5: unexpected EOF reading ${length} bytes at address ${addr}`,
      );
    }
    return buf;
  }

  /**
   * Speculative metadata read: at least `min` bytes, up to `want` (fewer only
   * at EOF). Reading a whole structure at once avoids a dependent round trip
   * when it straddles an uncached page.
   */
  async readAtLeast(
    addr: number,
    min: number,
    want: number,
  ): Promise<Uint8Array> {
    const buf = await this.meta.read(this.base + addr, Math.max(min, want));
    if (buf.length < min) {
      throw new Error(
        `HDF5: unexpected EOF reading ${min} bytes at address ${addr}`,
      );
    }
    return buf;
  }

  /** `length` bytes at `addr`, reusing `head` (read at `headAddr`) when it covers them. */
  private async within(
    head: Uint8Array,
    headAddr: number,
    addr: number,
    length: number,
  ): Promise<Cursor> {
    const rel = addr - headAddr;
    if (rel >= 0 && rel + length <= head.length) {
      return this.wrap(head.subarray(rel, rel + length));
    }
    return this.cursor(addr, length);
  }

  async cursor(addr: number, length: number): Promise<Cursor> {
    return this.wrap(await this.read(addr, length));
  }

  wrap(buf: Uint8Array, pos = 0): Cursor {
    return new Cursor(buf, pos, this.sizeOfOffsets, this.sizeOfLengths);
  }

  private headers = new Map<number, Promise<RawMessage[]>>();

  /** All messages of the object header at `addr` (continuations followed). */
  objectHeader(addr: number): Promise<RawMessage[]> {
    let p = this.headers.get(addr);
    if (!p) {
      p = this.readObjectHeader(addr);
      this.headers.set(addr, p);
      p.catch(() => this.headers.delete(addr));
    }
    return p;
  }

  private async readObjectHeader(addr: number): Promise<RawMessage[]> {
    // Most object headers fit in a few hundred bytes: fetch them in one go.
    const sig = await this.readAtLeast(addr, 16, OBJECT_HEADER_GUESS);
    if (
      sig[0] === 0x4f &&
      sig[1] === 0x48 &&
      sig[2] === 0x44 &&
      sig[3] === 0x52
    ) {
      return this.readHeaderV2(addr, sig);
    }
    if (sig[0] === 1) return this.readHeaderV1(addr, sig);
    throw new Error(`HDF5: no object header at address ${addr}`);
  }

  /** Version 1 object header: 16-byte prefix, 8-byte aligned messages. */
  private async readHeaderV1(
    addr: number,
    head: Uint8Array,
  ): Promise<RawMessage[]> {
    const p = this.wrap(head);
    p.skip(2); // version, reserved
    const nMessages = p.u16();
    p.skip(4); // reference count
    const headerSize = p.u32();
    const out: RawMessage[] = [];
    const blocks: Array<[number, number]> = [[addr + 16, headerSize]];
    while (blocks.length > 0 && out.length < nMessages) {
      const [a, len] = blocks.shift()!;
      const c = await this.within(head, addr, a, len);
      while (c.remaining >= 8 && out.length < nMessages) {
        const type = c.u16();
        const size = c.u16();
        const flags = c.u8();
        c.skip(3);
        const data = c.bytes(Math.min(size, c.remaining));
        out.push({ type, flags, data });
        if (type === MSG.CONTINUATION) {
          const cc = this.wrap(data);
          const off = cc.offset();
          const clen = cc.length();
          if (off !== null) blocks.push([off, clen]);
        }
      }
    }
    return out;
  }

  /** Version 2 object header ("OHDR") and its "OCHK" continuation blocks. */
  private async readHeaderV2(
    addr: number,
    head: Uint8Array,
  ): Promise<RawMessage[]> {
    const pre = this.wrap(head);
    pre.skip(5); // signature, version
    const flags = pre.u8();
    let prefix = 6;
    if (flags & 0x20) prefix += 16; // access/modification/change/birth times
    if (flags & 0x10) prefix += 4; // max compact / min dense attributes
    const sizeLen = 1 << (flags & 0x03);
    pre.pos = prefix;
    const chunk0Size = pre.uN(sizeLen);
    prefix += sizeLen;
    const trackOrder = (flags & 0x04) !== 0;
    const out: RawMessage[] = [];
    const parse = (c: Cursor, blocks: Array<[number, number]>) => {
      const hdr = trackOrder ? 6 : 4;
      while (c.remaining >= hdr) {
        const type = c.u8();
        const size = c.u16();
        const mflags = c.u8();
        if (trackOrder) c.skip(2);
        if (size > c.remaining) break;
        const data = c.bytes(size);
        out.push({ type, flags: mflags, data });
        if (type === MSG.CONTINUATION) {
          const cc = this.wrap(data);
          const off = cc.offset();
          const len = cc.length();
          if (off !== null) blocks.push([off, len]);
        }
      }
    };
    const blocks: Array<[number, number]> = [];
    parse(await this.within(head, addr, addr + prefix, chunk0Size), blocks);
    while (blocks.length > 0) {
      const [a, len] = blocks.shift()!;
      // "OCHK" signature (4) ... messages ... checksum (4)
      parse(await this.cursor(a + 4, len - 8), blocks);
    }
    return out;
  }
}
