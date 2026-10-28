/** Fractal heap ("FRHP" header, "FHIB" indirect / "FHDB" direct blocks). */

import type { Hdf5Context } from "./context";
import { log2Floor } from "./cursor";

interface IndirectBlock {
  /** Child addresses, row-major (nrows x width); null = not allocated. */
  children: Array<number | null>;
  nrows: number;
}

export class FractalHeap {
  private iblocks = new Map<number, Promise<IndirectBlock>>();

  private constructor(
    private readonly ctx: Hdf5Context,
    readonly idLength: number,
    private readonly filtered: boolean,
    private readonly maxManagedSize: number,
    private readonly width: number,
    private readonly startBlockSize: number,
    private readonly maxDirectBlockSize: number,
    private readonly maxHeapBits: number,
    private readonly rootAddress: number | null,
    private readonly rootRows: number,
  ) {}

  /** Parse the fractal heap header at `addr`. */
  static async open(ctx: Hdf5Context, addr: number): Promise<FractalHeap> {
    const O = ctx.sizeOfOffsets;
    const L = ctx.sizeOfLengths;
    const c = await ctx.cursor(addr, 14 + 12 * L + 3 * O + 8);
    if (c.ascii(4) !== "FRHP") {
      throw new Error(`HDF5: expected fractal heap header at ${addr}`);
    }
    c.u8(); // version
    const idLength = c.u16();
    const filterLen = c.u16();
    c.u8(); // flags
    const maxManagedSize = c.u32();
    c.length(); // next huge object ID
    c.offset(); // huge object v2 B-tree
    c.length(); // free space in managed blocks
    c.offset(); // free space manager
    c.length(); // managed space
    c.length(); // allocated managed space
    c.length(); // direct block allocation iterator offset
    c.length(); // number of managed objects
    c.length(); // size of huge objects
    c.length(); // number of huge objects
    c.length(); // size of tiny objects
    c.length(); // number of tiny objects
    const width = c.u16();
    const startBlockSize = c.length();
    const maxDirectBlockSize = c.length();
    const maxHeapBits = c.u16();
    c.u16(); // starting number of rows in root indirect block
    const rootAddress = c.offset();
    const rootRows = c.u16();
    return new FractalHeap(
      ctx,
      idLength,
      filterLen > 0,
      maxManagedSize,
      width,
      startBlockSize,
      maxDirectBlockSize,
      maxHeapBits,
      rootAddress,
      rootRows,
    );
  }

  /** Bytes used for heap offsets in IDs and block headers. */
  private get offsetSize(): number {
    return Math.ceil(this.maxHeapBits / 8);
  }

  /** Bytes used for object lengths in managed heap IDs. */
  private get lengthSize(): number {
    const dirBits = log2Floor(this.maxDirectBlockSize);
    return Math.min(
      Math.ceil(dirBits / 8),
      Math.floor(log2Floor(Math.max(1, this.maxManagedSize)) / 8) + 1,
    );
  }

  /** Size of blocks in doubling-table row `r`. */
  private rowBlockSize(r: number): number {
    return r === 0 ? this.startBlockSize : this.startBlockSize * 2 ** (r - 1);
  }

  /** Number of rows holding direct blocks. */
  private get maxDirectRows(): number {
    return (
      log2Floor(this.maxDirectBlockSize) - log2Floor(this.startBlockSize) + 2
    );
  }

  /** Fetch the object referenced by a heap ID. */
  async get(id: Uint8Array): Promise<Uint8Array> {
    const kind = (id[0]! >> 4) & 0x03;
    if (kind === 2) {
      // Tiny object stored inline in the ID.
      const extended = this.idLength > 17;
      const len = extended
        ? (((id[0]! & 0x0f) << 8) | id[1]!) + 1
        : (id[0]! & 0x0f) + 1;
      const start = extended ? 2 : 1;
      return id.subarray(start, start + len);
    }
    if (kind !== 0)
      throw new Error("HDF5: huge fractal heap objects unsupported");
    if (this.filtered)
      throw new Error("HDF5: filtered fractal heaps unsupported");
    const c = this.ctx.wrap(id, 1);
    const offset = c.uN(this.offsetSize);
    const length = c.uN(this.lengthSize);
    const blockAddr = await this.directBlockFor(offset);
    return this.ctx.read(
      blockAddr.address + (offset - blockAddr.blockOffset),
      length,
    );
  }

  /** Locate the direct block containing heap-space `offset`. */
  private async directBlockFor(
    offset: number,
  ): Promise<{ address: number; blockOffset: number }> {
    if (this.rootAddress === null) throw new Error("HDF5: empty fractal heap");
    if (this.rootRows === 0) {
      return { address: this.rootAddress, blockOffset: 0 };
    }
    let iaddr = this.rootAddress;
    let nrows = this.rootRows;
    let base = 0; // heap offset of the current indirect block
    for (;;) {
      const ib = await this.indirectBlock(iaddr, nrows);
      let rel = offset - base;
      let row = 0;
      while (row < ib.nrows) {
        const span = this.rowBlockSize(row) * this.width;
        if (rel < span) break;
        rel -= span;
        row++;
      }
      if (row >= ib.nrows) {
        throw new Error(`HDF5: heap offset ${offset} outside fractal heap`);
      }
      const bs = this.rowBlockSize(row);
      const col = Math.floor(rel / bs);
      const child = ib.children[row * this.width + col];
      if (child === null || child === undefined) {
        throw new Error(`HDF5: heap offset ${offset} in unallocated block`);
      }
      const blockOffset = offset - (rel - col * bs);
      if (row < this.maxDirectRows) return { address: child, blockOffset };
      // Child indirect block covering `bs` bytes of heap space.
      iaddr = child;
      nrows =
        log2Floor(bs) -
        (log2Floor(this.startBlockSize) + log2Floor(this.width)) +
        1;
      base = blockOffset;
    }
  }

  private indirectBlock(addr: number, nrows: number): Promise<IndirectBlock> {
    let p = this.iblocks.get(addr);
    if (!p) {
      p = this.readIndirectBlock(addr, nrows);
      this.iblocks.set(addr, p);
    }
    return p;
  }

  private async readIndirectBlock(
    addr: number,
    nrows: number,
  ): Promise<IndirectBlock> {
    const O = this.ctx.sizeOfOffsets;
    const directRows = Math.min(nrows, this.maxDirectRows);
    const nDirect = directRows * this.width;
    const nIndirect = (nrows - directRows) * this.width;
    const headerSize = 5 + O + this.offsetSize;
    const c = await this.ctx.cursor(
      addr,
      headerSize + (nDirect + nIndirect) * O,
    );
    if (c.ascii(4) !== "FHIB") {
      throw new Error(`HDF5: expected fractal heap indirect block at ${addr}`);
    }
    c.skip(1 + O + this.offsetSize); // version, heap header address, block offset
    const children: Array<number | null> = [];
    for (let i = 0; i < nDirect + nIndirect; i++) children.push(c.offset());
    return { children, nrows };
  }
}
