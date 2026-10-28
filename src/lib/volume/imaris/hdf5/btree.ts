/** Version 1 ("TREE") and version 2 ("BTHD"/"BTIN"/"BTLF") B-trees. */

import type { Hdf5Context } from "./context";
import { limitEncSize } from "./cursor";

export interface BTree1Entry {
  /** Raw key preceding the child (key i of child i). */
  key: Uint8Array;
  /** Child address (leaf level: SNOD for groups, chunk for raw data). */
  child: number;
}

/**
 * Walk a version 1 B-tree and return all leaf-level entries. `keySize` is
 * the per-key byte size (type 0: size of lengths; type 1: 8 + 8*ndims).
 * `kHint` is the expected node K (nodes hold up to 2K children); whole nodes
 * of that size are fetched in one read. Sibling subtrees are read in parallel.
 */
export async function btree1Leaves(
  ctx: Hdf5Context,
  addr: number,
  keySize: number,
  kHint = 32,
): Promise<BTree1Entry[]> {
  const O = ctx.sizeOfOffsets;
  const headerSize = 8 + 2 * O;
  const guess = headerSize + 2 * kHint * (keySize + O) + keySize;
  const head = await ctx.readAtLeast(addr, headerSize, guess);
  const h = ctx.wrap(head);
  if (h.ascii(4) !== "TREE") {
    throw new Error(`HDF5: expected v1 B-tree node at ${addr}`);
  }
  h.u8(); // node type
  const level = h.u8();
  const used = h.u16();
  const bodySize = used * (keySize + O) + keySize;
  const c =
    headerSize + bodySize <= head.length
      ? ctx.wrap(head, headerSize)
      : await ctx.cursor(addr + headerSize, bodySize);
  const entries: BTree1Entry[] = [];
  for (let i = 0; i < used; i++) {
    const key = c.bytes(keySize);
    const child = c.offset();
    if (child !== null) entries.push({ key, child });
  }
  if (level === 0) return entries;
  const sub = await Promise.all(
    entries.map((e) => btree1Leaves(ctx, e.child, keySize, kHint)),
  );
  return sub.flat();
}

/** Return all records of a version 2 B-tree (raw record bytes). */
export async function btree2Records(
  ctx: Hdf5Context,
  addr: number,
): Promise<Uint8Array[]> {
  const O = ctx.sizeOfOffsets;
  const h = await ctx.cursor(addr, 16 + O + 2 + ctx.sizeOfLengths);
  if (h.ascii(4) !== "BTHD") {
    throw new Error(`HDF5: expected v2 B-tree header at ${addr}`);
  }
  h.u8(); // version
  h.u8(); // type
  const nodeSize = h.u32();
  const recordSize = h.u16();
  const depth = h.u16();
  h.skip(2); // split / merge percent
  const root = h.offset();
  const rootRecords = h.u16();
  if (root === null || rootRecords === 0) return [];

  // Field widths in internal-node child pointers (H5B2__hdr_init).
  const PREFIX = 10; // signature + version + type + checksum
  const maxNrec: number[] = [Math.floor((nodeSize - PREFIX) / recordSize)];
  const cumMaxNrec: number[] = [maxNrec[0]!];
  const cumMaxNrecSize: number[] = [0];
  const maxNrecSize = limitEncSize(maxNrec[0]!);
  for (let d = 1; d <= depth; d++) {
    const ptr = O + maxNrecSize + (d > 1 ? cumMaxNrecSize[d - 1]! : 0);
    const m = Math.floor((nodeSize - (PREFIX + ptr)) / (recordSize + ptr));
    maxNrec.push(m);
    cumMaxNrec.push((m + 1) * cumMaxNrec[d - 1]! + m);
    cumMaxNrecSize.push(limitEncSize(cumMaxNrec[d]!));
  }

  const walk = async (
    nodeAddr: number,
    nrec: number,
    d: number,
  ): Promise<Uint8Array[]> => {
    if (d === 0) {
      const c = await ctx.cursor(nodeAddr, 6 + nrec * recordSize);
      if (c.ascii(4) !== "BTLF") {
        throw new Error(`HDF5: expected v2 B-tree leaf at ${nodeAddr}`);
      }
      c.skip(2);
      const recs: Uint8Array[] = [];
      for (let i = 0; i < nrec; i++) recs.push(c.bytes(recordSize));
      return recs;
    }
    const totalSize = d > 1 ? cumMaxNrecSize[d - 1]! : 0;
    const ptrSize = O + maxNrecSize + totalSize;
    const c = await ctx.cursor(
      nodeAddr,
      6 + nrec * recordSize + (nrec + 1) * ptrSize,
    );
    if (c.ascii(4) !== "BTIN") {
      throw new Error(`HDF5: expected v2 B-tree internal node at ${nodeAddr}`);
    }
    c.skip(2);
    const recs: Uint8Array[] = [];
    for (let i = 0; i < nrec; i++) recs.push(c.bytes(recordSize));
    const children: Array<Promise<Uint8Array[]>> = [];
    for (let i = 0; i <= nrec; i++) {
      const child = c.offset();
      const n = c.uN(maxNrecSize);
      if (totalSize) c.skip(totalSize);
      if (child !== null && n > 0) children.push(walk(child, n, d - 1));
    }
    const sub = await Promise.all(children);
    return [...recs, ...sub.flat()];
  };
  return walk(root, rootRecords, depth);
}
