/** HDF5 file / group / dataset API over an async RangeSource. */

import { float16BitsToFloat32 } from "../util/float16";
import { btree1Leaves, btree2Records } from "./btree";
import { Hdf5Context } from "./context";
import { Cursor, decodeCString, decodeUtf8, product } from "./cursor";
import type { Hdf5Filter } from "./filters";
import { FractalHeap } from "./fractal-heap";
import {
  type AttributeMsg,
  type Datatype,
  type DenseInfo,
  DT,
  type Layout,
  type Link,
  MSG,
  parseAttribute,
  parseAttributeInfo,
  parseDataspace,
  parseDatatype,
  parseFilters,
  parseLayout,
  parseLink,
  parseLinkInfo,
  parseSymbolTable,
  type RawMessage,
  type SymbolTableMsg,
} from "./messages";
import {
  type CachedRangeSourceOptions,
  cachedRangeSource,
  type RangeSource,
} from "./source";

export type Hdf5AttrValue = string | number | bigint | number[] | string[];

export interface Hdf5Dtype {
  class: "int" | "uint" | "float" | "string" | "other";
  size: number;
  /** Byte order as stored on disk. */
  littleEndian: boolean;
}

export interface Hdf5ChunkRef {
  /** Chunk origin in element coordinates, dataset order (e.g. [z, y, x]). */
  offset: number[];
  /** Absolute file byte offset. */
  address: number;
  /** Stored (compressed) byte size. */
  size: number;
  filterMask: number;
}

export interface Hdf5OpenOptions {
  /** Pre-built metadata source (default: cachedRangeSource(src, cache)). */
  meta?: RangeSource;
  /** Options for the default metadata page cache. */
  cache?: CachedRangeSourceOptions;
}

const SIGNATURE = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/** Memoized promise that is forgotten on failure (so e.g. network errors can be retried). */
function retryable<T>(p: Promise<T>, forget: () => void): Promise<T> {
  p.catch(forget);
  return p;
}

function childPath(parent: string, name: string): string {
  return parent === "/" ? `/${name}` : `${parent}/${name}`;
}

export class Hdf5File {
  private nodes = new Map<string, Promise<Hdf5Group | Hdf5Dataset | null>>();

  private constructor(
    /** @internal */ readonly ctx: Hdf5Context,
    private readonly rootAddress: number,
  ) {}

  static async open(
    src: RangeSource,
    opts: Hdf5OpenOptions = {},
  ): Promise<Hdf5File> {
    const meta = opts.meta ?? cachedRangeSource(src, opts.cache);
    // The superblock is at 0 or a power of two >= 512 (after a user block).
    let sbPos = -1;
    for (let pos = 0; pos <= 1 << 24; pos = pos === 0 ? 512 : pos * 2) {
      const b = await meta.read(pos, 8);
      if (b.length < 8) break;
      if (SIGNATURE.every((v, i) => b[i] === v)) {
        sbPos = pos;
        break;
      }
    }
    if (sbPos < 0) throw new Error("HDF5: superblock signature not found");
    const buf = await meta.read(sbPos, 256);
    const c = new Cursor(buf, 8);
    const version = c.u8();
    let O: number;
    let L: number;
    const k = { groupLeaf: 4, groupInternal: 16, chunk: 32 };
    let base: number;
    let root: number | null;
    if (version === 0 || version === 1) {
      // Superblock v0/v1 (spec II.A): versions, sizes, B-tree Ks, addresses,
      // then the root group symbol table entry.
      c.skip(4); // free-space, root symtab, reserved, shared header versions
      O = c.u8();
      L = c.u8();
      c.skip(1);
      k.groupLeaf = c.u16();
      k.groupInternal = c.u16();
      c.u32(); // file consistency flags
      if (version === 1) {
        k.chunk = c.u16(); // indexed storage internal node K
        c.skip(2);
      }
      const sc = new Cursor(buf, c.pos, O, L);
      base = sc.offset() ?? 0;
      sc.offset(); // free-space info
      sc.offset(); // end of file
      sc.offset(); // driver info
      sc.offset(); // root entry: link name offset
      root = sc.offset(); // root entry: object header address
    } else if (version === 2 || version === 3) {
      O = c.u8();
      L = c.u8();
      c.u8(); // flags
      const sc = new Cursor(buf, c.pos, O, L);
      base = sc.offset() ?? 0;
      sc.offset(); // superblock extension
      sc.offset(); // end of file
      root = sc.offset();
    } else {
      throw new Error(`HDF5: unsupported superblock version ${version}`);
    }
    if (root === null) throw new Error("HDF5: root group address undefined");
    return new Hdf5File(new Hdf5Context(meta, base, O, L, k), root);
  }

  async root(): Promise<Hdf5Group> {
    const node = await this.openAt(this.rootAddress, "/");
    if (!(node instanceof Hdf5Group))
      throw new Error("HDF5: root is not a group");
    return node;
  }

  /** Resolve an absolute path, e.g. "/DataSet/ResolutionLevel 0". */
  async get(path: string): Promise<Hdf5Group | Hdf5Dataset | null> {
    const parts = path.split("/").filter((p) => p.length > 0 && p !== ".");
    let node: Hdf5Group | Hdf5Dataset | null = await this.root();
    for (const part of parts) {
      if (!(node instanceof Hdf5Group)) return null;
      node = await node.get(part);
    }
    return node;
  }

  /** @internal Open (and cache by path) the object whose header is at `addr`. */
  openAt(addr: number, path: string): Promise<Hdf5Group | Hdf5Dataset | null> {
    let p = this.nodes.get(path);
    if (!p) {
      p = this.ctx.objectHeader(addr).then((msgs) => {
        const isDataset = msgs.some((m) => m.type === MSG.LAYOUT);
        return isDataset
          ? new Hdf5Dataset(this, path, msgs)
          : new Hdf5Group(this, path, msgs);
      });
      this.nodes.set(path, p);
      p.catch(() => this.nodes.delete(path));
    }
    return p;
  }
}

abstract class Hdf5Object {
  private attrsP?: Promise<Record<string, Hdf5AttrValue>>;

  constructor(
    protected readonly file: Hdf5File,
    readonly path: string,
    protected readonly messages: RawMessage[],
  ) {}

  protected get ctx(): Hdf5Context {
    return this.file.ctx;
  }

  protected message(type: number): RawMessage | undefined {
    return this.messages.find((m) => m.type === type);
  }

  /** Attributes (compact and dense storage). */
  attrs(): Promise<Record<string, Hdf5AttrValue>> {
    this.attrsP ??= retryable(this.readAttrs(), () => {
      this.attrsP = undefined;
    });
    return this.attrsP;
  }

  private async readAttrs(): Promise<Record<string, Hdf5AttrValue>> {
    const ctx = this.ctx;
    const msgs: AttributeMsg[] = this.messages
      .filter((m) => m.type === MSG.ATTRIBUTE && !(m.flags & 0x02))
      .map((m) => parseAttribute(ctx.wrap(m.data)));
    const info = this.message(MSG.ATTRIBUTE_INFO);
    if (info) {
      const dense = parseAttributeInfo(ctx.wrap(info.data));
      // Attribute name index records (v2 B-tree type 8): heap ID, flags,
      // creation order, name hash.
      const objs = await denseObjects(ctx, dense, (rec, idLen) =>
        rec[idLen]! & 0x01 ? null : rec.subarray(0, idLen),
      );
      for (const o of objs) msgs.push(parseAttribute(ctx.wrap(o)));
    }
    const values = await Promise.all(msgs.map((a) => attributeValue(ctx, a)));
    const out: Record<string, Hdf5AttrValue> = {};
    msgs.forEach((a, i) => {
      out[a.name] = values[i]!;
    });
    return out;
  }
}

export class Hdf5Group extends Hdf5Object {
  private linksP?: Promise<Map<string, Link>>;

  /** Child names, sorted (like h5py). */
  async keys(): Promise<string[]> {
    const links = await this.links();
    return [...links.keys()].sort();
  }

  /** Child by name (a relative path with "/" is also accepted). */
  async get(name: string): Promise<Hdf5Group | Hdf5Dataset | null> {
    if (name.includes("/")) {
      return name.startsWith("/")
        ? this.file.get(name)
        : this.file.get(childPath(this.path, name));
    }
    const link = (await this.links()).get(name);
    if (!link) return null;
    if (link.linkType === 0 && link.address !== null) {
      return this.file.openAt(link.address, childPath(this.path, name));
    }
    if (link.linkType === 1 && link.target) {
      const t = link.target;
      return this.file.get(t.startsWith("/") ? t : childPath(this.path, t));
    }
    return null; // external / dangling links
  }

  private links(): Promise<Map<string, Link>> {
    this.linksP ??= retryable(this.readLinks(), () => {
      this.linksP = undefined;
    });
    return this.linksP;
  }

  private async readLinks(): Promise<Map<string, Link>> {
    const ctx = this.ctx;
    const links: Link[] = this.messages
      .filter((m) => m.type === MSG.LINK)
      .map((m) => parseLink(ctx.wrap(m.data)));
    const info = this.message(MSG.LINK_INFO);
    if (info) {
      const dense = parseLinkInfo(ctx.wrap(info.data));
      // Link name index records (v2 B-tree type 5): hash (4) + heap ID.
      const objs = await denseObjects(ctx, dense, (rec) => rec.subarray(4));
      for (const o of objs) links.push(parseLink(ctx.wrap(o)));
    }
    const st = this.message(MSG.SYMBOL_TABLE);
    if (st) {
      links.push(
        ...(await symbolTableLinks(ctx, parseSymbolTable(ctx.wrap(st.data)))),
      );
    }
    return new Map(links.map((l) => [l.name, l]));
  }
}

export class Hdf5Dataset extends Hdf5Object {
  readonly shape: number[];
  readonly dtype: Hdf5Dtype;
  /** Chunk dims in elements (no trailing element-size dim); null if not chunked. */
  readonly chunkShape: number[] | null;
  readonly filters: Hdf5Filter[];
  private readonly maxShape: number[];
  private readonly datatype: Datatype;
  private readonly layout: Layout;

  /** @internal */
  constructor(file: Hdf5File, path: string, messages: RawMessage[]) {
    super(file, path, messages);
    const ctx = file.ctx;
    const need = (type: number, what: string) => {
      const m = this.message(type);
      if (!m) throw new Error(`HDF5: dataset ${path} has no ${what} message`);
      if (m.flags & 0x02) {
        throw new Error(`HDF5: shared ${what} message in ${path} unsupported`);
      }
      return ctx.wrap(m.data);
    };
    const space = parseDataspace(need(MSG.DATASPACE, "dataspace"));
    this.shape = space.dims;
    this.maxShape = space.maxDims;
    this.datatype = parseDatatype(need(MSG.DATATYPE, "datatype"));
    this.dtype = publicDtype(this.datatype);
    this.layout = parseLayout(need(MSG.LAYOUT, "layout"));
    const f = this.message(MSG.FILTERS);
    this.filters = f ? parseFilters(ctx.wrap(f.data)) : [];
    if (this.layout.kind === "chunked") {
      const dims = this.layout.dims;
      this.chunkShape =
        dims.length === this.shape.length + 1
          ? dims.slice(0, -1)
          : dims.slice();
    } else {
      this.chunkShape = null;
    }
  }

  /** Bytes in one decoded chunk. */
  private get chunkBytes(): number {
    return product(this.chunkShape ?? []) * this.dtype.size;
  }

  /**
   * All allocated chunks. Not memoised: callers keep their own (compact)
   * index, and the dataset stays cached by its file.
   */
  async chunks(): Promise<Hdf5ChunkRef[]> {
    const layout = this.layout;
    if (layout.kind !== "chunked" || !this.chunkShape) return [];
    const ctx = this.ctx;
    const rank = this.shape.length;
    const idx = layout.index;
    if (idx.type === "unsupported") {
      throw new Error(
        `HDF5: unsupported chunk index (${idx.name}) in ${this.path}`,
      );
    }
    if (idx.address === null) return [];
    switch (idx.type) {
      case "btree1": {
        // Chunk B-tree (node type 1) keys: size, filter mask, offsets[ndims].
        const ndims = layout.dims.length;
        const leaves = await btree1Leaves(
          ctx,
          idx.address,
          8 + 8 * ndims,
          ctx.k.chunk,
        );
        return leaves.map((e) => {
          const k = ctx.wrap(e.key);
          const size = k.u32();
          const filterMask = k.u32();
          const offset: number[] = [];
          for (let i = 0; i < rank; i++) offset.push(k.u64());
          return { offset, address: ctx.base + e.child, size, filterMask };
        });
      }
      case "single":
        return [
          {
            offset: new Array(rank).fill(0),
            address: ctx.base + idx.address,
            size: idx.filteredSize ?? this.chunkBytes,
            filterMask: idx.filterMask,
          },
        ];
      case "implicit": {
        const grid = this.indexGrid();
        const n = product(grid);
        const refs: Hdf5ChunkRef[] = [];
        for (let i = 0; i < n; i++) {
          const offset = this.gridOffset(i, grid);
          if (offset) {
            refs.push({
              offset,
              address: ctx.base + idx.address + i * this.chunkBytes,
              size: this.chunkBytes,
              filterMask: 0,
            });
          }
        }
        return refs;
      }
      case "farray":
        return this.fixedArrayChunks(idx.address);
    }
  }

  /** Chunk grid used to linearize implicit / fixed-array indexes (max dims). */
  private indexGrid(): number[] {
    const cs = this.chunkShape!;
    return this.shape.map((d, i) => {
      const m = this.maxShape[i];
      const ext = m !== undefined && Number.isFinite(m) && m >= d ? m : d;
      return Math.ceil(ext / cs[i]!);
    });
  }

  /** Element offset of linear chunk index `i`, or null if outside the shape. */
  private gridOffset(i: number, grid: number[]): number[] | null {
    const cs = this.chunkShape!;
    const off = new Array<number>(grid.length);
    for (let d = grid.length - 1; d >= 0; d--) {
      const g = grid[d]!;
      off[d] = (i % g) * cs[d]!;
      i = Math.floor(i / g);
    }
    return off.every((o, d) => o < this.shape[d]!) ? off : null;
  }

  /** Fixed array chunk index ("FAHD" header + "FADB" data block, optionally paged). */
  private async fixedArrayChunks(addr: number): Promise<Hdf5ChunkRef[]> {
    const ctx = this.ctx;
    const O = ctx.sizeOfOffsets;
    const h = await ctx.cursor(addr, 4 + 1 + 1 + 1 + 1 + ctx.sizeOfLengths + O);
    if (h.ascii(4) !== "FAHD")
      throw new Error(`HDF5: expected FAHD at ${addr}`);
    h.u8(); // version
    const clientId = h.u8(); // 0 = plain chunks, 1 = filtered chunks
    const elemSize = h.u8();
    const pageBits = h.u8();
    const n = h.length();
    const dblock = h.offset();
    if (dblock === null || n === 0) return [];
    const pageElems = 2 ** pageBits;
    const paged = n > pageElems;
    const npages = Math.ceil(n / pageElems);
    const bitmapBytes = paged ? Math.ceil(npages / 8) : 0;
    const prefix = 4 + 1 + 1 + O + bitmapBytes; // before elements / checksum
    const head = await ctx.cursor(dblock, prefix);
    if (head.ascii(4) !== "FADB")
      throw new Error(`HDF5: expected FADB at ${dblock}`);
    head.skip(2 + O);
    const bitmap = head.bytes(bitmapBytes);

    const parse = (
      c: Cursor,
      count: number,
      first: number,
      out: Hdf5ChunkRef[],
    ) => {
      const grid = this.indexGrid();
      for (let j = 0; j < count; j++) {
        const start = c.pos;
        const address = c.offset();
        let size = this.chunkBytes;
        let filterMask = 0;
        if (clientId === 1) {
          size = c.uN(elemSize - O - 4);
          filterMask = c.u32();
        }
        c.pos = start + elemSize;
        if (address === null) continue;
        const offset = this.gridOffset(first + j, grid);
        if (offset)
          out.push({ offset, address: ctx.base + address, size, filterMask });
      }
    };

    const out: Hdf5ChunkRef[] = [];
    if (!paged) {
      parse(await ctx.cursor(dblock + prefix, n * elemSize), n, 0, out);
      return out;
    }
    const pageStride = pageElems * elemSize + 4; // elements + checksum
    const pagesStart = dblock + prefix + 4; // after the data block checksum
    const pages = await Promise.all(
      Array.from({ length: npages }, async (_, p) => {
        if (!(bitmap[p >> 3]! & (0x80 >> (p & 7)))) return [];
        const count = Math.min(pageElems, n - p * pageElems);
        const refs: Hdf5ChunkRef[] = [];
        const c = await ctx.cursor(
          pagesStart + p * pageStride,
          count * elemSize,
        );
        parse(c, count, p * pageElems, refs);
        return refs;
      }),
    );
    return pages.flat();
  }
}

function publicDtype(dt: Datatype): Hdf5Dtype {
  const base = dt.cls === DT.ENUM && dt.base ? dt.base : dt;
  let cls: Hdf5Dtype["class"] = "other";
  if (base.cls === DT.FIXED) cls = base.signed ? "int" : "uint";
  else if (base.cls === DT.FLOAT) cls = "float";
  else if (
    base.cls === DT.STRING ||
    (base.cls === DT.VLEN && base.vlenString)
  ) {
    cls = "string";
  }
  return { class: cls, size: dt.size, littleEndian: dt.littleEndian };
}

// ------------------------------------------------------------ dense storage

/**
 * Objects of a dense link/attribute store: walk the name-index v2 B-tree and
 * fetch each record's heap ID from the fractal heap.
 */
async function denseObjects(
  ctx: Hdf5Context,
  info: DenseInfo,
  heapIdOf: (record: Uint8Array, idLength: number) => Uint8Array | null,
): Promise<Uint8Array[]> {
  if (info.heapAddress === null || info.nameIndexAddress === null) return [];
  const [heap, records] = await Promise.all([
    FractalHeap.open(ctx, info.heapAddress),
    btree2Records(ctx, info.nameIndexAddress),
  ]);
  const ids = records
    .map((r) => heapIdOf(r, heap.idLength))
    .filter((id): id is Uint8Array => id !== null);
  return Promise.all(ids.map((id) => heap.get(id)));
}

// ------------------------------------------------------- old-style groups

/** Symbol-table group: v1 B-tree (type 0) of "SNOD" nodes + local "HEAP". */
async function symbolTableLinks(
  ctx: Hdf5Context,
  st: SymbolTableMsg,
): Promise<Link[]> {
  const O = ctx.sizeOfOffsets;
  const L = ctx.sizeOfLengths;
  const heapData = (async () => {
    const h = await ctx.cursor(st.heapAddress, 8 + 2 * L + O);
    if (h.ascii(4) !== "HEAP") throw new Error("HDF5: expected local heap");
    h.skip(4);
    const size = h.length();
    h.length(); // free list offset
    const dataAddr = h.offset();
    return dataAddr === null ? new Uint8Array(0) : ctx.read(dataAddr, size);
  })();
  const leaves = await btree1Leaves(
    ctx,
    st.btreeAddress,
    L,
    ctx.k.groupInternal,
  );
  const entrySize = 2 * O + 24;
  const nodes = await Promise.all(
    leaves.map(async (e) => {
      // SNOD: signature, version, reserved, count, then up to 2K entries.
      const guess = 8 + 2 * ctx.k.groupLeaf * entrySize;
      const head = await ctx.readAtLeast(e.child, 8, guess);
      const h = ctx.wrap(head);
      if (h.ascii(4) !== "SNOD") {
        throw new Error("HDF5: expected symbol table node");
      }
      h.skip(2);
      const n = h.u16();
      return 8 + n * entrySize <= head.length
        ? ctx.wrap(head.subarray(8, 8 + n * entrySize))
        : ctx.cursor(e.child + 8, n * entrySize);
    }),
  );
  const heap = await heapData;
  const links: Link[] = [];
  for (const c of nodes) {
    while (c.remaining >= entrySize) {
      const nameOff = c.length();
      const address = c.offset();
      const cacheType = c.u32();
      c.skip(4);
      const scratch = c.bytes(16);
      const name = decodeCString(heap.subarray(nameOff));
      if (cacheType === 2) {
        const off = new DataView(
          scratch.buffer,
          scratch.byteOffset,
          4,
        ).getUint32(0, true);
        links.push({
          name,
          linkType: 1,
          address: null,
          target: decodeCString(heap.subarray(off)),
        });
      } else {
        links.push({ name, linkType: 0, address });
      }
    }
  }
  return links;
}

// ---------------------------------------------------------- attribute values

const globalHeaps = new WeakMap<
  Hdf5Context,
  Map<number, Promise<Map<number, Uint8Array>>>
>();

/** Global heap collection ("GCOL"), used by variable-length data. */
function globalHeap(
  ctx: Hdf5Context,
  addr: number,
): Promise<Map<number, Uint8Array>> {
  let cache = globalHeaps.get(ctx);
  if (!cache) {
    cache = new Map();
    globalHeaps.set(ctx, cache);
  }
  let p = cache.get(addr);
  if (!p) {
    p = (async () => {
      const L = ctx.sizeOfLengths;
      const h = await ctx.cursor(addr, 8 + L);
      if (h.ascii(4) !== "GCOL") throw new Error("HDF5: expected global heap");
      h.skip(4);
      const size = h.length();
      const c = await ctx.cursor(addr, size);
      c.pos = 8 + L;
      const objs = new Map<number, Uint8Array>();
      while (c.remaining >= 8 + L) {
        const index = c.u16();
        c.skip(6); // reference count, reserved
        const osize = c.length();
        if (index === 0) break; // free space
        objs.set(index, c.bytes(osize));
        c.align(8);
      }
      return objs;
    })();
    cache.set(addr, p);
  }
  return p;
}

function readNumber(
  view: DataView,
  off: number,
  dt: Datatype,
): number | bigint {
  const le = dt.littleEndian;
  if (dt.cls === DT.FLOAT) {
    if (dt.size === 4) return view.getFloat32(off, le);
    if (dt.size === 8) return view.getFloat64(off, le);
    if (dt.size === 2) return float16BitsToFloat32(view.getUint16(off, le));
    return Number.NaN;
  }
  switch (dt.size) {
    case 1:
      return dt.signed ? view.getInt8(off) : view.getUint8(off);
    case 2:
      return dt.signed ? view.getInt16(off, le) : view.getUint16(off, le);
    case 4:
      return dt.signed ? view.getInt32(off, le) : view.getUint32(off, le);
    case 8: {
      const v = dt.signed
        ? view.getBigInt64(off, le)
        : view.getBigUint64(off, le);
      const n = Number(v);
      return Number.isSafeInteger(n) ? n : v;
    }
    default:
      return Number.NaN;
  }
}

function stripString(b: Uint8Array, pad: number): string {
  let s = decodeCString(b);
  if (pad === 2) s = s.replace(/ +$/, "");
  return s;
}

async function attributeValue(
  ctx: Hdf5Context,
  a: AttributeMsg,
): Promise<Hdf5AttrValue> {
  if (a.shared) return "";
  const dt = a.datatype;
  const n = a.dataspace.isNull ? 0 : product(a.dataspace.dims);
  if (n === 0 || a.data.length < n * dt.size) return "";
  if (dt.cls === DT.STRING) {
    const strs: string[] = [];
    for (let i = 0; i < n; i++) {
      strs.push(
        stripString(a.data.subarray(i * dt.size, (i + 1) * dt.size), dt.strPad),
      );
    }
    // Imaris stores text as arrays of 1-byte strings: join them.
    return dt.size === 1 || n === 1 ? strs.join("") : strs;
  }
  if (dt.cls === DT.VLEN) {
    if (!dt.vlenString) return "";
    try {
      const strs = await Promise.all(
        Array.from({ length: n }, async (_, i) => {
          const c = ctx.wrap(a.data, i * dt.size);
          const len = c.u32();
          const coll = c.offset();
          const index = c.u32();
          if (coll === null || len === 0) return "";
          const obj = (await globalHeap(ctx, coll)).get(index);
          return obj
            ? decodeUtf8(obj.subarray(0, len)).replace(/\0+$/, "")
            : "";
        }),
      );
      return n === 1 ? strs[0]! : strs;
    } catch {
      return "";
    }
  }
  let base = dt;
  let count = n;
  if (dt.cls === DT.ARRAY && dt.base) {
    base = dt.base;
    count = n * product(dt.arrayDims ?? []);
  }
  if (base.cls === DT.ENUM && base.base) base = base.base;
  if (base.cls !== DT.FIXED && base.cls !== DT.FLOAT) return "";
  const view = new DataView(
    a.data.buffer,
    a.data.byteOffset,
    a.data.byteLength,
  );
  const vals: Array<number | bigint> = [];
  for (let i = 0; i < count; i++)
    vals.push(readNumber(view, i * base.size, base));
  // Scalars and single-element arrays → number (bigint only beyond 2^53).
  if (count === 1 && dt.cls !== DT.ARRAY) return vals[0]!;
  return vals.map((v) => Number(v));
}
