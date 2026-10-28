/** Object header messages (HDF5 spec section IV.A.2). */

import { type Cursor, decodeUtf8 } from "./cursor";
import { FILTER_NAMES, type Hdf5Filter } from "./filters";

export const MSG = {
  NIL: 0x0000,
  DATASPACE: 0x0001,
  LINK_INFO: 0x0002,
  DATATYPE: 0x0003,
  FILL_OLD: 0x0004,
  FILL: 0x0005,
  LINK: 0x0006,
  EXTERNAL_FILES: 0x0007,
  LAYOUT: 0x0008,
  GROUP_INFO: 0x000a,
  FILTERS: 0x000b,
  ATTRIBUTE: 0x000c,
  CONTINUATION: 0x0010,
  SYMBOL_TABLE: 0x0011,
  ATTRIBUTE_INFO: 0x0015,
} as const;

export interface RawMessage {
  type: number;
  flags: number;
  data: Uint8Array;
}

// ---------------------------------------------------------------- Dataspace

export interface Dataspace {
  /** Dimension sizes; [] for scalar. */
  dims: number[];
  /** Max dimension sizes (Infinity = unlimited); [] when not stored. */
  maxDims: number[];
  /** Null dataspace (no elements). */
  isNull: boolean;
}

/** Dataspace message (0x0001), versions 1 and 2. */
export function parseDataspace(c: Cursor): Dataspace {
  const version = c.u8();
  const rank = c.u8();
  const flags = c.u8();
  let isNull = false;
  if (version === 1) {
    c.skip(5);
  } else if (version === 2) {
    isNull = c.u8() === 2;
  } else {
    throw new Error(`HDF5: unsupported dataspace version ${version}`);
  }
  const dims: number[] = [];
  for (let i = 0; i < rank; i++) dims.push(c.length());
  const maxDims: number[] = [];
  if (flags & 0x01) {
    for (let i = 0; i < rank; i++) {
      maxDims.push(c.lengthOrUndefined() ?? Number.POSITIVE_INFINITY);
    }
  }
  // Permutation indices (v1, flags bit 1) are never used by the library.
  return { dims, maxDims, isNull };
}

export function dataspaceSize(ds: Dataspace): number {
  if (ds.isNull) return 0;
  let n = 1;
  for (const d of ds.dims) n *= d;
  return n;
}

// ----------------------------------------------------------------- Datatype

export const DT = {
  FIXED: 0,
  FLOAT: 1,
  TIME: 2,
  STRING: 3,
  BITFIELD: 4,
  OPAQUE: 5,
  COMPOUND: 6,
  REFERENCE: 7,
  ENUM: 8,
  VLEN: 9,
  ARRAY: 10,
} as const;

export interface Datatype {
  cls: number;
  size: number;
  littleEndian: boolean;
  signed: boolean;
  /** String padding: 0 null-terminated, 1 null-padded, 2 space-padded. */
  strPad: number;
  /** Variable-length: true for vlen strings, false for vlen sequences. */
  vlenString?: boolean;
  /** Base type for array / vlen / enum. */
  base?: Datatype;
  /** Array dimensions. */
  arrayDims?: number[];
}

/** Datatype message (0x0003). Compound/reference/etc. are kept as opaque. */
export function parseDatatype(c: Cursor): Datatype {
  const classAndVersion = c.u8();
  const cls = classAndVersion & 0x0f;
  const version = classAndVersion >> 4;
  const bits = c.u8() | (c.u8() << 8) | (c.u8() << 16);
  const size = c.u32();
  const dt: Datatype = {
    cls,
    size,
    littleEndian: (bits & 1) === 0,
    signed: false,
    strPad: 0,
  };
  switch (cls) {
    case DT.FIXED:
      dt.signed = (bits & 0x08) !== 0;
      c.skip(4); // bit offset, bit precision
      break;
    case DT.FLOAT:
      dt.signed = true;
      c.skip(12);
      break;
    case DT.STRING:
      dt.strPad = bits & 0x0f;
      dt.littleEndian = true;
      break;
    case DT.ENUM:
      // Base type first; member names/values follow (not needed).
      dt.base = parseDatatype(c);
      dt.littleEndian = dt.base.littleEndian;
      dt.signed = dt.base.signed;
      break;
    case DT.VLEN:
      dt.vlenString = (bits & 0x0f) === 1;
      dt.strPad = (bits >> 4) & 0x0f;
      dt.base = parseDatatype(c);
      break;
    case DT.ARRAY: {
      const ndims = c.u8();
      if (version < 3) c.skip(3);
      const dims: number[] = [];
      for (let i = 0; i < ndims; i++) dims.push(c.u32());
      if (version < 3) c.skip(4 * ndims); // permutation indices
      dt.arrayDims = dims;
      dt.base = parseDatatype(c);
      dt.littleEndian = dt.base.littleEndian;
      dt.signed = dt.base.signed;
      break;
    }
    default:
      // time, bitfield, opaque, compound, reference: properties not parsed.
      break;
  }
  return dt;
}

// ------------------------------------------------------------------- Layout

export type ChunkIndex =
  | { type: "btree1"; address: number | null }
  | {
      type: "single";
      address: number | null;
      filteredSize: number | null;
      filterMask: number;
    }
  | { type: "implicit"; address: number | null }
  | { type: "farray"; address: number | null }
  | { type: "unsupported"; name: string };

export type Layout =
  | { kind: "compact"; data: Uint8Array }
  | { kind: "contiguous"; address: number | null; size: number }
  | {
      kind: "chunked";
      /** Chunk dims as stored (includes the trailing element-size dim for v1-3). */
      dims: number[];
      index: ChunkIndex;
    }
  | { kind: "virtual" };

/** Data layout message (0x0008), versions 1-4. */
export function parseLayout(c: Cursor): Layout {
  const version = c.u8();
  if (version === 1 || version === 2) {
    const ndims = c.u8();
    const cls = c.u8();
    c.skip(5);
    const address = cls !== 0 ? c.offset() : null;
    const dims: number[] = [];
    for (let i = 0; i < ndims; i++) dims.push(c.u32());
    if (cls === 0) {
      const size = c.u32();
      return { kind: "compact", data: c.bytes(size) };
    }
    if (cls === 1) {
      let size = 1;
      for (const d of dims) size *= d;
      return { kind: "contiguous", address, size };
    }
    return { kind: "chunked", dims, index: { type: "btree1", address } };
  }
  if (version < 3 || version > 5) {
    throw new Error(`HDF5: unsupported layout message version ${version}`);
  }
  const cls = c.u8();
  if (cls === 0) {
    const size = c.u16();
    return { kind: "compact", data: c.bytes(size) };
  }
  if (cls === 1) {
    return { kind: "contiguous", address: c.offset(), size: c.length() };
  }
  if (cls === 3) return { kind: "virtual" };
  if (cls !== 2) throw new Error(`HDF5: unknown layout class ${cls}`);
  if (version === 3) {
    const ndims = c.u8();
    const address = c.offset();
    const dims: number[] = [];
    for (let i = 0; i < ndims; i++) dims.push(c.u32());
    return { kind: "chunked", dims, index: { type: "btree1", address } };
  }
  // Version 4 (and 5, HDF5 2.x) chunked layout.
  const flags = c.u8();
  const ndims = c.u8();
  const encLen = c.u8();
  const dims: number[] = [];
  for (let i = 0; i < ndims; i++) dims.push(c.uN(encLen));
  const indexType = c.u8();
  let index: ChunkIndex;
  switch (indexType) {
    case 1: {
      // Single chunk; filtered size + mask present when flags bit 1 set.
      let filteredSize: number | null = null;
      let filterMask = 0;
      if (flags & 0x02) {
        filteredSize = c.length();
        filterMask = c.u32();
      }
      index = { type: "single", address: c.offset(), filteredSize, filterMask };
      break;
    }
    case 2:
      index = { type: "implicit", address: c.offset() };
      break;
    case 3:
      c.skip(1); // page bits (also stored in the fixed array header)
      index = { type: "farray", address: c.offset() };
      break;
    case 4:
      index = { type: "unsupported", name: "extensible array" };
      break;
    case 5:
      index = { type: "unsupported", name: "version 2 B-tree" };
      break;
    default:
      index = { type: "unsupported", name: `type ${indexType}` };
  }
  return { kind: "chunked", dims, index };
}

// ------------------------------------------------------------ Filter pipeline

/** Filter pipeline message (0x000B), versions 1 and 2. */
export function parseFilters(c: Cursor): Hdf5Filter[] {
  const version = c.u8();
  const n = c.u8();
  if (version === 1) c.skip(6);
  const out: Hdf5Filter[] = [];
  for (let i = 0; i < n; i++) {
    const id = c.u16();
    const nameLen = version === 1 || id >= 256 ? c.u16() : 0;
    const flags = c.u16();
    const nValues = c.u16();
    let name = "";
    if (nameLen > 0) {
      const raw = c.bytes(nameLen);
      const z = raw.indexOf(0);
      name = decodeUtf8(z >= 0 ? raw.subarray(0, z) : raw);
      if (version === 1) c.skip((8 - (nameLen % 8)) % 8);
    }
    const clientData: number[] = [];
    for (let k = 0; k < nValues; k++) clientData.push(c.u32());
    if (version === 1 && nValues % 2 === 1) c.skip(4);
    out.push({
      id,
      name: name || FILTER_NAMES[id] || `filter ${id}`,
      clientData,
      optional: (flags & 1) !== 0,
    });
  }
  return out;
}

// ---------------------------------------------------------------- Attribute

export interface AttributeMsg {
  name: string;
  datatype: Datatype;
  dataspace: Dataspace;
  data: Uint8Array;
  /** Datatype/dataspace stored as shared messages (not resolved). */
  shared: boolean;
}

/** Attribute message (0x000C), versions 1-3. */
export function parseAttribute(c: Cursor): AttributeMsg {
  const version = c.u8();
  const flags = c.u8(); // reserved in v1
  const nameSize = c.u16();
  const dtSize = c.u16();
  const dsSize = c.u16();
  if (version === 3) c.skip(1); // name character set
  const pad = (n: number) => (version === 1 ? Math.ceil(n / 8) * 8 : n);
  const nameBytes = c.bytes(nameSize);
  c.skip(pad(nameSize) - nameSize);
  const z = nameBytes.indexOf(0);
  const name = decodeUtf8(z >= 0 ? nameBytes.subarray(0, z) : nameBytes);
  const shared = version > 1 && (flags & 0x03) !== 0;
  const dtStart = c.pos;
  const datatype = parseDatatype(c.sub(dtStart, dtSize));
  c.pos = dtStart + pad(dtSize);
  const dsStart = c.pos;
  const dataspace = shared
    ? { dims: [], maxDims: [], isNull: true }
    : parseDataspace(c.sub(dsStart, dsSize));
  c.pos = dsStart + pad(dsSize);
  const n = dataspaceSize(dataspace);
  const data = c.bytes(Math.min(c.remaining, n * datatype.size));
  return { name, datatype, dataspace, data, shared };
}

// --------------------------------------------------------------------- Links

export interface Link {
  name: string;
  /** 0 hard, 1 soft, 64 external. */
  linkType: number;
  address: number | null;
  /** Soft link target path. */
  target?: string;
}

/** Link message (0x0006). Also the encoding of dense-storage heap objects. */
export function parseLink(c: Cursor): Link {
  const version = c.u8();
  if (version !== 1)
    throw new Error(`HDF5: unsupported link version ${version}`);
  const flags = c.u8();
  const linkType = flags & 0x08 ? c.u8() : 0;
  if (flags & 0x04) c.skip(8); // creation order
  if (flags & 0x10) c.skip(1); // charset
  const nameLen = c.uN(1 << (flags & 0x03));
  const name = decodeUtf8(c.bytes(nameLen));
  if (linkType === 0) return { name, linkType, address: c.offset() };
  if (linkType === 1) {
    const len = c.u16();
    return { name, linkType, address: null, target: decodeUtf8(c.bytes(len)) };
  }
  return { name, linkType, address: null };
}

/** Link info (0x0002) / attribute info (0x0015): dense storage addresses. */
export interface DenseInfo {
  heapAddress: number | null;
  nameIndexAddress: number | null;
}

export function parseLinkInfo(c: Cursor): DenseInfo {
  c.u8(); // version
  const flags = c.u8();
  if (flags & 0x01) c.skip(8); // max creation index
  return { heapAddress: c.offset(), nameIndexAddress: c.offset() };
}

export function parseAttributeInfo(c: Cursor): DenseInfo {
  c.u8(); // version
  const flags = c.u8();
  if (flags & 0x01) c.skip(2); // max creation index
  return { heapAddress: c.offset(), nameIndexAddress: c.offset() };
}

/** Symbol table message (0x0011): old-style group. */
export interface SymbolTableMsg {
  btreeAddress: number;
  heapAddress: number;
}

export function parseSymbolTable(c: Cursor): SymbolTableMsg {
  const btreeAddress = c.offset();
  const heapAddress = c.offset();
  if (btreeAddress === null || heapAddress === null) {
    throw new Error("HDF5: symbol table message with undefined address");
  }
  return { btreeAddress, heapAddress };
}
