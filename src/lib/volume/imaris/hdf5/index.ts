/**
 * Minimal pure-TypeScript, async HDF5 reader (enough for Imaris .ims files):
 * groups, attributes, dataset chunk indexes and chunk decoding over HTTP
 * Range requests or local Blobs.
 */

export { decodeChunk, type Hdf5Filter } from "./filters";
export {
  type Hdf5AttrValue,
  type Hdf5ChunkRef,
  Hdf5Dataset,
  type Hdf5Dtype,
  Hdf5File,
  Hdf5Group,
  type Hdf5OpenOptions,
} from "./reader";
export {
  blobRangeSource,
  type CachedRangeSource,
  type CachedRangeSourceOptions,
  cachedRangeSource,
  httpRangeSource,
  type RangeSource,
} from "./source";
