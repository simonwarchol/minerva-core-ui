/**
 * ImarisDataProvider — streams Imaris 5.5 (.ims) files over async HTTP Range
 * requests (or a local File) with a pure-TS HDF5 reader, as a DataProvider
 * for upstream kiln-render.
 *
 * Lifted from simonwarchol/kiln-render (src/data/imaris-provider.ts), where it
 * lived inside kiln. Differences:
 *
 * - Voxels go through kiln's raw-value path: each channel is normalised to
 *   [0, 1] by its histogram range and stored as float16, with an identity
 *   `dataRange` of [0, 1], so windows are fractions of a channel's range.
 * - Up to 4 channels (kiln's limit) are active at once; switch them in place
 *   with `setSourceSelection`.
 * - Brick ghost borders always read their neighbouring chunks: upstream kiln
 *   has no hook to reload a brick later, which the fork used to edge-clamp
 *   borders whose chunks weren't cached yet.
 * - The virtual isotropic pyramid (ims-layout.ts) is kiln's legacy 2:1
 *   model: open with `pyramid: "legacy"`.
 */

import type {
  BrickLoadResult,
  BrickStats,
  DataProvider,
  NetworkStats,
  SourceSelection,
  VolumeMetadata,
} from "kiln-upstream";
import { UnsupportedDatasetError } from "kiln-upstream";
import { ChunkFetcher } from "./chunk-fetcher";
import { imsBrickIsEmpty } from "./empty-bricks";
import {
  blobRangeSource,
  type CachedRangeSource,
  cachedRangeSource,
  type Hdf5AttrValue,
  Hdf5Dataset,
  Hdf5File,
  Hdf5Group,
  httpRangeSource,
  type RangeSource,
} from "./hdf5/index";
import type {
  ImsAssembleRequest,
  ImsBrickResponse,
  ImsExtractRequest,
  ImsExtractResponse,
  ImsInitRequest,
  ImsPiece,
} from "./ims-brick-worker";
import {
  axisSampling,
  DECODED_CHUNK_BUDGET,
  LOGICAL,
  overlappingChunks,
  PHYSICAL,
  planVirtualPyramid,
  type SourceLevel,
  type Vec3,
  type VirtualLevel,
  WORKER_CHUNK_BUDGET,
} from "./ims-layout";
import { ByteLru } from "./util/byte-lru";
import { NetworkTracker, RollingAvg } from "./util/network-tracker";

/** Kiln renders at most this many channels at once. */
export const MAX_CHANNELS = 4;

const MAX_STATS_ENTRIES = 200_000;
/**
 * Compressed chunk bytes kept in memory. Remote files also persist chunks in
 * OPFS, so revisits are cheap either way; local files re-read from disk.
 */
const REMOTE_CHUNK_BUDGET = 192 * 1024 * 1024;
const LOCAL_CHUNK_BUDGET = 64 * 1024 * 1024;
/** Metadata page cache: generous while chunk indexes load, small after. */
const META_PAGES = 8192; // × 16 KB
const META_PAGES_AFTER_INDEX = 256;

type PipelineTimings = ReturnType<
  NonNullable<DataProvider["getPipelineTimings"]>
>;

/** One channel as described by the file (all channels, not just active ones). */
export interface ImsChannelInfo {
  index: number;
  name: string;
  /** Display colour, linear 0–1 */
  color: [number, number, number];
  /** Normalisation range used for this channel (raw units) */
  range: [number, number];
  /** Imaris display range (raw units) */
  displayRange: [number, number] | null;
}

export interface ImarisProviderOptions {
  /** Source channel indices to load (≤ 4). Default: the first ones. */
  channels?: number[];
  /** Time point to load (default 0). */
  timepoint?: number;
  /**
   * Worker count (default: half the cores, up to 4, for URLs; cores − 1, up
   * to 8, for local files). Workers start with the first brick load.
   */
  workers?: number;
  /** Keep fetched chunks in the Origin Private File System across sessions (default true). */
  persistentCache?: boolean;
}

function createWorker(): Worker {
  return new Worker(new URL("./ims-brick-worker.ts", import.meta.url), {
    type: "module",
  });
}

function attrString(v: Hdf5AttrValue | undefined): string | undefined {
  if (v === undefined) return undefined;
  if (typeof v === "string") return v.replace(/\0+$/, "").trim();
  if (Array.isArray(v)) return v.map(String).join("").trim();
  return String(v);
}

function attrNumber(v: Hdf5AttrValue | undefined): number | undefined {
  const s = attrString(v);
  if (s === undefined || s === "") return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

function attrNumbers(v: Hdf5AttrValue | undefined): number[] | undefined {
  const s = attrString(v);
  if (!s) return undefined;
  const parts = s
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);
  return parts.length > 0 && parts.every(Number.isFinite) ? parts : undefined;
}

/** Chunk → owning worker (each chunk is decoded and cached by one worker). */
function ownerOf(address: number, n: number): number {
  let h = Math.floor(address / 4096) ^ (address % 4096);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) % n;
}

/**
 * Validated active channels, as kiln's own providers resolve a selection:
 * 1–4 distinct indices below `count`; default the first ones.
 */
function resolveChannels(
  requested: readonly number[] | undefined,
  count: number,
): number[] {
  if (requested === undefined) {
    return Array.from({ length: Math.min(MAX_CHANNELS, count) }, (_, i) => i);
  }
  if (
    requested.length < 1 ||
    requested.length > MAX_CHANNELS ||
    new Set(requested).size !== requested.length ||
    requested.some((c) => !Number.isInteger(c) || c < 0 || c >= count)
  ) {
    throw new RangeError(
      `Select 1–${MAX_CHANNELS} distinct channels between 0 and ${count - 1}`,
    );
  }
  return [...requested];
}

/**
 * A channel's display window as a fraction of its normalisation range (the
 * stored voxel values).
 */
function windowFor(ch: ImsChannelInfo) {
  const [lo, hi] = ch.range;
  const span = hi - lo;
  const [a, b] = ch.displayRange ?? ch.range;
  const start = Math.min(1, Math.max(0, (a - lo) / span));
  const end = Math.min(1, Math.max(start + 1e-3, (b - lo) / span));
  return { start, end, min: 0, max: 1 };
}

/**
 * One dataset's chunk index as flat arrays over the chunk grid (linear index
 * (cz * grid[1] + cy) * grid[0] + cx): ~12 bytes per chunk instead of a ref
 * object per chunk.
 */
interface DatasetIndex {
  dataset: Hdf5Dataset;
  /** chunk shape [x, y, z] */
  chunkShape: Vec3;
  /** chunk grid [x, y, z] */
  grid: Vec3;
  /** File address per chunk; -1 when unallocated (reads as zeros) */
  address: Float64Array;
  /** Stored (compressed) bytes per chunk */
  size: Uint32Array;
  /** The rare non-zero filter masks */
  filterMask: Map<number, number>;
}

const chunkIndex = (idx: DatasetIndex, cx: number, cy: number, cz: number) =>
  (cz * idx.grid[1] + cy) * idx.grid[0] + cx;

export class ImarisDataProvider implements DataProvider {
  private readonly tracker = new NetworkTracker();
  private file: Hdf5File | null = null;
  private raw: RangeSource | null = null;
  private meta: CachedRangeSource | null = null;
  private metadata: VolumeMetadata | null = null;
  private pyramid: VirtualLevel[] = [];
  private sourceLevels: SourceLevel[] = [];
  private readonly brickStatsCache = new Map<string, BrickStats>();
  private readonly datasetIndex = new Map<string, Promise<DatasetIndex>>();
  private format: "f16" | "u8" = "f16";
  private workers: Worker[] = [];
  private fetcher: ChunkFetcher | null = null;
  private requestId = 0;
  private readonly pending = new Map<
    number,
    {
      worker: number;
      resolve: (r: ImsBrickResponse | ImsExtractResponse) => void;
      reject: (e: Error) => void;
    }
  >();
  /** Per worker: chunks it holds decoded (address → bytes), as an LRU like its own. */
  private ownerMirror: Array<ByteLru<number, number>> = [];
  /** Per worker: requests posted and not answered yet. */
  private load: number[] = [];
  private readonly fetchAvg = new RollingAvg();
  private readonly assemblyAvg = new RollingAvg();
  private readonly queueAvg = new RollingAvg();
  private readonly timepoint: number;
  /** Channels asked for before initialize() (options, then setSourceSelection). */
  private requestedChannels: number[] | undefined;

  /** Every channel in the file (names, colours, ranges). */
  allChannels: ImsChannelInfo[] = [];
  /** Source channel index for each active slot. */
  activeChannels: number[] = [];

  constructor(
    private readonly source: string | File,
    private readonly options: ImarisProviderOptions = {},
  ) {
    this.timepoint = options.timepoint ?? 0;
    this.requestedChannels = options.channels;
  }

  /** Called by kiln after initialize() with the atlas format it chose. */
  setTargetFormat(format: "r8unorm" | "r16unorm" | "r16float"): void {
    this.format = format === "r8unorm" ? "u8" : "f16";
  }

  /**
   * Choose the active channels. Before initialize() it shapes the metadata;
   * afterwards it switches the live metadata in place and kiln re-streams.
   */
  setSourceSelection(selection: SourceSelection): void {
    if ((selection.timepoint ?? this.timepoint) !== this.timepoint) {
      throw new RangeError("Switching Imaris time points isn't supported");
    }
    if (!this.metadata) {
      this.requestedChannels = selection.sourceChannels
        ? [...selection.sourceChannels]
        : undefined;
      return;
    }
    this.activeChannels = resolveChannels(
      selection.sourceChannels,
      this.allChannels.length,
    );
    Object.assign(this.metadata, this.channelMetadata());
    if (this.fetcher) this.prefetchIndexes(this.activeChannels);
  }

  /** Metadata fields that follow the active channels. */
  private channelMetadata(): Partial<VolumeMetadata> {
    const active = this.activeChannels.map((c) => this.allChannels[c]!);
    return {
      window: active[0] ? windowFor(active[0]) : undefined,
      channelWindows: active.map(windowFor),
      channels: active.map((c) => ({
        label: c.name,
        color: c.color,
        active: true,
      })),
      numChannels: Math.max(1, active.length),
      sourceChannels: [...this.activeChannels],
      sourceChannelMetadata: this.allChannels.map((c) => ({
        label: c.name,
        color: c.color,
        active: this.activeChannels.includes(c.index),
      })),
    };
  }

  private rangeSource(): RangeSource {
    if (typeof this.source !== "string") return blobRangeSource(this.source);
    // Retry transient network failures (TypeError "Failed to fetch", 5xx).
    const inner = httpRangeSource(this.source);
    return {
      async read(offset, length, signal) {
        let lastErr: unknown;
        for (let attempt = 0; attempt < 4; attempt++) {
          try {
            return await inner.read(offset, length, signal);
          } catch (e) {
            lastErr = e;
            if (signal?.aborted) throw e;
            const msg = e instanceof Error ? e.message : String(e);
            if (/HTTP 4\d\d|ignored the HTTP Range/.test(msg)) throw e;
            await new Promise((r) => setTimeout(r, 250 * 2 ** attempt));
          }
        }
        throw lastErr;
      },
    };
  }

  async initialize(): Promise<VolumeMetadata> {
    if (this.metadata) return this.metadata;
    const raw = this.rangeSource();
    const meta = cachedRangeSource(raw, {
      pageSize: 16 * 1024,
      maxPages: META_PAGES,
    });
    this.raw = raw;
    this.meta = meta;
    // Imaris writes its group/attribute metadata at the head of the file:
    // one up-front read turns ~30 serial round trips into one.
    await meta.read(0, 1024 * 1024).catch(() => undefined);
    const file = await Hdf5File.open(raw, { meta });
    this.file = file;

    const reasons: string[] = [];
    const dataSet = await file.get("/DataSet");
    if (!(dataSet instanceof Hdf5Group)) {
      throw new UnsupportedDatasetError([
        "Not an Imaris 5.5 file (missing /DataSet group)",
      ]);
    }
    const levelNames = (await dataSet.keys())
      .map((k) => ({ k, m: /^Resolution ?Level (\d+)$/.exec(k) }))
      .filter((e) => e.m)
      .sort((a, b) => Number(a.m![1]) - Number(b.m![1]));
    const tp = `TimePoint ${this.timepoint}`;

    // Source levels (ImageSize attrs of channel 0)
    const sources: SourceLevel[] = [];
    const ch0Attrs: Array<Record<string, Hdf5AttrValue>> = [];
    const levelAttrs = await Promise.all(
      levelNames.map(async ({ k }) => {
        const g = await file.get(`/DataSet/${k}/${tp}/Channel 0`);
        return g instanceof Hdf5Group ? g.attrs() : null;
      }),
    );
    for (let i = 0; i < levelNames.length; i++) {
      const a = levelAttrs[i];
      if (!a) break;
      const x = attrNumber(a.ImageSizeX);
      const y = attrNumber(a.ImageSizeY);
      const z = attrNumber(a.ImageSizeZ);
      if (!x || !y || !z) break;
      sources.push({ index: Number(levelNames[i]!.m![1]), dims: [x, y, z] });
      ch0Attrs.push(a);
    }
    if (sources.length === 0) {
      throw new UnsupportedDatasetError([
        `IMS file has no ResolutionLevel / ${tp} / Channel 0 data`,
      ]);
    }
    this.sourceLevels = sources;
    if (sources[0]!.dims[2] < 2) {
      reasons.push("IMS volume is not 3D (ImageSizeZ must be greater than 1)");
    }

    const tpGroup = await file.get(`/DataSet/${levelNames[0]!.k}/${tp}`);
    const channelCount =
      tpGroup instanceof Hdf5Group
        ? (await tpGroup.keys()).filter((k) => /^Channel \d+$/.test(k)).length
        : 1;

    // Header only: the chunk index is read lazily, per level, on first use.
    const probe = await file.get(
      `/DataSet/ResolutionLevel ${sources[0]!.index}/TimePoint ${this.timepoint}/Channel 0/Data`,
    );
    if (!(probe instanceof Hdf5Dataset)) {
      throw new Error("IMS dataset missing for the finest level");
    }
    const dt = probe.dtype;
    if (
      !(
        (dt.class === "uint" && (dt.size === 1 || dt.size === 2)) ||
        (dt.class === "int" && dt.size === 2) ||
        (dt.class === "float" && dt.size === 4)
      ) ||
      !dt.littleEndian
    ) {
      reasons.push(
        `Data type ${dt.class}${dt.size * 8}${dt.littleEndian ? "" : " (big-endian)"} is not supported`,
      );
    }
    if (reasons.length > 0) throw new UnsupportedDatasetError(reasons);

    // Channel descriptions
    const info = await file.get("/DataSetInfo");
    const infoGroup = info instanceof Hdf5Group ? info : null;
    const typeMax = dt.class === "float" ? 1 : 2 ** (dt.size * 8) - 1;
    this.allChannels = await Promise.all(
      Array.from({ length: channelCount }, async (_, c) => {
        const levelAttrs: Record<string, Hdf5AttrValue> =
          c === 0
            ? ch0Attrs[0]!
            : await (async () => {
                const g = await file.get(
                  `/DataSet/${levelNames[0]!.k}/${tp}/Channel ${c}`,
                );
                return g instanceof Hdf5Group ? g.attrs() : {};
              })();
        const infoCh = infoGroup ? await infoGroup.get(`Channel ${c}`) : null;
        const ia = infoCh instanceof Hdf5Group ? await infoCh.attrs() : {};
        const hMin = attrNumber(levelAttrs.HistogramMin);
        const hMax = attrNumber(levelAttrs.HistogramMax);
        let lo = hMin ?? 0;
        let hi = hMax ?? typeMax;
        if (dt.class !== "float") lo = Math.max(0, lo);
        if (!(hi > lo)) hi = lo + 1;
        const rgb = attrNumbers(ia.Color);
        const color: [number, number, number] =
          rgb && rgb.length >= 3
            ? [
                Math.min(1, Math.max(0, rgb[0]!)),
                Math.min(1, Math.max(0, rgb[1]!)),
                Math.min(1, Math.max(0, rgb[2]!)),
              ]
            : [1, 1, 1];
        const cr = attrNumbers(ia.ColorRange);
        const name = attrString(ia.Name);
        return {
          index: c,
          name: name && name !== "(name not specified)" ? name : `Channel ${c}`,
          color,
          range: [lo, hi] as [number, number],
          displayRange:
            cr && cr.length >= 2
              ? ([cr[0]!, cr[1]!] as [number, number])
              : null,
        };
      }),
    );

    this.activeChannels = resolveChannels(this.requestedChannels, channelCount);

    // Geometry
    let voxelSpacing: [number, number, number] | undefined;
    let name = "ims";
    const image = infoGroup ? await infoGroup.get("Image") : null;
    if (image instanceof Hdf5Group) {
      const a = await image.attrs();
      const [sx, sy, sz] = sources[0]!.dims;
      const ext = (i: number) => {
        const lo = attrNumber(a[`ExtMin${i}`]);
        const hi = attrNumber(a[`ExtMax${i}`]);
        return lo !== undefined && hi !== undefined
          ? Math.abs(hi - lo)
          : undefined;
      };
      const dx = ext(0);
      const dy = ext(1);
      const dz = ext(2);
      if (dx && dy && dz) voxelSpacing = [dx / sx, dy / sy, dz / sz];
      const n = attrString(a.Name);
      if (n && n !== "(name not specified)") name = n;
    }
    if (name === "ims" && typeof this.source === "string") {
      name = decodeURIComponent(this.source.split("/").pop() ?? "ims");
    } else if (name === "ims" && typeof this.source !== "string") {
      name = this.source.name;
    }

    this.pyramid = planVirtualPyramid(sources);
    this.metadata = {
      name,
      dimensions: sources[0]!.dims,
      voxelSpacing,
      brickSize: LOGICAL,
      physicalBrickSize: PHYSICAL,
      maxLod: this.pyramid.length - 1,
      levels: this.pyramid.map((l) => ({
        lod: l.lod,
        dimensions: l.dims,
        brickGrid: l.brickGrid,
        brickCount: l.brickGrid[0] * l.brickGrid[1] * l.brickGrid[2],
      })),
      bitDepth: 16,
      isFloat: true,
      dataRange: [0, 1],
      sourceChannelCount: channelCount,
      dtype: `${dt.class}${dt.size * 8}`,
      compression: probe.filters.map((f) => f.name).join("+") || "none",
      numChannels: 1,
      ...this.channelMetadata(),
    };

    // The coarsest level's chunk index now: the base layer needs it first,
    // and an unreadable index fails here rather than on every brick.
    // Workers, the chunk fetcher and the other indexes wait for the first
    // brick (start()), so a metadata-only open stays cheap.
    const coarse = this.pyramid[this.pyramid.length - 1]!.source;
    await Promise.all(
      this.activeChannels.map((c) => this.openDataset(coarse, c)),
    );
    return this.metadata;
  }

  /** Workers, chunk fetcher and index prefetch, on the first brick load. */
  private start(): void {
    if (this.fetcher || !this.raw) return;
    // Decoding keeps up with a remote link on a few workers; more only add
    // memory (each has a decoded-chunk cache) and contend for cores. Local
    // files read at disk speed and can use more.
    const remote = typeof this.source === "string";
    const cores = navigator.hardwareConcurrency || 4;
    const n = Math.max(
      1,
      Math.min(
        this.options.workers ?? (remote ? Math.ceil(cores / 2) : cores - 1),
        remote ? 4 : 8,
      ),
    );
    this.fetcher = new ChunkFetcher(this.raw, {
      budgetBytes: remote ? REMOTE_CHUNK_BUDGET : LOCAL_CHUNK_BUDGET,
      persistKey:
        remote && this.options.persistentCache !== false
          ? new URL(this.source as string, location.href).href
          : undefined,
      onFetched: (bytes, requests) => this.tracker.record(bytes, requests),
    });
    const chunkBudget = Math.min(WORKER_CHUNK_BUDGET, DECODED_CHUNK_BUDGET / n);
    for (let i = 0; i < n; i++) {
      const w = createWorker();
      w.onmessage = (e: MessageEvent<ImsBrickResponse | ImsExtractResponse>) =>
        this.onWorkerMessage(e.data);
      w.onerror = (e) => console.error("[Imaris] worker error", e.message);
      w.postMessage({ type: "init", chunkBudget } satisfies ImsInitRequest);
      this.workers.push(w);
      this.load.push(0);
      this.ownerMirror.push(new ByteLru(chunkBudget * 0.9, (b) => b));
    }
    this.prefetchIndexes(this.activeChannels);
  }

  /**
   * Open chunk indexes for `channels` at every level in the background,
   * coarse to fine, so a first zoom doesn't wait on B-tree reads queued
   * behind chunk downloads. Then shrink the metadata page cache: once the
   * indexes are built, its pages are not read again.
   */
  private prefetchIndexes(channels: readonly number[]): void {
    const file = this.file;
    const levels = [...new Set(this.pyramid.map((l) => l.source))].reverse();
    void (async () => {
      for (const level of levels) {
        if (this.file !== file) return; // disposed
        await Promise.all(
          channels.map((c) => this.openDataset(level, c).catch(() => null)),
        );
      }
      if (this.file === file) this.meta?.setMaxPages(META_PAGES_AFTER_INDEX);
    })();
  }

  private onWorkerMessage(msg: ImsBrickResponse | ImsExtractResponse): void {
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    this.load[p.worker]!--;
    if (msg.error) p.reject(new Error(msg.error));
    else p.resolve(msg);
  }

  private call<T extends ImsBrickResponse | ImsExtractResponse>(
    worker: number,
    msg: ImsExtractRequest | ImsAssembleRequest,
    transfer: Transferable[] = [],
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(msg.id, {
        worker,
        resolve: resolve as (r: ImsBrickResponse | ImsExtractResponse) => void,
        reject,
      });
      this.load[worker]!++;
      this.workers[worker]!.postMessage(msg, transfer);
    });
  }

  /** The worker with the fewest requests outstanding. */
  private leastLoaded(): number {
    let best = 0;
    for (let i = 1; i < this.load.length; i++)
      if (this.load[i]! < this.load[best]!) best = i;
    return best;
  }

  /**
   * Get the sub-region of one chunk (normalised float16) from its owner,
   * sending the compressed bytes only when the owner doesn't have it.
   */
  private async extract(
    idx: DatasetIndex,
    chunk: number,
    region: [number, number, number, number, number, number],
    norm: [number, number],
    signal: AbortSignal | undefined,
    priority: number,
  ): Promise<Uint16Array> {
    const address = idx.address[chunk]!;
    const size = idx.size[chunk]!;
    const w = ownerOf(address, this.workers.length);
    const decodedBytes =
      idx.chunkShape[0] * idx.chunkShape[1] * idx.chunkShape[2] * 2;
    const send = async (withBytes: boolean): Promise<ImsExtractResponse> => {
      let bytes: Uint8Array | null = null;
      if (withBytes) {
        if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
        bytes = await this.fetcher!.get({ address, size }, signal, priority);
      }
      const msg: ImsExtractRequest = {
        type: "extract",
        id: ++this.requestId,
        address,
        bytes,
        filters: idx.dataset.filters,
        filterMask: idx.filterMask.get(chunk) ?? 0,
        dtype: {
          class: idx.dataset.dtype.class as "int" | "uint" | "float",
          size: idx.dataset.dtype.size,
        },
        chunkShape: idx.chunkShape,
        normLo: norm[0],
        normHi: norm[1],
        region,
      };
      return this.call<ImsExtractResponse>(w, msg);
    };
    let res = await send(!this.ownerMirror[w]!.has(address));
    if (res.missing) {
      this.ownerMirror[w]!.delete(address);
      res = await send(true);
    }
    if (res.error || !res.data)
      throw new Error(res.error ?? "IMS chunk extract failed");
    this.ownerMirror[w]!.set(address, decodedBytes);
    return new Uint16Array(res.data);
  }

  private openDataset(level: number, channel: number): Promise<DatasetIndex> {
    const key = `${level}:${channel}`;
    let p = this.datasetIndex.get(key);
    if (!p) {
      p = (async () => {
        const file = this.file!;
        const ds = await file.get(
          `/DataSet/ResolutionLevel ${level}/TimePoint ${this.timepoint}/Channel ${channel}/Data`,
        );
        if (!(ds instanceof Hdf5Dataset)) {
          throw new Error(
            `IMS dataset missing for level ${level} channel ${channel}`,
          );
        }
        const cs = ds.chunkShape ?? ds.shape;
        const chunkShape: Vec3 = [cs[2]!, cs[1]!, cs[0]!];
        const grid: Vec3 = [
          Math.ceil(ds.shape[2]! / chunkShape[0]),
          Math.ceil(ds.shape[1]! / chunkShape[1]),
          Math.ceil(ds.shape[0]! / chunkShape[2]),
        ];
        const n = grid[0] * grid[1] * grid[2];
        const idx: DatasetIndex = {
          dataset: ds,
          chunkShape,
          grid,
          address: new Float64Array(n).fill(-1),
          size: new Uint32Array(n),
          filterMask: new Map(),
        };
        if (ds.chunkShape) {
          for (const ref of await ds.chunks()) {
            const i = chunkIndex(
              idx,
              Math.floor(ref.offset[2]! / chunkShape[0]),
              Math.floor(ref.offset[1]! / chunkShape[1]),
              Math.floor(ref.offset[0]! / chunkShape[2]),
            );
            if (i >= n) continue;
            idx.address[i] = ref.address;
            idx.size[i] = ref.size;
            if (ref.filterMask !== 0) idx.filterMask.set(i, ref.filterMask);
          }
        }
        return idx;
      })();
      p.catch(() => this.datasetIndex.delete(key));
      this.datasetIndex.set(key, p);
    }
    return p;
  }

  getMetadata(): VolumeMetadata {
    if (!this.metadata)
      throw new Error("Metadata not loaded. Call initialize() first.");
    return this.metadata;
  }

  getBrickGrid(lod: number): [number, number, number] {
    const level = this.pyramid[lod];
    if (!level) throw new Error(`LOD level ${lod} not found`);
    return level.brickGrid;
  }

  async loadBrick(
    lod: number,
    bx: number,
    by: number,
    bz: number,
    channelIndex = 0,
    signal?: AbortSignal,
    priority = 0,
  ): Promise<BrickLoadResult | null> {
    const level = this.pyramid[lod];
    const channel = this.activeChannels[channelIndex];
    if (!this.metadata || !level || channel === undefined) return null;
    const [gx, gy, gz] = level.brickGrid;
    if (bx < 0 || by < 0 || bz < 0 || bx >= gx || by >= gy || bz >= gz)
      return null;
    this.start();

    try {
      const idx = await this.openDataset(level.source, channel);
      const src = this.sourceLevels.find((s) => s.index === level.source)!;
      const sx = axisSampling(bx, level.dims[0], src.dims[0], level.ratio[0]);
      const sy = axisSampling(by, level.dims[1], src.dims[1], level.ratio[1]);
      const sz = axisSampling(bz, level.dims[2], src.dims[2], level.ratio[2]);
      const lo: Vec3 = [sx.lo, sy.lo, sz.lo];
      const hi: Vec3 = [sx.hi, sy.hi, sz.hi];
      const P = PHYSICAL;
      const info = this.allChannels[channel]!;
      const norm: [number, number] = [info.range[0], info.range[1]];
      const t0 = performance.now();
      const jobs: Array<Promise<ImsPiece>> = [];
      const [csx, csy, csz] = idx.chunkShape;
      for (const [cx, cy, cz] of overlappingChunks(lo, hi, idx.chunkShape)) {
        const index = chunkIndex(idx, cx, cy, cz);
        if (!(idx.address[index]! >= 0)) continue; // unallocated → zeros
        const x0 = Math.max(lo[0], cx * csx);
        const y0 = Math.max(lo[1], cy * csy);
        const z0 = Math.max(lo[2], cz * csz);
        const x1 = Math.min(hi[0], (cx + 1) * csx);
        const y1 = Math.min(hi[1], (cy + 1) * csy);
        const z1 = Math.min(hi[2], (cz + 1) * csz);
        const region: [number, number, number, number, number, number] = [
          x0 - cx * csx,
          y0 - cy * csy,
          z0 - cz * csz,
          x1 - cx * csx,
          y1 - cy * csy,
          z1 - cz * csz,
        ];
        jobs.push(
          this.extract(idx, index, region, norm, signal, priority).then(
            (data) => ({
              x0,
              y0,
              z0,
              sx: x1 - x0,
              sy: y1 - y0,
              sz: z1 - z0,
              data,
            }),
          ),
        );
      }
      if (jobs.length === 0) {
        const empty = { min: 0, max: 0, avg: 0 };
        this.cacheBrickStats(lod, bx, by, bz, empty);
        return {
          data:
            this.format === "u8"
              ? new Uint8Array(P * P * P)
              : new Uint16Array(P * P * P),
          ...empty,
        };
      }
      const pieces = await Promise.all(jobs);
      if (signal?.aborted) return null;
      const tPieces = performance.now();
      const result = await this.call<ImsBrickResponse>(
        this.leastLoaded(),
        {
          type: "assemble",
          id: ++this.requestId,
          pieces,
          lo,
          hi,
          fills: [],
          rx: sx.ranges,
          ry: sy.ranges,
          rz: sz.ranges,
          format: this.format,
        },
        pieces.map((p) => p.data.buffer as ArrayBuffer),
      );
      if (result.error || !result.data)
        throw new Error(result.error ?? "IMS brick assembly failed");
      this.fetchAvg.add(tPieces - t0);
      this.assemblyAvg.add(result.assemblyMs);
      this.queueAvg.add(performance.now() - tPieces - result.assemblyMs);
      const brickStats = { min: result.min, max: result.max, avg: result.avg };
      this.cacheBrickStats(lod, bx, by, bz, brickStats);
      const data =
        this.format === "u8"
          ? new Uint8Array(result.data)
          : new Uint16Array(result.data);
      return { data, ...brickStats };
    } catch (e) {
      if (signal?.aborted) return null;
      if (e instanceof Error && e.message === "cancelled") return null;
      console.warn(
        `[Imaris] brick lod${lod}:${bx}-${by}-${bz} ch${channelIndex} failed:`,
        e,
      );
      return null;
    }
  }

  async isBrickEmpty(
    lod: number,
    bx: number,
    by: number,
    bz: number,
    maxThreshold?: number,
  ): Promise<boolean> {
    return imsBrickIsEmpty({
      lod,
      bx,
      by,
      bz,
      maxLod: this.metadata?.maxLod ?? lod,
      threshold: maxThreshold ?? 1,
      getStats: (l, x, y, z) =>
        this.brickStatsCache.get(`${l}:${x}/${y}/${z}`) ?? null,
    });
  }

  async getBrickStats(
    lod: number,
    bx: number,
    by: number,
    bz: number,
  ): Promise<BrickStats | null> {
    return this.brickStatsCache.get(`${lod}:${bx}/${by}/${bz}`) ?? null;
  }

  getNetworkStats(): NetworkStats {
    return this.tracker.getStats();
  }

  getPipelineTimings(): PipelineTimings {
    return {
      avgQueueMs: this.queueAvg.value,
      avgFetchMs: this.fetchAvg.value,
      avgAssemblyMs: this.assemblyAvg.value,
      avgUploadMs: 0, // measured by kiln
      sampleCount: this.fetchAvg.count,
    };
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.ownerMirror = [];
    this.load = [];
    this.fetcher?.dispose();
    this.fetcher = null;
    for (const p of this.pending.values()) p.reject(new Error("cancelled"));
    this.pending.clear();
    this.brickStatsCache.clear();
    this.datasetIndex.clear();
    this.metadata = null;
    this.file = null;
    this.raw = null;
    this.meta = null;
  }

  private cacheBrickStats(
    lod: number,
    bx: number,
    by: number,
    bz: number,
    stats: BrickStats,
  ): void {
    const key = `${lod}:${bx}/${by}/${bz}`;
    const existing = this.brickStatsCache.get(key);
    this.brickStatsCache.set(
      key,
      existing
        ? {
            min: Math.min(existing.min, stats.min),
            max: Math.max(existing.max, stats.max),
            avg: Math.max(existing.avg, stats.avg),
          }
        : stats,
    );
    while (this.brickStatsCache.size > MAX_STATS_ENTRIES) {
      const oldest = this.brickStatsCache.keys().next().value;
      if (oldest === undefined) break;
      this.brickStatsCache.delete(oldest);
    }
  }
}
