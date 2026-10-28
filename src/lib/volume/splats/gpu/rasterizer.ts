/**
 * Tile-based Gaussian splat rasterizer (forward pass) on WebGPU.
 *
 * Pipeline per frame:
 *   project → depth radix sort → per-Gaussian tile counts → scan →
 *   emit (tile, gaussian) pairs in depth order → stable sort by tile →
 *   tile ranges → per-tile front-to-back compositing.
 *
 * Output is a storage buffer of premultiplied RGBA (one vec4 per pixel).
 * Everything stays on the GPU; the intersection total is read back
 * asynchronously only to grow buffers.
 */

import type { SplatCamera } from "../cpu-reference";
import { type DepthCue, NO_DEPTH_CUE } from "../depth-cue";
import {
  bindGroup,
  createBuffer,
  createKernel,
  dispatch1d,
  type Kernel,
  READBACK,
  readBuffer,
  STORAGE_RW,
  withMapped,
} from "./kernel";
import { GpuRadixSort } from "./radix-sort";
import {
  CAMERA_UNIFORM_SIZE,
  MAX_SPLAT_CHANNELS,
  PROJ_VEC4S,
  PROJECT_WGSL,
  RANGES_WGSL,
  RASTERIZE_WGSL,
  TILE,
  TILES_WGSL,
} from "./raster-wgsl";
import { GpuScan } from "./scan";

interface RasterKernels {
  project: Kernel;
  countTiles: Kernel;
  emit: Kernel;
  rangesSetup: Kernel;
  ranges: Kernel;
  /** Front-to-back compositing (plain clouds, channelMode 1) */
  rasterize: Kernel;
  /** Per-channel additive compositing (channelMode 2) */
  rasterizeAdd: Kernel;
}

const kernelCache = new WeakMap<GPUDevice, RasterKernels>();

/** Minimum interval between intersection-capacity readbacks (display). */
const CAPACITY_CHECK_MS = 500;

function rasterKernels(device: GPUDevice): RasterKernels {
  let k = kernelCache.get(device);
  if (k) return k;
  const tileBindings = [
    "uniform",
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
  ] as const;
  const rangeBindings = [
    "uniform",
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
  ] as const;
  const rasterBindings = [
    "uniform",
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "storage",
  ] as const;
  k = {
    project: createKernel(device, "splat:project", PROJECT_WGSL, "project", [
      "uniform",
      "read-only-storage",
      "storage",
      "storage",
      "storage",
    ]),
    countTiles: createKernel(
      device,
      "splat:count",
      TILES_WGSL,
      "countTiles",
      tileBindings,
    ),
    emit: createKernel(
      device,
      "splat:emit",
      TILES_WGSL,
      "emitIntersections",
      tileBindings,
    ),
    rangesSetup: createKernel(
      device,
      "splat:ranges-setup",
      RANGES_WGSL,
      "rangesSetup",
      rangeBindings,
    ),
    ranges: createKernel(
      device,
      "splat:ranges",
      RANGES_WGSL,
      "tileRangesKernel",
      rangeBindings,
    ),
    rasterize: createKernel(
      device,
      "splat:rasterize",
      RASTERIZE_WGSL,
      "rasterize",
      rasterBindings,
    ),
    rasterizeAdd: createKernel(
      device,
      "splat:rasterize-add",
      RASTERIZE_WGSL,
      "rasterize",
      rasterBindings,
      { additive: 1 },
    ),
  };
  kernelCache.set(device, k);
  return k;
}

function bitsFor(n: number): number {
  return Math.max(1, Math.ceil(Math.log2(Math.max(2, n))));
}

type Groups = Record<keyof RasterKernels, GPUBindGroup>;

export class SplatRasterizer {
  readonly device: GPUDevice;
  private readonly k: RasterKernels;

  // Per-Gaussian
  private gaussianCapacity = 0;
  private count = 0;
  /** Count last written to countBuf (−1: not written yet) */
  private countWritten = -1;
  private params: GPUBuffer | null = null;
  private projected!: GPUBuffer;
  private depthKeys!: GPUBuffer;
  /** Gaussian indices in depth order after a forward pass (tests read it). */
  depthIdx!: GPUBuffer;
  private offsets!: GPUBuffer;
  private countBuf!: GPUBuffer;
  private depthSort!: GpuRadixSort;
  private scan!: GpuScan;

  // Per-intersection
  private isectCapacity = 0;
  private isectKeys!: GPUBuffer;
  private isectVals!: GPUBuffer;
  private isectSort!: GpuRadixSort;

  // Per-pixel / per-tile
  private width = 0;
  private height = 0;
  private tilesX = 0;
  private tilesY = 0;
  /** Premultiplied RGBA, one vec4f per pixel (row-major). */
  image!: GPUBuffer;
  private tileRanges!: GPUBuffer;
  private readonly camera: GPUBuffer;
  private readonly rangesIndirect: GPUBuffer;
  private readonly dummy: GPUBuffer;
  private readonly cameraF32 = new Float32Array(CAMERA_UNIFORM_SIZE / 4);
  private readonly cameraU32 = new Uint32Array(this.cameraF32.buffer);
  /** Persistent staging buffer for capacity checks. */
  private readonly totalReadback: GPUBuffer;

  private groups: Groups | null = null;
  /**
   * Mip-Splatting opacity compensation for the screen-space dilation. On for
   * Kiln-trained clouds; off for ordinary 3DGS files trained without it.
   */
  mipFilter = true;
  /** See `channelMode` in the WGSL Camera struct (0 = plain cloud). */
  channelMode: 0 | 1 | 2 = 0;
  /** Per-channel tint, rgba × MAX_SPLAT_CHANNELS (a = visible). */
  readonly channelTint = new Float32Array(MAX_SPLAT_CHANNELS * 4).fill(1);
  /** Depth cue (core/depth-cue.ts); off by default (training, evaluation). */
  depthCue: DepthCue = NO_DEPTH_CUE;

  private lastCapacityCheck = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.k = rasterKernels(device);
    this.camera = createBuffer(
      device,
      "splat:camera",
      CAMERA_UNIFORM_SIZE,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.dummy = createBuffer(device, "splat:dummy", 16, STORAGE_RW);
    this.rangesIndirect = createBuffer(
      device,
      "splat:ranges-indirect",
      12,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
    this.totalReadback = createBuffer(device, "splat:isect-total", 4, READBACK);
  }

  get gaussianCount(): number {
    return this.count;
  }

  /** Point the rasterizer at a params buffer (PARAM_STRIDE floats per Gaussian). */
  setGaussians(params: GPUBuffer, count: number): void {
    // Bind groups only depend on buffers, not on the count (a uniform).
    if (params !== this.params) this.groups = null;
    this.params = params;
    this.count = count;
    if (count > this.gaussianCapacity) {
      this.allocGaussians(Math.max(count, 1024));
    }
    this.ensureIsectCapacity(Math.max(1 << 20, count * 8));
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.tilesX = Math.ceil(width / TILE);
    this.tilesY = Math.ceil(height / TILE);
    this.image?.destroy();
    this.tileRanges?.destroy();
    this.image = createBuffer(
      this.device,
      "splat:image",
      width * height * 16,
      STORAGE_RW,
    );
    this.tileRanges = createBuffer(
      this.device,
      "splat:tile-ranges",
      this.tilesX * this.tilesY * 8,
      STORAGE_RW,
    );
    this.groups = null;
  }

  private allocGaussians(capacity: number): void {
    this.groups = null;
    this.gaussianCapacity = capacity;
    for (const b of [
      this.projected,
      this.depthKeys,
      this.depthIdx,
      this.offsets,
      this.countBuf,
    ])
      b?.destroy();
    this.depthSort?.destroy();
    this.scan?.destroy();
    const d = this.device;
    this.projected = createBuffer(
      d,
      "splat:projected",
      capacity * PROJ_VEC4S * 16,
      STORAGE_RW,
    );
    this.depthKeys = createBuffer(
      d,
      "splat:depth-keys",
      capacity * 4,
      STORAGE_RW,
    );
    this.depthIdx = createBuffer(
      d,
      "splat:depth-idx",
      capacity * 4,
      STORAGE_RW,
    );
    this.offsets = createBuffer(
      d,
      "splat:offsets",
      (capacity + 1) * 4,
      STORAGE_RW,
    );
    this.countBuf = createBuffer(d, "splat:count", 16, STORAGE_RW);
    this.countWritten = -1;
    this.depthSort = new GpuRadixSort(d, capacity);
    this.scan = new GpuScan(d, capacity + 1);
  }

  private ensureIsectCapacity(capacity: number): void {
    if (capacity <= this.isectCapacity) return;
    this.isectCapacity = capacity;
    this.isectKeys?.destroy();
    this.isectVals?.destroy();
    this.isectSort?.destroy();
    this.isectKeys = createBuffer(
      this.device,
      "splat:isect-keys",
      capacity * 4,
      STORAGE_RW,
    );
    this.isectVals = createBuffer(
      this.device,
      "splat:isect-vals",
      capacity * 4,
      STORAGE_RW,
    );
    this.isectSort = new GpuRadixSort(this.device, capacity);
    this.groups = null;
  }

  private writeCamera(cam: SplatCamera): void {
    const f = this.cameraF32;
    const u = this.cameraU32;
    for (let i = 0; i < 16; i++) f[i] = cam.view[i]!;
    f[16] = cam.fx;
    f[17] = cam.fy;
    f[18] = cam.cx;
    f[19] = cam.cy;
    u[20] = this.width;
    u[21] = this.height;
    u[22] = this.tilesX;
    u[23] = this.tilesY;
    u[24] = this.count;
    u[25] = this.isectCapacity;
    u[26] = this.mipFilter ? 1 : 0;
    u[27] = this.channelMode;
    f.set(this.channelTint, 28);
    f[28 + MAX_SPLAT_CHANNELS * 4] = this.depthCue.near;
    f[29 + MAX_SPLAT_CHANNELS * 4] = this.depthCue.scale;
    this.device.queue.writeBuffer(this.camera, 0, f);
    if (this.countWritten !== this.count) {
      // [0]: Gaussians (depth sort), [1]: offsets entries (scan)
      this.device.queue.writeBuffer(
        this.countBuf,
        0,
        new Uint32Array([this.count, this.count + 1]),
      );
      this.countWritten = this.count;
    }
  }

  private bindGroups(): Groups {
    if (this.groups) return this.groups;
    const d = this.device;
    const k = this.k;
    if (!this.params) throw new Error("SplatRasterizer: no Gaussians set");
    const tiles = [
      this.camera,
      this.projected,
      this.depthIdx,
      this.offsets,
      this.isectKeys,
      this.isectVals,
    ];
    const ranges = [this.camera, this.offsets, this.isectKeys, this.tileRanges];
    const raster = [
      this.camera,
      this.projected,
      this.isectVals,
      this.tileRanges,
      this.image,
    ];
    this.groups = {
      project: bindGroup(d, k.project, [
        this.camera,
        this.params,
        this.projected,
        this.depthKeys,
        this.depthIdx,
      ]),
      countTiles: bindGroup(d, k.countTiles, tiles),
      emit: bindGroup(d, k.emit, tiles),
      rangesSetup: bindGroup(d, k.rangesSetup, [
        ...ranges,
        this.rangesIndirect,
      ]),
      // The indirect buffer can't be bound while it drives this dispatch.
      ranges: bindGroup(d, k.ranges, [...ranges, this.dummy]),
      rasterize: bindGroup(d, k.rasterize, raster),
      rasterizeAdd: bindGroup(d, k.rasterizeAdd, raster),
    };
    return this.groups;
  }

  /** Encode the full forward pass. Output lands in `image`. */
  encodeForward(encoder: GPUCommandEncoder, cam: SplatCamera): void {
    if (!this.params || this.width === 0) return;
    this.writeCamera(cam);
    const g = this.bindGroups();
    const k = this.k;
    const n = this.count;
    encoder.clearBuffer(this.tileRanges);
    if (n === 0) {
      encoder.clearBuffer(this.image);
      return;
    }
    // One pass: dispatches in a pass see each other's writes.
    const pass = encoder.beginComputePass({ label: "splat:forward" });
    dispatch1d(pass, k.project, g.project, Math.ceil(n / 256));
    this.depthSort.encode(
      pass,
      this.depthKeys,
      this.depthIdx,
      { buffer: this.countBuf, index: 0 },
      32,
    );
    dispatch1d(pass, k.countTiles, g.countTiles, Math.ceil((n + 1) / 256));
    this.scan.encode(pass, this.offsets, { buffer: this.countBuf, index: 1 });
    dispatch1d(pass, k.emit, g.emit, Math.ceil(n / 256));
    this.isectSort.encode(
      pass,
      this.isectKeys,
      this.isectVals,
      { buffer: this.offsets, index: n },
      bitsFor(this.tilesX * this.tilesY),
    );
    pass.setPipeline(k.rangesSetup.pipeline);
    pass.setBindGroup(0, g.rangesSetup);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(k.ranges.pipeline);
    pass.setBindGroup(0, g.ranges);
    pass.dispatchWorkgroupsIndirect(this.rangesIndirect, 0);
    const additive = this.channelMode === 2;
    pass.setPipeline((additive ? k.rasterizeAdd : k.rasterize).pipeline);
    pass.setBindGroup(0, additive ? g.rasterizeAdd : g.rasterize);
    pass.dispatchWorkgroups(this.tilesX, this.tilesY, 1);
    pass.end();
  }

  /** Copy the last forward pass's intersection total (u32) into `dst`. */
  copyIntersectionTotal(
    encoder: GPUCommandEncoder,
    dst: GPUBuffer,
    dstOffset: number,
  ): void {
    encoder.copyBufferToBuffer(this.offsets, this.count * 4, dst, dstOffset, 4);
  }

  /**
   * After submitting a forward pass: read the intersection total back
   * asynchronously and grow buffers when close to capacity.
   */
  checkCapacity(): void {
    const buf = this.totalReadback;
    if (this.count === 0 || buf.mapState !== "unmapped") return;
    // Capacity drifts slowly; a readback per displayed frame isn't needed.
    const now = performance.now();
    if (now - this.lastCapacityCheck < CAPACITY_CHECK_MS) return;
    this.lastCapacityCheck = now;
    const enc = this.device.createCommandEncoder();
    this.copyIntersectionTotal(enc, buf, 0);
    this.device.queue.submit([enc.finish()]);
    withMapped(buf, (d) => new Uint32Array(d, 0, 1)[0]!)
      .then((m) => {
        if (m > this.isectCapacity * 0.85) {
          this.ensureIsectCapacity(Math.ceil(m * 1.5));
        }
      })
      .catch(() => {});
  }

  /** Await the exact intersection total (offline evaluation). */
  async intersectionTotal(): Promise<number> {
    const buf = await readBuffer(this.device, this.offsets, this.count * 4, 4);
    return new Uint32Array(buf)[0] ?? 0;
  }

  /** Grow intersection buffers for at least `m` entries. Returns true if grown. */
  reserveIntersections(m: number): boolean {
    if (m <= this.isectCapacity) return false;
    this.ensureIsectCapacity(Math.ceil(m * 1.3));
    return true;
  }

  destroy(): void {
    for (const b of [
      this.projected,
      this.depthKeys,
      this.depthIdx,
      this.offsets,
      this.countBuf,
      this.isectKeys,
      this.isectVals,
      this.image,
      this.tileRanges,
      this.camera,
      this.rangesIndirect,
      this.dummy,
      this.totalReadback,
    ])
      b?.destroy();
    this.depthSort?.destroy();
    this.isectSort?.destroy();
    this.scan?.destroy();
  }
}
