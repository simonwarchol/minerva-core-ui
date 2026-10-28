/**
 * Volume → Gaussian splats, end to end, beside upstream kiln-render:
 *
 *  1. fit     — moment-matched Gaussians straight from the voxels, classified
 *               with the shown channel colours / windows / TF / density /
 *               clip box (volume-fit.ts, in a worker);
 *  2. capture — render training views of the *same* volume into an
 *               rgba16float texture array (premultiplied, transparent
 *               background) with a headless KilnEngine of its own, so the
 *               viewer keeps running untouched;
 *  3. train   — photometric refinement of the fitted Gaussians against those
 *               views with Brush (brush-trainer.ts).
 *
 * Lifted from simonwarchol/kiln-render (src/splat/splat-builder.ts and
 * KilnViewer.generateSplats), where capture borrowed the viewer's own
 * renderer and canvas. Tiled builds (`tileSplits`) are not ported.
 */

import type {
  DataProvider,
  KilnViewer,
  Renderer,
  UpAxis,
  VolumeMetadata,
} from "kiln-upstream";
import { Camera, KilnEngine } from "kiln-upstream";
import { LOGICAL } from "../imaris/ims-layout";
import { getFloat16ToFloat32Lut } from "../imaris/util/float16";
import { throwIfAborted, trainWithBrush } from "./brush-trainer";
import { cloudFromParams, type GaussianCloud, mergeClouds } from "./cloud";
import { intrinsicsFor, quatToRot, type SplatCamera } from "./cpu-reference";
import { createBuffer, READBACK, STORAGE_RW, withMapped } from "./gpu/kernel";
import { SplatRasterizer } from "./gpu/rasterizer";
import type { ClassifyParams } from "./volume-fit";
import { VolumeFitter } from "./volume-fit";
import type { FitWorkerRequest, FitWorkerResponse } from "./volume-fit-worker";

export type OrbitState = [number, number, number, number, number, number];

export type BuildStage = "fit" | "capture" | "train" | "done";

export interface BuildProgress {
  stage: BuildStage;
  /** 0–1 within the stage */
  fraction: number;
  label: string;
  gaussians?: number;
  loss?: number;
  /** Mean PSNR (dB) of the splats vs. the volume on held-out views */
  psnr?: number;
}

export interface SplatBuildOptions {
  /** Gaussians produced by the fit (training may densify up to 2×). Default 1M. */
  targetCount?: number;
  /** Refine photometrically against volume renders with Brush. Default true. */
  train?: boolean;
  /** Training iterations (default 3000) */
  iterations?: number;
  /** Training views (default 96) */
  views?: number;
  /** Square training image size in pixels (default 640) */
  resolution?: number;
  /** TAA frames accumulated per captured view (default 16) */
  settleFrames?: number;
  /** Seed for training-view placement; set to get a different (held-out) set */
  viewSeed?: number;
  /** Held-out views to score the result on (PSNR vs. the volume); 0 = skip */
  evaluateViews?: number;
  /**
   * Extra training cameras — e.g. the views a story will show — rendered and
   * trained on alongside the generated ones.
   */
  extraViews?: OrbitState[];
  /** Cameras to score the result at instead of generated held-out views. */
  evaluateAt?: OrbitState[];
  /** Fit pyramid level (default: finest that fits the memory budget) */
  fitLod?: number;
  /** Fit memory budget in bytes (default 512 MB) */
  fitMemoryBytes?: number;
  onProgress?: (p: BuildProgress) => void;
  signal?: AbortSignal;
}

export interface GenerateSplatsOptions extends SplatBuildOptions {
  /** Render mode the splats reproduce (default: the displayed one, or volume rendering). */
  renderMode?: "dvr" | "mip" | "iso";
  /** Volume-rendering density (default: the displayed one). */
  density?: number;
  /** Clip box to build from, 0–1 per axis (default: the displayed one). */
  region?: { min: [number, number, number]; max: [number, number, number] };
  /**
   * Train each shown channel on its own, rendered in white, and return one
   * cloud tagged by channel (`GaussianCloud.channelComposite`), tinted with
   * the live channel colours at display. `targetCount` is shared.
   */
  perChannel?: boolean;
  /** Called as each build starts (one per channel). */
  onTile?: (index: number, count: number) => void;
}

/** Atlas budget of the capture engine (the viewer keeps its own). */
const CAPTURE_ATLAS_BYTES = 512 * 1024 * 1024;

/**
 * Fit density for MIP builds. MIP ignores density, but the fit derives
 * opacity from it; this gives training a starting cloud near MIP brightness.
 */
const MIP_FIT_DENSITY = 1;

/** Background the viewer composites the volume over (kiln's default). */
const DISPLAY_BACKGROUND = [0.05, 0.05, 0.05];

/** Everything a build needs: a headless engine holding the look to reproduce. */
export interface SplatBuildHost {
  engine: KilnEngine;
  /** Places capture views (kiln's orbit camera, on no visible canvas). */
  camera: Camera;
}

/**
 * Yield to the event loop without depending on requestAnimationFrame (which
 * stops entirely while the page is occluded, e.g. a hidden pane or tab
 * behind another window — long builds should keep going).
 */
function yieldTask(delayMs = 0): Promise<void> {
  if (delayMs > 0) return new Promise((r) => setTimeout(r, delayMs));
  return new Promise((r) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => r();
    ch.port2.postMessage(null);
  });
}

function createFitWorker(): Worker {
  return new Worker(new URL("./volume-fit-worker.ts", import.meta.url), {
    type: "module",
  });
}

/** World-space box of a 0–1 clip box (the volume is centred at the origin). */
function clipToWorld(
  normalizedSize: readonly number[],
  clipMin: ArrayLike<number>,
  clipMax: ArrayLike<number>,
): { lo: [number, number, number]; hi: [number, number, number] } {
  const at = (c: ArrayLike<number>, a: number, d: number) =>
    ((c[a] ?? d) - 0.5) * normalizedSize[a]!;
  return {
    lo: [at(clipMin, 0, 0), at(clipMin, 1, 0), at(clipMin, 2, 0)],
    hi: [at(clipMax, 0, 1), at(clipMax, 1, 1), at(clipMax, 2, 1)],
  };
}

function classifyParamsFor(host: SplatBuildHost): ClassifyParams {
  const r = host.engine.renderer;
  const transfer = host.engine.transferFunction as unknown as {
    getColor(t: number): [number, number, number];
    sampleOpacity(t: number): number; // private in kiln
  };
  const tf = new Float32Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [cr, cg, cb] = transfer.getColor(i / 255);
    tf.set([cr, cg, cb, transfer.sampleOpacity(i / 255)], i * 4);
  }
  return {
    numChannels: r.numChannels,
    channelColors: Array.from(r.channelColors),
    channelWindowCenter: Array.from(r.channelWindowCenter),
    channelWindowWidth: Array.from(r.channelWindowWidth),
    floatMin: r.floatMin,
    floatMax: r.floatMax,
    windowCenter: r.windowCenter,
    windowWidth: r.windowWidth,
    tf,
    densityScale: r.densityScale,
    maxDim0: Math.max(...host.engine.config.dimensions),
    clipMin: Array.from(r.clipMin),
    clipMax: Array.from(r.clipMax),
  };
}

/** Finest LOD whose fit fits in the memory budget. */
function chooseFitLod(metadata: VolumeMetadata, budget: number): number {
  for (const level of [...metadata.levels].sort((a, b) => a.lod - b.lod)) {
    if (VolumeFitter.estimateBytes(level.dimensions) <= budget)
      return level.lod;
  }
  return metadata.maxLod;
}

/** Physical size of one normalised unit (for PLY metadata). */
function worldScaleOf(metadata: VolumeMetadata): number {
  const s = metadata.voxelSpacing ?? [1, 1, 1];
  return Math.max(
    metadata.dimensions[0] * s[0],
    metadata.dimensions[1] * s[1],
    metadata.dimensions[2] * s[2],
  );
}

/** Small RPC wrapper around a fit worker. */
class FitWorker {
  readonly worker: Worker;
  private nextId = 0;
  private readonly pending = new Map<
    number,
    { resolve: (r: FitWorkerResponse) => void; reject: (e: Error) => void }
  >();

  constructor() {
    this.worker = createFitWorker();
    this.worker.onmessage = (e: MessageEvent<FitWorkerResponse>) => {
      const r = e.data;
      const p = this.pending.get(r.id);
      if (!p) return;
      this.pending.delete(r.id);
      if (r.type === "error") p.reject(new Error(r.message));
      else p.resolve(r);
    };
  }

  call(
    msg: FitWorkerRequest,
    transfer: Transferable[] = [],
  ): Promise<FitWorkerResponse> {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  terminate(): void {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error("terminated"));
  }
}

/** Idle fit workers are kept this long for the next fit (channels). */
const POOL_IDLE_MS = 5 * 60 * 1000;
let idlePool: FitWorker[] | null = null;
let idleTimer: ReturnType<typeof setTimeout> | undefined;

function takePool(): FitWorker[] {
  clearTimeout(idleTimer);
  const pool = idlePool;
  idlePool = null;
  if (pool) return pool;
  const n = Math.max(1, Math.min(6, (navigator.hardwareConcurrency || 4) - 2));
  return Array.from({ length: n }, () => new FitWorker());
}

function releasePool(pool: FitWorker[]): void {
  if (idlePool) {
    for (const w of pool) w.terminate();
    return;
  }
  idlePool = pool;
  idleTimer = setTimeout(() => {
    for (const w of pool) w.terminate();
    idlePool = null;
  }, POOL_IDLE_MS);
}

/** Stage 1: direct fit from bricks (classification parallelised over workers). */
export async function fitVolume(
  host: SplatBuildHost,
  opts: SplatBuildOptions = {},
): Promise<GaussianCloud> {
  const { engine } = host;
  const meta = engine.metadata;
  const renderer = engine.renderer;
  const lod =
    opts.fitLod ?? chooseFitLod(meta, opts.fitMemoryBytes ?? 512 * 1024 * 1024);
  const level = meta.levels.find((l) => l.lod === lod)!;
  const onProgress = opts.onProgress ?? (() => {});
  const dims0 = engine.config.dimensions;
  // The renderer places LOD-L voxel j over LOD-0 voxels [j·2^L, (j+1)·2^L).
  const lodScale = 2 ** lod;
  const grid = {
    dims: level.dimensions,
    normalizedSize: engine.config.normalizedSize,
    dims0,
    lodScale,
  };
  const params = classifyParamsFor(host);
  const pool = takePool();
  let stop = false;
  try {
    await Promise.all(
      pool.map((w) => w.call({ type: "init", id: 0, grid, params })),
    );
    // The merged node array lives here; workers only produce per-brick blocks.
    const merged = new VolumeFitter(grid, params);
    // Only bricks overlapping the clip box (the fitter also clips per voxel).
    const clipMin = renderer.clipMin;
    const clipMax = renderer.clipMax;
    const span = (a: number, n: number) => {
      // Clip-space extent of one brick
      const b = (LOGICAL * lodScale) / dims0[a]!;
      return [
        Math.max(0, Math.floor((clipMin[a] ?? 0) / b)),
        Math.min(n - 1, Math.ceil((clipMax[a] ?? 1) / b) - 1),
      ] as const;
    };
    const [gx, gy, gz] = level.brickGrid;
    const [x0, x1] = span(0, gx);
    const [y0, y1] = span(1, gy);
    const [z0, z1] = span(2, gz);
    const bricks: Array<[number, number, number]> = [];
    for (let z = z0; z <= z1; z++)
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++) bricks.push([x, y, z]);
    const nch = renderer.numChannels;
    const visible = Array.from({ length: nch }, (_, ch) =>
      nch === 1 ? true : (renderer.channelColors[ch * 4 + 3] ?? 0) > 0,
    );
    const threshold = engine.config.emptyBrickThreshold;
    const provider = engine.dataProvider;
    let done = 0;
    let next = 0;
    let rr = 0;
    const loader = async () => {
      while (!stop && next < bricks.length) {
        const [bx, by, bz] = bricks[next++]!;
        throwIfAborted(opts.signal);
        if (!(await provider.isBrickEmpty(lod, bx, by, bz, threshold))) {
          const results = await Promise.all(
            visible.map((v, ch) =>
              v ? provider.loadBrick(lod, bx, by, bz, ch, opts.signal) : null,
            ),
          );
          const channels = results.map((r) => (r ? r.data.slice() : null));
          const w = pool[rr++ % pool.length]!;
          const res = await w.call(
            { type: "classify", id: 0, bx, by, bz, channels },
            channels.flatMap((c) => (c ? [c.buffer as ArrayBuffer] : [])),
          );
          if (res.type === "block" && res.block) {
            merged.addBlock(bx, by, bz, res.block, res.occupied);
          }
        }
        done++;
        onProgress({
          stage: "fit",
          fraction: (0.9 * done) / bricks.length,
          label: `Reading LOD ${lod} bricks ${done}/${bricks.length}`,
        });
      }
    };
    await Promise.all(Array.from({ length: 12 }, loader));
    onProgress({ stage: "fit", fraction: 0.92, label: "Fitting Gaussians…" });
    const nodes = merged.takeNodes();
    const res = await pool[0]!.call(
      {
        type: "finish",
        id: 0,
        nodes,
        occupied: merged.occupiedVoxels,
        options: { targetCount: opts.targetCount ?? 1_000_000 },
      },
      [nodes.buffer],
    );
    if (res.type !== "done") throw new Error("fit worker returned no result");
    const cloud = cloudFromParams(res.params, res.count, {
      worldScale: worldScaleOf(meta),
      worldUnit: "micron",
    });
    onProgress({
      stage: "fit",
      fraction: 1,
      label: `Fitted ${res.count.toLocaleString()} Gaussians from ${merged.occupiedVoxels.toLocaleString()} voxels (LOD ${lod})`,
      gaussians: res.count,
    });
    return cloud;
  } finally {
    // Loaders still running after a failure stop at their next brick.
    stop = true;
    releasePool(pool);
  }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/** Fibonacci directions on the unit sphere. */
export function sphereDirections(n: number): Array<[number, number, number]> {
  const out: Array<[number, number, number]> = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - ((i + 0.5) / n) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    out.push([Math.cos(golden * i) * r, y, Math.sin(golden * i) * r]);
  }
  return out;
}

/** Orbit angles (kiln Camera convention) that place the eye along `dir`. */
function orbitForDirection(
  dir: [number, number, number],
  upAxis: string,
): { rotX: number; rotY: number } {
  const sign = upAxis.startsWith("-") ? -1 : 1;
  const base = upAxis.replace("-", "");
  const lim = Math.PI / 2 - 0.02;
  const clampX = (v: number) => Math.max(-lim, Math.min(lim, v));
  const asin = (v: number) => Math.asin(Math.max(-1, Math.min(1, v)));
  if (base === "x")
    return {
      rotX: clampX(asin(dir[0] * sign)),
      rotY: Math.atan2(dir[2], dir[1]),
    };
  if (base === "z")
    return {
      rotX: clampX(asin(dir[2] * sign)),
      rotY: Math.atan2(dir[1], dir[0]),
    };
  return {
    rotX: clampX(asin(dir[1] * sign)),
    rotY: Math.atan2(dir[0], dir[2]),
  };
}

/** Frames rendered before a view counts as settled (lets ray feedback ask for bricks). */
const MIN_SETTLE_FRAMES = 12;

/**
 * Render `camera` until the bricks it needs are resident. Kiln picks bricks
 * from the rays it marched, so frames are rendered while waiting.
 */
async function waitForBricks(
  engine: KilnEngine,
  target: GPUTextureView,
  view: ReturnType<Camera["getViewParams"]>,
  timeoutMs = 120_000,
): Promise<void> {
  const t0 = performance.now();
  let frames = 0;
  let idle = 0;
  while (performance.now() - t0 < timeoutMs) {
    engine.update(view);
    engine.render(target, view);
    frames++;
    const s = engine.streamingManager.getStats();
    const settled =
      s.pendingCount === 0 && engine.streamingManager.baseLodLoaded;
    idle = settled ? idle + 1 : 0;
    if (frames >= MIN_SETTLE_FRAMES && idle >= 3) return;
    await engine.device.queue.onSubmittedWorkDone();
    await yieldTask(16);
  }
}

/** Deterministic uniform [0, 1) generator (LCG) — reproducible view layouts. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Deterministic random rotation (row-major 3×3) from a seed. */
function randomRotation(seed: number): number[] {
  const r = lcg(seed);
  // Uniform random unit quaternion (Shoemake)
  const u1 = r();
  const u2 = r() * 2 * Math.PI;
  const u3 = r() * 2 * Math.PI;
  const a = Math.sqrt(1 - u1);
  const b = Math.sqrt(u1);
  return quatToRot(
    b * Math.cos(u3),
    a * Math.sin(u2),
    a * Math.cos(u2),
    b * Math.sin(u3),
  );
}

/** Training views: rendered targets and the cameras they were taken with. */
export interface CapturedViews {
  /**
   * rgba16float array texture, premultiplied, one layer per view (usage
   * TEXTURE_BINDING | COPY_SRC)
   */
  texture: GPUTexture;
  cameras: SplatCamera[];
  width: number;
  height: number;
}

/**
 * Generated training cameras around the clip box: a whole-volume orbit, a
 * closer orbit, and close-ups aimed at random points inside.
 */
function planViews(
  host: SplatBuildHost,
  opts: SplatBuildOptions,
): OrbitState[] {
  const n = Math.max(4, Math.min(256, opts.views ?? 96));
  const renderer = host.engine.renderer;
  const { lo, hi } = clipToWorld(
    host.engine.config.normalizedSize,
    renderer.clipMin,
    renderer.clipMax,
  );
  const center: [number, number, number] = [
    (lo[0]! + hi[0]!) / 2,
    (lo[1]! + hi[1]!) / 2,
    (lo[2]! + hi[2]!) / 2,
  ];
  const radius =
    0.5 * Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!);
  const fov = Math.PI / 4;
  const outer = Math.min(9.5, (radius / Math.sin(fov / 2)) * 1.02);
  const inner = Math.max(radius * 1.05, outer * 0.55);
  // Three kinds of views: whole-volume orbit, a closer orbit, and close-up
  // "detail" views aimed at random points inside the volume — the latter are
  // what lets training recover full-resolution structure.
  const nMid = Math.floor(n * 0.3);
  const nDetail = Math.floor(n * 0.3);
  const seed = opts.viewSeed ?? 12345;
  // Held-out views (viewSeed set) use the same layout under a random rotation.
  const rot = opts.viewSeed === undefined ? null : randomRotation(seed);
  const turn = (d: [number, number, number]): [number, number, number] =>
    rot
      ? [
          rot[0]! * d[0] + rot[1]! * d[1] + rot[2]! * d[2],
          rot[3]! * d[0] + rot[4]! * d[1] + rot[5]! * d[2],
          rot[6]! * d[0] + rot[7]! * d[1] + rot[8]! * d[2],
        ]
      : d;
  const rand = lcg(seed);
  const shots: Array<{
    dir: [number, number, number];
    dist: number;
    target: [number, number, number];
  }> = [];
  for (const dir of sphereDirections(n - nMid - nDetail))
    shots.push({ dir: turn(dir), dist: outer, target: center });
  for (const dir of sphereDirections(nMid))
    shots.push({ dir: turn(dir), dist: inner, target: center });
  const detailDist = Math.max(0.12, outer * 0.32);
  const detailDirs = sphereDirections(nDetail);
  const detail: typeof shots = [];
  for (let i = 0; i < nDetail; i++) {
    const t: [number, number, number] = [
      lo[0]! + (0.15 + 0.7 * rand()) * (hi[0]! - lo[0]!),
      lo[1]! + (0.15 + 0.7 * rand()) * (hi[1]! - lo[1]!),
      lo[2]! + (0.15 + 0.7 * rand()) * (hi[2]! - lo[2]!),
    ];
    detail.push({
      dir: turn(detailDirs[(i * 7) % nDetail]!),
      dist: detailDist,
      target: t,
    });
  }
  // Visit close-ups in spatial (Morton) order so consecutive views reuse the
  // same full-resolution bricks instead of thrashing the caches.
  const morton = (t: [number, number, number]) => {
    let code = 0;
    const q = [0, 1, 2].map((a) =>
      Math.min(
        1023,
        Math.max(
          0,
          Math.floor(
            ((t[a]! - lo[a]!) / Math.max(1e-9, hi[a]! - lo[a]!)) * 1024,
          ),
        ),
      ),
    );
    for (let bit = 9; bit >= 0; bit--)
      for (let a = 0; a < 3; a++) code = code * 2 + ((q[a]! >> bit) & 1);
    return code;
  };
  detail.sort((a, b) => morton(a.target) - morton(b.target));
  shots.push(...detail);
  return shots.map(({ dir, dist, target }) => {
    const { rotX, rotY } = orbitForDirection(dir, host.camera.getUpAxis());
    return [rotX, rotY, dist, target[0], target[1], target[2]];
  });
}

/** Stage 2: render training views: the generated ones, then `extraViews`. */
export function captureViews(
  host: SplatBuildHost,
  opts: SplatBuildOptions = {},
): Promise<CapturedViews> {
  return captureViewsAt(
    host,
    [...planViews(host, opts), ...(opts.extraViews ?? [])],
    opts,
  );
}

/** Render the volume from each camera into a training-target texture array. */
export async function captureViewsAt(
  host: SplatBuildHost,
  orbits: readonly OrbitState[],
  opts: SplatBuildOptions = {},
): Promise<CapturedViews> {
  const { engine, camera } = host;
  const device = engine.device;
  const shots = orbits.slice(0, device.limits.maxTextureArrayLayers);
  const n = shots.length;
  const size = Math.max(64, opts.resolution ?? 640);
  const settle = Math.max(1, opts.settleFrames ?? 16);
  const onProgress = opts.onProgress ?? (() => {});

  // The engine renders straight into the layers (its output format is
  // rgba16float and its background transparent).
  const texture = device.createTexture({
    label: "splat:train-targets",
    size: [size, size, n],
    format: "rgba16float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC,
  });
  const cameras: SplatCamera[] = [];
  engine.setSize(size, size);
  for (const [index, shot] of shots.entries()) {
    throwIfAborted(opts.signal);
    camera.setOrbitState(shot);
    // Never "interacting": that drops to a lower render scale.
    const view = { ...camera.getViewParams(size, size), interacting: false };
    const layer = texture.createView({
      dimension: "2d",
      baseArrayLayer: index,
      arrayLayerCount: 1,
    });
    await waitForBricks(engine, layer, view);
    engine.renderer.resetAccumulation();
    for (let f = 0; f < settle; f++) {
      engine.render(layer, view);
      if (f % 4 === 3) await yieldTask();
    }
    cameras.push(intrinsicsFor(Float32Array.from(view.view), size, size));
    // Keep the CPU in step with the GPU: without this, submission runs far
    // ahead and progress reads "done" while rendering is still queued.
    await device.queue.onSubmittedWorkDone();
    onProgress({
      stage: "capture",
      fraction: (index + 1) / n,
      label: `Rendering training view ${index + 1}/${n}`,
    });
  }
  return { texture, cameras, width: size, height: size };
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

/** Stage 3: photometric refinement with Brush. */
export function trainCloud(
  host: SplatBuildHost,
  init: GaussianCloud,
  views: CapturedViews,
  opts: SplatBuildOptions = {},
): Promise<GaussianCloud> {
  const iterations = Math.max(1, opts.iterations ?? 3000);
  return trainWithBrush(host.engine.device, init, views, {
    iterations,
    maxSplats: Math.ceil((opts.targetCount ?? init.count) * 2),
    signal: opts.signal,
    onProgress: (fraction, label, gaussians) =>
      opts.onProgress?.({ stage: "train", fraction, label, gaussians }),
  });
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

/**
 * Mean PSNR (dB) of `cloud` rendered from each captured view against the
 * volume render, both composited over `background` as displayed.
 */
export async function evaluateCloud(
  device: GPUDevice,
  cloud: GaussianCloud,
  views: CapturedViews,
  background: ArrayLike<number>,
): Promise<number> {
  const r = new SplatRasterizer(device);
  const { width: w, height: h } = views;
  const imageBytes = w * h * 16;
  const rowBytes = Math.ceil((w * 8) / 256) * 256;
  const targetBytes = rowBytes * h;
  const paramBytes = cloud.params.byteLength;
  const params = createBuffer(
    device,
    "splat:eval-params",
    paramBytes,
    STORAGE_RW,
  );
  device.queue.writeBuffer(
    params,
    0,
    cloud.params.buffer,
    cloud.params.byteOffset,
    paramBytes,
  );
  // One readback per view: splat image, volume target, intersection total.
  const staging = createBuffer(
    device,
    "splat:eval-readback",
    imageBytes + targetBytes + 4,
    READBACK,
  );
  const f16 = getFloat16ToFloat32Lut();
  const bg = [background[0] ?? 0, background[1] ?? 0, background[2] ?? 0];
  const psnrOf = (data: ArrayBuffer): number | null => {
    const m = new Uint32Array(data, imageBytes + targetBytes, 1)[0]!;
    // Intersections overflowed: grow and render the view again.
    if (r.reserveIntersections(m)) return null;
    const img = new Float32Array(data, 0, w * h * 4);
    const tgt = new Uint16Array(data, imageBytes, targetBytes / 2);
    let se = 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const p = (y * w + x) * 4;
        const t = (y * rowBytes) / 2 + x * 4;
        const pa = img[p + 3]!;
        const ta = f16[tgt[t + 3]!]!;
        for (let c = 0; c < 3; c++) {
          const a = Math.min(1, img[p + c]! + bg[c]! * (1 - pa));
          const b = Math.min(1, f16[tgt[t + c]!]! + bg[c]! * (1 - ta));
          se += (a - b) * (a - b);
        }
      }
    const mse = se / (w * h * 3);
    return 10 * Math.log10(1 / Math.max(1e-10, mse));
  };
  let psnrSum = 0;
  try {
    r.resize(w, h);
    r.mipFilter = cloud.antialiased ?? false;
    r.setGaussians(params, cloud.count);
    for (let v = 0; v < views.cameras.length; v++) {
      const enc = device.createCommandEncoder();
      r.encodeForward(enc, views.cameras[v]!);
      enc.copyBufferToBuffer(r.image, 0, staging, 0, imageBytes);
      enc.copyTextureToBuffer(
        { texture: views.texture, origin: [0, 0, v] },
        { buffer: staging, offset: imageBytes, bytesPerRow: rowBytes },
        [w, h, 1],
      );
      r.copyIntersectionTotal(enc, staging, imageBytes + targetBytes);
      device.queue.submit([enc.finish()]);
      const psnr = await withMapped(staging, psnrOf);
      if (psnr === null) v--;
      else psnrSum += psnr;
    }
  } finally {
    r.destroy();
    staging.destroy();
    params.destroy();
  }
  return psnrSum / views.cameras.length;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/** Fit (+ train, + evaluate) inside the capture engine's clip box. */
async function buildSplats(
  host: SplatBuildHost,
  opts: SplatBuildOptions,
): Promise<GaussianCloud> {
  let views: CapturedViews | null = null;
  let evalViews: CapturedViews | null = null;
  try {
    let cloud = await fitVolume(host, opts);
    const train = opts.train !== false && cloud.count > 0;
    const nEval = cloud.count > 0 ? (opts.evaluateViews ?? 0) : 0;
    const scoreAt = cloud.count > 0 ? (opts.evaluateAt ?? []) : [];
    if (train) views = await captureViews(host, opts);
    const evalOpts: SplatBuildOptions = {
      ...opts,
      views: nEval,
      viewSeed: 7919,
      onProgress: (p) =>
        opts.onProgress?.({
          ...p,
          label: `Held-out ${p.label.toLowerCase()}`,
        }),
    };
    if (scoreAt.length > 0) {
      evalViews = await captureViewsAt(host, scoreAt, evalOpts);
    } else if (nEval > 0) {
      evalViews = await captureViews(host, evalOpts);
    }
    if (views) cloud = await trainCloud(host, cloud, views, opts);
    const measured = evalViews
      ? await evaluateCloud(
          host.engine.device,
          cloud,
          evalViews,
          DISPLAY_BACKGROUND,
        )
      : undefined;
    // Reported only when it is a number (callers store it as one).
    const psnr = Number.isFinite(measured) ? measured : undefined;
    opts.onProgress?.({
      stage: "done",
      fraction: 1,
      label:
        `${cloud.count.toLocaleString()} Gaussians` +
        (psnr !== undefined ? ` · ${psnr.toFixed(1)} dB PSNR vs volume` : ""),
      gaussians: cloud.count,
      psnr,
    });
    return cloud;
  } finally {
    views?.texture.destroy();
    evalViews?.texture.destroy();
  }
}

/**
 * Turn the volume `viewer` shows — its channels, colours, windows and density
 * (each overridable) — into Gaussian splats. Fitting and capture run on a
 * headless KilnEngine on the viewer's GPU device, streaming from
 * `openProvider()` (a second provider of the same volume and channels), so
 * the viewer itself is never touched.
 */
export async function generateSplats(
  viewer: KilnViewer,
  openProvider: () => DataProvider,
  opts: GenerateSplatsOptions = {},
): Promise<GaussianCloud> {
  const shown = viewer.renderer;
  const mode =
    opts.renderMode ??
    (shown.volumeRenderMode === "mip" || shown.volumeRenderMode === "iso"
      ? shown.volumeRenderMode
      : "dvr");
  const engine = await KilnEngine.create(viewer.device, openProvider(), {
    pyramid: "legacy",
    outputFormat: "rgba16float",
    atlasBudgetBytes: CAPTURE_ATLAS_BYTES,
    renderScale: 1,
  });
  try {
    const r: Renderer = engine.renderer;
    r.syncLook(shown);
    r.volumeRenderMode = mode;
    r.densityScale =
      mode === "mip" ? MIP_FIT_DENSITY : (opts.density ?? shown.densityScale);
    if (opts.region) {
      r.clipMin.set(opts.region.min);
      r.clipMax.set(opts.region.max);
    }
    r.showWireframe = false;
    r.showAxis = false;
    r.background.set([0, 0, 0, 0]);
    const camera = new Camera(document.createElement("canvas"));
    camera.setUpAxis(viewer.camera.getUpAxis() as UpAxis);
    const host: SplatBuildHost = { engine, camera };

    // Channel slots to build one by one (null: all together, as displayed).
    const colors = Float32Array.from(r.channelColors);
    const channels: Array<number | null> = [];
    if (opts.perChannel && r.numChannels > 1) {
      for (let ch = 0; ch < r.numChannels; ch++)
        if ((colors[ch * 4 + 3] ?? 0) > 0) channels.push(ch);
    }
    if (channels.length === 0) channels.push(null);
    const targetCount = Math.ceil(
      (opts.targetCount ?? 1_000_000) / channels.length,
    );
    const built: GaussianCloud[] = [];
    for (const [i, ch] of channels.entries()) {
      opts.onTile?.(i, channels.length);
      if (ch !== null) {
        // This channel alone, in white: colour is applied at display.
        r.channelColors.fill(0);
        r.channelColors.set([1, 1, 1, 1], ch * 4);
      }
      r.resetAccumulation();
      built.push(await buildSplats(host, { ...opts, targetCount }));
    }
    return channels[0] === null
      ? built[0]!
      : mergeClouds(
          built,
          channels as number[],
          mode === "mip" ? "add" : "over",
        );
  } finally {
    engine.dispose();
  }
}
