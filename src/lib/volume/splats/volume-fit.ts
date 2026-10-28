/**
 * Direct volume → Gaussian fitting (no images, no training).
 *
 * Every voxel is classified exactly like Kiln's DVR shader does it (per-channel
 * window → additive colour, max-density extinction; or the 1D transfer
 * function for single-channel data), giving an emission colour c and an
 * extinction coefficient σ per voxel. Mass m = σ·V is accumulated with colour
 * and spatial moments into an octree of 2³ / 4³ / 8³ voxel nodes. A cut
 * through the tree is chosen by error threshold (bisected to hit a target
 * Gaussian count), and each selected node becomes one moment-matched
 * anisotropic Gaussian:
 *
 *   μ = centroid, Σ = mass-weighted covariance (+ voxel-extent regulariser),
 *   colour = mass-weighted emission, opacity = 1 − exp(−M / (2π·|Σ|^{1/3})).
 *
 * The last formula uses the fact that the optical depth of a Gaussian density
 * integrated over the image plane equals its total mass, so a splat with that
 * peak opacity absorbs the same amount of light as the voxels it replaces
 * (exactly for thin media, approximately otherwise). The result is an
 * excellent initialisation for photometric refinement.
 */

import { LOGICAL, PHYSICAL } from "../imaris/ims-layout";
import { getFloat16ToFloat32Lut } from "../imaris/util/float16";
import { createCloud, type GaussianCloud, setGaussian } from "./cloud";

/** Kiln renders at most this many channels. */
const MAX_CH = 4;

/** Rendering state the fit must reproduce (mirrors the DVR uniforms). */
export interface ClassifyParams {
  numChannels: number;
  /** RGBA per channel (alpha = visibility/intensity weight) */
  channelColors: ArrayLike<number>;
  channelWindowCenter: ArrayLike<number>;
  channelWindowWidth: ArrayLike<number>;
  floatMin: number;
  floatMax: number;
  /** single-channel window */
  windowCenter: number;
  windowWidth: number;
  /** single-channel transfer function, 256 × RGBA (0–1) */
  tf?: Float32Array;
  densityScale: number;
  /** max(dataset dims at LOD 0) — DVR scales extinction by it */
  maxDim0: number;
  clipMin: ArrayLike<number>;
  clipMax: ArrayLike<number>;
}

export interface FitGrid {
  /** voxel dims at the fitting LOD */
  dims: [number, number, number];
  normalizedSize: [number, number, number];
  /**
   * LOD-0 dims and the LOD's 2^L scale. The renderer places LOD-L voxel j
   * over LOD-0 voxels [j·2^L, (j+1)·2^L), so voxel size is
   * normalizedSize / dims0 · 2^L (not normalizedSize / dims when dims0 isn't
   * divisible by 2^L). Default: dims / 1.
   */
  dims0?: [number, number, number];
  lodScale?: number;
}

// Node accumulator layout (floats)
const N_STRIDE = 16;
const A_M = 0; // Σ m
const A_S1 = 1; // Σ m d (3)  — d relative to node centre
const A_S2 = 4; // Σ m d dᵀ  xx yy zz xy xz yz (6)
const A_C1 = 10; // Σ m c (3)
const A_C2 = 13; // Σ m |c|²
const A_N = 14; // occupied voxel count
const A_Q = 15; // Σ m σ

export interface FitOptions {
  /** Target number of Gaussians (default 1M) */
  targetCount?: number;
  /** Weight of the empty-space term in the split error */
  emptyWeight?: number;
  /** Weight of the density-heterogeneity term */
  densityWeight?: number;
  /** Weight of the node-size term (prefers splitting big nodes) */
  sizeWeight?: number;
  /** Minimum opacity to keep a Gaussian */
  minOpacity?: number;
}

export class VolumeFitter {
  readonly dims: [number, number, number];
  readonly h: [number, number, number];
  private readonly origin: [number, number, number];
  private readonly nsize: [number, number, number];
  /** finest nodes are 2³ voxels */
  private readonly grid1: [number, number, number];
  private nodes1: Float32Array;
  private readonly f16 = getFloat16ToFloat32Lut();
  /** Raw voxel value → window weight, per channel and storage type. */
  private readonly weightLuts = new Map<string, Float64Array>();
  occupiedVoxels = 0;

  /** Nodes per brick axis (a 64³ brick covers 32³ 2³-nodes). */
  static readonly BLOCK = LOGICAL / 2;

  /**
   * @param allocate false for classify-only instances (worker pools) that
   *   never hold the full node array.
   */
  constructor(
    grid: FitGrid,
    private readonly params: ClassifyParams,
    allocate = true,
  ) {
    this.dims = grid.dims;
    const d0 = grid.dims0 ?? grid.dims;
    const k = grid.dims0 ? (grid.lodScale ?? 1) : 1;
    this.h = [
      (grid.normalizedSize[0] / d0[0]) * k,
      (grid.normalizedSize[1] / d0[1]) * k,
      (grid.normalizedSize[2] / d0[2]) * k,
    ];
    this.nsize = grid.normalizedSize;
    this.origin = [
      -grid.normalizedSize[0] / 2,
      -grid.normalizedSize[1] / 2,
      -grid.normalizedSize[2] / 2,
    ];
    this.grid1 = [
      Math.ceil(grid.dims[0] / 2),
      Math.ceil(grid.dims[1] / 2),
      Math.ceil(grid.dims[2] / 2),
    ];
    this.nodes1 = new Float32Array(
      allocate ? this.grid1[0] * this.grid1[1] * this.grid1[2] * N_STRIDE : 0,
    );
  }

  /** Hand the node array over (e.g. to transfer it to a worker). */
  takeNodes(): Float32Array {
    const n = this.nodes1;
    this.nodes1 = new Float32Array(0);
    return n;
  }

  /** Adopt node moments accumulated elsewhere (see classifyBrick/addBlock). */
  setNodes(nodes: Float32Array, occupied: number): void {
    this.nodes1 = nodes;
    this.occupiedVoxels = occupied;
  }

  /** Accumulate one brick (classify + merge). */
  addBrick(
    bx: number,
    by: number,
    bz: number,
    channels: ArrayLike<Uint16Array | Uint8Array | null | undefined>,
  ): void {
    const r = this.classifyBrick(bx, by, bz, channels);
    if (r) this.addBlock(bx, by, bz, r.block, r.occupied);
  }

  /** Copy a brick's node block (BLOCK³ × N_STRIDE) into the full node array. */
  addBlock(
    bx: number,
    by: number,
    bz: number,
    block: Float32Array,
    occupied: number,
  ): void {
    const B = VolumeFitter.BLOCK;
    const [g1x, g1y, g1z] = this.grid1;
    const nodes = this.nodes1;
    for (let z = 0; z < B; z++) {
      const nz = bz * B + z;
      if (nz >= g1z) break;
      for (let y = 0; y < B; y++) {
        const ny = by * B + y;
        if (ny >= g1y) break;
        const nx0 = bx * B;
        const len = Math.min(B, g1x - nx0);
        if (len <= 0) continue;
        const src = (z * B + y) * B * N_STRIDE;
        const dst = ((nz * g1y + ny) * g1x + nx0) * N_STRIDE;
        nodes.set(block.subarray(src, src + len * N_STRIDE), dst);
      }
    }
    this.occupiedVoxels += occupied;
  }

  /** Memory needed for a fit at these dims (bytes). */
  static estimateBytes(dims: [number, number, number]): number {
    return (
      Math.ceil(dims[0] / 2) *
      Math.ceil(dims[1] / 2) *
      Math.ceil(dims[2] / 2) *
      N_STRIDE *
      4 *
      1.15
    );
  }

  /**
   * Classify one brick into a local block of 2³-node moments
   * (BLOCK³ × N_STRIDE floats). `channels[ch]` is the 66³ brick (float16 bits
   * or 8-bit unorm) per atlas channel; null/undefined = all zero. Pure — safe
   * to run on many workers in parallel.
   */
  classifyBrick(
    bx: number,
    by: number,
    bz: number,
    channels: ArrayLike<Uint16Array | Uint8Array | null | undefined>,
  ): { block: Float32Array; occupied: number } | null {
    const p = this.params;
    const nch = Math.min(MAX_CH, Math.max(1, p.numChannels));
    const multi = nch > 1;
    const lower = new Float32Array(MAX_CH);
    const invW = new Float32Array(MAX_CH);
    const vis: number[] = [];
    for (let ch = 0; ch < nch; ch++) {
      const ww = Math.max(1e-4, p.channelWindowWidth[ch] ?? 1);
      lower[ch] = (p.channelWindowCenter[ch] ?? 0.5) - ww * 0.5;
      invW[ch] = 1 / ww;
      if (!multi || (p.channelColors[ch * 4 + 3] ?? 0) > 0) {
        if (channels[ch]) vis.push(ch);
      }
    }
    if (vis.length === 0) return null;
    const B = VolumeFitter.BLOCK;
    const nodes = new Float32Array(B * B * B * N_STRIDE);
    let occupied = 0;
    const sigmaScale = p.densityScale * p.maxDim0 * 0.5;
    const [dx, dy, dz] = this.dims;
    const [hx, hy, hz] = this.h;
    const V = hx * hy * hz;
    const tf = p.tf;
    // Visible channels' data, weight tables and colours (rgb, a).
    const data = vis.map((ch) => channels[ch]!);
    const luts = vis.map((ch, k) =>
      multi
        ? this.weightLut(data[k]!, `${ch}`, lower[ch]!, invW[ch]!)
        : this.weightLut(
            data[k]!,
            "single",
            p.windowCenter - p.windowWidth / 2,
            1 / Math.max(1e-3, p.windowWidth),
          ),
    );
    const colour = (ch: number, c: number) =>
      p.channelColors[ch * 4 + c] ?? (c === 3 ? 1 : 0);
    const cr = vis.map((ch) => colour(ch, 0));
    const cg = vis.map((ch) => colour(ch, 1));
    const cb = vis.map((ch) => colour(ch, 2));
    const ca = vis.map((ch) => colour(ch, 3));
    const nv = vis.length;
    const [cminX, cminY, cminZ] = [0, 1, 2].map((a) => p.clipMin[a] ?? 0);
    const [cmaxX, cmaxY, cmaxZ] = [0, 1, 2].map((a) => p.clipMax[a] ?? 1);

    for (let lz = 0; lz < LOGICAL; lz++) {
      const vz = bz * LOGICAL + lz;
      if (vz >= dz) break;
      const uz = ((vz + 0.5) * hz) / this.nsize[2];
      if (uz < cminZ! || uz > cmaxZ!) continue;
      for (let ly = 0; ly < LOGICAL; ly++) {
        const vy = by * LOGICAL + ly;
        if (vy >= dy) break;
        const uy = ((vy + 0.5) * hy) / this.nsize[1];
        if (uy < cminY! || uy > cmaxY!) continue;
        for (let lx = 0; lx < LOGICAL; lx++) {
          const vx = bx * LOGICAL + lx;
          if (vx >= dx) break;
          const ux = ((vx + 0.5) * hx) / this.nsize[0];
          if (ux < cminX! || ux > cmaxX!) continue;
          const idx = ((lz + 1) * PHYSICAL + (ly + 1)) * PHYSICAL + (lx + 1);
          let r = 0;
          let g = 0;
          let b = 0;
          let dens = 0;
          if (multi) {
            for (let k = 0; k < nv; k++) {
              const d = luts[k]![data[k]![idx]!]!;
              if (d <= 0) continue;
              const a = ca[k]!;
              r += d * cr[k]! * a;
              g += d * cg[k]! * a;
              b += d * cb[k]! * a;
              if (d > dens) dens = d;
            }
            if (dens <= 0.01) continue;
            r /= dens;
            g /= dens;
            b /= dens;
          } else {
            const w = luts[0]![data[0]![idx]!]!;
            if (w <= 0.01) continue;
            if (tf) {
              const t = Math.min(255, Math.max(0, w * 255));
              const i0 = Math.floor(t);
              const i1 = Math.min(255, i0 + 1);
              const f = t - i0;
              r = tf[i0 * 4]! * (1 - f) + tf[i1 * 4]! * f;
              g = tf[i0 * 4 + 1]! * (1 - f) + tf[i1 * 4 + 1]! * f;
              b = tf[i0 * 4 + 2]! * (1 - f) + tf[i1 * 4 + 2]! * f;
              dens = tf[i0 * 4 + 3]! * (1 - f) + tf[i1 * 4 + 3]! * f;
            } else {
              r = g = b = w;
              dens = w;
            }
            if (dens <= 0) continue;
          }
          const sigma = dens * sigmaScale;
          const m = sigma * V;
          // offset of the voxel centre from its 2³ node centre (±h/2)
          const ox = ((vx & 1) - 0.5) * hx;
          const oy = ((vy & 1) - 0.5) * hy;
          const oz = ((vz & 1) - 0.5) * hz;
          const n = (((lz >> 1) * B + (ly >> 1)) * B + (lx >> 1)) * N_STRIDE;
          nodes[n + A_M] = nodes[n + A_M]! + m;
          nodes[n + A_S1] = nodes[n + A_S1]! + m * ox;
          nodes[n + A_S1 + 1] = nodes[n + A_S1 + 1]! + m * oy;
          nodes[n + A_S1 + 2] = nodes[n + A_S1 + 2]! + m * oz;
          nodes[n + A_S2] = nodes[n + A_S2]! + m * ox * ox;
          nodes[n + A_S2 + 1] = nodes[n + A_S2 + 1]! + m * oy * oy;
          nodes[n + A_S2 + 2] = nodes[n + A_S2 + 2]! + m * oz * oz;
          nodes[n + A_S2 + 3] = nodes[n + A_S2 + 3]! + m * ox * oy;
          nodes[n + A_S2 + 4] = nodes[n + A_S2 + 4]! + m * ox * oz;
          nodes[n + A_S2 + 5] = nodes[n + A_S2 + 5]! + m * oy * oz;
          nodes[n + A_C1] = nodes[n + A_C1]! + m * r;
          nodes[n + A_C1 + 1] = nodes[n + A_C1 + 1]! + m * g;
          nodes[n + A_C1 + 2] = nodes[n + A_C1 + 2]! + m * b;
          nodes[n + A_C2] = nodes[n + A_C2]! + m * (r * r + g * g + b * b);
          nodes[n + A_N] = nodes[n + A_N]! + 1;
          nodes[n + A_Q] = nodes[n + A_Q]! + m * sigma;
          occupied++;
        }
      }
    }
    return occupied > 0 ? { block: nodes, occupied } : null;
  }

  /**
   * Window weight of every raw value of `data`'s type (8-bit unorm or
   * float16 bits): clamp((normalised value − lo) · inv, 0, 1), as the DVR
   * shader applies it. Built once per channel and type.
   */
  private weightLut(
    data: Uint16Array | Uint8Array,
    channel: string,
    lo: number,
    inv: number,
  ): Float64Array {
    const u8 = data instanceof Uint8Array;
    const key = `${channel}:${u8 ? 8 : 16}`;
    let lut = this.weightLuts.get(key);
    if (lut) return lut;
    const p = this.params;
    const floatInv = 1 / Math.max(1e-4, p.floatMax - p.floatMin);
    lut = new Float64Array(u8 ? 256 : 65536);
    for (let v = 0; v < lut.length; v++) {
      const raw = u8 ? v / 255 : this.f16[v]!;
      const norm = Math.min(1, Math.max(0, (raw - p.floatMin) * floatInv));
      lut[v] = Math.min(1, Math.max(0, (norm - lo) * inv));
    }
    this.weightLuts.set(key, lut);
    return lut;
  }

  /** Build coarser node levels (4³, 8³) by merging children with parallel-axis shifts. */
  private buildLevels(): Array<{
    grid: [number, number, number];
    nodes: Float32Array;
    size: number;
  }> {
    const levels = [{ grid: this.grid1, nodes: this.nodes1, size: 2 }];
    for (let l = 1; l < 3; l++) {
      const prev = levels[l - 1]!;
      const grid: [number, number, number] = [
        Math.ceil(prev.grid[0] / 2),
        Math.ceil(prev.grid[1] / 2),
        Math.ceil(prev.grid[2] / 2),
      ];
      const nodes = new Float32Array(grid[0] * grid[1] * grid[2] * N_STRIDE);
      const childSize = prev.size;
      for (let cz = 0; cz < prev.grid[2]; cz++)
        for (let cy = 0; cy < prev.grid[1]; cy++)
          for (let cx = 0; cx < prev.grid[0]; cx++) {
            const c = ((cz * prev.grid[1] + cy) * prev.grid[0] + cx) * N_STRIDE;
            const M = prev.nodes[c + A_M]!;
            if (M <= 0) continue;
            const pn =
              (((cz >> 1) * grid[1] + (cy >> 1)) * grid[0] + (cx >> 1)) *
              N_STRIDE;
            // child centre − parent centre = ±childSize·h/2
            const d0 = ((cx & 1) - 0.5) * childSize * this.h[0];
            const d1 = ((cy & 1) - 0.5) * childSize * this.h[1];
            const d2 = ((cz & 1) - 0.5) * childSize * this.h[2];
            const s0 = prev.nodes[c + A_S1]!;
            const s1 = prev.nodes[c + A_S1 + 1]!;
            const s2 = prev.nodes[c + A_S1 + 2]!;
            nodes[pn + A_M] = nodes[pn + A_M]! + M;
            nodes[pn + A_S1] = nodes[pn + A_S1]! + s0 + M * d0;
            nodes[pn + A_S1 + 1] = nodes[pn + A_S1 + 1]! + s1 + M * d1;
            nodes[pn + A_S1 + 2] = nodes[pn + A_S1 + 2]! + s2 + M * d2;
            nodes[pn + A_S2] =
              nodes[pn + A_S2]! +
              prev.nodes[c + A_S2]! +
              2 * s0 * d0 +
              M * d0 * d0;
            nodes[pn + A_S2 + 1] =
              nodes[pn + A_S2 + 1]! +
              prev.nodes[c + A_S2 + 1]! +
              2 * s1 * d1 +
              M * d1 * d1;
            nodes[pn + A_S2 + 2] =
              nodes[pn + A_S2 + 2]! +
              prev.nodes[c + A_S2 + 2]! +
              2 * s2 * d2 +
              M * d2 * d2;
            nodes[pn + A_S2 + 3] =
              nodes[pn + A_S2 + 3]! +
              prev.nodes[c + A_S2 + 3]! +
              s0 * d1 +
              s1 * d0 +
              M * d0 * d1;
            nodes[pn + A_S2 + 4] =
              nodes[pn + A_S2 + 4]! +
              prev.nodes[c + A_S2 + 4]! +
              s0 * d2 +
              s2 * d0 +
              M * d0 * d2;
            nodes[pn + A_S2 + 5] =
              nodes[pn + A_S2 + 5]! +
              prev.nodes[c + A_S2 + 5]! +
              s1 * d2 +
              s2 * d1 +
              M * d1 * d2;
            for (let k = A_C1; k < N_STRIDE; k++)
              nodes[pn + k] = nodes[pn + k]! + prev.nodes[c + k]!;
          }
      levels.push({ grid, nodes, size: childSize * 2 });
    }
    return levels;
  }

  /** Produce the Gaussian cloud. */
  finish(opts: FitOptions = {}): GaussianCloud {
    const target = Math.max(1, opts.targetCount ?? 1_000_000);
    const emptyW = opts.emptyWeight ?? 0.5;
    const densW = opts.densityWeight ?? 0.25;
    const sizeW = opts.sizeWeight ?? 0.002;
    const minOpacity = opts.minOpacity ?? 0.004;
    const levels = this.buildLevels();
    const V = this.h[0] * this.h[1] * this.h[2];

    // Per-node split error for levels 1..2 (the finest level never splits).
    const errors = levels.map(({ nodes, size }, li) => {
      if (li === 0) return null;
      const count = nodes.length / N_STRIDE;
      const e = new Float32Array(count);
      const nTot = size * size * size;
      for (let i = 0; i < count; i++) {
        const b = i * N_STRIDE;
        const M = nodes[b + A_M]!;
        if (M <= 0) continue;
        const cr = nodes[b + A_C1]! / M;
        const cg = nodes[b + A_C1 + 1]! / M;
        const cb = nodes[b + A_C1 + 2]! / M;
        const colorVar = Math.max(
          0,
          nodes[b + A_C2]! / M - (cr * cr + cg * cg + cb * cb),
        );
        const nOcc = nodes[b + A_N]!;
        const empty = 1 - nOcc / nTot;
        const meanSigmaByMass = nodes[b + A_Q]! / M;
        const meanSigma = M / (nOcc * V);
        const hetero = Math.max(
          0,
          meanSigmaByMass / Math.max(1e-12, meanSigma) - 1,
        );
        // The size term spends spare budget on splitting large uniform nodes
        // (one Gaussian per big block is blurrier than several small ones).
        e[i] = M * (colorVar + emptyW * empty + densW * hetero + sizeW * size);
      }
      return e;
    });

    const thr = splitThreshold(levels, errors[1]!, errors[2]!, target);
    // Descend from the top level where error > thr.
    const selected: Array<[number, number]> = [];
    const rec = (li: number, x: number, y: number, z: number) => {
      const L = levels[li]!;
      if (x >= L.grid[0] || y >= L.grid[1] || z >= L.grid[2]) return;
      const i = (z * L.grid[1] + y) * L.grid[0] + x;
      if (L.nodes[i * N_STRIDE + A_M]! <= 0) return;
      if (li > 0 && errors[li]![i]! > thr) {
        for (let k = 0; k < 8; k++)
          rec(
            li - 1,
            x * 2 + (k & 1),
            y * 2 + ((k >> 1) & 1),
            z * 2 + (k >> 2),
          );
        return;
      }
      selected.push([li, i]);
    };
    const top = levels[2]!;
    for (let z = 0; z < top.grid[2]; z++)
      for (let y = 0; y < top.grid[1]; y++)
        for (let x = 0; x < top.grid[0]; x++) rec(2, x, y, z);
    const cloud = createCloud(selected.length);
    let out = 0;
    // Voxel-extent regulariser so single-voxel nodes aren't degenerate.
    const reg = [this.h[0] ** 2 / 4, this.h[1] ** 2 / 4, this.h[2] ** 2 / 4];
    for (const [li, i] of selected) {
      const L = levels[li]!;
      const b = i * N_STRIDE;
      const nd = L.nodes;
      const M = nd[b + A_M]!;
      const x = i % L.grid[0];
      const y = Math.floor(i / L.grid[0]) % L.grid[1];
      const z = Math.floor(i / (L.grid[0] * L.grid[1]));
      const cx = this.origin[0] + (x + 0.5) * L.size * this.h[0];
      const cy = this.origin[1] + (y + 0.5) * L.size * this.h[1];
      const cz = this.origin[2] + (z + 0.5) * L.size * this.h[2];
      const m0 = nd[b + A_S1]! / M;
      const m1 = nd[b + A_S1 + 1]! / M;
      const m2 = nd[b + A_S1 + 2]! / M;
      const cov = [
        nd[b + A_S2]! / M - m0 * m0 + reg[0]!,
        nd[b + A_S2 + 3]! / M - m0 * m1,
        nd[b + A_S2 + 4]! / M - m0 * m2,
        nd[b + A_S2 + 3]! / M - m0 * m1,
        nd[b + A_S2 + 1]! / M - m1 * m1 + reg[1]!,
        nd[b + A_S2 + 5]! / M - m1 * m2,
        nd[b + A_S2 + 4]! / M - m0 * m2,
        nd[b + A_S2 + 5]! / M - m1 * m2,
        nd[b + A_S2 + 2]! / M - m2 * m2 + reg[2]!,
      ];
      const { values, vectors } = symmetricEigen3(cov);
      const l0 = Math.max(values[0]!, 1e-12);
      const l1 = Math.max(values[1]!, 1e-12);
      const l2 = Math.max(values[2]!, 1e-12);
      const tau = M / (2 * Math.PI * Math.cbrt(l0 * l1 * l2));
      const opacity = Math.min(0.98, 1 - Math.exp(-tau));
      if (opacity < minOpacity) continue;
      const q = rotationToQuat(vectors);
      const clamp = (v: number) => Math.max(0.002, v);
      setGaussian(cloud, out++, {
        x: cx + m0,
        y: cy + m1,
        z: cz + m2,
        sx: Math.sqrt(l0),
        sy: Math.sqrt(l1),
        sz: Math.sqrt(l2),
        qw: q[0],
        qx: q[1],
        qy: q[2],
        qz: q[3],
        r: clamp(nd[b + A_C1]! / M),
        g: clamp(nd[b + A_C1 + 1]! / M),
        b: clamp(nd[b + A_C1 + 2]! / M),
        opacity,
      });
    }
    cloud.count = out;
    cloud.params = cloud.params.subarray(0, out * 16);
    return cloud;
  }
}

interface NodeLevel {
  grid: [number, number, number];
  nodes: Float32Array;
}

const bitsF32 = new Float32Array(1);
const bitsU32 = new Uint32Array(bitsF32.buffer);
/** Bits of +Infinity: larger patterns are NaN (errors are never negative). */
const INF_BITS = 0x7f800000;

/**
 * Smallest split threshold whose cut (descend where error > threshold) has
 * at most `target` Gaussians — 0 if the full cut fits, the largest error
 * (no splits) if not even the top level does.
 *
 * The cut size is a step function: N(thr) = T + Σ wᵢ·[keyᵢ > thr], with T
 * the non-empty top nodes and one event per non-empty node — a top node
 * splitting adds (non-empty children − 1), and a level-1 node, which splits
 * only once its parent has, adds (non-empty children − 1) below
 * min(its error, its parent's). Sorting the events gives N exactly.
 */
function splitThreshold(
  levels: readonly NodeLevel[],
  e1: Float32Array,
  e2: Float32Array,
  target: number,
): number {
  const [l0, l1, l2] = levels as [NodeLevel, NodeLevel, NodeLevel];
  const filled = (L: NodeLevel, i: number) => L.nodes[i * N_STRIDE + A_M]! > 0;
  /** Calls fn(node, parent node) for every non-empty node of `child`. */
  const eachFilled = (
    child: NodeLevel,
    parent: NodeLevel,
    fn: (i: number, parent: number) => void,
  ) => {
    const [gx, gy, gz] = child.grid;
    const [px, py] = parent.grid;
    let i = 0;
    for (let z = 0; z < gz; z++)
      for (let y = 0; y < gy; y++) {
        const row = ((z >> 1) * py + (y >> 1)) * px;
        for (let x = 0; x < gx; x++, i++)
          if (filled(child, i)) fn(i, row + (x >> 1));
      }
  };
  // Non-empty children per node.
  const kids1 = new Uint8Array(e1.length);
  const kids2 = new Uint8Array(e2.length);
  eachFilled(l0, l1, (_, up) => {
    kids1[up] = kids1[up]! + 1;
  });
  eachFilled(l1, l2, (_, up) => {
    kids2[up] = kids2[up]! + 1;
  });

  const bits1 = new Uint32Array(e1.buffer, e1.byteOffset, e1.length);
  const bits2 = new Uint32Array(e2.buffer, e2.byteOffset, e2.length);
  // Event = key bits · 8 + weight (weights are 0…7): sorts by key.
  const events: number[] = [];
  let T = 0;
  let total = 0;
  let maxBits = 0;
  const push = (key: number, w: number) => {
    // Error 0 never splits; NaN never compares greater.
    if (w <= 0 || key === 0 || key > INF_BITS) return;
    events.push(key * 8 + w);
    total += w;
  };
  for (let i = 0; i < e2.length; i++) {
    if (!filled(l2, i)) continue;
    T++;
    if (bits2[i]! <= INF_BITS) maxBits = Math.max(maxBits, bits2[i]!);
    push(bits2[i]!, kids2[i]! - 1);
  }
  eachFilled(l1, l2, (i, up) => {
    const own = bits1[i]!;
    const parent = bits2[up]!;
    if (own <= INF_BITS) maxBits = Math.max(maxBits, own);
    if (parent <= INF_BITS) push(Math.min(own, parent), kids1[i]! - 1);
  });
  if (T + total <= target) return 0;
  const sorted = Float64Array.from(events).sort();
  const toFloat = (b: number) => {
    bitsU32[0] = b;
    return bitsF32[0]!;
  };
  // Walk keys downwards: n = N(key) = T + weights of larger keys.
  let thr = toFloat(maxBits);
  let n = T;
  for (let i = sorted.length - 1; i >= 0 && n <= target; ) {
    const key = Math.floor(sorted[i]! / 8);
    thr = toFloat(key);
    for (; i >= 0 && Math.floor(sorted[i]! / 8) === key; i--)
      n += sorted[i]! % 8;
  }
  return thr;
}

/**
 * Jacobi eigen-decomposition of a symmetric 3×3 (row-major). Eigenvectors are
 * the columns of `vectors` (row-major), forming a proper rotation.
 */
export function symmetricEigen3(a: number[]): {
  values: [number, number, number];
  vectors: number[];
} {
  const m = a.slice();
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 12; sweep++) {
    const off = m[1]! ** 2 + m[2]! ** 2 + m[5]! ** 2;
    if (off < 1e-30) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      const apq = m[p * 3 + q]!;
      if (Math.abs(apq) < 1e-30) continue;
      const app = m[p * 3 + p]!;
      const aqq = m[q * 3 + q]!;
      const theta = (aqq - app) / (2 * apq);
      const t =
        Math.sign(theta || 1) /
        (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const mkp = m[k * 3 + p]!;
        const mkq = m[k * 3 + q]!;
        m[k * 3 + p] = c * mkp - s * mkq;
        m[k * 3 + q] = s * mkp + c * mkq;
      }
      for (let k = 0; k < 3; k++) {
        const mpk = m[p * 3 + k]!;
        const mqk = m[q * 3 + k]!;
        m[p * 3 + k] = c * mpk - s * mqk;
        m[q * 3 + k] = s * mpk + c * mqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p]!;
        const vkq = v[k * 3 + q]!;
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  // Ensure a right-handed basis (det = +1).
  const det =
    v[0]! * (v[4]! * v[8]! - v[5]! * v[7]!) -
    v[1]! * (v[3]! * v[8]! - v[5]! * v[6]!) +
    v[2]! * (v[3]! * v[7]! - v[4]! * v[6]!);
  if (det < 0) {
    v[2] = -v[2]!;
    v[5] = -v[5]!;
    v[8] = -v[8]!;
  }
  return { values: [m[0]!, m[4]!, m[8]!], vectors: v };
}

/** Rotation matrix (row-major) → unit quaternion (w, x, y, z). */
export function rotationToQuat(r: number[]): [number, number, number, number] {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = r as [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  const tr = m00 + m11 + m22;
  let w: number;
  let x: number;
  let y: number;
  let z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z) || 1;
  return [w / n, x / n, y / n, z / n];
}
