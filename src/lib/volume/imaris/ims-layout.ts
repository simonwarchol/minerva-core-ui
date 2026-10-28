/**
 * Pure planning helpers for Imaris bricks.
 *
 * Kiln's renderer assumes an isotropic pyramid: LOD L is exactly 2^L coarser
 * than LOD 0 on every axis. Imaris pyramids are not (Z is only halved once it
 * is long enough), so the provider exposes a *virtual* isotropic pyramid and
 * box-resamples each brick from the coarsest Imaris level that is still at
 * least as fine as the virtual level on every axis.
 *
 * Axis order here is always [x, y, z]; HDF5 datasets are stored [z, y, x].
 */

/** Kiln's brick geometry (64³ voxels stored with a one-voxel border); kiln doesn't export it. */
export const LOGICAL = 64;
export const PHYSICAL = 66;

/** Decoded (float16) chunk cache per brick worker… */
export const WORKER_CHUNK_BUDGET = 96 * 1024 * 1024;
/** …and across all of a provider's workers (more workers get less each). */
export const DECODED_CHUNK_BUDGET = 384 * 1024 * 1024;

export type Vec3 = [number, number, number];

export interface SourceLevel {
  /** Imaris ResolutionLevel index */
  index: number;
  /** ImageSizeX/Y/Z (valid voxels; datasets may be padded beyond this) */
  dims: Vec3;
}

export interface VirtualLevel {
  lod: number;
  dims: Vec3;
  brickGrid: Vec3;
  /** Imaris level sampled for this LOD */
  source: number;
  /** Source voxels per virtual voxel along each axis (≥ 1) */
  ratio: Vec3;
}

/**
 * Build the virtual pyramid: dims_L = ceil(dims0 / 2^L) until the whole
 * volume fits in one brick, each mapped to the best Imaris source level.
 */
export function planVirtualPyramid(sources: SourceLevel[]): VirtualLevel[] {
  if (sources.length === 0) return [];
  const base = sources[0]!.dims;
  const out: VirtualLevel[] = [];
  for (let lod = 0; lod < 16; lod++) {
    const f = 2 ** lod;
    const dims: Vec3 = [
      Math.max(1, Math.ceil(base[0] / f)),
      Math.max(1, Math.ceil(base[1] / f)),
      Math.max(1, Math.ceil(base[2] / f)),
    ];
    // Coarsest source whose downsampling factor ≤ 2^L on every axis.
    let best = sources[0]!;
    for (const s of sources) {
      const ok = [0, 1, 2].every(
        (a) => base[a]! / Math.max(1, s.dims[a]!) <= f * 1.001,
      );
      if (
        ok &&
        s.dims[0] * s.dims[1] * s.dims[2] <
          best.dims[0] * best.dims[1] * best.dims[2]
      ) {
        best = s;
      }
    }
    const ratio: Vec3 = [
      best.dims[0] / dims[0],
      best.dims[1] / dims[1],
      best.dims[2] / dims[2],
    ];
    out.push({
      lod,
      dims,
      brickGrid: [
        Math.ceil(dims[0] / LOGICAL),
        Math.ceil(dims[1] / LOGICAL),
        Math.ceil(dims[2] / LOGICAL),
      ],
      source: best.index,
      ratio,
    });
    if (dims[0] <= LOGICAL && dims[1] <= LOGICAL && dims[2] <= LOGICAL) break;
  }
  return out;
}

/** One axis of a brick's sampling (see axisSampling). */
export interface AxisSampling {
  ranges: Int32Array;
  lo: number;
  hi: number;
}

/**
 * For each output voxel along one axis of a brick (PHYSICAL samples incl.
 * the 1-voxel ghost border, edge-clamped), the [start, end) range of source
 * voxels it averages. Returns flat [start0, end0, start1, end1, …] plus the
 * overall source span.
 */
export function axisSampling(
  brick: number,
  virtualDim: number,
  sourceDim: number,
  ratio: number,
): AxisSampling {
  const ranges = new Int32Array(PHYSICAL * 2);
  let lo = Number.POSITIVE_INFINITY;
  let hi = 0;
  for (let i = 0; i < PHYSICAL; i++) {
    const v = Math.min(virtualDim - 1, Math.max(0, brick * LOGICAL - 1 + i));
    let s0 = Math.floor(v * ratio + 1e-6);
    let s1 = Math.ceil((v + 1) * ratio - 1e-6);
    s0 = Math.min(sourceDim - 1, Math.max(0, s0));
    s1 = Math.min(sourceDim, Math.max(s0 + 1, s1));
    ranges[i * 2] = s0;
    ranges[i * 2 + 1] = s1;
    lo = Math.min(lo, s0);
    hi = Math.max(hi, s1);
  }
  return { ranges, lo, hi };
}

/** Ghost-border face bit: axis 0–2 (x, y, z), side 0 (low) or 1 (high). */
export const faceBit = (axis: number, side: 0 | 1): number =>
  (1 + side) << (2 * axis);

/**
 * The ghost sample on one side repeats its interior neighbour (edge
 * clamping, as at the volume boundary), so the source span stops at the
 * interior.
 */
export function clampGhost(s: AxisSampling, side: 0 | 1): AxisSampling {
  const ranges = s.ranges.slice();
  const ghost = side === 0 ? 0 : PHYSICAL - 1;
  const inner = side === 0 ? 1 : PHYSICAL - 2;
  ranges[ghost * 2] = ranges[inner * 2]!;
  ranges[ghost * 2 + 1] = ranges[inner * 2 + 1]!;
  return side === 0
    ? { ranges, lo: ranges[2]!, hi: s.hi }
    : { ranges, lo: s.lo, hi: ranges[inner * 2 + 1]! };
}

export interface BorderPlan {
  /** Faces (faceBit mask) to fill by edge clamping for now */
  clamped: number;
  /**
   * Uncached edge / corner chunks between kept faces: not fetched, their
   * voxels are clamped into the interior instead (see `fills` in the worker)
   */
  filled: Vec3[];
  /**
   * Uncached chunks in groups: once every chunk of a group has arrived,
   * loading the brick again improves it
   */
  waiting: Vec3[][];
}

/**
 * Which ghost-border faces of a brick to fill from neighbouring chunks.
 *
 * The 1-voxel border widens a brick's source box into chunks that hold
 * nothing but border voxels: with 128×128×16 chunks, a 64³ brick touches 20
 * chunks instead of 4. A face is filled for real once every chunk bordering
 * only that face is `cached`; otherwise it is edge-clamped. Edge and corner
 * chunks never hold a face back — they often belong to a diagonal neighbour
 * that is never loaded — and are clamp-filled while missing.
 */
export function planBorders(
  axes: readonly [AxisSampling, AxisSampling, AxisSampling],
  chunk: Vec3,
  cached: (cx: number, cy: number, cz: number) => boolean,
): BorderPlan {
  // Per axis: chunk index span of the whole box and of the brick interior.
  const c0: Vec3 = [0, 0, 0];
  const c1: Vec3 = [0, 0, 0];
  const k0: Vec3 = [0, 0, 0];
  const k1: Vec3 = [0, 0, 0];
  let faces = 0;
  for (let a = 0; a < 3; a++) {
    const s = axes[a]!;
    const cs = chunk[a]!;
    c0[a] = Math.floor(s.lo / cs);
    c1[a] = Math.floor((s.hi - 1) / cs);
    k0[a] = Math.floor(s.ranges[2]! / cs);
    k1[a] = Math.floor((s.ranges[(PHYSICAL - 2) * 2 + 1]! - 1) / cs);
    if (c0[a]! < k0[a]!) faces |= faceBit(a, 0);
    if (c1[a]! > k1[a]!) faces |= faceBit(a, 1);
  }
  // Uncached border-only chunks: per face for those bordering one face,
  // with their face mask for edges and corners.
  const faceMissing = new Map<number, Vec3[]>();
  const edgeMissing: Array<{ mask: number; at: Vec3 }> = [];
  if (faces !== 0) {
    for (let cz = c0[2]; cz <= c1[2]; cz++)
      for (let cy = c0[1]; cy <= c1[1]; cy++)
        for (let cx = c0[0]; cx <= c1[0]; cx++) {
          const at: Vec3 = [cx, cy, cz];
          let mask = 0;
          for (let a = 0; a < 3; a++) {
            if (at[a]! < k0[a]!) mask |= faceBit(a, 0);
            else if (at[a]! > k1[a]!) mask |= faceBit(a, 1);
          }
          if (mask === 0 || cached(cx, cy, cz)) continue;
          if ((mask & (mask - 1)) !== 0) edgeMissing.push({ mask, at });
          else faceMissing.set(mask, [...(faceMissing.get(mask) ?? []), at]);
        }
  }
  let clamped = 0;
  for (const f of faceMissing.keys()) clamped |= f;
  // Edges touching a clamped face fall outside the narrowed box anyway.
  const filled = edgeMissing
    .filter((m) => (m.mask & clamped) === 0)
    .map((m) => m.at);
  return {
    clamped,
    filled,
    waiting: [...faceMissing.values(), ...filled.map((at) => [at])],
  };
}

/** Chunk grid coordinates (x, y, z) overlapping a source box [lo, hi). */
export function overlappingChunks(lo: Vec3, hi: Vec3, chunk: Vec3): Vec3[] {
  const out: Vec3[] = [];
  const c0 = [0, 1, 2].map((a) => Math.floor(lo[a]! / chunk[a]!));
  const c1 = [0, 1, 2].map((a) => Math.floor((hi[a]! - 1) / chunk[a]!));
  for (let cz = c0[2]!; cz <= c1[2]!; cz++)
    for (let cy = c0[1]!; cy <= c1[1]!; cy++)
      for (let cx = c0[0]!; cx <= c1[0]!; cx++) out.push([cx, cy, cz]);
  return out;
}
