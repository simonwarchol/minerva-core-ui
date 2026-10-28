/**
 * CPU reference for the Gaussian splat rasterizer (forward). Slow and simple
 * on purpose — it documents the exact math the WGSL kernels implement and
 * lets tests check GPU output against it.
 *
 * Conventions (shared with the GPU path):
 * - Parameters per Gaussian (PARAM_STRIDE floats): mean xyz, log-scale xyz,
 *   quaternion wxyz (unnormalised), log colour rgb, opacity logit.
 *   Colour is exp(param): unbounded above, because Kiln's additive
 *   multichannel DVR emits colours > 1 where channels overlap.
 * - View matrix is column-major (wgpu-matrix lookAt): camera looks down -z.
 * - Pixel (x, y) has centre (x + 0.5, y + 0.5); y grows downwards.
 *   u = fx·tx/z + cx, v = cy − fy·ty/z with z = −t.z.
 * - Rendered output is premultiplied colour C = Σ cᵢαᵢTᵢ and alpha A = 1 − T.
 */

export const PARAM_STRIDE = 16;
export const P_MEAN = 0;
export const P_SCALE = 3;
export const P_QUAT = 6;
export const P_COLOR = 10;
export const P_OPACITY = 13;
/** Channel slot of a per-channel cloud (GaussianCloud.channelComposite). */
export const P_CHANNEL = 14;

export const ALPHA_MIN = 1 / 255;
export const ALPHA_MAX = 0.99;
export const T_MIN = 1e-4;
export const COV2D_DILATION = 0.3;
export const NEAR_Z = 0.01;
export const FRUSTUM_SLACK = 1.3;

/** Kiln's vertical field of view. */
const FOV_Y = Math.PI / 4;

/** Pinhole camera of a splat render. */
export interface SplatCamera {
  /** Column-major 4×4 view matrix */
  view: ArrayLike<number>;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Inverse of the colour mapping (colour = exp(param)). */
export function colorParam(c: number): number {
  return Math.log(Math.max(1e-4, c));
}

export function logit(p: number): number {
  const c = Math.min(1 - 1e-6, Math.max(1e-6, p));
  return Math.log(c / (1 - c));
}

/** Intrinsics of Kiln's camera (vertical FOV, square pixels) for an image size. */
export function intrinsicsFor(
  view: ArrayLike<number>,
  width: number,
  height: number,
  fovY: number = FOV_Y,
): SplatCamera {
  const f = (0.5 * height) / Math.tan(fovY / 2);
  return { view, fx: f, fy: f, cx: width / 2, cy: height / 2, width, height };
}

type Mat3 = number[]; // row-major 3×3

/** Rotation matrix (row-major) of the unit quaternion (w, x, y, z). */
export function quatToRot(w: number, x: number, y: number, z: number): Mat3 {
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
}

interface Projected {
  visible: boolean;
  u: number;
  v: number;
  /** conic = inverse 2D covariance (A, B, C) */
  conic: [number, number, number];
  depth: number;
  color: [number, number, number];
  /** effective opacity o·ρ used for compositing */
  opacity: number;
  /** Half-extents of the α ≥ ALPHA_MIN ellipse's bounding box (px) */
  extentX: number;
  extentY: number;
}

function projectGaussian(
  p: ArrayLike<number>,
  base: number,
  cam: SplatCamera,
): Projected {
  const V = cam.view;
  const mx = p[base + P_MEAN]!;
  const my = p[base + P_MEAN + 1]!;
  const mz = p[base + P_MEAN + 2]!;
  const W: Mat3 = [
    V[0]!,
    V[4]!,
    V[8]!,
    V[1]!,
    V[5]!,
    V[9]!,
    V[2]!,
    V[6]!,
    V[10]!,
  ];
  const t: [number, number, number] = [
    W[0]! * mx + W[1]! * my + W[2]! * mz + V[12]!,
    W[3]! * mx + W[4]! * my + W[5]! * mz + V[13]!,
    W[6]! * mx + W[7]! * my + W[8]! * mz + V[14]!,
  ];
  const z = -t[2];
  const color: [number, number, number] = [
    Math.exp(p[base + P_COLOR]!),
    Math.exp(p[base + P_COLOR + 1]!),
    Math.exp(p[base + P_COLOR + 2]!),
  ];
  const opacity = sigmoid(p[base + P_OPACITY]!);
  const empty: Projected = {
    visible: false,
    u: 0,
    v: 0,
    conic: [0, 0, 0],
    depth: z,
    color,
    opacity,
    extentX: 0,
    extentY: 0,
  };
  if (z < NEAR_Z) return empty;

  const s = [
    Math.exp(p[base + P_SCALE]!),
    Math.exp(p[base + P_SCALE + 1]!),
    Math.exp(p[base + P_SCALE + 2]!),
  ];
  const qw = p[base + P_QUAT]!;
  const qx = p[base + P_QUAT + 1]!;
  const qy = p[base + P_QUAT + 2]!;
  const qz = p[base + P_QUAT + 3]!;
  const qlen = Math.max(1e-12, Math.hypot(qw, qx, qy, qz));
  const R = quatToRot(qw / qlen, qx / qlen, qy / qlen, qz / qlen);
  const M: Mat3 = [];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) M[i * 3 + j] = R[i * 3 + j]! * s[j]!;
  const sigma: Mat3 = [];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let acc = 0;
      for (let k = 0; k < 3; k++) acc += M[i * 3 + k]! * M[j * 3 + k]!;
      sigma[i * 3 + j] = acc;
    }

  const limX = (FRUSTUM_SLACK * (0.5 * cam.width)) / cam.fx;
  const limY = (FRUSTUM_SLACK * (0.5 * cam.height)) / cam.fy;
  const txz = Math.min(limX, Math.max(-limX, t[0] / z));
  const tyz = Math.min(limY, Math.max(-limY, t[1] / z));
  const J = [
    cam.fx / z,
    0,
    (cam.fx * txz) / z,
    0,
    -cam.fy / z,
    (-cam.fy * tyz) / z,
  ];
  const Tm: number[] = []; // 2×3 row-major (J·W)
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 3; c++) {
      let acc = 0;
      for (let k = 0; k < 3; k++) acc += J[r * 3 + k]! * W[k * 3 + c]!;
      Tm[r * 3 + c] = acc;
    }
  const TS: number[] = [];
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 3; c++) {
      let acc = 0;
      for (let k = 0; k < 3; k++) acc += Tm[r * 3 + k]! * sigma[k * 3 + c]!;
      TS[r * 3 + c] = acc;
    }
  const cov = (r: number, c: number) => {
    let acc = 0;
    for (let k = 0; k < 3; k++) acc += TS[r * 3 + k]! * Tm[c * 3 + k]!;
    return acc;
  };
  const a = cov(0, 0) + COV2D_DILATION;
  const b = cov(0, 1);
  const c = cov(1, 1) + COV2D_DILATION;
  const det = a * c - b * b;
  if (!(det > 0)) return empty;
  const conic: [number, number, number] = [c / det, -b / det, a / det];
  // Mip-Splatting's 2D filter: the dilation is an anti-aliasing blur, so
  // scale opacity to conserve the splat's energy (sub-pixel splats fade
  // instead of growing into bright speckle).
  const det0 = Math.max(
    1e-20,
    (a - COV2D_DILATION) * (c - COV2D_DILATION) - b * b,
  );
  const effOpacity = opacity * Math.sqrt(det0 / det);
  if (effOpacity < ALPHA_MIN) return empty;
  // α ≥ ALPHA_MIN inside the ellipse dᵀΣ⁻¹d ≤ extent²; its bounding box.
  const extent = Math.sqrt(2 * Math.log(effOpacity / ALPHA_MIN));
  return {
    ...empty,
    visible: extent > 0,
    u: cam.fx * (t[0] / z) + cam.cx,
    v: cam.cy - cam.fy * (t[1] / z),
    conic,
    opacity: effOpacity,
    extentX: Math.ceil(extent * Math.sqrt(a)),
    extentY: Math.ceil(extent * Math.sqrt(c)),
  };
}

/** Pixel-exact forward rasterisation (front-to-back by view depth). */
export function renderReference(
  params: ArrayLike<number>,
  count: number,
  cam: SplatCamera,
): { image: Float64Array; order: number[] } {
  const projected: Projected[] = [];
  for (let g = 0; g < count; g++)
    projected.push(projectGaussian(params, g * PARAM_STRIDE, cam));
  const order = projected
    .map((_, i) => i)
    .filter((i) => projected[i]!.visible)
    .sort((a, b) => projected[a]!.depth - projected[b]!.depth || a - b);
  const { width, height } = cam;
  // premultiplied rgb + alpha, 4 floats per pixel
  const image = new Float64Array(width * height * 4);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const pix = py * width + px;
      let T = 1;
      let r = 0;
      let gC = 0;
      let bC = 0;
      for (const g of order) {
        const pr = projected[g]!;
        const dx = pr.u - (px + 0.5);
        const dy = pr.v - (py + 0.5);
        // Outside the footprint's bounding box α < ALPHA_MIN anyway.
        if (Math.abs(dx) > pr.extentX + 0.5 || Math.abs(dy) > pr.extentY + 0.5)
          continue;
        const power =
          -0.5 * (pr.conic[0] * dx * dx + pr.conic[2] * dy * dy) -
          pr.conic[1] * dx * dy;
        if (power > 0) continue;
        const alpha = Math.min(ALPHA_MAX, pr.opacity * Math.exp(power));
        if (alpha < ALPHA_MIN) continue;
        const nextT = T * (1 - alpha);
        if (nextT < T_MIN) break;
        r += pr.color[0] * alpha * T;
        gC += pr.color[1] * alpha * T;
        bC += pr.color[2] * alpha * T;
        T = nextT;
      }
      image[pix * 4] = r;
      image[pix * 4 + 1] = gC;
      image[pix * 4 + 2] = bC;
      image[pix * 4 + 3] = 1 - T;
    }
  }
  return { image, order };
}
