/**
 * WGSL for the tile-based Gaussian splat rasterizer (forward pass).
 * Mirrors src/splat/cpu-reference.ts — the constants are shared with it.
 */

import {
  ALPHA_MAX,
  ALPHA_MIN,
  COV2D_DILATION,
  FRUSTUM_SLACK,
  NEAR_Z,
  P_CHANNEL,
  P_COLOR,
  P_OPACITY,
  P_QUAT,
  P_SCALE,
  PARAM_STRIDE,
  T_MIN,
} from "../cpu-reference";
import { WGSL_LINEAR_WORKGROUP } from "./kernel";

export const TILE = 16;
/**
 * vec4s per projected Gaussian: [uv, bounding-box half-extents (0 = culled)],
 * [conic, opacity], [rgb, channel].
 */
export const PROJ_VEC4S = 3;

/** Up to this many channels can be tinted per cloud (Kiln's MAX_CHANNELS). */
export const MAX_SPLAT_CHANNELS = 6;

/**
 * Camera uniform layout (bytes): 28 scalars, the channel tints, then the
 * depth cue (2 scalars, padded to 16 bytes).
 */
export const CAMERA_UNIFORM_SIZE = (28 + MAX_SPLAT_CHANNELS * 4 + 4) * 4;

/** A float literal WGSL accepts as f32. */
const f32 = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

const CAMERA_WGSL = /* wgsl */ `
const TILE: u32 = ${TILE}u;

struct Camera {
  view: mat4x4f,
  fx: f32,
  fy: f32,
  cx: f32,
  cy: f32,
  width: u32,
  height: u32,
  tilesX: u32,
  tilesY: u32,
  count: u32,
  isectCapacity: u32,
  /** 1 = Mip-Splatting opacity compensation for the dilation (Kiln clouds) */
  mipFilter: u32,
  /**
   * 0: plain cloud. 1: per-channel cloud (channel index in param P_CHANNEL),
   * tinted, composited front to back. 2: tinted, each channel composited on
   * its own and the results added — as a maximum-intensity projection
   * combines them.
   */
  channelMode: u32,
  /** Per-channel colour (rgb) and visibility (a) */
  channelTint: array<vec4f, ${MAX_SPLAT_CHANNELS}>,
  /** Depth cue (core/depth-cue.ts): colour × exp(−scale · max(0, d − near)) */
  depthCueNear: f32,
  depthCueScale: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;

${WGSL_LINEAR_WORKGROUP}
`;

/** Bindings: 0 camera, 1 params, 2 projected, 3 depthKeys, 4 depthIdx */
export const PROJECT_WGSL = /* wgsl */ `
${CAMERA_WGSL}
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> depthIdx: array<u32>;

const PARAM_STRIDE: u32 = ${PARAM_STRIDE}u;
const ALPHA_MIN: f32 = ${f32(ALPHA_MIN)};
const COV2D_DILATION: f32 = ${f32(COV2D_DILATION)};
const NEAR_Z: f32 = ${f32(NEAR_Z)};
const FRUSTUM_SLACK: f32 = ${f32(FRUSTUM_SLACK)};

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

/** Row-major rotation entries packed as WGSL columns: R[col][row]. */
fn quatToRot(q: vec4f) -> mat3x3f {
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
  );
}

fn param3(b: u32) -> vec3f { return vec3f(params[b], params[b + 1u], params[b + 2u]); }

struct Projected {
  visible: bool,
  uv: vec2f,
  conic: vec3f,
  depth: f32,
  /** Half-extents of the α ≥ ALPHA_MIN ellipse's bounding box (px) */
  extentXY: vec2f,
  color: vec3f,
  /** effective opacity o·ρ used for compositing */
  opacity: f32,
}

fn projectGaussian(g: u32) -> Projected {
  let b = g * PARAM_STRIDE;
  var p: Projected;
  p.visible = false;
  let mean = param3(b);
  let t = (camera.view * vec4f(mean, 1.0)).xyz;
  let z = -t.z;
  p.depth = z;
  // Depth cue: dim with distance from the camera, as the volume renderer does.
  let fade = exp(-camera.depthCueScale * max(0.0, length(t) - camera.depthCueNear));
  p.color = exp(param3(b + ${P_COLOR}u)) * fade;
  let opacityRaw = sigmoid(params[b + ${P_OPACITY}u]);
  p.opacity = opacityRaw;
  if (z < NEAR_Z || opacityRaw < ALPHA_MIN) { return p; }

  let s = exp(param3(b + ${P_SCALE}u));
  let q = vec4f(params[b + ${P_QUAT}u], param3(b + ${P_QUAT + 1}u));
  let R = quatToRot(q / max(1e-12, length(q)));
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let sigma = M * transpose(M);
  let W = mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let limX = FRUSTUM_SLACK * 0.5 * f32(camera.width) / camera.fx;
  let limY = FRUSTUM_SLACK * 0.5 * f32(camera.height) / camera.fy;
  let rawX = t.x / z;
  let rawY = t.y / z;
  let txz = clamp(rawX, -limX, limX);
  let tyz = clamp(rawY, -limY, limY);
  let J = mat3x2f(
    vec2f(camera.fx / z, 0.0),
    vec2f(0.0, -camera.fy / z),
    vec2f(camera.fx * txz / z, -camera.fy * tyz / z),
  );
  let Tm = J * W;
  let cov = Tm * sigma * transpose(Tm);
  let a = cov[0][0] + COV2D_DILATION;
  let bb = cov[0][1];
  let c = cov[1][1] + COV2D_DILATION;
  let det = a * c - bb * bb;
  if (!(det > 0.0)) { return p; }
  p.conic = vec3f(c / det, -bb / det, a / det);
  // Mip-Splatting 2D filter: the dilation is an anti-aliasing blur, so scale
  // opacity to conserve energy (sub-pixel splats fade, not speckle).
  let det0 = max(1e-20, (a - COV2D_DILATION) * (c - COV2D_DILATION) - bb * bb);
  let rho = select(1.0, sqrt(det0 / det), camera.mipFilter != 0u);
  p.opacity = opacityRaw * rho;
  if (p.opacity < ALPHA_MIN) { return p; }
  // α ≥ ALPHA_MIN inside the ellipse dᵀΣ⁻¹d ≤ extent². Its exact bounding
  // box is tighter than a circle for elongated splats: fewer tile hits.
  let extent = sqrt(2.0 * log(p.opacity / ALPHA_MIN));
  p.extentXY = ceil(extent * sqrt(vec2f(a, c)));
  p.uv = vec2f(camera.fx * rawX + camera.cx, camera.cy - camera.fy * rawY);
  // Screen-space cull: the footprint must overlap the image.
  let e = p.extentXY;
  if (p.uv.x + e.x < 0.0 || p.uv.x - e.x > f32(camera.width) ||
      p.uv.y + e.y < 0.0 || p.uv.y - e.y > f32(camera.height)) {
    return p;
  }
  p.visible = extent > 0.0;
  return p;
}

@compute @workgroup_size(256)
fn project(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let g = linearWorkgroup(wid, nwg) * 256u + lid;
  if (g >= camera.count) { return; }
  var p = projectGaussian(g);
  var ch = 0u;
  if (camera.channelMode != 0u) {
    ch = min(u32(max(params[g * PARAM_STRIDE + ${P_CHANNEL}u], 0.0)), ${MAX_SPLAT_CHANNELS - 1}u);
    let tint = camera.channelTint[ch];
    p.color *= tint.rgb;
    p.visible = p.visible && tint.a > 0.0;
  }
  let o = g * ${PROJ_VEC4S}u;
  // Zero half-extents mark a culled Gaussian.
  projected[o] = vec4f(p.uv, select(vec2f(0.0), p.extentXY, p.visible));
  projected[o + 1u] = vec4f(p.conic, p.opacity);
  projected[o + 2u] = vec4f(p.color, f32(ch));
  // Positive floats sort correctly as u32; culled Gaussians go last.
  depthKeys[g] = select(0xffffffffu, bitcast<u32>(p.depth), p.visible);
  depthIdx[g] = g;
}
`;

/**
 * Bindings: 0 camera, 1 projected, 2 depthIdx (sorted), 3 counts/offsets (N+1),
 * 4 isectKeys, 5 isectVals
 */
export const TILES_WGSL = /* wgsl */ `
${CAMERA_WGSL}
@group(0) @binding(1) var<storage, read> projected: array<vec4f>;
@group(0) @binding(2) var<storage, read> depthIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> isectKeys: array<u32>;
@group(0) @binding(5) var<storage, read_write> isectVals: array<u32>;

/** Tile rect [min, max) covered by a projected footprint (uv, half-extents). */
fn tileRect(fp: vec4f) -> vec4u {
  let tiles = vec2f(f32(camera.tilesX), f32(camera.tilesY));
  let lo = clamp(floor((fp.xy - fp.zw) / f32(TILE)), vec2f(0.0), tiles);
  let hi = clamp(ceil((fp.xy + fp.zw) / f32(TILE)), vec2f(0.0), tiles);
  return vec4u(vec2u(lo), vec2u(hi));
}

@compute @workgroup_size(256)
fn countTiles(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let k = linearWorkgroup(wid, nwg) * 256u + lid;
  if (k > camera.count) { return; }
  if (k == camera.count) { offsets[k] = 0u; return; }
  let fp = projected[depthIdx[k] * ${PROJ_VEC4S}u];
  if (fp.z <= 0.0) { offsets[k] = 0u; return; }
  let r = tileRect(fp);
  offsets[k] = (r.z - r.x) * (r.w - r.y);
}

@compute @workgroup_size(256)
fn emitIntersections(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let k = linearWorkgroup(wid, nwg) * 256u + lid;
  if (k >= camera.count) { return; }
  let g = depthIdx[k];
  let fp = projected[g * ${PROJ_VEC4S}u];
  if (fp.z <= 0.0) { return; }
  let r = tileRect(fp);
  var idx = offsets[k];
  for (var ty = r.y; ty < r.w; ty++) {
    for (var tx = r.x; tx < r.z; tx++) {
      if (idx < camera.isectCapacity) {
        isectKeys[idx] = ty * camera.tilesX + tx;
        isectVals[idx] = g;
      }
      idx++;
    }
  }
}
`;

/** Bindings: 0 camera, 1 offsets (total at [count]), 2 isectKeys, 3 tileRanges, 4 indirect */
export const RANGES_WGSL = /* wgsl */ `
${CAMERA_WGSL}
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> isectKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> tileRanges: array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(1)
fn rangesSetup() {
  let m = min(offsets[camera.count], camera.isectCapacity);
  let blocks = (m + 255u) / 256u;
  indirect[0] = min(blocks, 65535u);
  indirect[1] = select(1u, (blocks + 65534u) / 65535u, blocks > 65535u);
  indirect[2] = 1u;
}

@compute @workgroup_size(256)
fn tileRangesKernel(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let m = min(offsets[camera.count], camera.isectCapacity);
  let i = linearWorkgroup(wid, nwg) * 256u + lid;
  if (i >= m) { return; }
  let tile = isectKeys[i];
  if (i == 0u || isectKeys[i - 1u] != tile) { tileRanges[tile * 2u] = i; }
  if (i == m - 1u || isectKeys[i + 1u] != tile) { tileRanges[tile * 2u + 1u] = i + 1u; }
}
`;

/**
 * Bindings: 0 camera, 1 projected, 2 isectVals, 3 tileRanges, 4 image (vec4 per
 * pixel, premultiplied). Override `additive` selects channelMode 2's per-channel
 * compositing; otherwise everything composites front to back.
 */
export const RASTERIZE_WGSL = /* wgsl */ `
${CAMERA_WGSL}
@group(0) @binding(1) var<storage, read> projected: array<vec4f>;
@group(0) @binding(2) var<storage, read> isectVals: array<u32>;
@group(0) @binding(3) var<storage, read> tileRanges: array<u32>;
@group(0) @binding(4) var<storage, read_write> image: array<vec4f>;

override additive: bool = false;

const ALPHA_MIN: f32 = ${f32(ALPHA_MIN)};
const ALPHA_MAX: f32 = ${f32(ALPHA_MAX)};
const T_MIN: f32 = ${f32(T_MIN)};

var<workgroup> shUv: array<vec2f, 256>;
var<workgroup> shConicO: array<vec4f, 256>;
/** rgb + channel */
var<workgroup> shColor: array<vec4f, 256>;
var<workgroup> shRange: vec2u;
var<workgroup> doneCount: atomic<u32>;
var<workgroup> doneFlag: u32;

@compute @workgroup_size(${TILE}, ${TILE})
fn rasterize(
  @builtin(local_invocation_id) lid3: vec3u,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
) {
  let tileId = wid.y * camera.tilesX + wid.x;
  let px = wid.xy * TILE + lid3.xy;
  let inside = px.x < camera.width && px.y < camera.height;
  let pixf = vec2f(px) + vec2f(0.5);
  if (lid == 0u) {
    shRange = vec2u(tileRanges[tileId * 2u], tileRanges[tileId * 2u + 1u]);
    atomicStore(&doneCount, 0u);
  }
  workgroupBarrier();
  let range = workgroupUniformLoad(&shRange);

  // Transmittance: one for front-to-back, one per channel when additive.
  var T = 1.0;
  var Tch = array<f32, ${MAX_SPLAT_CHANNELS}>(${Array(MAX_SPLAT_CHANNELS).fill("1.0").join(", ")});
  var C = vec3f(0.0);
  var done = !inside;
  if (done) { atomicAdd(&doneCount, 1u); }

  for (var batch = range.x; batch < range.y; batch += 256u) {
    workgroupBarrier();
    if (lid == 0u) { doneFlag = atomicLoad(&doneCount); }
    let i = batch + lid;
    if (i < range.y) {
      let o = isectVals[i] * ${PROJ_VEC4S}u;
      shUv[lid] = projected[o].xy;
      shConicO[lid] = projected[o + 1u];
      shColor[lid] = projected[o + 2u];
    }
    let nDone = workgroupUniformLoad(&doneFlag);
    if (nDone >= 256u) { break; }
    let n = min(256u, range.y - batch);
    if (!done) {
      for (var j = 0u; j < n; j++) {
        let d = shUv[j] - pixf;
        let co = shConicO[j];
        let power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
        if (power > 0.0) { continue; }
        let alpha = min(ALPHA_MAX, co.w * exp(power));
        if (alpha < ALPHA_MIN) { continue; }
        let col = shColor[j];
        if (additive) {
          // Channels don't occlude each other; a saturated channel adds nothing.
          let k = u32(col.w);
          let Tk = Tch[k];
          if (Tk < T_MIN) { continue; }
          C += col.rgb * (alpha * Tk);
          Tch[k] = Tk * (1.0 - alpha);
        } else {
          let nextT = T * (1.0 - alpha);
          if (nextT < T_MIN) { done = true; break; }
          C += col.rgb * (alpha * T);
          T = nextT;
        }
      }
      if (done) { atomicAdd(&doneCount, 1u); }
    }
  }
  if (inside) {
    if (additive) {
      for (var k = 0u; k < ${MAX_SPLAT_CHANNELS}u; k++) { T *= Tch[k]; }
    }
    image[px.y * camera.width + px.x] = vec4f(C, 1.0 - T);
  }
}
`;
