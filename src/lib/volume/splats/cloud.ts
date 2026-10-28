/**
 * GaussianCloud — CPU-side Gaussian splat set in Kiln's training layout
 * (PARAM_STRIDE floats per Gaussian: mean, log-scale, quaternion wxyz,
 * colour logit, opacity logit), plus standard 3DGS PLY import/export.
 *
 * Positions live in Kiln's normalised volume space (centred at the origin,
 * longest physical axis = 1), so a cloud lines up with the volume it came
 * from. `worldScale` (e.g. µm per normalised unit) is written to the PLY so
 * other tools can recover physical units.
 */

import {
  colorParam,
  logit,
  P_CHANNEL,
  P_COLOR,
  P_MEAN,
  P_OPACITY,
  P_QUAT,
  P_SCALE,
  PARAM_STRIDE,
} from "./cpu-reference";

/** Zeroth-order spherical-harmonic basis constant (DC colour). */
export const SH_C0 = 0.28209479177387814;

export interface GaussianCloud {
  count: number;
  /** count × PARAM_STRIDE floats */
  params: Float32Array;
  /**
   * Trained with Mip-Splatting dilation compensation (Kiln clouds). Ordinary
   * 3DGS files are not and should be rendered without it.
   */
  antialiased?: boolean;
  /** Physical units per normalised unit (optional metadata) */
  worldScale?: number;
  worldUnit?: string;
  /**
   * Set on per-channel clouds: param P_CHANNEL holds each Gaussian's channel
   * (the viewer's channel slot) and colours are that channel rendered in
   * white, tinted with the live channel colours at display. "over" composites
   * front to back (volume / surface); "add" composites each channel on its
   * own and sums them, as a maximum-intensity projection does.
   */
  channelComposite?: "over" | "add";
}

/**
 * Concatenate clouds (metadata from the first). With `slots`, the result is
 * a per-channel cloud: part i's Gaussians are tagged with channel slots[i]
 * in the same copy, and it composites as `composite` (default: the first
 * part's, else "over").
 */
export function mergeClouds(
  parts: readonly GaussianCloud[],
  slots?: readonly number[],
  composite?: "over" | "add",
): GaussianCloud {
  const first = parts[0];
  if (!first) return createCloud(0);
  const count = parts.reduce((n, c) => n + c.count, 0);
  const params = new Float32Array(count * PARAM_STRIDE);
  let o = 0;
  parts.forEach((c, i) => {
    const n = c.count * PARAM_STRIDE;
    params.set(c.params.subarray(0, n), o);
    const slot = slots?.[i];
    if (slot !== undefined)
      for (let k = o + P_CHANNEL; k < o + n; k += PARAM_STRIDE)
        params[k] = slot;
    o += n;
  });
  const merged = { ...first, count, params };
  if (slots)
    merged.channelComposite = composite ?? first.channelComposite ?? "over";
  return merged;
}

/** Copy of `cloud` with every Gaussian in channel `slot`. */
export function withChannel(
  cloud: GaussianCloud,
  slot: number,
  composite: "over" | "add",
): GaussianCloud {
  return mergeClouds([cloud], [slot], composite);
}

/** The per-channel sub-clouds of a channel cloud, by channel slot. */
export function splitByChannel(
  cloud: GaussianCloud,
): Map<number, GaussianCloud> {
  const src = cloud.params;
  const S = PARAM_STRIDE;
  // Runs of one slot (a merged cloud is one run per channel), copied whole.
  const runs: number[] = []; // start, end pairs
  const counts = new Map<number, number>();
  for (let i = 0; i < cloud.count; ) {
    const slot = src[i * S + P_CHANNEL]!;
    let j = i + 1;
    while (j < cloud.count && src[j * S + P_CHANNEL] === slot) j++;
    runs.push(i, j);
    counts.set(slot, (counts.get(slot) ?? 0) + j - i);
    i = j;
  }
  const out = new Map<number, GaussianCloud>();
  for (const [slot, n] of counts)
    out.set(slot, { ...cloud, count: 0, params: new Float32Array(n * S) });
  for (let r = 0; r < runs.length; r += 2) {
    const i = runs[r]!;
    const j = runs[r + 1]!;
    const part = out.get(src[i * S + P_CHANNEL]!)!;
    part.params.set(src.subarray(i * S, j * S), part.count * S);
    part.count += j - i;
  }
  return out;
}

export function createCloud(count: number): GaussianCloud {
  return { count, params: new Float32Array(count * PARAM_STRIDE) };
}

/** A Kiln-built cloud (always Mip-Splatting compensated) around `params`. */
export function cloudFromParams(
  params: Float32Array,
  count: number,
  meta: Pick<GaussianCloud, "worldScale" | "worldUnit"> = {},
): GaussianCloud {
  return {
    count,
    params,
    antialiased: true,
    ...(meta.worldScale !== undefined ? { worldScale: meta.worldScale } : {}),
    ...(meta.worldUnit !== undefined ? { worldUnit: meta.worldUnit } : {}),
  };
}

interface GaussianInit {
  x: number;
  y: number;
  z: number;
  /** linear scale (std-dev) per local axis */
  sx: number;
  sy: number;
  sz: number;
  /** rotation quaternion w, x, y, z */
  qw: number;
  qx: number;
  qy: number;
  qz: number;
  /** emission colour, linear, ≥ 0 (may exceed 1) */
  r: number;
  g: number;
  b: number;
  opacity: number;
}

export function setGaussian(
  cloud: GaussianCloud,
  i: number,
  g: GaussianInit,
): void {
  const p = cloud.params;
  const b = i * PARAM_STRIDE;
  p[b + P_MEAN] = g.x;
  p[b + P_MEAN + 1] = g.y;
  p[b + P_MEAN + 2] = g.z;
  p[b + P_SCALE] = Math.log(Math.max(1e-7, g.sx));
  p[b + P_SCALE + 1] = Math.log(Math.max(1e-7, g.sy));
  p[b + P_SCALE + 2] = Math.log(Math.max(1e-7, g.sz));
  p[b + P_QUAT] = g.qw;
  p[b + P_QUAT + 1] = g.qx;
  p[b + P_QUAT + 2] = g.qy;
  p[b + P_QUAT + 3] = g.qz;
  p[b + P_COLOR] = colorParam(g.r);
  p[b + P_COLOR + 1] = colorParam(g.g);
  p[b + P_COLOR + 2] = colorParam(g.b);
  p[b + P_OPACITY] = logit(g.opacity);
}

const PLY_PROPS = [
  "x",
  "y",
  "z",
  "nx",
  "ny",
  "nz",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
] as const;

/**
 * Encode as a binary little-endian INRIA-style 3DGS PLY (SH degree 0), with
 * colours multiplied by `colorScale`. The header is padded to a multiple of
 * four bytes so the float body can be read in place.
 */
export function writePly(cloud: GaussianCloud, colorScale = 1): Uint8Array {
  const n = cloud.count;
  const channels = !!cloud.channelComposite;
  const lines = [
    "ply",
    "format binary_little_endian 1.0",
    "comment kiln-render gaussian splats",
  ];
  if (cloud.antialiased) lines.push("comment kiln mip_filter 1");
  if (cloud.channelComposite)
    lines.push(`comment kiln channel_composite ${cloud.channelComposite}`);
  if (cloud.worldScale) {
    lines.push(`comment kiln world_scale ${cloud.worldScale}`);
    if (cloud.worldUnit)
      lines.push(`comment kiln world_unit ${cloud.worldUnit}`);
  }
  lines.push(`element vertex ${n}`);
  for (const p of PLY_PROPS) lines.push(`property float ${p}`);
  if (channels) lines.push("property float channel");
  lines.push("end_header\n");
  const utf8 = new TextEncoder();
  let header = utf8.encode(lines.join("\n"));
  if (header.length % 4) {
    lines[2] += " ".repeat(4 - (header.length % 4));
    header = utf8.encode(lines.join("\n"));
  }
  const stride = PLY_PROPS.length + (channels ? 1 : 0);
  const out = new Uint8Array(header.length + n * stride * 4);
  out.set(header);
  const body = new Float32Array(out.buffer, header.length, n * stride);
  const p = cloud.params;
  for (let i = 0; i < n; i++) {
    const b = i * PARAM_STRIDE;
    const o = i * stride;
    body[o] = p[b + P_MEAN]!;
    body[o + 1] = p[b + P_MEAN + 1]!;
    body[o + 2] = p[b + P_MEAN + 2]!;
    // [3..5] normals stay 0
    for (let c = 0; c < 3; c++) {
      body[o + 6 + c] =
        (Math.exp(p[b + P_COLOR + c]!) * colorScale - 0.5) / SH_C0;
    }
    body[o + 9] = p[b + P_OPACITY]!;
    body[o + 10] = p[b + P_SCALE]!;
    body[o + 11] = p[b + P_SCALE + 1]!;
    body[o + 12] = p[b + P_SCALE + 2]!;
    const qw = p[b + P_QUAT]!;
    const qx = p[b + P_QUAT + 1]!;
    const qy = p[b + P_QUAT + 2]!;
    const qz = p[b + P_QUAT + 3]!;
    const ql = Math.hypot(qw, qx, qy, qz) || 1;
    body[o + 13] = qw / ql;
    body[o + 14] = qx / ql;
    body[o + 15] = qy / ql;
    body[o + 16] = qz / ql;
    if (channels) body[o + 17] = p[b + P_CHANNEL]!;
  }
  return out;
}

const TYPE_SIZE: Record<string, number> = {
  char: 1,
  uchar: 1,
  int8: 1,
  uint8: 1,
  short: 2,
  ushort: 2,
  int16: 2,
  uint16: 2,
  int: 4,
  uint: 4,
  int32: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

function reader(type: string): (v: DataView, o: number) => number {
  switch (type) {
    case "char":
    case "int8":
      return (v, o) => v.getInt8(o);
    case "uchar":
    case "uint8":
      return (v, o) => v.getUint8(o);
    case "short":
    case "int16":
      return (v, o) => v.getInt16(o, true);
    case "ushort":
    case "uint16":
      return (v, o) => v.getUint16(o, true);
    case "int":
    case "int32":
      return (v, o) => v.getInt32(o, true);
    case "uint":
    case "uint32":
      return (v, o) => v.getUint32(o, true);
    case "double":
    case "float64":
      return (v, o) => v.getFloat64(o, true);
    default:
      return (v, o) => v.getFloat32(o, true);
  }
}

/** PLY properties readPly uses (any others, e.g. higher-order SH, are skipped). */
const READ_PROPS = [
  "x",
  "y",
  "z",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "red",
  "green",
  "blue",
  "opacity",
  "channel",
] as const;

const DEFAULT_LOG_SCALE = Math.log(0.002);
const DEFAULT_OPACITY = logit(0.9);

/**
 * Parse a binary little-endian 3DGS PLY (INRIA layout; extra properties such
 * as higher-order SH are ignored — only the DC colour is kept).
 */
export function readPly(bytes: Uint8Array): GaussianCloud {
  const headText = new TextDecoder().decode(
    bytes.subarray(0, Math.min(bytes.length, 64 * 1024)),
  );
  const end = headText.indexOf("end_header");
  if (!headText.startsWith("ply") || end < 0) throw new Error("Not a PLY file");
  // Byte length (the header may hold multi-byte UTF-8, e.g. a "µm" unit).
  const headerLen = new TextEncoder().encode(
    headText.slice(0, headText.indexOf("\n", end) + 1),
  ).length;
  const lines = headText.slice(0, end).split(/\r?\n/);
  if (!lines.some((l) => l.startsWith("format binary_little_endian"))) {
    throw new Error("Only binary little-endian PLY files are supported");
  }
  let count = 0;
  let inVertex = false;
  let worldScale: number | undefined;
  let worldUnit: string | undefined;
  let antialiased = false;
  let channelComposite: GaussianCloud["channelComposite"];
  const props: Array<{ name: string; type: string; offset: number }> = [];
  let stride = 0;
  for (const line of lines) {
    const t = line.trim().split(/\s+/);
    if (t[0] === "comment" && t[1] === "kiln" && t[2] === "world_scale") {
      worldScale = Number(t[3]);
    } else if (t[0] === "comment" && t[1] === "kiln" && t[2] === "mip_filter") {
      antialiased = t[3] === "1";
    } else if (
      t[0] === "comment" &&
      t[1] === "kiln" &&
      t[2] === "channel_composite"
    ) {
      if (t[3] === "over" || t[3] === "add") channelComposite = t[3];
    } else if (t[0] === "comment" && t[1] === "kiln" && t[2] === "world_unit") {
      worldUnit = t[3];
    } else if (t[0] === "element") {
      inVertex = t[1] === "vertex";
      if (inVertex) count = Number(t[2]);
      else if (count > 0 && props.length > 0) break;
    } else if (t[0] === "property" && inVertex) {
      if (t[1] === "list")
        throw new Error("PLY list properties are not supported");
      const type = t[1]!;
      props.push({ name: t[2]!, type, offset: stride });
      stride += TYPE_SIZE[type] ?? 4;
    }
  }
  const used = READ_PROPS.map((name) => props.find((q) => q.name === name));
  // Rows of floats: all-float32 files (Kiln's, standard 3DGS) are read in
  // place; others have the used properties converted first.
  let rows: Float32Array;
  let s: number;
  let col: number[];
  const bodyOffset = bytes.byteOffset + headerLen;
  if (
    stride % 4 === 0 &&
    props.every((q) => TYPE_SIZE[q.type] === 4 && /^float/.test(q.type))
  ) {
    rows =
      bodyOffset % 4 === 0
        ? new Float32Array(bytes.buffer, bodyOffset, (count * stride) / 4)
        : new Float32Array(
            bytes.slice(headerLen, headerLen + count * stride).buffer,
          );
    s = stride / 4;
    col = used.map((q) => (q ? q.offset / 4 : -1));
  } else {
    const view = new DataView(bytes.buffer, bodyOffset);
    s = READ_PROPS.length;
    rows = new Float32Array(count * s);
    col = used.map((q, k) => {
      if (!q) return -1;
      const read = reader(q.type);
      for (let i = 0; i < count; i++)
        rows[i * s + k] = read(view, i * stride + q.offset);
      return k;
    });
  }
  const at = (name: (typeof READ_PROPS)[number]) =>
    col[READ_PROPS.indexOf(name)]!;
  const x = at("x");
  const y = at("y");
  const z = at("z");
  if (x < 0 || y < 0 || z < 0) throw new Error("PLY has no x/y/z");
  const scale = [at("scale_0"), at("scale_1"), at("scale_2")];
  const rot = [at("rot_0"), at("rot_1"), at("rot_2"), at("rot_3")];
  const dc = [at("f_dc_0"), at("f_dc_1"), at("f_dc_2")];
  const rgb = [at("red"), at("green"), at("blue")];
  const opacity = at("opacity");
  const channel = at("channel");
  const cloud = createCloud(count);
  cloud.worldScale = worldScale;
  cloud.worldUnit = worldUnit;
  cloud.antialiased = antialiased;
  if (channelComposite) cloud.channelComposite = channelComposite;
  const p = cloud.params;
  for (let i = 0; i < count; i++) {
    const r = i * s;
    const b = i * PARAM_STRIDE;
    p[b] = rows[r + x]!;
    p[b + 1] = rows[r + y]!;
    p[b + 2] = rows[r + z]!;
    for (let k = 0; k < 3; k++) {
      const c = scale[k]!;
      p[b + P_SCALE + k] = c >= 0 ? rows[r + c]! : DEFAULT_LOG_SCALE;
    }
    if (rot[0]! >= 0) {
      for (let k = 0; k < 4; k++) {
        const c = rot[k]!;
        p[b + P_QUAT + k] = c >= 0 ? rows[r + c]! : 0;
      }
    } else {
      p[b + P_QUAT] = 1;
    }
    for (let k = 0; k < 3; k++) {
      let c = 1;
      if (dc[k]! >= 0) c = 0.5 + SH_C0 * rows[r + dc[k]!]!;
      else if (rgb[k]! >= 0) c = rows[r + rgb[k]!]! / 255;
      p[b + P_COLOR + k] = colorParam(c);
    }
    p[b + P_OPACITY] = opacity >= 0 ? rows[r + opacity]! : DEFAULT_OPACITY;
    if (channel >= 0) p[b + P_CHANNEL] = rows[r + channel]!;
  }
  return cloud;
}
