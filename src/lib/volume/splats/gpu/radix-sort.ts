/**
 * Stable LSD radix sort of (u32 key, u32 value) pairs on the GPU.
 *
 * Each pass = per-block digit histogram → global exclusive scan (digit-major,
 * so one scan yields every block's output base) → stable scatter. The element
 * count is read on the GPU, so the sort can follow kernels whose output size
 * the CPU never sees. Passes are always even so results land back in the
 * caller's buffers.
 *
 * Two scatter variants:
 * - **subgroups** (preferred): 8-bit digits. A key's rank among equal digits
 *   in its subgroup comes from 8 ballots; per-subgroup digit counts in shared
 *   memory give the offset across subgroups. Half the passes of 4-bit digits
 *   and a cheap scatter.
 * - **portable**: 4-bit digits, rank from a packed one-hot Hillis–Steele scan
 *   (16 digits × 16-bit counters in 8 words).
 *
 * The subgroup variant assumes subgroups are contiguous runs of
 * local_invocation_index (true on Metal, D3D12 and Vulkan for 1D workgroups).
 */

import {
  bindGroup,
  createBuffer,
  createKernel,
  type Kernel,
  STORAGE_RW,
  WGSL_LINEAR_WORKGROUP,
} from "./kernel";
import { type CountSource, GpuScan } from "./scan";

const BLOCK = 256;
const UNIFORM_STRIDE = 256;
const MAX_PASSES = 8;

interface Variant {
  /** bits per digit */
  bits: number;
  radix: number;
  subgroups: boolean;
  /** max subgroups per workgroup (subgroup variant) */
  maxSubgroups: number;
}

function setupWgsl(radix: number): string {
  return /* wgsl */ `
struct SetupParams { srcIndex: u32, capacity: u32, _a: u32, _b: u32 }
struct Info { n: u32, numBlocks: u32, histSize: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> setupParams: SetupParams;
@group(0) @binding(1) var<storage, read> countSrc: array<u32>;
@group(0) @binding(2) var<storage, read_write> setupInfo: Info;
@group(0) @binding(3) var<storage, read_write> setupIndirect: array<u32>;

@compute @workgroup_size(1)
fn setup() {
  let n = min(countSrc[setupParams.srcIndex], setupParams.capacity);
  let blocks = (n + ${BLOCK - 1}u) / ${BLOCK}u;
  setupInfo.n = n;
  setupInfo.numBlocks = blocks;
  setupInfo.histSize = blocks * ${radix}u;
  setupIndirect[0] = min(blocks, 65535u);
  setupIndirect[1] = select(1u, (blocks + 65534u) / 65535u, blocks > 65535u);
  setupIndirect[2] = 1u;
}
`;
}

function passHeader(v: Variant): string {
  return /* wgsl */ `
${v.subgroups ? "enable subgroups;\ndiagnostic(off, subgroup_uniformity);" : ""}
${WGSL_LINEAR_WORKGROUP}

struct Info { n: u32, numBlocks: u32, histSize: u32, _pad: u32 }
struct PassParams { shift: u32, _a: u32, _b: u32, _c: u32 }

@group(0) @binding(0) var<storage, read> info: Info;
@group(0) @binding(1) var<uniform> passParams: PassParams;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valsIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valsOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> hist: array<u32>;

const RADIX: u32 = ${v.radix}u;
const MASK: u32 = ${v.radix - 1}u;
`;
}

function histogramWgsl(v: Variant): string {
  return /* wgsl */ `
${passHeader(v)}
var<workgroup> localHist: array<atomic<u32>, ${v.radix}>;

@compute @workgroup_size(${BLOCK})
fn histogram(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let block = linearWorkgroup(wid, nwg);
  for (var d = lid; d < RADIX; d += ${BLOCK}u) { atomicStore(&localHist[d], 0u); }
  workgroupBarrier();
  let i = block * ${BLOCK}u + lid;
  if (i < info.n) {
    atomicAdd(&localHist[(keysIn[i] >> passParams.shift) & MASK], 1u);
  }
  workgroupBarrier();
  if (block < info.numBlocks) {
    for (var d = lid; d < RADIX; d += ${BLOCK}u) {
      hist[d * info.numBlocks + block] = atomicLoad(&localHist[d]);
    }
  }
}
`;
}

function scatterSubgroupWgsl(v: Variant): string {
  const tableSize = v.maxSubgroups * v.radix;
  return /* wgsl */ `
${passHeader(v)}
// counts[sg * RADIX + digit] for this block
var<workgroup> counts: array<u32, ${tableSize}>;

fn lanesBelow(lane: u32) -> vec4u {
  var m = vec4u(0u);
  for (var k = 0u; k < 4u; k++) {
    let below = clamp(i32(lane) - i32(k * 32u), 0, 32);
    if (below == 32) { m[k] = 0xffffffffu; }
    else if (below > 0) { m[k] = (1u << u32(below)) - 1u; }
  }
  return m;
}

fn popcount4(m: vec4u) -> u32 {
  let c = countOneBits(m);
  return c.x + c.y + c.z + c.w;
}

@compute @workgroup_size(${BLOCK})
fn scatter(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let block = linearWorkgroup(wid, nwg);
  let numSg = (${BLOCK}u + sgSize - 1u) / sgSize;
  for (var t = lid; t < numSg * RADIX; t += ${BLOCK}u) { counts[t] = 0u; }
  workgroupBarrier();

  let i = block * ${BLOCK}u + lid;
  let valid = i < info.n;
  var key = 0u;
  var val = 0u;
  if (valid) {
    key = keysIn[i];
    val = valsIn[i];
  }
  let d = (key >> passParams.shift) & MASK;
  // Lanes in this subgroup holding the same digit.
  var same = subgroupBallot(valid);
  for (var b = 0u; b < ${v.bits}u; b++) {
    let bit = ((d >> b) & 1u) == 1u;
    let ballot = subgroupBallot(bit);
    same &= select(~ballot, ballot, bit);
  }
  let rank = popcount4(same & lanesBelow(lane));
  let sg = lid / sgSize;
  if (valid && rank == 0u) { counts[sg * RADIX + d] = popcount4(same); }
  workgroupBarrier();
  if (valid) {
    var before = 0u;
    for (var s = 0u; s < sg; s++) { before += counts[s * RADIX + d]; }
    let dst = hist[d * info.numBlocks + block] + before + rank;
    keysOut[dst] = key;
    valsOut[dst] = val;
  }
}
`;
}

function scatterPortableWgsl(v: Variant): string {
  return /* wgsl */ `
${passHeader(v)}
// 256 threads × 8 packed words (16 digits × 16-bit counters)
var<workgroup> packed: array<u32, ${BLOCK * 8}>;

@compute @workgroup_size(${BLOCK})
fn scatter(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let block = linearWorkgroup(wid, nwg);
  let i = block * ${BLOCK}u + lid;
  let valid = i < info.n;
  var key = 0u;
  var val = 0u;
  if (valid) {
    key = keysIn[i];
    val = valsIn[i];
  }
  let d = (key >> passParams.shift) & MASK;
  let word = d >> 1u;
  let half = (d & 1u) * 16u;
  for (var w = 0u; w < 8u; w++) {
    var x = 0u;
    if (valid && w == word) { x = 1u << half; }
    packed[lid * 8u + w] = x;
  }
  workgroupBarrier();
  // Inclusive Hillis–Steele scan of the packed one-hot vectors.
  for (var off = 1u; off < ${BLOCK}u; off <<= 1u) {
    var t: array<u32, 8>;
    if (lid >= off) {
      for (var w = 0u; w < 8u; w++) { t[w] = packed[(lid - off) * 8u + w]; }
    }
    workgroupBarrier();
    if (lid >= off) {
      for (var w = 0u; w < 8u; w++) { packed[lid * 8u + w] += t[w]; }
    }
    workgroupBarrier();
  }
  if (valid) {
    let inclusive = (packed[lid * 8u + word] >> half) & 0xffffu;
    let dst = hist[d * info.numBlocks + block] + inclusive - 1u;
    keysOut[dst] = key;
    valsOut[dst] = val;
  }
}
`;
}

interface RadixKernels {
  variant: Variant;
  setup: Kernel;
  histogram: Kernel;
  scatter: Kernel;
}

const kernelCache = new WeakMap<GPUDevice, RadixKernels>();

const PASS_BINDINGS = [
  "read-only-storage",
  "uniform-dynamic",
  "read-only-storage",
  "read-only-storage",
  "storage",
  "storage",
  "storage",
] as const;

function pickVariant(device: GPUDevice): Variant {
  const info = (
    device as GPUDevice & { adapterInfo?: { subgroupMinSize?: number } }
  ).adapterInfo;
  const minSize = info?.subgroupMinSize ?? 0;
  const maxSubgroups = minSize > 0 ? Math.ceil(BLOCK / minSize) : 0;
  // 8-bit digits need maxSubgroups × 256 counters in shared memory.
  if (
    device.features.has("subgroups") &&
    maxSubgroups > 0 &&
    maxSubgroups <= 16
  ) {
    return { bits: 8, radix: 256, subgroups: true, maxSubgroups };
  }
  return { bits: 4, radix: 16, subgroups: false, maxSubgroups: 0 };
}

function radixKernels(device: GPUDevice): RadixKernels {
  let k = kernelCache.get(device);
  if (!k) {
    const v = pickVariant(device);
    k = {
      variant: v,
      setup: createKernel(device, "radix:setup", setupWgsl(v.radix), "setup", [
        "uniform",
        "read-only-storage",
        "storage",
        "storage",
      ]),
      histogram: createKernel(
        device,
        "radix:histogram",
        histogramWgsl(v),
        "histogram",
        PASS_BINDINGS,
      ),
      scatter: createKernel(
        device,
        "radix:scatter",
        v.subgroups ? scatterSubgroupWgsl(v) : scatterPortableWgsl(v),
        "scatter",
        PASS_BINDINGS,
      ),
    };
    kernelCache.set(device, k);
  }
  return k;
}

/** Passes (always even) needed to sort keys with `bits` significant bits. */
function radixPassCount(bits: number, digitBits: number): number {
  const p = Math.ceil(Math.max(1, Math.min(32, bits)) / digitBits);
  return p % 2 === 0 ? p : p + 1;
}

export class GpuRadixSort {
  readonly capacity: number;
  private readonly kernels: RadixKernels;
  private readonly tmpKeys: GPUBuffer;
  private readonly tmpVals: GPUBuffer;
  private readonly hist: GPUBuffer;
  private readonly info: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly setupParams: GPUBuffer;
  private readonly passParams: GPUBuffer;
  private readonly scan: GpuScan;
  private readonly groupCache = new Map<
    string,
    { forward: GPUBindGroup[]; backward: GPUBindGroup[] }
  >();
  private readonly setupCache = new Map<GPUBuffer, GPUBindGroup>();
  private lastSrcIndex = -1;
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 1;

  constructor(
    private readonly device: GPUDevice,
    capacity: number,
  ) {
    this.capacity = Math.max(BLOCK, capacity);
    this.kernels = radixKernels(device);
    const radix = this.kernels.variant.radix;
    const maxBlocks = Math.ceil(this.capacity / BLOCK);
    this.tmpKeys = createBuffer(
      device,
      "radix:tmp-keys",
      this.capacity * 4,
      STORAGE_RW,
    );
    this.tmpVals = createBuffer(
      device,
      "radix:tmp-vals",
      this.capacity * 4,
      STORAGE_RW,
    );
    this.hist = createBuffer(
      device,
      "radix:hist",
      maxBlocks * radix * 4,
      STORAGE_RW,
    );
    this.info = createBuffer(device, "radix:info", 16, STORAGE_RW);
    this.indirect = createBuffer(
      device,
      "radix:indirect",
      12,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
    this.setupParams = createBuffer(
      device,
      "radix:setup-params",
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.passParams = createBuffer(
      device,
      "radix:pass-params",
      UNIFORM_STRIDE * MAX_PASSES,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const digitBits = this.kernels.variant.bits;
    const pp = new Uint32Array((UNIFORM_STRIDE * MAX_PASSES) / 4);
    for (let p = 0; p < MAX_PASSES; p++) {
      pp[(p * UNIFORM_STRIDE) / 4] = Math.min(31, p * digitBits);
    }
    device.queue.writeBuffer(this.passParams, 0, pp);
    this.scan = new GpuScan(device, maxBlocks * radix);
  }

  private id(buf: GPUBuffer): number {
    let id = this.bufferIds.get(buf);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(buf, id);
    }
    return id;
  }

  /**
   * Sort `keys`/`vals` (first n entries, n = count.buffer[count.index]) by
   * the low `bits` bits of each key. Encode inside a compute pass.
   */
  encode(
    pass: GPUComputePassEncoder,
    keys: GPUBuffer,
    vals: GPUBuffer,
    count: CountSource,
    bits: number,
  ): void {
    const k = this.kernels;
    if (count.index !== this.lastSrcIndex) {
      this.device.queue.writeBuffer(
        this.setupParams,
        0,
        new Uint32Array([count.index, this.capacity, 0, 0]),
      );
      this.lastSrcIndex = count.index;
    }
    let sg = this.setupCache.get(count.buffer);
    if (!sg) {
      sg = bindGroup(this.device, k.setup, [
        this.setupParams,
        count.buffer,
        this.info,
        this.indirect,
      ]);
      if (this.setupCache.size > 8) this.setupCache.clear();
      this.setupCache.set(count.buffer, sg);
    }
    pass.setPipeline(k.setup.pipeline);
    pass.setBindGroup(0, sg);
    pass.dispatchWorkgroups(1);

    const groups = this.groupsFor(keys, vals);
    const passes = radixPassCount(bits, k.variant.bits);
    for (let p = 0; p < passes; p++) {
      const set = p % 2 === 0 ? groups.forward : groups.backward;
      const offset = [p * UNIFORM_STRIDE];
      pass.setPipeline(k.histogram.pipeline);
      pass.setBindGroup(0, set[0]!, offset);
      pass.dispatchWorkgroupsIndirect(this.indirect, 0);
      this.scan.encode(pass, this.hist, { buffer: this.info, index: 2 });
      pass.setPipeline(k.scatter.pipeline);
      pass.setBindGroup(0, set[1]!, offset);
      pass.dispatchWorkgroupsIndirect(this.indirect, 0);
    }
  }

  private groupsFor(
    keys: GPUBuffer,
    vals: GPUBuffer,
  ): { forward: GPUBindGroup[]; backward: GPUBindGroup[] } {
    const key = `${this.id(keys)}:${this.id(vals)}`;
    let g = this.groupCache.get(key);
    if (g) return g;
    const uniform = {
      buffer: this.passParams,
      offset: 0,
      size: 16,
    } as GPUBufferBinding;
    const make = (
      kIn: GPUBuffer,
      vIn: GPUBuffer,
      kOut: GPUBuffer,
      vOut: GPUBuffer,
    ) =>
      [this.kernels.histogram, this.kernels.scatter].map((kernel) =>
        bindGroup(this.device, kernel, [
          this.info,
          uniform,
          kIn,
          vIn,
          kOut,
          vOut,
          this.hist,
        ]),
      );
    g = {
      forward: make(keys, vals, this.tmpKeys, this.tmpVals),
      backward: make(this.tmpKeys, this.tmpVals, keys, vals),
    };
    if (this.groupCache.size > 8) this.groupCache.clear();
    this.groupCache.set(key, g);
    return g;
  }

  destroy(): void {
    this.tmpKeys.destroy();
    this.tmpVals.destroy();
    this.hist.destroy();
    this.info.destroy();
    this.indirect.destroy();
    this.setupParams.destroy();
    this.passParams.destroy();
    this.scan.destroy();
  }
}
