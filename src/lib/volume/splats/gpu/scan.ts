/**
 * GPU-driven exclusive prefix sum (u32, in place).
 *
 * The element count is read from a GPU buffer, and every dispatch is
 * indirect, so a scan can follow a kernel whose output size is only known on
 * the GPU (e.g. tile/Gaussian intersection counts) without a CPU round trip.
 * Reduce-then-scan over levels of 1024-element blocks.
 */

import {
  bindGroup,
  createBuffer,
  createKernel,
  type Kernel,
  STORAGE_RW,
  WGSL_LINEAR_WORKGROUP,
} from "./kernel";

const SCAN_BLOCK = 1024;
const MAX_LEVELS = 8;
const UNIFORM_STRIDE = 256;

const SCAN_WGSL = /* wgsl */ `
${WGSL_LINEAR_WORKGROUP}

struct SetupParams { srcIndex: u32, numLevels: u32, capacity: u32, _pad: u32 }
struct LevelParams { level: u32, writeSums: u32, _a: u32, _b: u32 }

@group(0) @binding(0) var<uniform> setupParams: SetupParams;
@group(0) @binding(1) var<storage, read> countSrc: array<u32>;
@group(0) @binding(2) var<storage, read_write> setupInfo: array<u32>;
@group(0) @binding(3) var<storage, read_write> setupIndirect: array<u32>;

@compute @workgroup_size(1)
fn setup() {
  var n = min(countSrc[setupParams.srcIndex], setupParams.capacity);
  for (var l = 0u; l < ${MAX_LEVELS}u; l++) {
    let blocks = (n + ${SCAN_BLOCK - 1}u) / ${SCAN_BLOCK}u;
    if (l < setupParams.numLevels) {
      setupInfo[l] = n;
      let x = min(blocks, 65535u);
      let y = select(1u, (blocks + 65534u) / 65535u, blocks > 65535u);
      setupIndirect[l * 3u] = x;
      setupIndirect[l * 3u + 1u] = y;
      setupIndirect[l * 3u + 2u] = 1u;
    }
    n = blocks;
  }
}
`;

const SCAN_LEVEL_WGSL = /* wgsl */ `
${WGSL_LINEAR_WORKGROUP}

struct LevelParams { level: u32, writeSums: u32, _a: u32, _b: u32 }

@group(0) @binding(0) var<storage, read> info: array<u32>;
@group(0) @binding(1) var<uniform> lvl: LevelParams;
@group(0) @binding(2) var<storage, read_write> data: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<u32>;

var<workgroup> wsum: array<u32, 256>;

@compute @workgroup_size(256)
fn scanLocal(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let n = info[lvl.level];
  let block = linearWorkgroup(wid, nwg);
  let base = block * ${SCAN_BLOCK}u + lid * 4u;
  var v: array<u32, 4>;
  var s = 0u;
  for (var k = 0u; k < 4u; k++) {
    let idx = base + k;
    var x = 0u;
    if (idx < n) { x = data[idx]; }
    v[k] = x;
    s += x;
  }
  wsum[lid] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var t = 0u;
    if (lid >= off) { t = wsum[lid - off]; }
    workgroupBarrier();
    wsum[lid] += t;
    workgroupBarrier();
  }
  var run = wsum[lid] - s;
  for (var k = 0u; k < 4u; k++) {
    let idx = base + k;
    if (idx < n) { data[idx] = run; }
    run += v[k];
  }
  if (lid == 255u && lvl.writeSums != 0u) {
    sums[block] = wsum[255];
  }
}

@compute @workgroup_size(256)
fn scanAdd(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let n = info[lvl.level];
  let block = linearWorkgroup(wid, nwg);
  let add = sums[block];
  let base = block * ${SCAN_BLOCK}u + lid * 4u;
  for (var k = 0u; k < 4u; k++) {
    let idx = base + k;
    if (idx < n) { data[idx] += add; }
  }
}
`;

interface ScanKernels {
  setup: Kernel;
  local: Kernel;
  add: Kernel;
}

const kernelCache = new WeakMap<GPUDevice, ScanKernels>();

function scanKernels(device: GPUDevice): ScanKernels {
  let k = kernelCache.get(device);
  if (!k) {
    k = {
      setup: createKernel(device, "scan:setup", SCAN_WGSL, "setup", [
        "uniform",
        "read-only-storage",
        "storage",
        "storage",
      ]),
      local: createKernel(device, "scan:local", SCAN_LEVEL_WGSL, "scanLocal", [
        "read-only-storage",
        "uniform-dynamic",
        "storage",
        "storage",
      ]),
      add: createKernel(device, "scan:add", SCAN_LEVEL_WGSL, "scanAdd", [
        "read-only-storage",
        "uniform-dynamic",
        "storage",
        "storage",
      ]),
    };
    kernelCache.set(device, k);
  }
  return k;
}

export interface CountSource {
  buffer: GPUBuffer;
  /** u32 element index of the count inside `buffer`. */
  index: number;
}

/** Number of reduction levels needed so the top level fits one block. */
function scanLevelCount(capacity: number): number {
  let levels = 1;
  let n = Math.max(1, capacity);
  while (n > SCAN_BLOCK) {
    n = Math.ceil(n / SCAN_BLOCK);
    levels++;
  }
  if (levels > MAX_LEVELS) throw new Error("scan capacity too large");
  return levels;
}

export class GpuScan {
  readonly capacity: number;
  private readonly kernels: ScanKernels;
  private readonly levels: number;
  private readonly sums: GPUBuffer[] = [];
  private readonly info: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly setupParams: GPUBuffer;
  private readonly levelParams: GPUBuffer;
  private readonly dummy: GPUBuffer;
  private readonly setupGroups = new Map<GPUBuffer, GPUBindGroup>();
  private readonly levelGroups = new Map<GPUBuffer, GPUBindGroup[][]>();
  private srcIndexWritten = -1;

  constructor(
    private readonly device: GPUDevice,
    capacity: number,
  ) {
    this.capacity = Math.max(1, capacity);
    this.kernels = scanKernels(device);
    this.levels = scanLevelCount(this.capacity);
    let n = this.capacity;
    for (let l = 1; l < this.levels; l++) {
      n = Math.ceil(n / SCAN_BLOCK);
      this.sums.push(createBuffer(device, `scan:sums${l}`, n * 4, STORAGE_RW));
    }
    this.info = createBuffer(device, "scan:info", MAX_LEVELS * 4, STORAGE_RW);
    this.indirect = createBuffer(
      device,
      "scan:indirect",
      MAX_LEVELS * 12,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
    this.setupParams = createBuffer(
      device,
      "scan:setup-params",
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.levelParams = createBuffer(
      device,
      "scan:level-params",
      UNIFORM_STRIDE * MAX_LEVELS,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const lp = new Uint32Array((UNIFORM_STRIDE * MAX_LEVELS) / 4);
    for (let l = 0; l < this.levels; l++) {
      lp[(l * UNIFORM_STRIDE) / 4] = l;
      lp[(l * UNIFORM_STRIDE) / 4 + 1] = l < this.levels - 1 ? 1 : 0;
    }
    device.queue.writeBuffer(this.levelParams, 0, lp);
    this.dummy = createBuffer(device, "scan:dummy", 16, STORAGE_RW);
  }

  /**
   * Exclusive scan of `data[0..n)` in place, where n = `count.buffer[count.index]`
   * (clamped to capacity). Must be encoded inside a compute pass.
   */
  encode(
    pass: GPUComputePassEncoder,
    data: GPUBuffer,
    count: CountSource,
  ): void {
    if (count.index !== this.srcIndexWritten) {
      this.device.queue.writeBuffer(
        this.setupParams,
        0,
        new Uint32Array([count.index, this.levels, this.capacity, 0]),
      );
      this.srcIndexWritten = count.index;
    }
    const k = this.kernels;
    let sg = this.setupGroups.get(count.buffer);
    if (!sg) {
      sg = bindGroup(this.device, k.setup, [
        this.setupParams,
        count.buffer,
        this.info,
        this.indirect,
      ]);
      if (this.setupGroups.size > 8) this.setupGroups.clear();
      this.setupGroups.set(count.buffer, sg);
    }
    pass.setPipeline(k.setup.pipeline);
    pass.setBindGroup(0, sg);
    pass.dispatchWorkgroups(1);

    const groups = this.groupsFor(data);
    // Up-sweep: scan each level, emitting block sums into the next.
    for (let l = 0; l < this.levels; l++) {
      pass.setPipeline(k.local.pipeline);
      pass.setBindGroup(0, groups[l]![0]!, [l * UNIFORM_STRIDE]);
      pass.dispatchWorkgroupsIndirect(this.indirect, l * 12);
    }
    // Down-sweep: add scanned block sums back.
    for (let l = this.levels - 2; l >= 0; l--) {
      pass.setPipeline(k.add.pipeline);
      pass.setBindGroup(0, groups[l]![1]!, [l * UNIFORM_STRIDE]);
      pass.dispatchWorkgroupsIndirect(this.indirect, l * 12);
    }
  }

  private groupsFor(data: GPUBuffer): GPUBindGroup[][] {
    let g = this.levelGroups.get(data);
    if (g) return g;
    g = [];
    for (let l = 0; l < this.levels; l++) {
      const levelData = l === 0 ? data : this.sums[l - 1]!;
      const levelSums = l < this.levels - 1 ? this.sums[l]! : this.dummy;
      const uniform = {
        buffer: this.levelParams,
        offset: 0,
        size: 16,
      } as GPUBufferBinding;
      g.push([
        bindGroup(this.device, this.kernels.local, [
          this.info,
          uniform,
          levelData,
          levelSums,
        ]),
        bindGroup(this.device, this.kernels.add, [
          this.info,
          uniform,
          levelData,
          levelSums,
        ]),
      ]);
    }
    if (this.levelGroups.size > 8) this.levelGroups.clear();
    this.levelGroups.set(data, g);
    return g;
  }

  destroy(): void {
    for (const b of this.sums) b.destroy();
    this.info.destroy();
    this.indirect.destroy();
    this.setupParams.destroy();
    this.levelParams.destroy();
    this.dummy.destroy();
  }
}
