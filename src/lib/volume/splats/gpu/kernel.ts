/**
 * Small helpers for compute kernels with explicit bind-group layouts.
 *
 * Explicit layouts (instead of `layout: "auto"`) let one bind group carry
 * bindings an entry point doesn't statically use, so kernels that share a
 * module can share resource lists without "binding not in layout" errors.
 */

type BindingKind =
  | "uniform"
  | "uniform-dynamic"
  | "storage"
  | "read-only-storage"
  | "texture-unfilterable-array";

export interface Kernel {
  readonly label: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
}

function layoutEntry(
  binding: number,
  kind: BindingKind,
  visibility: number,
): GPUBindGroupLayoutEntry {
  switch (kind) {
    case "uniform":
      return { binding, visibility, buffer: { type: "uniform" } };
    case "uniform-dynamic":
      return {
        binding,
        visibility,
        buffer: { type: "uniform", hasDynamicOffset: true },
      };
    case "storage":
      return { binding, visibility, buffer: { type: "storage" } };
    case "read-only-storage":
      return { binding, visibility, buffer: { type: "read-only-storage" } };
    case "texture-unfilterable-array":
      return {
        binding,
        visibility,
        texture: {
          sampleType: "unfilterable-float",
          viewDimension: "2d-array",
        },
      };
  }
}

const moduleCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

function shaderModule(
  device: GPUDevice,
  label: string,
  code: string,
): GPUShaderModule {
  let cache = moduleCache.get(device);
  if (!cache) {
    cache = new Map();
    moduleCache.set(device, cache);
  }
  let mod = cache.get(code);
  if (!mod) {
    mod = device.createShaderModule({ label, code });
    cache.set(code, mod);
  }
  return mod;
}

/** Build a compute kernel whose bindings are 0..n-1 in group 0. */
export function createKernel(
  device: GPUDevice,
  label: string,
  code: string,
  entryPoint: string,
  bindings: readonly BindingKind[],
  constants?: Record<string, number>,
): Kernel {
  const layout = device.createBindGroupLayout({
    label: `${label}:layout`,
    entries: bindings.map((kind, i) =>
      layoutEntry(i, kind, GPUShaderStage.COMPUTE),
    ),
  });
  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: {
      module: shaderModule(device, label, code),
      entryPoint,
      constants,
    },
  });
  return { label, pipeline, layout };
}

/** Bind resources[i] to binding i of the kernel's layout. */
export function bindGroup(
  device: GPUDevice,
  kernel: Kernel,
  resources: ReadonlyArray<GPUBuffer | GPUBindingResource>,
): GPUBindGroup {
  return device.createBindGroup({
    label: `${kernel.label}:bg`,
    layout: kernel.layout,
    entries: resources.map((r, i) => ({
      binding: i,
      resource: r instanceof GPUBuffer ? { buffer: r } : r,
    })),
  });
}

const MAX_WORKGROUPS_PER_DIM = 65535;

export function dispatch1d(
  pass: GPUComputePassEncoder,
  kernel: Kernel,
  group: GPUBindGroup,
  workgroups: number,
): void {
  if (workgroups <= 0) return;
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, group);
  // Split into a 2D grid that respects the per-dimension limit.
  const n = Math.ceil(workgroups);
  if (n <= MAX_WORKGROUPS_PER_DIM) pass.dispatchWorkgroups(n, 1, 1);
  else
    pass.dispatchWorkgroups(
      MAX_WORKGROUPS_PER_DIM,
      Math.ceil(n / MAX_WORKGROUPS_PER_DIM),
      1,
    );
}

/** WGSL snippet: linear workgroup index for 2D-split 1D dispatches. */
export const WGSL_LINEAR_WORKGROUP = /* wgsl */ `
fn linearWorkgroup(wid: vec3u, nwg: vec3u) -> u32 {
  return wid.x + wid.y * nwg.x;
}
`;

export function createBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: number,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(16, Math.ceil(size / 4) * 4),
    usage,
  });
}

export const STORAGE_RW =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** Usage of a reusable staging buffer that GPU copies land in for reading. */
export const READBACK = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

/**
 * Map a READBACK buffer, run `fn` on its contents (valid only during the
 * call) and unmap it again.
 */
export async function withMapped<T>(
  buffer: GPUBuffer,
  fn: (data: ArrayBuffer) => T,
): Promise<T> {
  await buffer.mapAsync(GPUMapMode.READ);
  try {
    return fn(buffer.getMappedRange());
  } finally {
    buffer.unmap();
  }
}

/** Read a GPU buffer back to the CPU (debug / occasional reads only). */
export async function readBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  byteOffset = 0,
  byteLength: number = src.size - byteOffset,
): Promise<ArrayBuffer> {
  const size = Math.ceil(byteLength / 4) * 4;
  const staging = createBuffer(device, "readback", size, READBACK);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, byteOffset, staging, 0, size);
  device.queue.submit([enc.finish()]);
  try {
    return await withMapped(staging, (d) => d.slice(0, byteLength));
  } finally {
    staging.destroy();
  }
}
