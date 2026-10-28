/**
 * SplatLayer — draws a Gaussian cloud with the same camera as kiln's volume
 * renderer (its view matrix; perspective, kiln's fixed 45° field of view),
 * using the tile rasterizer.
 */

import type { GaussianCloud } from "./cloud";
import { intrinsicsFor, PARAM_STRIDE, type SplatCamera } from "./cpu-reference";
import { type DepthCue, NO_DEPTH_CUE } from "./depth-cue";
import { createBuffer, STORAGE_RW } from "./gpu/kernel";
import { SplatRasterizer } from "./gpu/rasterizer";

const COMPOSITE_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  splitX: f32,
  _pad: f32,
  background: vec4f,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image: array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  if (pos.x < params.splitX) { discard; }
  let px = min(u32(pos.x), params.width - 1u);
  let py = min(u32(pos.y), params.height - 1u);
  let c = image[py * params.width + px];
  let bg = params.background;
  return vec4f(c.rgb + bg.rgb * (1.0 - c.a), 1.0);
}
`;

/** Camera intrinsics matching kiln's volume renderer for a view matrix and size. */
export function splatCameraFor(
  view: ArrayLike<number>,
  width: number,
  height: number,
): SplatCamera {
  return intrinsicsFor(Float32Array.from(view), width, height);
}

export class SplatLayer {
  readonly rasterizer: SplatRasterizer;
  private readonly pipeline: GPURenderPipeline;
  private readonly uniform: GPUBuffer;
  private ownParams: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private boundImage: GPUBuffer | null = null;
  private count = 0;
  private readonly uniformData = new ArrayBuffer(32);
  /** Cloud currently displayed (null when none). */
  cloud: GaussianCloud | null = null;
  /** Bumped whenever the Gaussians change, so the viewer re-renders. */
  version = 0;

  constructor(
    readonly device: GPUDevice,
    format: GPUTextureFormat,
  ) {
    this.rasterizer = new SplatRasterizer(device);
    const module = device.createShaderModule({
      label: "splat:composite",
      code: COMPOSITE_WGSL,
    });
    this.pipeline = device.createRenderPipeline({
      label: "splat:composite",
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.uniform = createBuffer(
      device,
      "splat:composite-params",
      32,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
  }

  /** GPU parameters of the cloud set with {@link setCloud} (null otherwise). */
  get paramsBuffer(): GPUBuffer | null {
    return this.ownParams;
  }

  get gaussianCount(): number {
    return this.count;
  }

  /** Display a CPU cloud (uploaded to a buffer owned by the layer). */
  setCloud(cloud: GaussianCloud | null): void {
    this.cloud = cloud;
    this.ownParams?.destroy();
    this.ownParams = null;
    if (!cloud || cloud.count === 0) {
      this.count = 0;
      this.version++;
      return;
    }
    const bytes = cloud.count * PARAM_STRIDE * 4;
    this.ownParams = createBuffer(
      this.device,
      "splat:layer-params",
      bytes,
      STORAGE_RW,
    );
    this.device.queue.writeBuffer(
      this.ownParams,
      0,
      cloud.params.buffer,
      cloud.params.byteOffset,
      bytes,
    );
    this.count = cloud.count;
    this.rasterizer.mipFilter = cloud.antialiased ?? false;
    this.rasterizer.channelMode =
      cloud.channelComposite === "add"
        ? 2
        : cloud.channelComposite === "over"
          ? 1
          : 0;
    this.rasterizer.setGaussians(this.ownParams, cloud.count);
    this.version++;
  }

  /**
   * Rasterise and composite onto `target`. With `splitX > 0`, only pixels at
   * or right of splitX are drawn (the volume stays visible on the left).
   * `channelColors` (rgba per channel, a = shown) tint per-channel clouds;
   * `depthCue` fades them with distance (core/depth-cue.ts).
   */
  render(
    target: GPUTextureView,
    view: ArrayLike<number>,
    width: number,
    height: number,
    opts: {
      background: ArrayLike<number>;
      splitX?: number;
      clear?: boolean;
      channelColors?: ArrayLike<number>;
      depthCue?: DepthCue;
    },
  ): void {
    const r = this.rasterizer;
    r.resize(width, height);
    r.depthCue = opts.depthCue ?? NO_DEPTH_CUE;
    if (opts.channelColors) {
      const t = r.channelTint;
      for (let i = 0; i < t.length; i += 4) {
        const a = opts.channelColors[i + 3] ?? 0;
        // Volume rendering weights a channel's colour by its alpha.
        t[i] = (opts.channelColors[i] ?? 0) * a;
        t[i + 1] = (opts.channelColors[i + 1] ?? 0) * a;
        t[i + 2] = (opts.channelColors[i + 2] ?? 0) * a;
        t[i + 3] = a;
      }
    }
    const encoder = this.device.createCommandEncoder({ label: "splat:frame" });
    if (this.count > 0) {
      r.encodeForward(encoder, splatCameraFor(view, width, height));
    } else {
      encoder.clearBuffer(r.image);
    }
    const u = this.uniformData;
    const ui = new Uint32Array(u);
    const uf = new Float32Array(u);
    ui[0] = width;
    ui[1] = height;
    uf[2] = opts.splitX ?? 0;
    uf[4] = opts.background[0] ?? 0;
    uf[5] = opts.background[1] ?? 0;
    uf[6] = opts.background[2] ?? 0;
    uf[7] = 1;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    if (!this.bindGroup || this.boundImage !== r.image) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: r.image } },
        ],
      });
      this.boundImage = r.image;
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: target,
          loadOp: opts.clear === false ? "load" : "clear",
          clearValue: [0, 0, 0, 1],
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    if (this.count > 0) r.checkCapacity();
  }

  destroy(): void {
    this.ownParams?.destroy();
    this.uniform.destroy();
    this.rasterizer.destroy();
  }
}
