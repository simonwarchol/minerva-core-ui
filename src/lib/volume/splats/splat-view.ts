/**
 * Shows a Gaussian cloud on a canvas of its own laid over a KilnViewer's,
 * with the viewer's camera and live channel colours (which tint per-channel
 * clouds). The overlay should ignore pointer events so the viewer's canvas
 * keeps the camera controls.
 */
import type { KilnViewer } from "kiln-upstream";
import type { GaussianCloud } from "./cloud";
import { SplatLayer } from "./splat-layer";

/** Background the viewer composites the volume over (kiln's default). */
const BACKGROUND = [0.05, 0.05, 0.05];

export class SplatView {
  private readonly layer: SplatLayer;
  private readonly context: GPUCanvasContext;
  private raf = 0;
  /** What the last frame showed; redraw only when it changes. */
  private shown = "";

  /** `viewerCanvas` is the viewer's canvas (sets the size); `canvas` the overlay. */
  constructor(
    private readonly viewer: KilnViewer,
    private readonly viewerCanvas: HTMLCanvasElement,
    readonly canvas: HTMLCanvasElement,
  ) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("WebGPU canvas unavailable");
    context.configure({ device: viewer.device, format });
    this.context = context;
    this.layer = new SplatLayer(viewer.device, format);
    this.raf = requestAnimationFrame(this.frame);
  }

  get cloud(): GaussianCloud | null {
    return this.layer.cloud;
  }

  setCloud(cloud: GaussianCloud | null): void {
    this.layer.setCloud(cloud);
  }

  private readonly frame = () => {
    const { canvas } = this;
    const { width, height } = this.viewerCanvas;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const view = this.viewer.camera.getViewParams(width, height).view;
    const colors = this.viewer.renderer.channelColors;
    const key = `${this.layer.version}|${width}x${height}|${Array.from(view)}|${Array.from(colors)}`;
    if (key !== this.shown) {
      this.shown = key;
      this.layer.render(
        this.context.getCurrentTexture().createView(),
        view,
        width,
        height,
        { background: BACKGROUND, channelColors: colors },
      );
    }
    this.raf = requestAnimationFrame(this.frame);
  };

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.layer.destroy();
    this.context.unconfigure();
  }
}
