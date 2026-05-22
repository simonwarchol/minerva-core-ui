import { COORDINATE_SYSTEM } from "@deck.gl/core";
import { TileLayer, type TileLayerProps } from "@deck.gl/geo-layers";
import type { Loader } from "@/lib/imaging/viv";
import { MaskBitmapLayer, type MaskBitmapLayerProps } from "./MaskBitmapLayer";

type Selection = { t: number; z: number; c: number };

function tileRaster(tile: unknown): {
  data: ArrayLike<number>;
  width: number;
  height: number;
} | null {
  if (!tile || typeof tile !== "object") return null;
  const t = tile as {
    data?: ArrayLike<number>;
    width?: number;
    height?: number;
  };
  if (!t.data) return null;
  const width = t.width ?? Math.sqrt(t.data.length);
  const height = t.height ?? width;
  return { data: t.data, width, height };
}

/**
 * Pack the integer label tile into an RGBA `ImageData`. R+G+B carry the
 * 24-bit cell id (LSB first); A = 255 for foreground, 0 for background.
 * Done once per tile fetch — opacity / outlines never trigger a re-encode.
 */
function encodeLabelTile(
  raw: ArrayLike<number>,
  width: number,
  height: number,
): ImageData {
  const img = new ImageData(width, height);
  const out = img.data;
  for (let i = 0; i < width * height; i++) {
    const id = Math.round(raw[i] ?? 0);
    const o = i * 4;
    if (id <= 0) {
      out[o + 3] = 0;
      continue;
    }
    out[o] = id & 0xff;
    out[o + 1] = (id >> 8) & 0xff;
    out[o + 2] = (id >> 16) & 0xff;
    out[o + 3] = 255;
  }
  return img;
}

export type MaskOverlayOptions = {
  maskLoader: Loader;
  sourceImageId: string;
  opacity: number;
  outlines: boolean;
  id?: string;
};

type MaskTileLayerProps = TileLayerProps & {
  maskOpacity: number;
  maskOutlines: boolean;
};

/** Pushes live mask uniforms onto cached tile sublayers without re-fetching tiles. */
class MaskTileLayer extends TileLayer<MaskTileLayerProps> {
  static layerName = "MaskTileLayer";

  getSubLayerPropsByTile(_tile: unknown) {
    const { maskOpacity, maskOutlines } = this
      .props as unknown as MaskTileLayerProps;
    return {
      maskOpacity,
      maskOutlines,
    } as Partial<MaskBitmapLayerProps>;
  }
}

/**
 * Tile pyramid of `MaskBitmapLayer`s. Tiles are CPU-encoded once
 * (id → RGB packing); opacity and outlines are GPU uniforms.
 */
export function createMaskOverlayLayer(opts: MaskOverlayOptions) {
  const level = opts.maskLoader.data[0];
  if (!level) return null;

  const tileSize = level.tileSize ?? 256;
  const xIdx = level.labels.indexOf("x");
  const yIdx = level.labels.indexOf("y");
  const width = xIdx >= 0 ? level.shape[xIdx] : 0;
  const height = yIdx >= 0 ? level.shape[yIdx] : 0;
  if (width <= 0 || height <= 0) return null;

  return new MaskTileLayer({
    id: opts.id ?? `mask-overlay-${opts.sourceImageId}`,
    maskOpacity: opts.opacity,
    maskOutlines: opts.outlines,
    // Match Viv's MultiscaleImageLayerBase so `tile.bbox` is reported as
    // image-pixel `{left, top, right, bottom}` and overlays sit directly
    // on top of the underlying image tiles instead of being projected
    // through deck.gl's default LNGLAT coordinate system.
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    extent: [0, 0, width, height],
    tileSize,
    maxZoom: 0,
    minZoom: 0,
    pickable: false,
    opacity: 1,
    renderSubLayers: (props) => {
      const layerProps = props as MaskTileLayerProps;
      const bbox = props.tile.bbox as {
        left: number;
        bottom: number;
        right: number;
        top: number;
      };
      const { left, top } = bbox;
      let { right, bottom } = bbox;
      if ([left, bottom, right, top].some((v) => typeof v !== "number")) {
        return null;
      }
      const raster = tileRaster(props.data);
      if (!raster) return null;
      // Pixel dimensions come from the loader (authority); the world-coord
      // bbox is only used to position the bitmap. Trusting `right - left`
      // would explode for stripped masks where one tile covers a huge
      // extent (>30k px), which `new ImageData(...)` rejects.
      const w = raster.width;
      const h = raster.height;
      if (w <= 0 || h <= 0) return null;
      // Edge-tile clamp (mirrors Viv): when the loader returns a tile
      // smaller than `tileSize`, the tile's bbox still spans the full
      // tileSize so the BitmapLayer would stretch the partial raster.
      // Clamp `right`/`bottom` to the actual image extent so the bitmap
      // is placed at native pixel size — matching how Viv's image layer
      // positions its own bottom/right edge tiles.
      if (w < tileSize) right = Math.min(right, left + w);
      if (h < tileSize) bottom = Math.min(bottom, top + h);
      const image = encodeLabelTile(raster.data, w, h);
      return new MaskBitmapLayer(props as never, {
        image,
        bounds: [left, bottom, right, top],
        _imageCoordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        // Label images MUST sample with nearest neighbor — linear filtering
        // interpolates between two integer cell ids and decodes to garbage,
        // showing up as snow / static at zoomed-out scales (especially with
        // the outline shader doing 4-neighbor lookups).
        textureParameters: {
          minFilter: "nearest",
          magFilter: "nearest",
          mipmapFilter: "nearest",
        },
        pickable: false,
        maskOpacity: layerProps.maskOpacity,
        maskOutlines: layerProps.maskOutlines,
        texelSize: [1 / w, 1 / h],
      });
    },
    getTileData: async ({ index, signal }) => {
      const { x, y } = index;
      const sel: Selection = { t: 0, z: 0, c: 0 };
      return level.getTile({ x, y, selection: sel, signal });
    },
  } as never);
}
