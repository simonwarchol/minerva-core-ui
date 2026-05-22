/**
 * Mask OME-TIFF loader.
 *
 * Tries Viv's `loadOmeTiff` first (full pyramid + OME metadata path). When
 * that throws `GeoTIFFImageIndexError` ("No image at index N"), the file's
 * OME-XML declares more `<Image>` entries than the file actually contains
 * — common with mask writers (mcmicro, custom Python tools) that bundle a
 * synthetic OME header but only write one IFD. We then fall back to a raw
 * single-IFD GeoTIFF loader, shaped to look like a one-channel Viv loader
 * so {@link MaskOverlayLayer} can consume it unchanged.
 */

import { loadOmeTiff } from "@hms-dbmi/viv";
import { fromBlob, fromUrl } from "geotiff";
import type { Loader } from "./viv";
import type { PoolClass } from "./workers/Pool";

type GeoTiff = Awaited<ReturnType<typeof fromBlob>>;
type GeoImage = Awaited<ReturnType<GeoTiff["getImage"]>>;

function isMissingIfdError(err: unknown): boolean {
  if (err instanceof Error) return /No image at index/i.test(err.message);
  return false;
}

const VIV_DTYPE_BY_KEY = {
  Uint8: "Uint8",
  Uint16: "Uint16",
  Uint32: "Uint32",
  Int8: "Int8",
  Int16: "Int16",
  Int32: "Int32",
  Float32: "Float32",
  Float64: "Float64",
} as const;

function dtypeFromImage(image: GeoImage): string {
  const dir = (
    image as unknown as {
      fileDirectory: { SampleFormat?: number[]; BitsPerSample: number[] };
    }
  ).fileDirectory;
  const format = dir.SampleFormat?.[0] ?? 1;
  const bps = dir.BitsPerSample?.[0] ?? 8;
  switch (format) {
    case 3:
      return bps <= 32 ? VIV_DTYPE_BY_KEY.Float32 : VIV_DTYPE_BY_KEY.Float64;
    case 2:
      if (bps <= 8) return VIV_DTYPE_BY_KEY.Int8;
      if (bps <= 16) return VIV_DTYPE_BY_KEY.Int16;
      return VIV_DTYPE_BY_KEY.Int32;
    default:
      if (bps <= 8) return VIV_DTYPE_BY_KEY.Uint8;
      if (bps <= 16) return VIV_DTYPE_BY_KEY.Uint16;
      return VIV_DTYPE_BY_KEY.Uint32;
  }
}

/**
 * Default chunk size used when the source TIFF is stripped (not tiled) — i.e.
 * `getTileWidth()` returns the full image width. Without this, a 30000×30000
 * mask would produce a single TileLayer tile covering the whole image and
 * attempt to build a ~3.6 GB `ImageData`. 1024 keeps memory bounded while
 * still being friendly to `geotiff.js` strip reads.
 */
const STRIPPED_FALLBACK_TILE_SIZE = 1024;

async function loadSingleIfdAsLoader(
  src: { file: File } | { url: string },
): Promise<Loader> {
  const tiff =
    "file" in src ? await fromBlob(src.file) : await fromUrl(src.url);
  const image = await tiff.getImage(0);
  const width = image.getWidth();
  const height = image.getHeight();
  const rawTileW = image.getTileWidth() || 0;
  const rawTileH = image.getTileHeight() || 0;
  const looksStripped =
    rawTileW <= 0 || rawTileH <= 0 || rawTileW >= width || rawTileH >= height;
  const tileSize = looksStripped
    ? STRIPPED_FALLBACK_TILE_SIZE
    : Math.min(rawTileW, rawTileH);
  const dtype = dtypeFromImage(image);

  const labels: ["t", "c", "z", "y", "x"] = ["t", "c", "z", "y", "x"];
  const shape = [1, 1, 1, height, width];

  const level = {
    dtype,
    tileSize,
    shape,
    labels,
    onTileError(e: Error) {
      console.error("[mask] tile error:", e);
    },
    async getTile({
      x,
      y,
      signal,
    }: {
      x: number;
      y: number;
      selection?: { t: number; z: number; c: number };
      signal?: AbortSignal;
    }) {
      const x0 = x * tileSize;
      const y0 = y * tileSize;
      const x1 = Math.min(width, x0 + tileSize);
      const y1 = Math.min(height, y0 + tileSize);
      const w = x1 - x0;
      const h = y1 - y0;
      const raster = (await image.readRasters({
        window: [x0, y0, x1, y1],
        interleave: false,
        signal,
      } as unknown as Parameters<typeof image.readRasters>[0])) as
        | ArrayLike<number>
        | ArrayLike<number>[];
      const data = Array.isArray(raster) ? raster[0] : raster;
      return { data, width: w, height: h } as never;
    },
    async getRaster({
      signal,
    }: {
      selection?: { t: number; z: number; c: number };
      signal?: AbortSignal;
    }) {
      const raster = (await image.readRasters({
        interleave: false,
        signal,
      } as unknown as Parameters<typeof image.readRasters>[0])) as
        | ArrayLike<number>
        | ArrayLike<number>[];
      const data = Array.isArray(raster) ? raster[0] : raster;
      return { data, width, height } as never;
    },
  };

  const metadata = {
    ID: "Image:0",
    AquisitionDate: "",
    Description: "",
    Pixels: {
      Channels: [{ ID: "Channel:0:0", Name: "Mask", SamplesPerPixel: 1 }],
      ID: "Pixels:0",
      DimensionOrder: "XYZCT",
      Type: dtype.toLowerCase(),
      SizeT: 1,
      SizeC: 1,
      SizeZ: 1,
      SizeY: height,
      SizeX: width,
      PhysicalSizeX: 1,
      PhysicalSizeY: 1,
      PhysicalSizeXUnit: "px",
      PhysicalSizeYUnit: "px",
      PhysicalSizeZUnit: "px",
      BigEndian: false,
      TiffData: [],
    },
    ROIs: [],
  };

  return { data: [level], metadata } as unknown as Loader;
}

/**
 * Load an OME-TIFF segmentation mask from either a local `File` or a remote
 * URL. Tries Viv's pyramidal loader first; if the OME-XML declares phantom
 * IFDs we fall back to a single-IFD reader using `geotiff.fromBlob` /
 * `geotiff.fromUrl` as appropriate.
 */
export async function loadMaskTiff(
  args: ({ file: File } | { url: string }) & { pool?: PoolClass },
): Promise<Loader> {
  const { pool } = args;
  const source = "file" in args ? args.file : args.url;
  try {
    if (pool) {
      return (await loadOmeTiff(source, { pool: pool as never })) as Loader;
    }
    return (await loadOmeTiff(source)) as Loader;
  } catch (err) {
    if (!isMissingIfdError(err)) throw err;
    console.warn(
      "[mask] OME-TIFF metadata declared more images than the file " +
        "contains; falling back to a single-IFD loader.",
      err,
    );
    return loadSingleIfdAsLoader(
      "file" in args ? { file: args.file } : { url: args.url },
    );
  }
}
