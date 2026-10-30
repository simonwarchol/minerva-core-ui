import type { ImsChannelInfo } from "kiln-render";
import { urlFileName } from "@/lib/imaging/sourceUrl";
import type {
  Image,
  ImageChannel,
  ImageSourceVolume,
} from "@/lib/stores/documentSchema";
import { useDocumentStore } from "@/lib/stores/documentStore";

/** kiln-render is ~2 MB; only stories with a volume load it. */
export const loadKiln = () => import("./kiln");

/** `.ims` path (query / hash ignored). */
export function looksLikeImsUrl(url: string): boolean {
  try {
    return /\.ims$/i.test(new URL(url, "https://x.invalid").pathname);
  } catch {
    return /\.ims(?:[?#]|$)/i.test(url);
  }
}

export type VolumeProbe = {
  name: string;
  /** Finest-level voxels [x, y, z] */
  dimensions: [number, number, number];
  /** File voxel type, e.g. "float32", "uint16" */
  dtype: string;
  channels: ImsChannelInfo[];
};

/** Read Imaris metadata (header + attributes only; no voxel data). */
export async function probeImsVolume(url: string): Promise<VolumeProbe> {
  const { ImarisDataProvider } = await loadKiln();
  const provider = new ImarisDataProvider(url, { workers: 1 });
  try {
    const meta = await provider.initialize();
    return {
      name: meta.name,
      dimensions: meta.dimensions,
      dtype: meta.sourceDataType ?? "uint16",
      channels: provider.allChannels,
    };
  } finally {
    provider.dispose();
  }
}

/** Where a volume's saved chunk index is hosted next to it. */
export function chunkIndexBeside(url: string): string {
  const u = new URL(url, window.location.href);
  u.pathname += ".index";
  return u.href;
}

/**
 * A saved chunk index of `channels` to export with a story, or null when one
 * is hosted next to the volume (viewers read that one).
 */
export async function chunkIndexForExport(
  url: string,
  channels: number[],
  onProgress?: (done: number, total: number) => void,
): Promise<Uint8Array | null> {
  const { ImarisDataProvider } = await loadKiln();
  const provider = new ImarisDataProvider(url, { workers: 1 });
  try {
    await provider.initialize();
    if ((await provider.chunkIndexInUse()) === chunkIndexBeside(url)) {
      return null;
    }
    return await provider.buildChunkIndex(channels, onProgress);
  } finally {
    provider.dispose();
  }
}

/**
 * Document image for a volume import: channels carry the file's names and
 * display ranges. As with 2D imports there are no channel groups, and the
 * file's colours (mostly Imaris's default white) aren't used: the import
 * palette colours channels (`applyPaletteToFlatImportImages`).
 */
export function buildVolumeImport(url: string, probe: VolumeProbe): Image {
  const channels: ImageChannel[] = probe.channels.map((c) => {
    const [lo, hi] = c.displayRange ?? c.range;
    return {
      id: crypto.randomUUID(),
      index: c.index,
      name: c.name || `Channel ${c.index + 1}`,
      kind: "channel",
      sourceDataTypeId: probe.dtype,
      floatRange: { min: c.range[0], max: c.range[1] },
      gmmContrastLimits: { lower: lo, upper: hi },
      lowerLimit: lo,
      upperLimit: hi,
    };
  });
  const [sizeX, sizeY] = probe.dimensions;
  const leaf = decodeURIComponent(urlFileName(url));
  return {
    id: crypto.randomUUID(),
    sizeX,
    sizeY,
    sizeC: channels.length,
    omeXmlHash: `ims:${url}`,
    basename: leaf || probe.name || "volume.ims",
    contentRole: "intensity",
    rotationDegrees: 0,
    flipHorizontal: false,
    flipVertical: false,
    translateX: 0,
    translateY: 0,
    displayScale: 1,
    opacity: 1,
    channels,
    source: { kind: "volume", url, format: "ims" },
  };
}

export function isVolumeImage(
  im: Image | undefined,
): im is Image & { source: ImageSourceVolume } {
  return im?.source?.kind === "volume";
}

/** The story's 3D volume image, if any. */
export function useVolumeImage() {
  return useDocumentStore((s) => s.images.find(isVolumeImage));
}
