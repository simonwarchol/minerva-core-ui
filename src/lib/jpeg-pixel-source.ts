import type { JpegImage } from "./jpeg-image";

type Dtype =
  | "Uint8"
  | "Uint16"
  | "Uint32"
  | "Int8"
  | "Int16"
  | "Int32"
  | "Float32"
  | "Float64";
type Selection = Record<"z" | "t" | "c", number>;
type Labels = [...("y" | "x" | "z" | "t" | "c")[], "y", "x"];
type Shape = [number, number, number, number, number];
type ReadRasterProps = {
  x?: number;
  y?: number;
  height?: number;
  width?: number;
  signal?: AbortSignal;
};

class JpegPixelSource {
  _indexer: (s: Selection) => Promise<typeof JpegImage>;
  tileSize: number;
  labels: Labels;
  shape: Shape;
  dtype: Dtype;

  constructor(indexer, tileSize, shape) {
    this._indexer = indexer;
    this.tileSize = tileSize;
    this.labels = ["t", "c", "z", "y", "x"];
    this.dtype = "Uint16";
    this.shape = shape;
  }

  async getRaster({ selection, signal }) {
    return await this.getTile({ x: 0, y: 0, selection, signal });
  }

  async getTile({ x, y, selection, signal }) {
    const { height, width } = this._getTileExtent(x, y);

    const image = await this._indexer(selection);
    return this._readRasters(image, { x, y, width, height, signal });
  }

  async _readRasters(image: typeof JpegImage, props: ReadRasterProps = {}) {
    const raster = await image.readRasters({
      ...props,
    });

    if (props.signal?.aborted) {
      throw "__vivSignalAborted";
    }

    const extentW = props.width ?? this.tileSize;
    const extentH = props.height ?? this.tileSize;
    if (!raster?.data) {
      return {
        data: new Uint16Array(extentW * extentH),
        width: extentW,
        height: extentH,
      };
    }
    // Keep decoded JPEG dimensions; jpegRenderSubLayers places tiles via Deck bbox.
    return raster;
  }

  _getTileExtent(x, y) {
    const [levelHeight, levelWidth] = this.shape.slice(-2);
    let height = this.tileSize;
    let width = this.tileSize;
    const maxXTileCoord = Math.floor(levelWidth / this.tileSize);
    const maxYTileCoord = Math.floor(levelHeight / this.tileSize);

    if (x === maxXTileCoord && levelWidth % this.tileSize !== 0) {
      width = levelWidth % this.tileSize;
    }
    if (y === maxYTileCoord && levelHeight % this.tileSize !== 0) {
      height = levelHeight % this.tileSize;
    }
    return { height, width };
  }

  onTileError(err) {
    console.error(err);
  }
}

export { JpegPixelSource };
