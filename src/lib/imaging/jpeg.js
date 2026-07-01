import { MultiscaleImageLayer, XRLayer } from "@hms-dbmi/viv";
import { JpegImage } from "../jpeg-image";
import { JpegPixelSource } from "../jpeg-pixel-source";

/** Viv TileLayer busts its cache when `selections` ref changes (updateTriggers). */
let cachedSelectionsKey = "";
let cachedSelections = null;

function stableSelections(selections) {
  const key = selections.map(({ z, t, c }) => `${z},${t},${c}`).join("|");
  if (key === cachedSelectionsKey && cachedSelections) {
    return cachedSelections;
  }
  cachedSelectionsKey = key;
  cachedSelections = selections;
  return selections;
}

const BAKED_CONTRAST = Object.freeze([0, 65535]);

/**
 * Viv default renderSubLayers uses loader[0] (full-res) for edge-tile bounds. Our JPEG
 * tiles decode smaller than 1024², which triggers that path and stretches bounds to the
 * full 4096×4096 image — overlapping sublayers and duplicate tile labels after pan/zoom.
 */
function jpegRenderSubLayers(props) {
  const {
    bbox: { left, bottom, right, top },
    index: { x, y, z },
  } = props.tile;
  const { data, maxZoom } = props;
  if ([left, bottom, right, top].some((v) => v < 0) || !data) {
    return null;
  }
  return new XRLayer(props, {
    channelData: data,
    bounds: [left, bottom, right, top],
    id: `jpeg-tile-${z}-${x}-${y}`,
    tileId: { x, y, z },
    interpolation: z === maxZoom ? "nearest" : "linear",
  });
}

function getJpegImageLayerProps(meta) {
  const { channelsVisible, colors, contrastLimits, selections } = meta.settings;
  const visible = channelsVisible.some((x) => x);
  const { imagePath, jpegLoader } = meta;
  const imageID = imagePath.replace("/", "-");
  return {
    visible,
    loader: jpegLoader,
    excludeBackground: true,
    refinementStrategy: "no-overlap",
    renderSubLayers: jpegRenderSubLayers,
    // Stable id — must not change when channels toggle or deck orphans TileLayer children.
    id: imageID,
    channelsVisible,
    colors,
    contrastLimits: contrastLimits.map(() => BAKED_CONTRAST),
    selections: stableSelections(selections),
  };
}

function createJpegLayers(meta) {
  return new MultiscaleImageLayer(getJpegImageLayerProps(meta));
}

const toIndexer = (opts) => {
  const { imagePath, imageWidth, imageHeight } = opts;
  return (sel, level) => {
    return new JpegImage({
      imagePath,
      level,
      imageWidth,
      imageHeight,
      ...sel,
    });
  };
};

const getShapeForBinaryDownsampleLevel = (options) => {
  const { axes, level } = options;
  const xIndex = axes.labels.indexOf("x");
  const yIndex = axes.labels.indexOf("y");
  const resolutionShape = axes.shape.slice();
  resolutionShape[xIndex] = axes.shape[xIndex] >> level;
  resolutionShape[yIndex] = axes.shape[yIndex] >> level;
  return resolutionShape;
};

const loadJpeg = (meta) => {
  const { imagePath, imageWidth, imageHeight } = meta;
  const width = imageWidth;
  const height = imageHeight;
  const nChannels = 2; // TODO
  const tileSize = 1024; // TODO
  const levels = [0, 1, 2]; // TODO
  const pyramidIndexer = toIndexer({
    imagePath,
    imageWidth: width,
    imageHeight: height,
  });
  const data = levels.map((level) => {
    const axes = {
      labels: ["t", "c", "z", "y", "x"],
      shape: [1, nChannels, 1, height, width],
    };
    return new JpegPixelSource(
      (sel) => pyramidIndexer(sel, level),
      tileSize,
      getShapeForBinaryDownsampleLevel({
        axes,
        level,
      }),
    );
  });
  return {
    data,
    metadata: {
      Pixels: {
        Channels: [
          {
            ID: "DNA1",
            Name: "DNA1",
            SamplesPerPixel: 1,
          },
          {
            ID: "AF488",
            Name: "AF488",
            SamplesPerPixel: 1,
          },
        ],
        Type: "Uint16",
        // TODO  -- using placeholder data
        ID: "TODO",
        DimensionOrder: "TCZYX",
        SamplesPerPixel: 1,
        SizeT: 1,
        SizeC: 2,
        SizeZ: 1,
        SizeY: height,
        SizeX: width,
        PhysicalSizeX: width,
        PhysicalSizeY: height,
        PhysicalSizeZ: 1,
        PhysicalSizeXUnit: "µm",
        PhysicalSizeYUnit: "µm",
        PhysicalSizeZUnit: "µm",
        BigEndian: false,
        TiffData: null,
      },
      // TODO  -- using placeholder data
      ID: "TODO",
      AquisitionDate: new Date().toISOString().split("T")[0],
      Description: "",
      ROIs: [],
    },
  };
};

export { createJpegLayers, getJpegImageLayerProps, loadJpeg };
