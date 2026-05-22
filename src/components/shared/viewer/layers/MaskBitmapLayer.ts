/**
 * GPU-shaded segmentation mask sub-layer.
 *
 * Subclasses deck.gl's `BitmapLayer` (so we keep its mesh, picking,
 * coordinate-system, and bounds plumbing) and overrides only the fragment
 * shader and the per-frame uniform pack:
 *
 * 1. Each tile is encoded once on the CPU as RGBA8: bytes R+G+B pack the
 *    24-bit cell id (supports ~16M cells), A = 255 where mask > 0.
 * 2. The shader decodes the id, hashes it (Knuth `* 2654435761`) for a
 *    deterministic RGB color, and optionally writes outlines by sampling
 *    the 4-neighbor texels and flagging any differing id.
 * 3. Opacity and outline mode are *uniforms* — toggling them never
 *    re-encodes a tile, the GPU just re-shades the same texture.
 *
 * Inspired by SEAL (Warchol et al.,
 * https://github.com/simonwarchol/seal/blob/main/src/views/spotlight/SpotlightBitmaskLayer.js),
 * adapted from luma.gl 8 / GLSL 1.00 to luma.gl 9 / GLSL 3.00 ES.
 */

import { BitmapLayer, type BitmapLayerProps } from "@deck.gl/layers";
import type { ShaderModule } from "@luma.gl/shadertools";

export type MaskBitmapLayerProps = BitmapLayerProps & {
  /** 0-1 overlay opacity. */
  maskOpacity: number;
  /** True to draw white outlines between adjacent cell ids. */
  maskOutlines: boolean;
  /** Tile texel size in normalized texture coords (1 / tile-width). */
  texelSize: [number, number];
};

type MaskUniforms = {
  maskOpacity: number;
  maskOutlines: number;
  texelSize: [number, number];
};

const maskUniforms = {
  name: "mask",
  vs: `\
uniform maskUniforms {
  float maskOpacity;
  float maskOutlines;
  vec2 texelSize;
} mask;
`,
  fs: `\
uniform maskUniforms {
  float maskOpacity;
  float maskOutlines;
  vec2 texelSize;
} mask;
`,
  uniformTypes: {
    maskOpacity: "f32",
    maskOutlines: "f32",
    texelSize: "vec2<f32>",
  },
} as const satisfies ShaderModule<MaskUniforms>;

const fs = /* glsl */ `\
#version 300 es
#define SHADER_NAME mask-bitmap-layer-fragment-shader

precision highp float;

uniform sampler2D bitmapTexture;

in vec2 vTexCoord;
out vec4 fragColor;

float decodeId(vec4 rgba) {
  // 24-bit unsigned int packed across R+G+B bytes.
  return rgba.r * 255.0
       + rgba.g * 255.0 * 256.0
       + rgba.b * 255.0 * 65536.0;
}

vec3 colorForId(float id) {
  // Knuth integer hash, then take three byte slices for R/G/B.
  // Bitwise ops are GLSL 3.00 ES; cast the float id to uint first.
  uint h = uint(id) * 2654435761u;
  return vec3(
    float((h >> 16) & 255u) / 255.0,
    float((h >>  8) & 255u) / 255.0,
    float( h        & 255u) / 255.0
  );
}

bool differsFromId(vec4 neighbor, float id) {
  return neighbor.a < 0.5 || decodeId(neighbor) != id;
}

// Foreground pixel on the cell boundary (1 px inside the cell).
bool isInsideStroke(vec2 uv, float cellId) {
  vec4 left  = texture(bitmapTexture, uv - vec2(mask.texelSize.x, 0.0));
  vec4 right = texture(bitmapTexture, uv + vec2(mask.texelSize.x, 0.0));
  vec4 up    = texture(bitmapTexture, uv - vec2(0.0, mask.texelSize.y));
  vec4 down  = texture(bitmapTexture, uv + vec2(0.0, mask.texelSize.y));
  return differsFromId(left, cellId)
      || differsFromId(right, cellId)
      || differsFromId(up, cellId)
      || differsFromId(down, cellId);
}

// Background pixel directly outside a boundary foreground pixel (1 px outside).
bool isOutsideStroke(vec2 uv) {
  vec4 left  = texture(bitmapTexture, uv - vec2(mask.texelSize.x, 0.0));
  vec4 right = texture(bitmapTexture, uv + vec2(mask.texelSize.x, 0.0));
  vec4 up    = texture(bitmapTexture, uv - vec2(0.0, mask.texelSize.y));
  vec4 down  = texture(bitmapTexture, uv + vec2(0.0, mask.texelSize.y));
  if (left.a  >= 0.5 && isInsideStroke(uv - vec2(mask.texelSize.x, 0.0), decodeId(left)))  return true;
  if (right.a >= 0.5 && isInsideStroke(uv + vec2(mask.texelSize.x, 0.0), decodeId(right))) return true;
  if (up.a    >= 0.5 && isInsideStroke(uv - vec2(0.0, mask.texelSize.y), decodeId(up)))    return true;
  if (down.a  >= 0.5 && isInsideStroke(uv + vec2(0.0, mask.texelSize.y), decodeId(down))) return true;
  return false;
}

void main() {
  vec4 center = texture(bitmapTexture, vTexCoord);

  if (mask.maskOutlines > 0.5) {
    // 2 px stroke straddling the boundary: 1 px inside (foreground edge)
    // and 1 px outside (background hugging that edge).
    bool draw = false;
    if (center.a >= 0.5) {
      draw = isInsideStroke(vTexCoord, decodeId(center));
    } else {
      draw = isOutsideStroke(vTexCoord);
    }
    if (!draw) {
      discard;
    }
    fragColor = vec4(1.0, 1.0, 1.0, mask.maskOpacity);
    return;
  }

  if (center.a < 0.5) {
    discard;
  }
  fragColor = vec4(colorForId(decodeId(center)), mask.maskOpacity);
}
`;

export class MaskBitmapLayer extends BitmapLayer<MaskBitmapLayerProps> {
  static layerName = "MaskBitmapLayer";

  getShaders() {
    const shaders = super.getShaders();
    return {
      ...shaders,
      fs,
      modules: [...(shaders.modules ?? []), maskUniforms],
    };
  }

  draw(opts: Parameters<BitmapLayer["draw"]>[0]) {
    const { model } = this.state as {
      model: { shaderInputs?: { setProps: (p: unknown) => void } } | null;
    };
    const { maskOpacity, maskOutlines, texelSize } = this
      .props as MaskBitmapLayerProps;
    if (model?.shaderInputs) {
      model.shaderInputs.setProps({
        mask: {
          maskOpacity,
          maskOutlines: maskOutlines ? 1 : 0,
          texelSize,
        } satisfies MaskUniforms,
      });
    }
    super.draw(opts);
  }
}
