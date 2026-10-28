/**
 * Depth cueing: colour fades with distance from the camera, so structures
 * behind others read as farther away instead of stacking up to the same
 * brightness. Emission (not opacity) is scaled by
 *
 *   f(d) = exp(−scale · max(0, d − near)),  d = distance from the camera,
 *
 * where `near` is the box's nearest point and its farthest corner fades to
 * exp(−DEPTH_CUE_RANGE · strength). The volume
 * renderer and the splats use the same function.
 */

/** Strength 1 dims the far side of the box to e^−5 ≈ 0.7%. */
export const DEPTH_CUE_RANGE = 5;

export interface DepthCue {
  near: number;
  /** 0 = off */
  scale: number;
}

export const NO_DEPTH_CUE: DepthCue = { near: 0, scale: 0 };

/** Fade parameters for a box [lo, hi] seen from `eye` at `strength` (0–1). */
export function depthCueFor(
  eye: ArrayLike<number>,
  lo: ArrayLike<number>,
  hi: ArrayLike<number>,
  strength: number,
): DepthCue {
  if (!(strength > 0)) return NO_DEPTH_CUE;
  // Nearest point of the box (0 inside it) and farthest corner.
  let near2 = 0;
  let far2 = 0;
  for (let a = 0; a < 3; a++) {
    const e = eye[a] ?? 0;
    const l = lo[a] ?? 0;
    const h = hi[a] ?? 0;
    const inside = Math.max(l - e, 0, e - h);
    near2 += inside * inside;
    const corner = Math.max(Math.abs(e - l), Math.abs(e - h));
    far2 += corner * corner;
  }
  const near = Math.sqrt(near2);
  const far = Math.sqrt(far2);
  return {
    near,
    scale:
      (Math.min(1, strength) * DEPTH_CUE_RANGE) / Math.max(1e-6, far - near),
  };
}
