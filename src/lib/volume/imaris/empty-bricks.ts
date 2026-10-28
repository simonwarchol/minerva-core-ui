/**
 * IMS empty-brick checks. Coarse Imaris levels are subsampled, not max-pooled,
 * so a parent with max below the display threshold can still have bright
 * children. Inherit emptiness only when an ancestor is truly zero.
 */

export interface ImsBrickStatsLike {
  max: number;
}

export function imsBrickIsEmpty(args: {
  lod: number;
  bx: number;
  by: number;
  bz: number;
  maxLod: number;
  threshold: number;
  getStats: (
    lod: number,
    bx: number,
    by: number,
    bz: number,
  ) => ImsBrickStatsLike | null;
}): boolean {
  const self = args.getStats(args.lod, args.bx, args.by, args.bz);
  if (self) return self.max < args.threshold;

  let lod = args.lod + 1;
  let bx = args.bx >> 1;
  let by = args.by >> 1;
  let bz = args.bz >> 1;
  while (lod <= args.maxLod) {
    const parent = args.getStats(lod, bx, by, bz);
    if (parent && parent.max < 1) return true;
    lod += 1;
    bx >>= 1;
    by >>= 1;
    bz >>= 1;
  }
  return false;
}
