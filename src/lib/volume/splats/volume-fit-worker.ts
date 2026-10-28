/**
 * Volume-fit worker. A pool of these classifies bricks in parallel (each
 * brick maps to its own disjoint block of octree nodes); one of them then
 * runs the final cut/fit on the merged node array.
 */

import type { ClassifyParams, FitGrid, FitOptions } from "./volume-fit";
import { VolumeFitter } from "./volume-fit";

export type FitWorkerRequest =
  | { type: "init"; id: number; grid: FitGrid; params: ClassifyParams }
  | {
      type: "classify";
      id: number;
      bx: number;
      by: number;
      bz: number;
      channels: Array<Uint16Array | Uint8Array | null>;
    }
  | {
      type: "finish";
      id: number;
      nodes: Float32Array;
      occupied: number;
      options: FitOptions;
    };

export type FitWorkerResponse =
  | { type: "ack"; id: number }
  | { type: "block"; id: number; block: Float32Array | null; occupied: number }
  | { type: "done"; id: number; params: Float32Array; count: number }
  | { type: "error"; id: number; message: string };

let grid: FitGrid | null = null;
let params: ClassifyParams | null = null;
let classifier: VolumeFitter | null = null;

const post = postMessage as (m: FitWorkerResponse, t?: Transferable[]) => void;

self.onmessage = (e: MessageEvent<FitWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      grid = msg.grid;
      params = msg.params;
      classifier = new VolumeFitter(grid, params, false);
      post({ type: "ack", id: msg.id });
    } else if (msg.type === "classify") {
      const r = classifier!.classifyBrick(msg.bx, msg.by, msg.bz, msg.channels);
      post(
        {
          type: "block",
          id: msg.id,
          block: r?.block ?? null,
          occupied: r?.occupied ?? 0,
        },
        r ? [r.block.buffer] : [],
      );
    } else if (msg.type === "finish") {
      const fitter = new VolumeFitter(grid!, params!, false);
      fitter.setNodes(msg.nodes, msg.occupied);
      const cloud = fitter.finish(msg.options);
      post(
        { type: "done", id: msg.id, params: cloud.params, count: cloud.count },
        [cloud.params.buffer],
      );
    }
  } catch (err) {
    post({
      type: "error",
      id: msg.id,
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
