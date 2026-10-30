/**
 * Opt-in timings for the 3D view, observation only, switched on with kiln's
 * perf logging: `localStorage.setItem("kiln:perf", "1")`, then reload.
 */
export const perfOn: boolean = (() => {
  try {
    return globalThis.localStorage?.getItem("kiln:perf") === "1";
  } catch {
    return false;
  }
})();

export function perfLog(message: string): void {
  if (perfOn) console.log(`[minerva perf] ${message}`);
}

export const ms = (v: number) => `${v.toFixed(1)} ms`;
