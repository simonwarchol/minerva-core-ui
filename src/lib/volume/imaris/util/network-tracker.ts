import type { NetworkStats } from "kiln-upstream";

/**
 * Fixed-capacity rolling average over the last N samples.
 * Used for per-stage pipeline timing across all data providers.
 */
export class RollingAvg {
  private samples: number[] = [];
  constructor(private readonly capacity = 32) {}

  add(value: number): void {
    this.samples.push(value);
    if (this.samples.length > this.capacity) this.samples.shift();
  }

  get value(): number {
    if (!this.samples.length) return 0;
    return this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
  }

  get count(): number {
    return this.samples.length;
  }
}

/** Throughput is averaged over this trailing window. */
const WINDOW_MS = 10_000;

/**
 * Tracks download volume and throughput for a data provider.
 */
export class NetworkTracker {
  private totalBytesDownloaded = 0;
  private requestCount = 0;
  private recentDownloads: { timestamp: number; bytes: number }[] = [];

  record(bytes: number, requests = 1): void {
    this.totalBytesDownloaded += bytes;
    this.requestCount += requests;
    const now = performance.now();
    this.recentDownloads.push({ timestamp: now, bytes });
    // Prune here too (amortised: at most once per window), so the list stays
    // bounded when nobody polls getStats().
    if (this.recentDownloads[0]!.timestamp <= now - 2 * WINDOW_MS)
      this.prune(now);
  }

  private prune(now: number): void {
    const cutoff = now - WINDOW_MS;
    this.recentDownloads = this.recentDownloads.filter(
      (d) => d.timestamp > cutoff,
    );
  }

  getStats(): NetworkStats {
    const now = performance.now();
    this.prune(now);
    const recentBytes = this.recentDownloads.reduce(
      (sum, d) => sum + d.bytes,
      0,
    );
    const elapsed =
      this.recentDownloads.length > 0
        ? Math.max(1000, now - this.recentDownloads[0]!.timestamp)
        : 0;
    return {
      totalBytesDownloaded: this.totalBytesDownloaded,
      recentBytesPerSecond: elapsed > 0 ? (recentBytes / elapsed) * 1000 : 0,
      requestCount: this.requestCount,
    };
  }
}
