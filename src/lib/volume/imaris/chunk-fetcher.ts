/**
 * Shared fetcher for compressed HDF5 chunk bytes.
 *
 * One instance per file, on the main thread, in front of every brick worker:
 * - de-duplicates concurrent requests for the same chunk (neighbouring bricks
 *   share chunks through their ghost borders, often across workers);
 * - reads the most urgent chunks first (lowest priority value) and, when a
 *   connection frees up, coalesces nearby pending byte ranges into one Range
 *   read;
 * - keeps an LRU of compressed bytes in memory and, for remote files, a
 *   persistent OPFS copy so revisits skip the network;
 * - caps concurrent HTTP requests (browsers allow ~6 per host anyway; more
 *   just queues inside the network stack where we can't prioritise).
 */

import type { RangeSource } from "./hdf5/source";
import { OpfsChunkStore } from "./opfs-chunk-store";
import { ByteLru } from "./util/byte-lru";

export interface ChunkRequest {
  address: number;
  size: number;
}

export interface ChunkFetcherOptions {
  /** In-memory LRU budget for compressed bytes (default 128 MB) */
  budgetBytes?: number;
  /**
   * Persist chunks in the Origin Private File System under this key (the
   * file's URL); omit for local files.
   */
  persistKey?: string;
  /** Max simultaneous Range requests (default 6) */
  maxConcurrent?: number;
  /** Merge ranges separated by up to this many bytes (default 64 KB) */
  maxGap?: number;
  /**
   * Cap on a single coalesced request (default 2 MB). No chunk in a group
   * resolves before the whole group arrives, so big groups delay every chunk
   * in them on a slow link.
   */
  maxRequestBytes?: number;
  /** Called with the bytes and request count of every network read. */
  onFetched?: (bytes: number, requests: number) => void;
  /** Called when a chunk's bytes arrive (from the network or OPFS). */
  onCached?: (address: number) => void;
}

interface Entry {
  req: ChunkRequest;
  promise: Promise<Uint8Array>;
  resolve: (b: Uint8Array) => void;
  reject: (e: unknown) => void;
  /** Requesters' signals; null once anyone asked without one (never abandoned). */
  wanted: Set<AbortSignal> | null;
  /** Most urgent priority any requester gave (lower is read sooner). */
  priority: number;
  /** Request order, first come first served among equal priorities. */
  seq: number;
}

interface Group {
  start: number;
  end: number;
  members: Entry[];
}

const abandoned = (e: Entry) =>
  e.wanted !== null && [...e.wanted].every((s) => s.aborted);

const abortError = () => new DOMException("Chunk read cancelled", "AbortError");

const urgency = (a: Entry, b: Entry) =>
  a.priority - b.priority || a.seq - b.seq;

export class ChunkFetcher {
  private readonly lru: ByteLru<number, Uint8Array>;
  /** Chunks queued or being read, by address (requests share them). */
  private readonly entries = new Map<number, Entry>();
  /** Network reads not started yet; grouped when a connection frees up. */
  private pending: Entry[] = [];
  private active = 0;
  private pumpScheduled = false;
  private disposed = false;
  private seq = 0;
  private readonly maxConcurrent: number;
  private readonly store: OpfsChunkStore | null;
  private readonly maxGap: number;
  private readonly maxRequest: number;
  private readonly onFetched?: (bytes: number, requests: number) => void;
  private readonly onCached?: (address: number) => void;

  /** `source` should retry transient failures (the provider's does). */
  constructor(
    private readonly source: RangeSource,
    opts: ChunkFetcherOptions = {},
  ) {
    this.lru = new ByteLru(
      opts.budgetBytes ?? 128 * 1024 * 1024,
      (b) => b.byteLength,
    );
    this.maxConcurrent = opts.maxConcurrent ?? 6;
    this.maxGap = opts.maxGap ?? 64 * 1024;
    this.maxRequest = opts.maxRequestBytes ?? 2 * 1024 * 1024;
    this.onFetched = opts.onFetched;
    this.onCached = opts.onCached;
    this.store =
      opts.persistKey && OpfsChunkStore.supported()
        ? new OpfsChunkStore(opts.persistKey)
        : null;
  }

  /**
   * A chunk's compressed bytes. Lower `priority` values are read first. A
   * chunk whose every requester has aborted is skipped (rejecting with
   * AbortError) if its read hasn't started yet.
   */
  get(
    c: ChunkRequest,
    signal?: AbortSignal,
    priority = 0,
  ): Promise<Uint8Array> {
    const hit = this.lru.get(c.address);
    if (hit) return Promise.resolve(hit);
    if (this.disposed) return Promise.reject(abortError());
    const existing = this.entries.get(c.address);
    if (existing) {
      if (!signal) existing.wanted = null;
      else existing.wanted?.add(signal);
      existing.priority = Math.min(existing.priority, priority);
      return existing.promise;
    }
    let resolve!: (b: Uint8Array) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<Uint8Array>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    const entry: Entry = {
      req: c,
      promise,
      resolve,
      reject,
      wanted: signal ? new Set([signal]) : null,
      priority,
      seq: this.seq++,
    };
    this.entries.set(c.address, entry);
    promise.then(
      (b) => {
        this.entries.delete(c.address);
        if (this.disposed) return;
        this.lru.set(c.address, b);
        this.onCached?.(c.address);
      },
      () => this.entries.delete(c.address),
    );
    const store = this.store;
    if (!store) {
      this.enqueue(entry);
      return promise;
    }
    // Persistent store: misses go to the network right away; hits are read
    // on their own (and fall back to the network if the file is gone).
    void store.ready.then(() => {
      if (!store.has(c.address, c.size)) {
        this.enqueue(entry);
        return;
      }
      store
        .get(c.address, c.size, entry.priority, () => abandoned(entry))
        .then(
          (bytes) => (bytes ? entry.resolve(bytes) : this.enqueue(entry)),
          entry.reject,
        );
    });
    return promise;
  }

  /** In memory or on disk here: reading it needs no network. */
  has(address: number, size: number): boolean {
    return this.lru.has(address) || !!this.store?.has(address, size);
  }

  private enqueue(e: Entry): void {
    if (this.disposed) {
      e.reject(abortError());
      return;
    }
    this.pending.push(e);
    if (this.pumpScheduled) return;
    this.pumpScheduled = true;
    // Let callers in the same tick add their chunks first.
    setTimeout(() => {
      this.pumpScheduled = false;
      this.pump();
    }, 0);
  }

  /** Start reads while connections are free. */
  private pump(): void {
    while (this.active < this.maxConcurrent) {
      const group = this.nextGroup();
      if (!group) return;
      this.active++;
      void this.read(group).finally(() => {
        this.active--;
        this.pump();
      });
    }
  }

  /**
   * The most urgent pending chunk, plus pending chunks next to it in the
   * file that would be on the wire now anyway (as urgent as the
   * maxConcurrent-th most urgent) or that sit inside the read regardless.
   * Decided when a connection frees up, not at request time: the camera may
   * have moved on meanwhile, so abandoned chunks are dropped here.
   */
  private nextGroup(): Group | null {
    const live: Entry[] = [];
    for (const e of this.pending) {
      if (abandoned(e)) {
        e.reject(abortError());
      } else {
        live.push(e);
      }
    }
    if (live.length === 0) {
      this.pending = live;
      return null;
    }
    const byUrgency = [...live].sort(urgency);
    const top = byUrgency[0]!;
    const cutoff =
      byUrgency[Math.min(live.length, this.maxConcurrent) - 1]!.priority;
    const near = live
      .filter((e) => e.priority <= cutoff)
      .sort((a, b) => a.req.address - b.req.address);
    let lo = near.indexOf(top);
    let hi = lo;
    let start = top.req.address;
    let end = start + top.req.size;
    for (let grew = true; grew; ) {
      grew = false;
      const prev = near[lo - 1];
      if (
        prev &&
        start - (prev.req.address + prev.req.size) <= this.maxGap &&
        end - prev.req.address <= this.maxRequest
      ) {
        lo--;
        start = prev.req.address;
        grew = true;
      }
      const next = near[hi + 1];
      const nextEnd = next ? next.req.address + next.req.size : 0;
      if (
        next &&
        next.req.address - end <= this.maxGap &&
        Math.max(end, nextEnd) - start <= this.maxRequest
      ) {
        hi++;
        end = Math.max(end, nextEnd);
        grew = true;
      }
    }
    const members: Entry[] = [];
    this.pending = [];
    for (const e of live) {
      if (e.req.address >= start && e.req.address + e.req.size <= end)
        members.push(e);
      else this.pending.push(e);
    }
    return { start, end, members };
  }

  /** One Range read for a group of chunks. */
  private async read({ start, end, members }: Group): Promise<void> {
    const store = this.store;
    try {
      const bytes = await this.source.read(start, end - start);
      this.onFetched?.(end - start, 1);
      for (const m of members) {
        const off = m.req.address - start;
        // A lone chunk is the whole read; coalesced ones are copied out
        // so the cache doesn't pin the gaps between them.
        const chunk =
          members.length === 1 && off === 0 && bytes.length === m.req.size
            ? bytes
            : bytes.slice(off, off + m.req.size);
        m.resolve(chunk);
        store?.put(m.req.address, m.req.size, chunk);
      }
    } catch (e) {
      for (const m of members) m.reject(e);
    }
  }

  /** Drop cached bytes and cancel reads not started yet; later gets reject. */
  dispose(): void {
    this.disposed = true;
    this.lru.clear();
    for (const e of this.pending) e.reject(abortError());
    this.pending = [];
  }
}
