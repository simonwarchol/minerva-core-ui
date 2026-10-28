/**
 * Persistent store for compressed HDF5 chunks in the Origin Private File
 * System: one directory per source URL, one file per chunk.
 *
 * The directory is listed once on open, so a miss is answered without a
 * file lookup and the network read starts immediately. OPFS reads run at
 * hundreds of MB/s one at a time but slow sharply under concurrency, so reads
 * and writes each go through a small limiter.
 */

import { Limiter } from "./util/limiter";

const ROOT = "kiln-ims-chunks";
const READ_CONCURRENCY = 2;
const WRITE_CONCURRENCY = 2;

/** FNV-1a 32-bit — directory name for a URL. */
function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

const fileName = (address: number, size: number) => `${address}-${size}`;

export class OpfsChunkStore {
  /** Directory and the chunk files in it; null when OPFS is unusable. */
  private readonly opened: Promise<{
    dir: FileSystemDirectoryHandle;
    names: Set<string>;
  } | null>;
  /** Resolves once the directory has been listed; `has` is then accurate. */
  readonly ready: Promise<void>;
  private names: Set<string> | null = null;
  private readonly reads = new Limiter(READ_CONCURRENCY);
  private readonly writes = new Limiter(WRITE_CONCURRENCY);
  private enabled = true;

  constructor(url: string) {
    this.opened = (async () => {
      try {
        const root = await navigator.storage.getDirectory();
        const base = await root.getDirectoryHandle(ROOT, { create: true });
        const dir = await base.getDirectoryHandle(hash(url), { create: true });
        const names = new Set<string>();
        const listing = dir as unknown as { keys(): AsyncIterable<string> };
        for await (const name of listing.keys()) names.add(name);
        return { dir, names };
      } catch {
        this.enabled = false;
        return null;
      }
    })();
    this.ready = this.opened.then((o) => {
      this.names = o?.names ?? null;
    });
  }

  /** Stored (as of the last write), without touching the disk. */
  has(address: number, size: number): boolean {
    return this.enabled && !!this.names?.has(fileName(address, size));
  }

  static supported(): boolean {
    return (
      typeof navigator !== "undefined" &&
      typeof navigator.storage?.getDirectory === "function"
    );
  }

  /**
   * Stored bytes, or null if absent. Lower `priority` values are read first;
   * a read whose `cancelled()` is true by its turn rejects with AbortError.
   */
  async get(
    address: number,
    size: number,
    priority = 0,
    cancelled?: () => boolean,
  ): Promise<Uint8Array | null> {
    const o = await this.opened;
    const name = fileName(address, size);
    if (!o || !this.enabled || !o.names.has(name)) return null;
    return this.reads.run(async () => {
      if (cancelled?.()) {
        throw new DOMException("Chunk read cancelled", "AbortError");
      }
      try {
        const fh = await o.dir.getFileHandle(name);
        const buf = await (await fh.getFile()).arrayBuffer();
        return buf.byteLength === size ? new Uint8Array(buf) : null;
      } catch {
        return null; // removed meanwhile (e.g. storage cleared)
      }
    }, priority);
  }

  put(address: number, size: number, bytes: Uint8Array): void {
    void this.opened.then((o) => {
      if (!o || !this.enabled) return;
      const name = fileName(address, size);
      return this.writes.run(async () => {
        try {
          const fh = await o.dir.getFileHandle(name, { create: true });
          const w = await fh.createWritable();
          await w.write(bytes as Uint8Array<ArrayBuffer>);
          await w.close();
          o.names.add(name);
        } catch {
          // Quota exceeded or similar: stop persisting this session.
          this.enabled = false;
        }
      });
    });
  }
}
