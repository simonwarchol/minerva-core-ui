/**
 * Least-recently-used map capped by total size (bytes, or any weight given
 * by `sizeOf`). Lookups refresh recency; the newest entry is never evicted.
 */
export class ByteLru<K, V> {
  private readonly map = new Map<K, V>();
  private total = 0;

  constructor(
    private readonly budget: number,
    private readonly sizeOf: (v: V) => number,
  ) {}

  has(key: K): boolean {
    return this.map.has(key);
  }

  get(key: K): V | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }

  set(key: K, value: V): void {
    this.delete(key);
    this.map.set(key, value);
    this.total += this.sizeOf(value);
    while (this.total > this.budget && this.map.size > 1) {
      const [oldest, v] = this.map.entries().next().value as [K, V];
      this.map.delete(oldest);
      this.total -= this.sizeOf(v);
    }
  }

  delete(key: K): void {
    const v = this.map.get(key);
    if (v === undefined) return;
    this.map.delete(key);
    this.total -= this.sizeOf(v);
  }

  clear(): void {
    this.map.clear();
    this.total = 0;
  }
}
