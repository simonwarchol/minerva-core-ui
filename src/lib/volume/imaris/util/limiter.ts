/**
 * Runs at most `limit` async tasks at once. Waiting tasks start lowest
 * `priority` first, first come first served among equals.
 */
export class Limiter {
  private active = 0;
  private readonly waiting: Array<{ start: () => void; priority: number }> = [];

  constructor(private readonly limit: number) {}

  async run<T>(fn: () => Promise<T>, priority = 0): Promise<T> {
    if (this.active < this.limit) {
      this.active++;
    } else {
      await new Promise<void>((start) => {
        let i = this.waiting.length;
        while (i > 0 && this.waiting[i - 1]!.priority > priority) i--;
        this.waiting.splice(i, 0, { start, priority });
      });
    }
    try {
      return await fn();
    } finally {
      // Hand the slot straight on, so a newcomer can't take it in between.
      const next = this.waiting.shift();
      if (next) next.start();
      else this.active--;
    }
  }
}
