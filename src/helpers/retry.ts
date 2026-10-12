/**
 * Retry an async function with exponential backoff and full jitter. Honors a
 * provider-requested delay (`retryAfterMs` on the thrown error, which the http
 * provider fills from the `Retry-After` header).
 */
import { EmbeddingError } from '#src/errors.js';

/** Options for `withRetry`. */
export interface RetryOptions {
  /** Extra attempts after the first one. Defaults to 2. */
  retries?: number;
  /** Base delay for the backoff curve, in milliseconds. Defaults to 250. */
  baseDelayMs?: number;
  /** Upper bound for any single delay, in milliseconds. Defaults to 30000. */
  maxDelayMs?: number;
  /**
   * Decide whether an error is worth retrying. Defaults to `error.retryable`
   * for `EmbeddingError` and `true` for anything else.
   */
  shouldRetry?: (error: unknown, attempt: number) => boolean;
  /** Stops waiting and retrying when aborted. */
  signal?: AbortSignal;
}

/** Default retry predicate. */
function defaultShouldRetry(error: unknown): boolean {
  return error instanceof EmbeddingError ? error.retryable : true;
}

/** Read `retryAfterMs` from any error-like value. */
function retryAfterOf(error: unknown): number | undefined {
  if (typeof error === 'object' && error !== null && 'retryAfterMs' in error) {
    const value = error.retryAfterMs;
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value;
  }
  return undefined;
}

/** Sleep that rejects with the abort reason when the signal fires. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason as Error);
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal?.reason as Error);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Call `fn` until it succeeds or the retry budget runs out. `fn` receives the
 * zero-based attempt number. The last error is rethrown unchanged.
 *
 * @example
 * ```ts
 * const data = await withRetry(
 *   async () => {
 *     const res = await fetch(url);
 *     if (!res.ok) throw new Error(`status ${res.status}`);
 *     return res.json();
 *   },
 *   { retries: 3, baseDelayMs: 100 },
 * );
 * ```
 */
export async function withRetry<T>(fn: (attempt: number) => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const retries = options.retries ?? 2;
  const baseDelayMs = options.baseDelayMs ?? 250;
  const maxDelayMs = options.maxDelayMs ?? 30_000;
  const shouldRetry = options.shouldRetry ?? defaultShouldRetry;

  for (let attempt = 0; ; attempt++) {
    try {
      return await fn(attempt);
    } catch (error) {
      if (options.signal?.aborted) throw options.signal.reason as Error;
      if (attempt >= retries || !shouldRetry(error, attempt)) throw error;
      const requested = retryAfterOf(error);
      const backoff = Math.random() * Math.min(maxDelayMs, baseDelayMs * 2 ** attempt);
      await sleep(Math.min(maxDelayMs, requested ?? backoff), options.signal);
    }
  }
}
