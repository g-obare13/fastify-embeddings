/**
 * Internal: run async work over a list with bounded concurrency, keeping
 * results in input order. Stops starting new work after the first failure.
 */

/**
 * Map `items` through `fn` with at most `limit` calls in flight.
 *
 * @example
 * ```ts
 * const results = await mapWithConcurrency(batches, 2, (batch) => provider.embed(batch, 'document'));
 * ```
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  let failed = false;

  const worker = async (): Promise<void> => {
    while (!failed && next < items.length) {
      const index = next++;
      try {
        results[index] = await fn(items[index] as T, index);
      } catch (error) {
        failed = true;
        throw error;
      }
    }
  };

  const workers = Array.from({ length: Math.min(limit, items.length) }, () => worker());
  await Promise.all(workers);
  return results;
}
