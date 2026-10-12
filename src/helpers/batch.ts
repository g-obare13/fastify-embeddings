/**
 * Split a list into fixed-size chunks, keeping order.
 */

/**
 * Split `items` into chunks of at most `size` items. Order is preserved and the
 * last chunk may be shorter.
 *
 * @example
 * ```ts
 * batch(['a', 'b', 'c', 'd', 'e'], 2); // [['a', 'b'], ['c', 'd'], ['e']]
 * ```
 */
export function batch<T>(items: readonly T[], size: number): T[][] {
  if (!Number.isInteger(size) || size < 1) {
    throw new RangeError(`batch: size must be a positive integer, received ${String(size)}`);
  }
  const chunks: T[][] = [];
  for (let start = 0; start < items.length; start += size) {
    chunks.push(items.slice(start, start + size));
  }
  return chunks;
}
