/**
 * Internal: validates provider vectors and tracks their size. The size comes
 * from the configured `dimensions` or, if unset, from the first response.
 * Any later response of a different size is a dimension mismatch.
 */
import { EmbeddingError } from '#src/errors.js';
import type { Vector } from '#src/types.js';

/** Tracks the expected vector size for one embedder. */
export interface DimensionGuard {
  /** The known vector size, or `undefined` before the first response. */
  readonly value: number | undefined;
  /** Validate vectors and return their size. Throws on malformed data or mismatch. */
  check(vectors: readonly unknown[]): number;
}

/** True when `value` is a non-empty array of finite numbers. */
function isVector(value: unknown): value is Vector {
  return Array.isArray(value) && value.length > 0 && value.every((n) => typeof n === 'number' && Number.isFinite(n));
}

/**
 * Create a guard. `configured` comes from the provider's `dimensions`.
 *
 * @example
 * ```ts
 * const guard = createDimensionGuard(768);
 * guard.check(vectors); // throws EMBEDDING_DIMENSION_MISMATCH if any vector is not 768 long
 * ```
 */
export function createDimensionGuard(configured?: number): DimensionGuard {
  let known = configured;
  const source = (): string => (known === configured ? 'configured' : 'detected from the first response');

  return {
    get value() {
      return known;
    },
    check(vectors) {
      for (let i = 0; i < vectors.length; i++) {
        const vector = vectors[i];
        if (!isVector(vector)) {
          throw new EmbeddingError('EMBEDDING_PROVIDER_ERROR', `Provider returned a malformed vector at index ${String(i)}`);
        }
        if (known === undefined) {
          known = vector.length;
        } else if (vector.length !== known) {
          throw new EmbeddingError(
            'EMBEDDING_DIMENSION_MISMATCH',
            `Expected ${String(known)} dimensions (${source()}), received ${String(vector.length)}`,
          );
        }
      }
      return known ?? 0;
    },
  };
}
