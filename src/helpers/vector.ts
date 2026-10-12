/**
 * Pure vector math helpers. None of them mutate their inputs.
 */
import { EmbeddingError } from '#src/errors.js';
import type { Vector } from '#src/types.js';

/**
 * Return an L2-normalized copy of a vector (unit length). A zero vector is
 * returned unchanged because it has no direction.
 *
 * @example
 * ```ts
 * normalize([3, 4]); // [0.6, 0.8]
 * ```
 */
export function normalize(vector: readonly number[]): Vector {
  let sum = 0;
  for (const value of vector) sum += value * value;
  if (sum === 0) return [...vector];
  const length = Math.sqrt(sum);
  return vector.map((value) => value / length);
}

/**
 * Keep the first `dimensions` values of a vector. Useful for Matryoshka models
 * such as nomic-embed-text or text-embedding-3. Re-normalize afterwards if you
 * compare with dot products.
 *
 * @example
 * ```ts
 * truncate([0.1, 0.2, 0.3, 0.4], 2); // [0.1, 0.2]
 * normalize(truncate(vector, 256)); // shorter unit vector
 * ```
 */
export function truncate(vector: readonly number[], dimensions: number): Vector {
  if (!Number.isInteger(dimensions) || dimensions < 1) {
    throw new RangeError(`truncate: dimensions must be a positive integer, received ${String(dimensions)}`);
  }
  if (dimensions > vector.length) {
    throw new RangeError(`truncate: cannot keep ${String(dimensions)} of ${String(vector.length)} dimensions`);
  }
  return vector.slice(0, dimensions);
}

/**
 * Cosine similarity between two vectors, from -1 to 1. Returns 0 when either
 * vector is all zeros. Throws `EMBEDDING_DIMENSION_MISMATCH` when lengths differ.
 *
 * @example
 * ```ts
 * cosineSimilarity([1, 0], [1, 0]); // 1
 * cosineSimilarity([1, 0], [0, 1]); // 0
 * ```
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new EmbeddingError(
      'EMBEDDING_DIMENSION_MISMATCH',
      `cosineSimilarity: vectors have different lengths (${String(a.length)} and ${String(b.length)})`,
    );
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
