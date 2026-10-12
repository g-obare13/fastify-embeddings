/**
 * `defineProvider`: an identity helper that type-checks a custom provider.
 */
import type { EmbeddingProvider } from '#src/types.js';

/**
 * Type-check a custom provider object. Returns it unchanged.
 *
 * @example
 * ```ts
 * const provider = defineProvider({
 *   name: 'my-api',
 *   model: 'my-model-v1',
 *   dimensions: 512,
 *   async embed(texts, inputType, signal) {
 *     const res = await fetch('https://example.com/embed', {
 *       method: 'POST',
 *       body: JSON.stringify({ texts, kind: inputType }),
 *       signal,
 *     });
 *     const json = (await res.json()) as { vectors: number[][] };
 *     return { vectors: json.vectors };
 *   },
 * });
 * await app.register(embeddings, { provider });
 * ```
 */
export function defineProvider<T extends EmbeddingProvider>(provider: T): T {
  return provider;
}
