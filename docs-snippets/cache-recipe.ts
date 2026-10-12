import { createHash } from 'node:crypto';
import type { EmbeddingProvider, ProviderOutput, Vector } from 'fastify-embeddings';

const LONG_TEXT_THRESHOLD = 1024;

export function withCache(baseProvider: EmbeddingProvider, maxEntries = 500): EmbeddingProvider {
  if (!Number.isInteger(maxEntries) || maxEntries < 1) {
    throw new RangeError('maxEntries must be a positive integer');
  }

  const cache = new Map<string, Vector>();

  function cacheKey(text: string, inputType: 'query' | 'document'): string {
    const cacheText =
      text.length > LONG_TEXT_THRESHOLD
        ? `sha256:${createHash('sha256').update(text).digest('hex')}`
        : `text:${text}`;
    return JSON.stringify([baseProvider.model, inputType, cacheText]);
  }

  function store(key: string, vector: Vector): void {
    cache.delete(key);
    cache.set(key, [...vector]);
    if (cache.size > maxEntries) {
      const oldestKey = cache.keys().next().value;
      if (oldestKey !== undefined) cache.delete(oldestKey);
    }
  }

  return {
    name: `cached-${baseProvider.name}`,
    model: baseProvider.model,
    ...(baseProvider.dimensions === undefined ? {} : { dimensions: baseProvider.dimensions }),
    async embed(texts, inputType, signal): Promise<ProviderOutput> {
      const results: (Vector | undefined)[] = new Array(texts.length);
      const missingTexts: string[] = [];
      const missingIndices: number[] = [];
      const missingKeys: string[] = [];

      texts.forEach((text, index) => {
        const key = cacheKey(text, inputType);
        const cached = cache.get(key);
        if (cached !== undefined) {
          cache.delete(key);
          cache.set(key, cached);
          results[index] = [...cached];
        } else {
          missingTexts.push(text);
          missingIndices.push(index);
          missingKeys.push(key);
        }
      });

      let output: ProviderOutput | undefined;
      if (missingTexts.length > 0) {
        output = await baseProvider.embed(missingTexts, inputType, signal);
        if (output.vectors.length !== missingTexts.length) {
          throw new Error(`Cached provider returned ${String(output.vectors.length)} vectors for ${String(missingTexts.length)} inputs`);
        }
        output.vectors.forEach((vector, index) => {
          const originalIndex = missingIndices[index];
          const key = missingKeys[index];
          if (originalIndex === undefined || key === undefined) {
            throw new Error('Cached provider result could not be matched to its input');
          }
          store(key, vector);
          results[originalIndex] = [...vector];
        });
      }

      const vectors = results.map((vector, index) => {
        if (vector === undefined) throw new Error(`Missing cached vector for input ${String(index)}`);
        return [...vector];
      });
      return {
        vectors,
        ...(output?.usage === undefined ? {} : { usage: output.usage }),
        ...(output?.model === undefined ? {} : { model: output.model }),
      };
    },
    close: () => baseProvider.close?.(),
  };
}
