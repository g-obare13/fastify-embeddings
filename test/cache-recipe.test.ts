import { describe, expect, it } from 'vitest';
import { localProvider, mockProvider, presets } from 'fastify-embeddings';
import { withCache } from '../docs-snippets/cache-recipe.js';

describe('README cache recipe', () => {
  it('separates query and document entries with a prefixing preset and returns copies', async () => {
    const received: string[] = [];
    const provider = localProvider({
      preset: presets.nomic,
      load: async () => ({
        pipeline: async () => async (texts: string[]) => {
          received.push(...texts);
          return { tolist: () => texts.map((text) => [text.length, 1]) };
        },
      }),
    });
    const cached = withCache(provider);
    const query = await cached.embed(['same text'], 'query');
    const document = await cached.embed(['same text'], 'document');
    const originalQueryVector = [...query.vectors[0]!];
    query.vectors[0]![0] = -1;
    const queryHit = await cached.embed(['same text'], 'query');

    expect(provider.model).toBe('nomic-embed-text');
    expect(received).toEqual(['search_query: same text', 'search_document: same text']);
    expect(originalQueryVector).not.toEqual(document.vectors[0]);
    expect(queryHit.vectors[0]![0]).toBeGreaterThan(0);
    expect(queryHit.usage).toBeUndefined();
    await provider.close?.();
  });

  it('evicts the least-recently-used item at the configured maxEntries and preserves miss usage', async () => {
    const base = mockProvider({ dimensions: 4 });
    const cached = withCache(base, 2);
    const a = await cached.embed(['alpha'], 'document');
    const b = await cached.embed(['beta'], 'document');
    await cached.embed(['alpha'], 'document');
    await cached.embed(['gamma'], 'document');
    const callsAfterEviction = base.calls.length;

    expect(a.usage).toBeDefined();
    expect(b.usage).toBeDefined();
    const partialHit = await cached.embed(['alpha', 'beta'], 'document');
    expect(base.calls).toHaveLength(callsAfterEviction + 1);
    expect(partialHit.usage).toBeDefined();

    const fullHit = await cached.embed(['alpha'], 'document');
    expect(fullHit.usage).toBeUndefined();
    expect(base.calls).toHaveLength(callsAfterEviction + 1);
  });
});
