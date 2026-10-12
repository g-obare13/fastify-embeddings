import { describe, expect, it } from 'vitest';
import {
  createEmbedder,
  defineProvider,
  EmbeddingError,
  httpProvider,
  localProvider,
  mockProvider,
  resolveProvider,
} from 'fastify-embeddings';
import type { EmbeddingProvider } from 'fastify-embeddings';

describe('coverage edge cases', () => {
  it('covers transformInput returning invalid array or invalid length', async () => {
    const embedder = createEmbedder({
      provider: mockProvider({ dimensions: 4 }),
      transformInput: () => ['only-one'],
    });

    await expect(embedder.embed(['input-1', 'input-2'])).rejects.toThrow(
      'transformInput must return one string per input',
    );
  });

  it('covers transformOutput returning invalid array or invalid length', async () => {
    const embedder = createEmbedder({
      provider: mockProvider({ dimensions: 4 }),
      transformOutput: () => [],
    });

    await expect(embedder.embed(['input-1'])).rejects.toThrow(
      'transformOutput must return one vector per input',
    );
  });

  it('covers provider returning mismatched vector count or malformed vectors', async () => {
    const badCountProvider = defineProvider({
      name: 'bad-count',
      model: 'm1',
      async embed() {
        return { vectors: [[1, 2]] }; // returns 1 vector for 2 inputs
      },
    });

    const embedder1 = createEmbedder({ provider: badCountProvider });
    await expect(embedder1.embed(['a', 'b'])).rejects.toThrow(
      'returned 1 vectors for 2 inputs',
    );

    const nonArrayProvider = defineProvider({
      name: 'non-array',
      model: 'm1',
      async embed() {
        return { notVectors: true } as unknown as { vectors: number[][] };
      },
    });

    const embedder2 = createEmbedder({ provider: nonArrayProvider });
    await expect(embedder2.embed(['a'])).rejects.toThrowError(
      expect.objectContaining({ code: 'EMBEDDING_PROVIDER_ERROR' }),
    );
    await expect(embedder2.embed(['a'])).rejects.toThrow('returned an invalid result');

    const malformedVectorProvider = defineProvider({
      name: 'malformed-vector',
      model: 'm1',
      async embed() {
        return { vectors: ['not-a-vector' as unknown as number[]] };
      },
    });

    const embedder3 = createEmbedder({ provider: malformedVectorProvider });
    await expect(embedder3.embed(['a'])).rejects.toThrow(
      'Provider returned a malformed vector at index 0',
    );
  });

  it('rejects the legacy bare-array provider result shape with an actionable error', async () => {
    const provider = {
      name: 'legacy-array',
      model: 'm1',
      async embed() {
        return [[1, 2]];
      },
    } as unknown as EmbeddingProvider;

    const embedder = createEmbedder({ provider });
    const error: unknown = await embedder.embed('leaky-provider-input').then(
      () => undefined,
      (reason: unknown) => reason,
    );

    expect(error).toBeInstanceOf(EmbeddingError);
    expect(error).toMatchObject({ code: 'EMBEDDING_PROVIDER_ERROR', statusCode: 502 });
    const message = (error as EmbeddingError).message;
    expect(message).toContain('legacy-array');
    expect(message).toContain('{ vectors: number[][] }');
    expect(message).toContain('one vector per input');
    expect(message).toContain('bare array is not supported');
    expect(message).not.toContain('leaky-provider-input');
  });

  it('rejects invalid provider usage metadata', async () => {
    const provider = {
      name: 'bad-usage',
      model: 'm1',
      async embed() {
        return { vectors: [[1, 2]], usage: { promptTokens: 'one', totalTokens: 1 } };
      },
    } as unknown as EmbeddingProvider;

    const embedder = createEmbedder({ provider });
    await expect(embedder.embed('input')).rejects.toThrowError(
      expect.objectContaining({ code: 'EMBEDDING_PROVIDER_ERROR' }),
    );
    await expect(embedder.embed('input')).rejects.toThrow('returned an invalid result');
  });

  it('covers httpProvider 408 timeout status and HTTP date Retry-After', async () => {
    const futureDate = new Date(Date.now() + 5000).toUTCString();
    const provider = httpProvider({
      baseURL: 'https://mock.local',
      model: 'm1',
      fetch: async () => {
        return new Response('Timeout', {
          status: 408,
          headers: { 'retry-after': futureDate },
        });
      },
      retries: 0,
    });

    await expect(provider.embed(['test'], 'document')).rejects.toThrowError(
      expect.objectContaining({
        code: 'EMBEDDING_TIMEOUT',
        statusCode: 504,
      }),
    );
  });

  it('covers mockProvider with artificial latency', async () => {
    const provider = mockProvider({ dimensions: 4, latencyMs: 10 });
    const start = Date.now();
    await provider.embed(['hello'], 'document');
    expect(Date.now() - start).toBeGreaterThanOrEqual(8);
  });

  it('covers localProvider pipeline failure handling', async () => {
    const provider = localProvider({
      load: async () => ({
        pipeline: async () => {
          throw new Error('Pipeline init failed');
        },
      }),
      model: 'failing/model',
    });

    await expect(provider.embed(['test'], 'document')).rejects.toThrow(
      'Could not load local model failing/model',
    );
  });

  it('covers env schema validation failure in resolveProvider', () => {
    expect(() =>
      resolveProvider({
        env: {
          EMBEDDINGS_BASE_URL: 'not-a-valid-url',
        },
      }),
    ).toThrow('invalid env var "EMBEDDINGS_BASE_URL"');
  });
});
