import { describe, expect, it } from 'vitest';
import Fastify from 'fastify';
import { defineProvider, fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

describe('fastifyEmbeddings plugin', () => {
  it('decorates app.embeddings with embed, model, dimensions, provider, healthCheck, and cosineSimilarity', async () => {
    const app = Fastify();
    await app.register(fastifyEmbeddings, {
      provider: mockProvider({ dimensions: 16, model: 'test-model' }),
    });
    await app.ready();

    expect(app.embeddings).toBeDefined();
    expect(app.embeddings.provider).toBe('mock');
    expect(app.embeddings.model).toBe('test-model');
    expect(typeof app.embeddings.embed).toBe('function');
    expect(typeof app.embeddings.healthCheck).toBe('function');
    expect(typeof app.embeddings.cosineSimilarity).toBe('function');

    const result = await app.embeddings.embed('hello Fastify');
    expect(result.vectors.length).toBe(1);
    expect(result.dimensions).toBe(16);

    const health = await app.embeddings.healthCheck();
    expect(health.ok).toBe(true);

    const similarity = app.embeddings.cosineSimilarity([1, 0], [1, 0]);
    expect(similarity).toBeCloseTo(1, 5);

    await app.close();
  });

  it('supports named embedders accessed via app.embeddings.get(name)', async () => {
    const app = Fastify();

    await app.register(fastifyEmbeddings, {
      provider: mockProvider({ dimensions: 8, model: 'default-model' }),
    });

    await app.register(fastifyEmbeddings, {
      name: 'docs',
      provider: mockProvider({ dimensions: 32, model: 'docs-model' }),
    });

    await app.ready();

    // Default embedder
    expect(app.embeddings.model).toBe('default-model');
    const defaultRes = await app.embeddings.embed('doc 1');
    expect(defaultRes.dimensions).toBe(8);

    // Named embedder
    const docsEmbedder = app.embeddings.get('docs');
    expect(docsEmbedder).toBeDefined();
    expect(docsEmbedder?.model).toBe('docs-model');
    const docsRes = await docsEmbedder?.embed('doc 2');
    expect(docsRes?.dimensions).toBe(32);

    await app.close();
  });

  it('eager option runs healthCheck at ready and fails startup when provider is unreachable', async () => {
    const app = Fastify();
    const broken = defineProvider({
      name: 'broken',
      model: 'b1',
      async embed() {
        throw new Error('Connection refused');
      },
    });

    await app.register(fastifyEmbeddings, {
      provider: broken,
      eager: true,
    });

    await expect(app.ready()).rejects.toThrow();
    await app.close();
  });

  it('calls provider close on fastify app.close()', async () => {
    const app = Fastify();
    let closed = false;

    const providerWithClose = defineProvider({
      name: 'closable',
      model: 'c1',
      dimensions: 4,
      async embed() {
        return { vectors: [[1, 2, 3, 4]] };
      },
      close() {
        closed = true;
      },
    });

    await app.register(fastifyEmbeddings, {
      provider: providerWithClose,
    });

    await app.ready();
    expect(closed).toBe(false);

    await app.close();
    expect(closed).toBe(true);
  });

  describe('optional POST /embeddings route via app.inject', () => {
    it('is off by default', async () => {
      const app = Fastify();
      await app.register(fastifyEmbeddings, {
        provider: mockProvider({ dimensions: 4 }),
      });
      await app.ready();

      const response = await app.inject({
        method: 'POST',
        url: '/embeddings',
        payload: { input: 'test' },
      });

      expect(response.statusCode).toBe(404);
      await app.close();
    });

    it('enables route and returns standard EmbedResult shape', async () => {
      const app = Fastify();
      await app.register(fastifyEmbeddings, {
        provider: mockProvider({ dimensions: 8, model: 'mock-model' }),
        route: true,
      });
      await app.ready();

      const response = await app.inject({
        method: 'POST',
        url: '/embeddings',
        payload: { input: ['hello', 'world'], inputType: 'query' },
      });

      expect(response.statusCode).toBe(200);
      const json = response.json();
      expect(json.vectors.length).toBe(2);
      expect(json.dimensions).toBe(8);
      expect(json.model).toBe('mock-model');
      expect(json.usage).toBeDefined();

      await app.close();
    });

    it('returns 400 with EMBEDDING_INPUT_INVALID on invalid request body', async () => {
      const app = Fastify();
      await app.register(fastifyEmbeddings, {
        provider: mockProvider({ dimensions: 8 }),
        route: true,
      });
      await app.ready();

      const response = await app.inject({
        method: 'POST',
        url: '/embeddings',
        payload: { input: '' },
      });

      expect(response.statusCode).toBe(400);
      const json = response.json();
      expect(json.code).toBe('EMBEDDING_INPUT_INVALID');

      await app.close();
    });

    it('supports custom route path and bodyLimit configuration', async () => {
      const app = Fastify();
      await app.register(fastifyEmbeddings, {
        provider: mockProvider({ dimensions: 4 }),
        route: {
          path: '/api/v1/vectors',
          bodyLimit: 100,
        },
      });
      await app.ready();

      const okResponse = await app.inject({
        method: 'POST',
        url: '/api/v1/vectors',
        payload: { input: 'small' },
      });
      expect(okResponse.statusCode).toBe(200);

      const hugePayload = { input: 'a'.repeat(500) };
      const overflowResponse = await app.inject({
        method: 'POST',
        url: '/api/v1/vectors',
        payload: hugePayload,
      });
      expect(overflowResponse.statusCode).toBe(413);

      await app.close();
    });
  });
});
