import { describe, expect, it, vi } from 'vitest';
import {
  createEmbedder,
  defineProvider,
  mockProvider,
  normalize,
  presets,
  truncate,
} from 'fastify-embeddings';
import { withCache } from '../docs-snippets/cache-recipe.js';
import type { BatchInfo, EmbeddingsLogger, TransformOutputContext } from 'fastify-embeddings';

describe('createEmbedder core', () => {
  describe('input validation and return shape', () => {
    it('accepts single string or array of strings, returning identical EmbedResult shape', async () => {
      const embedder = createEmbedder({
        provider: mockProvider({ dimensions: 8 }),
      });

      const single = await embedder.embed('single string');
      expect(single.vectors.length).toBe(1);
      expect(single.dimensions).toBe(8);
      expect(single.model).toBe('mock-embedding');
      expect(single.usage).toBeDefined();

      const arrayResult = await embedder.embed(['item 1', 'item 2']);
      expect(arrayResult.vectors.length).toBe(2);
      expect(arrayResult.dimensions).toBe(8);
      expect(arrayResult.model).toBe('mock-embedding');
      expect(arrayResult.usage).toBeDefined();
    });

    it('throws EMBEDDING_INPUT_INVALID on invalid inputs', async () => {
      const embedder = createEmbedder({
        provider: mockProvider({ dimensions: 4 }),
      });

      await expect(embedder.embed('')).rejects.toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_INPUT_INVALID' }),
      );
      await expect(embedder.embed('   ')).rejects.toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_INPUT_INVALID' }),
      );
      await expect(embedder.embed([])).rejects.toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_INPUT_INVALID' }),
      );
      await expect(embedder.embed(['valid', ''])).rejects.toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_INPUT_INVALID' }),
      );
      // @ts-expect-error testing invalid type
      await expect(embedder.embed([123])).rejects.toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_INPUT_INVALID' }),
      );
      // @ts-expect-error testing invalid inputType
      await expect(embedder.embed('hello', { inputType: 'invalid' })).rejects.toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_INPUT_INVALID' }),
      );
    });
  });

  describe('batching and ordering', () => {
    it('splits inputs into ordered batches and preserves vector ordering', async () => {
      const mock = mockProvider({ dimensions: 4 });
      const embedder = createEmbedder({
        provider: mock,
        batchSize: 2,
        concurrency: 2,
      });

      const inputs = ['first', 'second', 'third', 'fourth', 'fifth'];
      const result = await embedder.embed(inputs);

      expect(result.vectors.length).toBe(5);
      expect(mock.calls).toEqual([
        { count: 2, inputType: 'document' },
        { count: 2, inputType: 'document' },
        { count: 1, inputType: 'document' },
      ]);
    });
  });

  describe('normalization', () => {
    it('normalizes vectors by default to unit length', async () => {
      const embedder = createEmbedder({
        provider: mockProvider({ dimensions: 4 }),
      });

      const result = await embedder.embed('test normalization');
      const vec = result.vectors[0]!;
      const length = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
      expect(length).toBeCloseTo(1, 5);
    });

    it('disables normalization when normalize: false is set', async () => {
      const unnormalized = defineProvider({
        name: 'raw',
        model: 'raw-1',
        dimensions: 2,
        async embed() {
          return { vectors: [[10, 20]] };
        },
      });

      const embedder = createEmbedder({
        provider: unnormalized,
        normalize: false,
      });

      const result = await embedder.embed('test');
      expect(result.vectors[0]).toEqual([10, 20]);
    });
  });

  describe('presets and prefix handling', () => {
    it('applies preset prefixes correctly and allows overriding them', () => {
      expect(presets.nomic.prefixes?.query).toBe('search_query: ');
      expect(presets.nomic.prefixes?.document).toBe('search_document: ');

      expect(presets.e5.prefixes?.query).toBe('query: ');
      expect(presets.e5.prefixes?.document).toBe('passage: ');

      expect(presets.bge.prefixes?.query).toContain('Represent this sentence');
      expect(presets.minilm.prefixes?.query).toBe('');
    });
  });

  describe('hooks', () => {
    it('runs transformInput, onBatch, and transformOutput in order', async () => {
      const executionOrder: string[] = [];
      const batchInfos: BatchInfo[] = [];

      const embedder = createEmbedder({
        provider: mockProvider({ dimensions: 8 }),
        batchSize: 2,
        transformInput: (texts) => {
          executionOrder.push('transformInput');
          return texts.map((t) => t.toUpperCase());
        },
        onBatch: (info) => {
          executionOrder.push('onBatch');
          batchInfos.push(info);
        },
        transformOutput: (vectors, ctx: TransformOutputContext) => {
          executionOrder.push('transformOutput');
          expect(ctx.model).toBe('mock-embedding');
          // Truncate to 4 dims and re-normalize
          return vectors.map((v) => normalize(truncate(v, 4)));
        },
      });

      const result = await embedder.embed(['item 1', 'item 2']);

      expect(executionOrder).toEqual(['transformInput', 'onBatch', 'transformOutput']);
      expect(result.vectors[0]!.length).toBe(4);
      expect(result.dimensions).toBe(4);
      expect(batchInfos.length).toBe(1);
      expect(batchInfos[0]!.count).toBe(2);
    });

    it('triggers onError hook on failure without raw text in error', async () => {
      const errorSpy = vi.fn();
      const failingProvider = defineProvider({
        name: 'failing',
        model: 'fail-v1',
        async embed() {
          throw new Error('connection reset');
        },
      });

      const embedder = createEmbedder({
        provider: failingProvider,
        onError: errorSpy,
      });

      await expect(embedder.embed('sensitive data')).rejects.toThrow();
      expect(errorSpy).toHaveBeenCalledTimes(1);
      const reportedError = errorSpy.mock.calls[0]![0];
      expect(reportedError.code).toBe('EMBEDDING_PROVIDER_ERROR');
      expect(reportedError.message).not.toContain('sensitive data');
    });
  });

  describe('truncated output dimensions across cached batches', () => {
    it('keeps the truncated dimension guard stable across repeated multi-batch calls and cache hits', async () => {
      const base = mockProvider({ dimensions: 4 });
      const cached = withCache(base);
      const embedder = createEmbedder({
        provider: cached,
        batchSize: 2,
        concurrency: 2,
        transformOutput: (vectors) => vectors.map((vector) => truncate(vector, 2)),
      });
      const firstInputs = Array.from({ length: 9 }, (_, index) => `item-${String(index)}`);

      const first = await embedder.embed(firstInputs);
      expect(first.dimensions).toBe(2);
      expect(first.vectors).toHaveLength(9);
      expect(first.vectors.every((vector) => vector.length === 2)).toBe(true);
      expect(base.calls).toHaveLength(5);

      const second = await embedder.embed(['item-0', 'new-a', 'item-2', 'new-b', 'item-8']);
      expect(second.dimensions).toBe(2);
      expect(second.vectors).toHaveLength(5);
      expect(second.vectors.every((vector) => vector.length === 2)).toBe(true);
      expect(base.calls).toHaveLength(7);
      expect(embedder.dimensions).toBe(2);
    });
  });

  describe('dimension mismatch detection', () => {
    it('throws EMBEDDING_DIMENSION_MISMATCH when output differs from configured dimensions', async () => {
      const mismatch = defineProvider({
        name: 'mismatch',
        model: 'm1',
        dimensions: 128,
        async embed() {
          return { vectors: [[0.1, 0.2]] }; // only 2 dims
        },
      });

      const embedder = createEmbedder({ provider: mismatch });
      await expect(embedder.embed('test')).rejects.toThrowError(
        expect.objectContaining({
          code: 'EMBEDDING_DIMENSION_MISMATCH',
        }),
      );
    });

    it('detects dimension on first call and throws on later dimension change', async () => {
      let callCount = 0;
      const changingDims = defineProvider({
        name: 'changing',
        model: 'm1',
        async embed() {
          callCount++;
          if (callCount === 1) return { vectors: [[0.1, 0.2, 0.3]] }; // 3 dims
          return { vectors: [[0.1, 0.2]] }; // 2 dims on 2nd call
        },
      });

      const embedder = createEmbedder({ provider: changingDims });

      // First call succeeds and detects 3 dimensions
      const res1 = await embedder.embed('first call');
      expect(res1.dimensions).toBe(3);
      expect(embedder.dimensions).toBe(3);

      // Second call returns 2 dimensions -> mismatch
      await expect(embedder.embed('second call')).rejects.toThrowError(
        expect.objectContaining({
          code: 'EMBEDDING_DIMENSION_MISMATCH',
        }),
      );
    });
  });

  describe('healthCheck', () => {
    it('returns ok: true with duration and dimensions on healthy provider', async () => {
      const embedder = createEmbedder({
        provider: mockProvider({ dimensions: 16 }),
      });

      const health = await embedder.healthCheck();
      expect(health.ok).toBe(true);
      if (health.ok) {
        expect(health.dimensions).toBe(16);
        expect(health.model).toBe('mock-embedding');
        expect(health.durationMs).toBeGreaterThanOrEqual(0);
      }
    });

    it('returns ok: false with error on unreachable provider', async () => {
      const broken = defineProvider({
        name: 'broken',
        model: 'b1',
        async embed() {
          throw new Error('Host not found');
        },
      });

      const embedder = createEmbedder({ provider: broken });
      const health = await embedder.healthCheck();
      expect(health.ok).toBe(false);
      if (!health.ok) {
        expect(health.error.code).toBe('EMBEDDING_PROVIDER_ERROR');
      }
    });
  });

  describe('logging invariants', () => {
    it('never logs raw inputs, vectors, or secrets', async () => {
      const loggedMessages: Array<{ details: object; msg: string }> = [];
      const testLogger: EmbeddingsLogger = {
        debug(details, msg) {
          loggedMessages.push({ details, msg });
        },
        warn(details, msg) {
          loggedMessages.push({ details, msg });
        },
      };

      const embedder = createEmbedder({
        provider: mockProvider({ dimensions: 8 }),
        logger: testLogger,
      });

      const secretText = 'super-secret-user-prompt-12345';
      await embedder.embed(secretText);

      expect(loggedMessages.length).toBeGreaterThan(0);
      for (const entry of loggedMessages) {
        const serialized = JSON.stringify(entry);
        expect(serialized).not.toContain(secretText);
        expect(serialized).not.toContain('vectors');
        // Only count, model, duration, batches etc.
        expect(entry.details).toHaveProperty('count');
        expect(entry.details).toHaveProperty('model');
        expect(entry.details).toHaveProperty('durationMs');
      }
    });
  });
});
