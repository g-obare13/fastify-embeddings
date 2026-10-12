import { describe, expect, it, vi } from 'vitest';
import {
  cosineSimilarity,
  createEmbedder,
  defineProvider,
  localProvider,
  mockProvider,
  mockVector,
} from 'fastify-embeddings';
import type { FeatureExtractor, TransformersModule } from 'fastify-embeddings';

describe('mock provider', () => {
  it('is deterministic: same text produces identical vectors', () => {
    const v1 = mockVector('hello fastify embeddings', 64);
    const v2 = mockVector('hello fastify embeddings', 64);
    expect(v1).toEqual(v2);
    expect(v1.length).toBe(64);
  });

  it('sentences sharing words score higher similarity than unrelated sentences', () => {
    const vRef = mockVector('quick brown fox jumps', 128);
    const vSimilar = mockVector('quick brown fox sleeps', 128);
    const vUnrelated = mockVector('aerospace engineering thermodynamics', 128);

    const simSimilar = cosineSimilarity(vRef, vSimilar);
    const simUnrelated = cosineSimilarity(vRef, vUnrelated);

    expect(simSimilar).toBeGreaterThan(simUnrelated);
  });

  it('records call count and inputType', async () => {
    const provider = mockProvider({ dimensions: 16 });
    expect(provider.calls).toEqual([]);
    expect(provider.closed).toBe(false);

    await provider.embed(['doc 1', 'doc 2'], 'document');
    await provider.embed(['query 1'], 'query');

    expect(provider.calls).toEqual([
      { count: 2, inputType: 'document' },
      { count: 1, inputType: 'query' },
    ]);

    await provider.close?.();
    expect(provider.closed).toBe(true);
  });
});

describe('local provider', () => {
  it('is lazy loaded: does not call load() upon creation', () => {
    const loadSpy = vi.fn();
    localProvider({ load: loadSpy });
    expect(loadSpy).not.toHaveBeenCalled();
  });

  it('handles missing peer dependency by throwing EMBEDDING_PROVIDER_UNREACHABLE', async () => {
    const provider = localProvider({
      load: async () => {
        throw new Error('Cannot find module @huggingface/transformers');
      },
    });

    await expect(provider.embed(['test'], 'document')).rejects.toThrowError(
      expect.objectContaining({
        code: 'EMBEDDING_PROVIDER_UNREACHABLE',
      }),
    );
  });

  it('loads pipeline with feature-extraction and runs inference', async () => {
    const disposeSpy = vi.fn();
    const extractorMock = vi.fn().mockResolvedValue({
      tolist: () => [[0.1, 0.2, 0.3]],
    }) as unknown as FeatureExtractor;
    extractorMock.dispose = disposeSpy;

    const pipelineSpy = vi.fn().mockResolvedValue(extractorMock);
    const mockModule: TransformersModule = {
      pipeline: pipelineSpy,
    };

    const provider = localProvider({
      load: async () => mockModule,
      model: 'custom/minilm',
      dimensions: 3,
    });

    const output = await provider.embed(['hello world'], 'document');
    expect(output).toEqual({ vectors: [[0.1, 0.2, 0.3]] });
    expect(pipelineSpy).toHaveBeenCalledWith('feature-extraction', 'custom/minilm', undefined);
    expect(extractorMock).toHaveBeenCalledWith(['hello world'], {
      pooling: 'mean',
      normalize: true,
    });

    await provider.close?.();
    expect(disposeSpy).toHaveBeenCalled();
  });
});

describe('custom provider extension point', () => {
  it('allows user-defined provider object via defineProvider', async () => {
    let closed = false;
    const custom = defineProvider({
      name: 'in-memory-hash',
      model: 'hash-v1',
      dimensions: 4,
      async embed(texts) {
        return { vectors: texts.map((t) => [t.length, 0.5, 0.25, 0.1]) };
      },
      close() {
        closed = true;
      },
    });

    const embedder = createEmbedder({ provider: custom });
    expect(embedder.provider).toBe('in-memory-hash');
    expect(embedder.model).toBe('hash-v1');

    const result = await embedder.embed(['cat', 'elephant']);
    expect(result.vectors.length).toBe(2);
    expect(result.dimensions).toBe(4);
    expect(result.model).toBe('hash-v1');

    await embedder.close();
    expect(closed).toBe(true);
  });
});
