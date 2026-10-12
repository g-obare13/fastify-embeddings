import { describe, expect, it } from 'vitest';
import * as allExports from '../src/index.js';
import * as testingExports from '../src/testing.js';

describe('barrel exports', () => {
  it('exports all expected public members from index.ts', () => {
    expect(allExports.fastifyEmbeddings).toBeDefined();
    expect(allExports.embeddings).toBeDefined();
    expect(allExports.default).toBeDefined();
    expect(allExports.createEmbedder).toBeDefined();
    expect(allExports.EmbeddingError).toBeDefined();
    expect(allExports.isEmbeddingError).toBeDefined();
    expect(allExports.presets).toBeDefined();
    expect(allExports.applyPrefixes).toBeDefined();
    expect(allExports.batch).toBeDefined();
    expect(allExports.withRetry).toBeDefined();
    expect(allExports.cosineSimilarity).toBeDefined();
    expect(allExports.normalize).toBeDefined();
    expect(allExports.truncate).toBeDefined();
    expect(allExports.defineProvider).toBeDefined();
    expect(allExports.httpProvider).toBeDefined();
    expect(allExports.localProvider).toBeDefined();
    expect(allExports.DEFAULT_LOCAL_MODEL).toBeDefined();
    expect(allExports.mockProvider).toBeDefined();
    expect(allExports.mockVector).toBeDefined();
    expect(allExports.formats).toBeDefined();
    expect(allExports.embedderOptionsSchema).toBeDefined();
    expect(allExports.pluginOptionsSchema).toBeDefined();
    expect(allExports.resolveProvider).toBeDefined();
    expect(allExports.validateOptions).toBeDefined();
    expect(allExports.registerRoute).toBeDefined();
  });

  it('exports test utilities from testing.ts', () => {
    expect(testingExports.mockProvider).toBeDefined();
    expect(testingExports.mockVector).toBeDefined();
    const mock = testingExports.mockProvider({ dimensions: 4 });
    expect(mock.dimensions).toBe(4);
  });
});
