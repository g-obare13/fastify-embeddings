import { describe, expect, it } from 'vitest';
import {
  isEmbeddingError,
  EmbeddingError,
  resolveProvider,
  fastifyEmbeddings,
} from 'fastify-embeddings';
import Fastify from 'fastify';

describe('options and env var resolution', () => {
  it('zero-config: resolves http provider when EMBEDDINGS_BASE_URL is set in env', () => {
    const provider = resolveProvider({
      env: {
        EMBEDDINGS_BASE_URL: 'http://localhost:11434',
        EMBEDDINGS_MODEL: 'nomic-embed-text',
        EMBEDDINGS_FORMAT: 'ollama',
        EMBEDDINGS_API_KEY: 'test-key',
      },
    });

    expect(provider.name).toBe('http');
    expect(provider.model).toBe('nomic-embed-text');
  });

  it('zero-config fallback: resolves local provider when EMBEDDINGS_BASE_URL is not set', () => {
    const provider = resolveProvider({
      env: {},
    });

    expect(provider.name).toBe('local');
    expect(provider.model).toBe('Xenova/all-MiniLM-L6-v2');
  });

  it('precedence: explicit option > env var > preset default', () => {
    const provider = resolveProvider({
      model: 'explicit-model',
      preset: 'nomic',
      env: {
        EMBEDDINGS_BASE_URL: 'http://localhost:11434',
        EMBEDDINGS_MODEL: 'env-model',
      },
    });

    // Explicit wins over env
    expect(provider.model).toBe('explicit-model');

    const providerFromEnv = resolveProvider({
      preset: 'nomic',
      env: {
        EMBEDDINGS_BASE_URL: 'http://localhost:11434',
        EMBEDDINGS_MODEL: 'env-model',
      },
    });

    // Env wins over preset default
    expect(providerFromEnv.model).toBe('env-model');

    const providerFromPreset = resolveProvider({
      preset: 'nomic',
      baseURL: 'http://localhost:11434',
      env: {},
    });

    // Preset fallback
    expect(providerFromPreset.model).toBe('nomic-embed-text');
  });

  it('uses the Nomic preset model without an explicit model', () => {
    const provider = resolveProvider({
      baseURL: 'http://localhost:11434',
      format: 'ollama',
      preset: 'nomic',
      env: {},
    });

    expect(provider.model).toBe('nomic-embed-text');
  });

  it('throws on missing baseURL for http provider without env', () => {
    expect(() =>
      resolveProvider({
        provider: 'http',
        env: {},
      }),
    ).toThrow('invalid option "baseURL"');
  });

  it('isEmbeddingError correctly identifies errors and codes', () => {
    const err = new EmbeddingError('EMBEDDING_RATE_LIMITED', 'Too many requests');
    expect(isEmbeddingError(err)).toBe(true);
    expect(isEmbeddingError(err, 'EMBEDDING_RATE_LIMITED')).toBe(true);
    expect(isEmbeddingError(err, 'EMBEDDING_TIMEOUT')).toBe(false);
    expect(isEmbeddingError(new Error('Generic error'))).toBe(false);
    expect(isEmbeddingError(null)).toBe(false);
  });

  it('decorator fallback branches when no default embedder is registered', async () => {
    const app = Fastify();
    await app.register(fastifyEmbeddings, {
      name: 'only-named',
      provider: 'mock',
    });
    await app.ready();

    // Default properties should be empty/undefined
    expect(app.embeddings.model).toBe('');
    expect(app.embeddings.dimensions).toBeUndefined();
    expect(app.embeddings.provider).toBe('');

    // Default actions should throw
    await expect(app.embeddings.embed('test')).rejects.toThrow('No default embedder registered');
    await expect(app.embeddings.healthCheck()).rejects.toThrow('No default embedder registered');

    // Named embedder is accessible
    expect(app.embeddings.get('only-named')).toBeDefined();
    expect(app.embeddings.get('default')).toBeUndefined();

    // Default close does not fail
    await expect(app.embeddings.close()).resolves.toBeUndefined();

    await app.close();
  });
});
