import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MockAgent, setGlobalDispatcher, getGlobalDispatcher } from 'undici';
import { httpProvider } from 'fastify-embeddings';
import type { Dispatcher } from 'undici';

describe('http provider', () => {
  let mockAgent: MockAgent;
  let originalDispatcher: Dispatcher;

  beforeEach(() => {
    originalDispatcher = getGlobalDispatcher();
    mockAgent = new MockAgent();
    mockAgent.disableNetConnect();
    setGlobalDispatcher(mockAgent);
  });

  afterEach(async () => {
    await mockAgent.close();
    setGlobalDispatcher(originalDispatcher);
  });

  describe('openai format', () => {
    it('sends correct request body, headers, and sorts vectors by index', async () => {
      const client = mockAgent.get('https://api.openai.com');

      client
        .intercept({
          path: '/v1/embeddings',
          method: 'POST',
        })
        .reply(200, {
          data: [
            { index: 1, embedding: [0.3, 0.4] },
            { index: 0, embedding: [0.1, 0.2] },
          ],
          model: 'text-embedding-3-small-2024',
          usage: {
            prompt_tokens: 12,
            total_tokens: 12,
          },
        });

      const provider = httpProvider({
        baseURL: 'https://api.openai.com/v1',
        model: 'text-embedding-3-small',
        apiKey: 'sk-secret-token',
        dimensions: 2,
      });

      const output = await provider.embed(['doc 1', 'doc 2'], 'document');
      expect(output).toEqual({
        vectors: [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
        model: 'text-embedding-3-small-2024',
        usage: { promptTokens: 12, totalTokens: 12 },
      });
    });

    it('omits dimensions parameter when not specified in options', async () => {
      const client = mockAgent.get('https://api.openai.com');

      client
        .intercept({
          path: '/v1/embeddings',
          method: 'POST',
        })
        .reply(200, {
          data: [{ index: 0, embedding: [0.5, 0.6] }],
        });

      const provider = httpProvider({
        baseURL: 'https://api.openai.com/v1',
        model: 'text-embedding-3-small',
      });

      const output = await provider.embed(['test'], 'document');
      expect(output).toEqual({ vectors: [[0.5, 0.6]] });
    });
  });

  describe('ollama format', () => {
    it('sends POST /api/embed and parses response with usage', async () => {
      const client = mockAgent.get('http://localhost:11434');

      client
        .intercept({
          path: '/api/embed',
          method: 'POST',
        })
        .reply(200, {
          embeddings: [[0.1, 0.2, 0.3]],
          model: 'nomic-embed-text:latest',
          prompt_eval_count: 5,
        });

      const provider = httpProvider({
        baseURL: 'http://localhost:11434',
        model: 'nomic-embed-text',
        format: 'ollama',
      });

      const output = await provider.embed(['hello'], 'document');
      expect(output).toEqual({
        vectors: [[0.1, 0.2, 0.3]],
        model: 'nomic-embed-text:latest',
        usage: { promptTokens: 5, totalTokens: 5 },
      });
    });
  });

  describe('retries and error handling', () => {
    it('retries on 429 rate limit and succeeds on retry', async () => {
      const client = mockAgent.get('https://api.example.com');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(429, { error: 'Rate limit exceeded' }, { headers: { 'retry-after': '0' } });

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(200, {
          data: [{ index: 0, embedding: [0.1, 0.2] }],
        });

      const provider = httpProvider({
        baseURL: 'https://api.example.com',
        model: 'm1',
        retries: 2,
        retryBaseDelayMs: 1,
      });

      const output = await provider.embed(['test'], 'document');
      expect(output).toEqual({ vectors: [[0.1, 0.2]] });
    });

    it('retries on 503 service unavailable and succeeds', async () => {
      const client = mockAgent.get('https://api.example.com');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(503, 'Service Unavailable');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(200, {
          data: [{ index: 0, embedding: [0.9] }],
        });

      const provider = httpProvider({
        baseURL: 'https://api.example.com',
        model: 'm1',
        retries: 1,
        retryBaseDelayMs: 1,
      });

      const output = await provider.embed(['test'], 'document');
      expect(output).toEqual({ vectors: [[0.9]] });
    });

    it('fails with EMBEDDING_RATE_LIMITED when 429 persists', async () => {
      const client = mockAgent.get('https://api.example.com');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(429, 'Rate limit', { headers: { 'retry-after': '0' } })
        .times(2);

      const provider = httpProvider({
        baseURL: 'https://api.example.com',
        model: 'm1',
        retries: 1,
        retryBaseDelayMs: 1,
      });

      await expect(provider.embed(['test'], 'document')).rejects.toThrowError(
        expect.objectContaining({
          code: 'EMBEDDING_RATE_LIMITED',
          statusCode: 429,
        }),
      );
    });

    it('throws EMBEDDING_TIMEOUT when request times out', async () => {
      const client = mockAgent.get('https://api.example.com');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(200, { data: [{ index: 0, embedding: [1] }] })
        .delay(100);

      const provider = httpProvider({
        baseURL: 'https://api.example.com',
        model: 'm1',
        timeoutMs: 20,
        retries: 0,
      });

      await expect(provider.embed(['test'], 'document')).rejects.toThrowError(
        expect.objectContaining({
          code: 'EMBEDDING_TIMEOUT',
          statusCode: 504,
        }),
      );
    });

    it('throws EMBEDDING_PROVIDER_ERROR on invalid JSON response', async () => {
      const client = mockAgent.get('https://api.example.com');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(200, '<html>502 Bad Gateway</html>', {
          headers: { 'content-type': 'text/html' },
        });

      const provider = httpProvider({
        baseURL: 'https://api.example.com',
        model: 'm1',
        retries: 0,
      });

      await expect(provider.embed(['test'], 'document')).rejects.toThrowError(
        expect.objectContaining({
          code: 'EMBEDDING_PROVIDER_ERROR',
        }),
      );
    });

    it('throws EMBEDDING_PROVIDER_ERROR on malformed response body', async () => {
      const client = mockAgent.get('https://api.example.com');

      client
        .intercept({ path: '/embeddings', method: 'POST' })
        .reply(200, { data: 'not-an-array' });

      const provider = httpProvider({
        baseURL: 'https://api.example.com',
        model: 'm1',
        retries: 0,
      });

      await expect(provider.embed(['test'], 'document')).rejects.toThrowError(
        expect.objectContaining({
          code: 'EMBEDDING_PROVIDER_ERROR',
        }),
      );
    });
  });

  describe('custom buildRequest and parseResponse escape hatches', () => {
    it('supports custom request path, payload shape, and response format', async () => {
      const client = mockAgent.get('https://api.custom-ai.internal');

      client
        .intercept({
          path: '/custom/v2/vectorize',
          method: 'POST',
        })
        .reply(200, {
          response: {
            items: [{ vector: [0.77, 0.88] }],
            meta: {
              activeModel: 'custom-embed-v2',
              tokens: 8,
            },
          },
        });

      const provider = httpProvider({
        baseURL: 'https://api.custom-ai.internal',
        model: 'custom-embed',
        headers: {
          'x-custom-header': 'custom-val',
        },
        buildRequest: ({ texts, inputType }) => ({
          url: '/custom/v2/vectorize',
          body: {
            sentences: texts,
            type: inputType === 'document' ? 'doc' : 'query',
          },
        }),
        parseResponse: (body) => {
          const res = body as {
            response: {
              items: Array<{ vector: number[] }>;
              meta: { activeModel: string; tokens: number };
            };
          };
          return {
            vectors: res.response.items.map((i) => i.vector),
            model: res.response.meta.activeModel,
            usage: {
              promptTokens: res.response.meta.tokens,
              totalTokens: res.response.meta.tokens,
            },
          };
        },
      });

      const output = await provider.embed(['hello custom'], 'document');
      expect(output).toEqual({
        vectors: [[0.77, 0.88]],
        model: 'custom-embed-v2',
        usage: { promptTokens: 8, totalTokens: 8 },
      });
    });

    it('sends expected JSON body shape and auth headers through fetch', async () => {
      let sentUrl = '';
      let sentBody: unknown;
      let sentAuth = '';

      const provider = httpProvider({
        baseURL: 'https://api.openai.com/v1',
        model: 'text-embedding-3-small',
        apiKey: 'sk-secret-token',
        dimensions: 1536,
        fetch: async (url, init) => {
          sentUrl = url;
          const headers = new Headers(init.headers);
          sentAuth = headers.get('authorization') ?? '';
          sentBody = JSON.parse(init.body as string);
          return new Response(
            JSON.stringify({
              data: [{ index: 0, embedding: [0.1, 0.2] }],
              model: 'text-embedding-3-small',
            }),
            { status: 200, headers: { 'content-type': 'application/json' } },
          );
        },
      });

      await provider.embed(['test string'], 'document');
      expect(sentUrl).toBe('https://api.openai.com/v1/embeddings');
      expect(sentAuth).toBe('Bearer sk-secret-token');
      expect(sentBody).toEqual({
        model: 'text-embedding-3-small',
        input: ['test string'],
        dimensions: 1536,
      });
    });
  });
});
