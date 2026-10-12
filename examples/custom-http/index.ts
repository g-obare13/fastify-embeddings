/**
 * Custom HTTP example: demonstrates buildRequest and parseResponse escape hatches
 * to communicate with proprietary or non-standard vector APIs.
 * Run with: `pnpm example:custom-http`
 */
import Fastify from 'fastify';
import { createEmbedder, httpProvider } from 'fastify-embeddings';

async function main(): Promise<void> {
  // 1. Spin up a temporary mock server that simulates an internal vector service
  const mockApiServer = Fastify();
  mockApiServer.post('/v3/internal/tokenize-and-embed', async (request) => {
    const body = request.body as { texts: string[]; mode: string };
    return {
      payload: {
        results: body.texts.map((_, i) => ({
          vec: [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)],
        })),
        consumed_units: 42,
      },
    };
  });

  const address = await mockApiServer.listen({ port: 0, host: '127.0.0.1' });

  // 2. Configure httpProvider with buildRequest and parseResponse hooks
  const provider = httpProvider({
    baseURL: address,
    model: 'internal-embed-v3',
    buildRequest: ({ texts, inputType }) => ({
      url: '/v3/internal/tokenize-and-embed',
      method: 'POST',
      body: {
        texts,
        mode: inputType === 'document' ? 'doc' : 'search',
      },
    }),
    parseResponse: (body) => {
      const data = body as {
        payload: {
          results: Array<{ vec: number[] }>;
          consumed_units: number;
        };
      };
      return {
        vectors: data.payload.results.map((r) => r.vec),
        usage: {
          promptTokens: data.payload.consumed_units,
          totalTokens: data.payload.consumed_units,
        },
      };
    },
  });

  const embedder = createEmbedder({ provider });
  const result = await embedder.embed(['First sentence', 'Second sentence']);

  process.stdout.write(
    `[Custom HTTP] Embedded ${String(result.vectors.length)} texts with model ${result.model}, total tokens: ${String(result.usage?.totalTokens)}\n`,
  );

  await embedder.close();
  await mockApiServer.close();
}

main().catch((err: unknown) => {
  process.stderr.write(String(err) + '\n');
  process.exit(1);
});
