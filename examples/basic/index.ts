/**
 * Basic example: registers fastify-embeddings on a Fastify server.
 * Run with: `pnpm example:basic`
 */
import Fastify from 'fastify';
import { fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

async function main(): Promise<void> {
  const app = Fastify({ logger: true });

  // If EMBEDDINGS_BASE_URL is not set in environment, use mock provider for offline demonstration
  const baseURL = process.env.EMBEDDINGS_BASE_URL;

  await app.register(fastifyEmbeddings, {
    ...(baseURL
      ? {
          baseURL,
          model: process.env.EMBEDDINGS_MODEL ?? 'text-embedding-3-small',
        }
      : {
          provider: mockProvider({ dimensions: 8 }),
        }),
    route: true,
  });

  await app.ready();

  app.log.info(
    { provider: app.embeddings.provider, model: app.embeddings.model },
    'Embeddings initialized',
  );

  // Generate embeddings for documents
  const documents = ['Fastify is a fast and low overhead web framework', 'Vectors enable semantic search'];
  const docResult = await app.embeddings.embed(documents, { inputType: 'document' });

  app.log.info(
    {
      dimensions: docResult.dimensions,
      vectorCount: docResult.vectors.length,
      usage: docResult.usage,
    },
    'Documents embedded successfully',
  );

  // Generate embedding for a search query
  const queryResult = await app.embeddings.embed('low overhead framework', { inputType: 'query' });
  const similarity = app.embeddings.cosineSimilarity(
    queryResult.vectors[0]!,
    docResult.vectors[0]!,
  );

  app.log.info({ similarity }, 'Cosine similarity between query and first document');

  // Exercise the optional POST /embeddings route
  const routeResponse = await app.inject({
    method: 'POST',
    url: '/embeddings',
    payload: {
      input: 'hello embeddings route',
      inputType: 'query',
    },
  });

  app.log.info({ statusCode: routeResponse.statusCode }, 'Route injection test completed');

  await app.close();
}

main().catch((err: unknown) => {
  process.stderr.write(String(err) + '\n');
  process.exit(1);
});
