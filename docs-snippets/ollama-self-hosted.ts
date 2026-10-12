import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  preset: 'nomic',
  batchSize: 8,
  concurrency: 1,
  timeoutMs: 120_000,
});
const result = await app.embeddings.embed('What is vector search?', { inputType: 'query' });
app.log.info({ model: result.model, dimensions: result.dimensions }, 'embedded with self-hosted Ollama');
await app.close();
