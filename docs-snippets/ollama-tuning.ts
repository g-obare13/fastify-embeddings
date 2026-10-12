import { createEmbedder } from 'fastify-embeddings';

const embedder = createEmbedder({
  baseURL: process.env.EMBEDDINGS_BASE_URL,
  format: 'ollama',
  model: process.env.EMBEDDINGS_MODEL,
  preset: 'nomic',
  batchSize: 8,
  concurrency: 1,
  timeoutMs: 120_000,
});

const health = await embedder.healthCheck();
if (health.ok) {
  process.stdout.write(`healthy: ${String(health.dimensions)} dimensions\n`);
}
await embedder.close();
