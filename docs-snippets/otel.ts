import Fastify from 'fastify';
import { fastifyEmbeddings, httpProvider, presets } from 'fastify-embeddings';
import { withTelemetry } from 'fastify-embeddings/otel';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  provider: withTelemetry(
    httpProvider({ baseURL: 'http://localhost:11434', format: 'ollama', preset: presets.nomic }),
  ),
  batchSize: 8,
  concurrency: 1,
});

const result = await app.embeddings.embed(['one document', 'another document']);
app.log.info({ vectors: result.vectors.length }, 'each provider batch produced one span');
await app.close();
