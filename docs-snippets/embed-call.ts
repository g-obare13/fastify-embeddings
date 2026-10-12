import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'http://localhost:11434',
  format: 'ollama',
  preset: 'nomic',
});

const { vectors, model, dimensions, usage } = await app.embeddings.embed(['some text', 'more text']);
app.log.info(
  { model, dimensions, vectors: vectors.length, tokens: usage?.totalTokens },
  'embedded two texts in one batch',
);
await app.close();
