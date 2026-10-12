import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'http://localhost:11434',
  format: 'ollama',
  preset: 'nomic',
});
const result = await app.embeddings.embed('What is vector search?', { inputType: 'query' });
void result;
await app.close();
