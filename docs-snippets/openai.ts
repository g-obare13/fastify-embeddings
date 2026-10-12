import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'https://api.openai.com/v1',
  model: 'text-embedding-3-small',
  apiKey: process.env.OPENAI_API_KEY,
  dimensions: 1536,
});
await app.close();
