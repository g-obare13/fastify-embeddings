import Fastify from 'fastify';
import { fastifyEmbeddings, normalize, truncate } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'https://api.openai.com/v1',
  model: 'text-embedding-3-small',
  apiKey: process.env.OPENAI_API_KEY,
  transformOutput: (vectors) => vectors.map((vector) => normalize(truncate(vector, 256))),
});
const result = await app.embeddings.embed('A Matryoshka embedding example');
app.log.info({ dimensions: result.dimensions }, 'transformed dimensions');
await app.close();
