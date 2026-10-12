import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings);
const result = await app.embeddings.embed(['Hello vector search']);
app.log.info(
  { model: result.model, dimensions: result.dimensions, count: result.vectors.length },
  'generated embeddings',
);
await app.close();
