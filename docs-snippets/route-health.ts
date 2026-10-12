import Fastify from 'fastify';
import { fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  provider: mockProvider({ dimensions: 8 }),
  route: { path: '/api/embeddings', bodyLimit: 16_384 },
});
await app.ready();
const health = await app.embeddings.healthCheck();
if (!health.ok) throw health.error;
const response = await app.inject({
  method: 'POST',
  url: '/api/embeddings',
  payload: { input: 'example text', inputType: 'query' },
});
void response;
await app.close();
