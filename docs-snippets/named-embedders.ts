import Fastify from 'fastify';
import { fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, { provider: mockProvider(), name: 'default' });
await app.register(fastifyEmbeddings, {
  provider: mockProvider({ model: 'documentation-model' }),
  name: 'docs',
});
await app.ready();
const docs = app.embeddings.get('docs');
if (docs !== undefined) await docs.embed('A page to index');
await app.close();
