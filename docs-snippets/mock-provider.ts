import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';
import { mockProvider } from 'fastify-embeddings/testing';

const app = Fastify();
const mock = mockProvider({ dimensions: 8 });
await app.register(fastifyEmbeddings, { provider: mock });
await app.ready();
await app.embeddings.embed(['Document A', 'Document B']);
process.stdout.write(JSON.stringify(mock.calls));
await app.close();
