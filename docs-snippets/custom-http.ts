import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'https://api.mycompany.internal',
  model: 'enterprise-v1',
  buildRequest: ({ texts, inputType }) => ({
    url: '/vectors/generate',
    method: 'POST',
    body: { passages: texts, mode: inputType === 'query' ? 'search' : 'index' },
  }),
  parseResponse: (body) => {
    const data = body as { items: Array<{ embedding: number[] }> };
    return { vectors: data.items.map((item) => item.embedding) };
  },
});
await app.close();
