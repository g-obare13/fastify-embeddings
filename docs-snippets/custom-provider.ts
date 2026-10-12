import Fastify from 'fastify';
import { defineProvider, fastifyEmbeddings } from 'fastify-embeddings';

const customProvider = defineProvider({
  name: 'my-custom-engine',
  model: 'custom-model',
  dimensions: 256,
  async embed(texts, inputType, signal) {
    const response = await fetch('https://custom-service/embed', {
      method: 'POST',
      body: JSON.stringify({ texts, inputType }),
      signal,
    });
    const json = (await response.json()) as { embeddings: number[][] };
    return { vectors: json.embeddings };
  },
  async close() {
    // Release connections or resources.
  },
});

const app = Fastify();
await app.register(fastifyEmbeddings, { provider: customProvider });
await app.close();
