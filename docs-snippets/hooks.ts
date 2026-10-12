import { createEmbedder, mockProvider, normalize, truncate } from 'fastify-embeddings';

const embedder = createEmbedder({
  provider: mockProvider({ dimensions: 8 }),
  transformInput: (texts, inputType) =>
    texts.map((text) => `${inputType === 'query' ? 'search: ' : 'document: '}${text}`),
  transformOutput: (vectors) => vectors.map((vector) => normalize(truncate(vector, 4))),
  onBatch: ({ count, durationMs, model, inputType }) => {
    process.stdout.write(`${model}: ${String(count)} ${inputType}s in ${String(durationMs)}ms\n`);
  },
  onError: (error) => {
    process.stderr.write(`${error.code}: ${error.message}\n`);
  },
});

await embedder.embed(['search this']);
await embedder.close();
