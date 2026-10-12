import type { EmbedResult } from 'fastify-embeddings';

const result: EmbedResult = {
  vectors: [[0.1, 0.2, 0.3]],
  model: 'example-model',
  dimensions: 3,
  usage: { promptTokens: 4, totalTokens: 4 },
};
void result;
