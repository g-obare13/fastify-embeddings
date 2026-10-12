import { createEmbedder, EmbeddingError, mockProvider } from 'fastify-embeddings';

const embedder = createEmbedder({ provider: mockProvider() });
try {
  await embedder.embed('');
} catch (error) {
  if (error instanceof EmbeddingError && error.code === 'EMBEDDING_INPUT_INVALID') {
    process.stderr.write(error.message);
  } else {
    throw error;
  }
}
await embedder.close();
