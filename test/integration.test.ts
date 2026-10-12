import { describe, expect, it } from 'vitest';
import { createEmbedder } from 'fastify-embeddings';

const baseURL = process.env.EMBEDDINGS_BASE_URL;

describe('real server integration test', () => {
  it.skipIf(!baseURL)(
    'connects to real embedding server and verifies semantic similarity ranking',
    async () => {
      const embedder = createEmbedder({
        baseURL,
        model: process.env.EMBEDDINGS_MODEL ?? 'text-embedding-3-small',
        format: (process.env.EMBEDDINGS_FORMAT as 'openai' | 'ollama') ?? 'openai',
        apiKey: process.env.EMBEDDINGS_API_KEY,
      });

      const ref = await embedder.embed('The canine barked loudly in the yard.', {
        inputType: 'document',
      });
      const similar = await embedder.embed('A dog was making loud barking sounds outside.', {
        inputType: 'query',
      });
      const unrelated = await embedder.embed('Quantum computing utilizes superposition and entanglement.', {
        inputType: 'query',
      });

      const simScore = embedder.cosineSimilarity(ref.vectors[0]!, similar.vectors[0]!);
      const unrelScore = embedder.cosineSimilarity(ref.vectors[0]!, unrelated.vectors[0]!);

      expect(simScore).toBeGreaterThan(unrelScore);
      await embedder.close();
    },
  );
});
