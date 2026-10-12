import { cosineSimilarity, normalize, truncate, withRetry } from 'fastify-embeddings';

const similarity = cosineSimilarity(normalize([3, 4]), normalize([3, 4]));
const shortened = truncate([0.1, 0.2, 0.3, 0.4], 2);
const data = await withRetry(async () => ({ ok: true }), {
  retries: 3,
  baseDelayMs: 100,
  maxDelayMs: 2_000,
  shouldRetry: (_error, attempt) => attempt < 2,
});
void similarity;
void shortened;
void data;
