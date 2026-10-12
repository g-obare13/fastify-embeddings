/**
 * Test utilities and mock provider for `fastify-embeddings`.
 * Deterministic hash-based vectors without network access or model downloads.
 */
export { mockProvider, mockVector } from '#src/providers/mock.js';
export type { MockCall, MockProvider, MockProviderOptions } from '#src/providers/mock.js';
