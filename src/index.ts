/**
 * fastify-embeddings: Fastify 5 plugin and standalone embedder that turns text
 * into vectors for vector search.
 */

// Plugin
export { fastifyEmbeddings, fastifyEmbeddings as embeddings, default } from './plugin.js';
export type { EmbeddingsDecorator } from './plugin.js';

// Core embedder
export { createEmbedder } from './embedder.js';
export type { Embedder } from './embedder.js';

// Errors
export { EmbeddingError, isEmbeddingError } from './errors.js';
export type { EmbeddingErrorCode, EmbeddingErrorOptions } from './errors.js';

// Public types
export type {
  BatchInfo,
  EmbedderHooks,
  EmbeddingsLogger,
  EmbedOptions,
  EmbedResult,
  EmbeddingProvider,
  HealthStatus,
  InputType,
  ProviderOutput,
  RouteOptions,
  TransformOutputContext,
  Usage,
  Vector,
} from './types.js';

// Presets
export { presets, applyPrefixes } from './presets.js';
export type { Prefixes, Preset, PresetName } from './presets.js';

// Standalone building blocks
export { batch } from '#src/helpers/batch.js';
export { withRetry } from '#src/helpers/retry.js';
export type { RetryOptions } from '#src/helpers/retry.js';
export { cosineSimilarity, normalize, truncate } from '#src/helpers/vector.js';

// Providers and helpers
export { defineProvider } from '#src/providers/define.js';
export { httpProvider } from '#src/providers/http.js';
export type { FetchLike, HttpProviderOptions } from '#src/providers/http.js';
export { localProvider, DEFAULT_LOCAL_MODEL } from '#src/providers/local.js';
export type { FeatureExtractor, LocalProviderOptions, TransformersModule } from '#src/providers/local.js';
export { mockProvider, mockVector } from '#src/providers/mock.js';
export type { MockCall, MockProvider, MockProviderOptions } from '#src/providers/mock.js';
export { formats } from '#src/providers/formats.js';
export type {
  BuildRequest,
  HttpFormat,
  HttpFormatDefinition,
  ParseResponse,
  RequestContext,
  RequestSpec,
} from '#src/providers/formats.js';

// Options and validation
export {
  embedderOptionsSchema,
  pluginOptionsSchema,
  resolveProvider,
  validateOptions,
} from './options.js';
export type { BuiltInProvider, EmbedderOptions, FastifyEmbeddingsOptions } from './options.js';

// Route helper
export { registerRoute } from './route.js';
