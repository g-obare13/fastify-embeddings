/**
 * `createEmbedder`: the Fastify-free core. Validates input, applies hooks,
 * splits work into ordered batches with bounded concurrency, checks vector
 * shapes and dimensions, normalizes, and returns one consistent result shape.
 * The Fastify plugin is a thin wrapper around this.
 */
import { EmbeddingError } from './errors.js';
import { batch } from '#src/helpers/batch.js';
import { createDimensionGuard } from '#src/helpers/dimensions.js';
import { mapWithConcurrency } from '#src/helpers/pool.js';
import { cosineSimilarity, normalize } from '#src/helpers/vector.js';
import { embedderOptionsSchema, resolveProvider, validateOptions } from './options.js';
import type { EmbedderOptions } from './options.js';
import type { EmbedOptions, EmbedResult, HealthStatus, InputType, ProviderOutput, Usage, Vector } from './types.js';

/** A ready-to-use embedder. Returned by `createEmbedder` and exposed as `app.embeddings`. */
export interface Embedder {
  /**
   * Embed one string or a list of strings. The result shape is the same for
   * both: `vectors` always has one entry per input, in input order.
   */
  embed(input: string | readonly string[], options?: EmbedOptions): Promise<EmbedResult>;
  /** Configured model id. */
  readonly model: string;
  /** Output vector size, once known (configured or after the first call). */
  readonly dimensions: number | undefined;
  /** Provider name, for example `'http'`, `'local'`, `'mock'`, or your provider's name. */
  readonly provider: string;
  /** Embed a short probe string and report whether the provider works. Never throws. */
  healthCheck(options?: { signal?: AbortSignal }): Promise<HealthStatus>;
  /** Cosine similarity between two vectors. Throws on length mismatch. */
  cosineSimilarity(a: readonly number[], b: readonly number[]): number;
  /** Release provider resources. Safe to call more than once. */
  close(): Promise<void>;
}

/** Output of one provider batch after validation. */
interface BatchOutput {
  vectors: Vector[];
  model: string;
  usage: Usage | undefined;
}

/** Turn `embed` input into a validated list of strings. */
function toTexts(input: unknown): string[] {
  if (typeof input === 'string') {
    if (input.trim() === '') throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'input must not be empty');
    return [input];
  }
  if (!Array.isArray(input)) {
    throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'input must be a string or an array of strings');
  }
  if (input.length === 0) throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'input must contain at least one string');
  input.forEach((text: unknown, i) => {
    if (typeof text !== 'string') throw new EmbeddingError('EMBEDDING_INPUT_INVALID', `input[${String(i)}] must be a string`);
    if (text.trim() === '') throw new EmbeddingError('EMBEDDING_INPUT_INVALID', `input[${String(i)}] must not be empty`);
  });
  return [...(input as string[])];
}

/** Validate the input type option. */
function toInputType(value: unknown): InputType {
  if (value === undefined) return 'document';
  if (value === 'query' || value === 'document') return value;
  throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'inputType must be "query" or "document"');
}

/** Sum usage across batches. Returns `undefined` when no batch reported usage. */
function sumUsage(outputs: readonly BatchOutput[]): Usage | undefined {
  const reported = outputs.flatMap((output) => (output.usage === undefined ? [] : [output.usage]));
  if (reported.length === 0) return undefined;
  return reported.reduce((total, usage) => ({
    promptTokens: total.promptTokens + usage.promptTokens,
    totalTokens: total.totalTokens + usage.totalTokens,
  }));
}

/** Check the required provider result shape before inspecting vector contents. */
function isProviderOutput(value: unknown): value is ProviderOutput {
  if (typeof value !== 'object' || value === null || Array.isArray(value) || !('vectors' in value) || !Array.isArray(value.vectors)) {
    return false;
  }
  if ('model' in value && value.model !== undefined && typeof value.model !== 'string') return false;
  if ('usage' in value && value.usage !== undefined) {
    const usage = value.usage;
    if (
      typeof usage !== 'object' ||
      usage === null ||
      Array.isArray(usage) ||
      !('promptTokens' in usage) ||
      typeof usage.promptTokens !== 'number' ||
      !('totalTokens' in usage) ||
      typeof usage.totalTokens !== 'number'
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Create an embedder without Fastify, for scripts, workers, and tests. Takes
 * the same options as the plugin (minus the Fastify-only ones).
 *
 * @example
 * ```ts
 * import { createEmbedder } from 'fastify-embeddings';
 *
 * const embedder = createEmbedder({ baseURL: 'http://localhost:11434', format: 'ollama', preset: 'nomic' });
 * const { vectors, model, dimensions } = await embedder.embed(['first doc', 'second doc']);
 * const query = await embedder.embed('what is first?', { inputType: 'query' });
 * await embedder.close();
 * ```
 */
export function createEmbedder(options: EmbedderOptions = {}): Embedder {
  validateOptions(embedderOptionsSchema, options);
  const provider = resolveProvider(options);
  const batchSize = options.batchSize ?? 32;
  const concurrency = options.concurrency ?? 2;
  const shouldNormalize = options.normalize ?? true;
  const { transformInput, transformOutput, onBatch, onError, logger } = options;
  const providerGuard = createDimensionGuard(provider.dimensions);
  const outputGuard = createDimensionGuard();
  let closing: Promise<void> | undefined;

  const runBatch = async (texts: string[], inputType: InputType, signal?: AbortSignal): Promise<BatchOutput> => {
    const started = performance.now();
    let output: unknown;
    try {
      output = await provider.embed(texts, inputType, signal);
    } catch (error) {
      if (error instanceof EmbeddingError || signal?.aborted) throw error;
      throw new EmbeddingError('EMBEDDING_PROVIDER_ERROR', `Provider "${provider.name}" failed`, { cause: error });
    }
    const durationMs = performance.now() - started;
    if (!isProviderOutput(output)) {
      throw new EmbeddingError(
        'EMBEDDING_PROVIDER_ERROR',
        `Provider "${provider.name}" returned an invalid result. Providers must return an object like { vectors: number[][] } holding one vector per input, with optional { usage, model }; returning a bare array is not supported.`,
      );
    }
    const vectors = output.vectors;
    if (vectors.length !== texts.length) {
      throw new EmbeddingError(
        'EMBEDDING_PROVIDER_ERROR',
        `Provider "${provider.name}" returned ${String(vectors.length)} vectors for ${String(texts.length)} inputs`,
      );
    }
    providerGuard.check(vectors);
    const model = typeof output.model === 'string' && output.model !== '' ? output.model : provider.model;
    onBatch?.({ count: texts.length, durationMs, model, inputType });
    return { vectors, model, usage: output.usage };
  };

  const embed = async (input: string | readonly string[], embedOptions: EmbedOptions = {}): Promise<EmbedResult> => {
    const started = performance.now();
    const { signal } = embedOptions;
    let count = 0;
    try {
      const inputType = toInputType(embedOptions.inputType);
      const texts = toTexts(input);
      count = texts.length;

      const prepared = transformInput === undefined ? texts : await transformInput([...texts], inputType);
      if (!Array.isArray(prepared) || prepared.length !== count || prepared.some((text) => typeof text !== 'string')) {
        throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'transformInput must return one string per input');
      }

      const batches = batch(prepared, batchSize);
      const outputs = await mapWithConcurrency(batches, concurrency, (texts) => runBatch(texts, inputType, signal));
      const model = outputs[0]?.model ?? provider.model;
      let vectors = outputs.flatMap((output) => output.vectors);
      if (shouldNormalize) vectors = vectors.map((vector) => normalize(vector));

      if (transformOutput !== undefined) {
        vectors = await transformOutput(vectors, { model, inputType });
        if (!Array.isArray(vectors) || vectors.length !== count) {
          throw new EmbeddingError('EMBEDDING_PROVIDER_ERROR', 'transformOutput must return one vector per input');
        }
      }
      const dimensions = outputGuard.check(vectors);
      const usage = sumUsage(outputs);
      const durationMs = Math.round(performance.now() - started);
      logger?.debug({ count, batches: batches.length, model, dimensions, durationMs }, 'embeddings generated');
      return { vectors, model, dimensions, ...(usage === undefined ? {} : { usage }) };
    } catch (error) {
      if (!(error instanceof EmbeddingError)) throw error;
      const durationMs = Math.round(performance.now() - started);
      logger?.warn({ code: error.code, count, model: provider.model, durationMs }, 'embedding failed');
      onError?.(error);
      throw error;
    }
  };

  return {
    embed,
    model: provider.model,
    get dimensions() {
      return outputGuard.value ?? (transformOutput === undefined ? providerGuard.value : undefined);
    },
    provider: provider.name,
    async healthCheck(healthOptions = {}) {
      const started = performance.now();
      try {
        const result = await embed('health check', {
          inputType: 'query',
          ...(healthOptions.signal === undefined ? {} : { signal: healthOptions.signal }),
        });
        return { ok: true, model: result.model, dimensions: result.dimensions, durationMs: Math.round(performance.now() - started) };
      } catch (error) {
        const failure =
          error instanceof EmbeddingError
            ? error
            : new EmbeddingError('EMBEDDING_PROVIDER_ERROR', 'Health check failed', { cause: error });
        return { ok: false, error: failure, durationMs: Math.round(performance.now() - started) };
      }
    },
    cosineSimilarity,
    close() {
      closing ??= (async () => {
        await provider.close?.();
      })();
      return closing;
    },
  };
}
