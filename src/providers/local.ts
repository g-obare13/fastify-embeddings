/**
 * The local provider: runs a sentence-transformer in process through the
 * optional peer dependency `@huggingface/transformers`. The model is loaded
 * lazily on the first call, so registering the plugin stays fast.
 */
import { EmbeddingError } from '#src/errors.js';
import { applyPrefixes } from '#src/presets.js';
import type { Prefixes, Preset } from '#src/presets.js';
import type { EmbeddingProvider, Vector } from '#src/types.js';

/** Default local model: small, fast, 384 dimensions. */
export const DEFAULT_LOCAL_MODEL = 'Xenova/all-MiniLM-L6-v2';

/** The callable returned by `pipeline('feature-extraction', ...)`. */
export interface FeatureExtractor {
  /** Run the model with pooling. */
  (texts: string[], options: { pooling: 'mean'; normalize: boolean }): Promise<{ tolist(): unknown }>;
  /** Free model memory. */
  dispose?(): Promise<unknown>;
}

/** The part of `@huggingface/transformers` this provider uses. */
export interface TransformersModule {
  /** Create a pipeline for a task and model. */
  pipeline(task: 'feature-extraction', model: string, options?: Record<string, unknown>): Promise<FeatureExtractor>;
}

/** Options for `localProvider`. */
export interface LocalProviderOptions {
  /** Hugging Face model id. Defaults to `preset.model`, then `Xenova/all-MiniLM-L6-v2`. */
  model?: string;
  /** Expected vector size. Defaults to 384 for the default model. */
  dimensions?: number;
  /** Query and document prefixes. Merged over `preset.prefixes`. */
  prefixes?: Prefixes;
  /** Model family defaults. */
  preset?: Preset;
  /** Extra options passed to `pipeline()`, for example `{ dtype: 'q8' }`. */
  pipelineOptions?: Record<string, unknown>;
  /** Override how the transformers module is loaded (custom builds, tests). */
  load?: () => Promise<TransformersModule>;
}

/** True when a transformer result is an array of numeric vectors. */
function isVectorList(value: unknown): value is Vector[] {
  return (
    Array.isArray(value) &&
    value.every(
      (vector: unknown) =>
        Array.isArray(vector) && vector.every((item: unknown) => typeof item === 'number'),
    )
  );
}

/** Import the optional peer dependency without a static reference. */
async function loadTransformers(): Promise<TransformersModule> {
  const specifier = '@huggingface/transformers';
  const loaded: unknown = await import(/* @vite-ignore */ specifier);
  return loaded as TransformersModule;
}

/**
 * Create the in-process local provider.
 *
 * @example
 * ```ts
 * // pnpm add @huggingface/transformers
 * const embedder = createEmbedder({ provider: localProvider() });
 * const { vectors } = await embedder.embed('hello world'); // 384 dimensions
 * ```
 */
export function localProvider(options: LocalProviderOptions = {}): EmbeddingProvider {
  const model = options.model ?? options.preset?.model ?? DEFAULT_LOCAL_MODEL;
  const dimensions = options.dimensions ?? (model === DEFAULT_LOCAL_MODEL ? 384 : undefined);
  const prefixes: Prefixes = { ...options.preset?.prefixes, ...options.prefixes };
  const load = options.load ?? loadTransformers;
  let extractor: Promise<FeatureExtractor> | undefined;

  const getExtractor = (): Promise<FeatureExtractor> => {
    extractor ??= (async () => {
      let transformers: TransformersModule;
      try {
        transformers = await load();
      } catch (error) {
        throw new EmbeddingError(
          'EMBEDDING_PROVIDER_UNREACHABLE',
          'The local provider needs the optional peer dependency @huggingface/transformers. Install it or set EMBEDDINGS_BASE_URL.',
          { cause: error, retryable: false },
        );
      }
      try {
        return await transformers.pipeline('feature-extraction', model, options.pipelineOptions);
      } catch (error) {
        throw new EmbeddingError('EMBEDDING_PROVIDER_UNREACHABLE', `Could not load local model ${model}`, {
          cause: error,
          retryable: false,
        });
      }
    })();
    extractor.catch(() => {
      extractor = undefined;
    });
    return extractor;
  };

  return {
    name: 'local',
    model,
    ...(dimensions === undefined ? {} : { dimensions }),
    async embed(texts, inputType) {
      const run = await getExtractor();
      const output = await run(applyPrefixes(texts, inputType, prefixes), { pooling: 'mean', normalize: true });
      const vectors = output.tolist();
      if (!isVectorList(vectors)) {
        throw new EmbeddingError('EMBEDDING_PROVIDER_ERROR', 'Local model returned malformed vectors');
      }
      return { vectors };
    },
    async close() {
      const current = extractor;
      extractor = undefined;
      if (current === undefined) return;
      const loaded = await current.catch(() => undefined);
      await loaded?.dispose?.();
    },
  };
}
