/**
 * Options for `createEmbedder`: their types, Zod validation with messages that
 * name the bad option, and resolution of the provider from options, env vars,
 * and presets (explicit option, then env var, then preset default).
 */
import { z } from 'zod';
import { presets } from './presets.js';
import type { Prefixes, Preset, PresetName } from './presets.js';
import type { BuildRequest, HttpFormat, ParseResponse } from '#src/providers/formats.js';
import { httpProvider } from '#src/providers/http.js';
import type { FetchLike } from '#src/providers/http.js';
import { localProvider } from '#src/providers/local.js';
import { mockProvider } from '#src/providers/mock.js';
import type { EmbedderHooks, EmbeddingProvider, EmbeddingsLogger, RouteOptions } from './types.js';

/** Names of the built-in providers. */
export type BuiltInProvider = 'http' | 'local' | 'mock';

/** Options accepted by `createEmbedder` (and by the plugin). */
export interface EmbedderOptions extends EmbedderHooks {
  /**
   * A built-in provider name or your own provider object. Defaults to `'http'`
   * when a base URL is set (option or `EMBEDDINGS_BASE_URL`), else `'local'`.
   */
  provider?: BuiltInProvider | EmbeddingProvider;
  /** API root for the http provider. Env: `EMBEDDINGS_BASE_URL`. */
  baseURL?: string;
  /** Model id. Env: `EMBEDDINGS_MODEL`. */
  model?: string;
  /** Wire format for the http provider. Env: `EMBEDDINGS_FORMAT`. Defaults to `'openai'`. */
  format?: HttpFormat;
  /** Bearer token for the http provider. Env: `EMBEDDINGS_API_KEY`. Never logged. */
  apiKey?: string;
  /** Extra HTTP headers. */
  headers?: Record<string, string>;
  /** Expected vector size from the provider. */
  dimensions?: number;
  /** Per-attempt HTTP timeout in milliseconds. Defaults to 30000. */
  timeoutMs?: number;
  /** HTTP retries after the first attempt. Defaults to 2. */
  retries?: number;
  /** Base delay for HTTP retry backoff, in milliseconds. Defaults to 250. */
  retryBaseDelayMs?: number;
  /** Query and document prefixes for the built-in http and local providers. */
  prefixes?: Prefixes;
  /** A preset name or object with model and prefix defaults. */
  preset?: PresetName | Preset;
  /** Replace the http provider's request builder. */
  buildRequest?: BuildRequest;
  /** Replace the http provider's response parser. */
  parseResponse?: ParseResponse;
  /** Custom `fetch` for the http provider. */
  fetch?: FetchLike;
  /** Texts per provider call. Defaults to 32. */
  batchSize?: number;
  /** Provider calls in flight at once. Defaults to 2. */
  concurrency?: number;
  /** L2-normalize vectors. Defaults to `true`. */
  normalize?: boolean;
  /** Logger for counts, model, and duration. Silent by default; the plugin passes `app.log`. */
  logger?: EmbeddingsLogger;
  /** Where to read `EMBEDDINGS_*` variables. Defaults to `process.env`. Pass `{}` to ignore env. */
  env?: Record<string, string | undefined>;
}

/** Options accepted by the Fastify plugin. */
export interface FastifyEmbeddingsOptions extends EmbedderOptions {
  /** Optional name to register as a named embedder accessible via `app.embeddings.get(name)`. */
  name?: string;
  /** Whether to mount the POST /embeddings route (or route configuration object). Off by default. */
  route?: boolean | RouteOptions;
  /** Whether to run a health check at startup and fail if unreachable. Defaults to false. */
  eager?: boolean;
}

const BUILT_IN_PROVIDERS: readonly string[] = ['http', 'local', 'mock'];

/** True for a value shaped like an `EmbeddingProvider`. */
function isProviderObject(value: unknown): value is EmbeddingProvider {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.name === 'string' &&
    v.name !== '' &&
    typeof v.model === 'string' &&
    typeof v.embed === 'function' &&
    (v.close === undefined || typeof v.close === 'function') &&
    (v.dimensions === undefined || (typeof v.dimensions === 'number' && Number.isInteger(v.dimensions) && v.dimensions > 0))
  );
}

const fn = z.custom<(...args: never[]) => unknown>((value) => typeof value === 'function', { error: 'Expected a function' });
const positiveInt = z.number().int().positive();
const prefixesSchema = z.strictObject({ query: z.string().optional(), document: z.string().optional() });

/** Zod schema for `EmbedderOptions`. Used for validation only; the original object is kept. */
export const embedderOptionsSchema = z.object({
  provider: z
    .custom((value) => (typeof value === 'string' ? BUILT_IN_PROVIDERS.includes(value) : isProviderObject(value)), {
      error: "Expected 'http', 'local', 'mock', or a provider object with name, model, and embed",
    })
    .optional(),
  baseURL: z.url({ protocol: /^https?$/, error: 'Expected an http or https URL' }).optional(),
  model: z.string().min(1).optional(),
  format: z.enum(['openai', 'ollama']).optional(),
  apiKey: z.string().optional(),
  headers: z.record(z.string(), z.string()).optional(),
  dimensions: positiveInt.optional(),
  timeoutMs: positiveInt.optional(),
  retries: z.number().int().min(0).max(10).optional(),
  retryBaseDelayMs: z.number().int().min(0).optional(),
  prefixes: prefixesSchema.optional(),
  preset: z
    .union([z.enum(['nomic', 'e5', 'bge', 'minilm']), z.strictObject({ model: z.string().min(1).optional(), prefixes: prefixesSchema.optional() })], {
      error: 'Expected nomic, e5, bge, minilm, or a preset object with model and prefixes',
    })
    .optional(),
  buildRequest: fn.optional(),
  parseResponse: fn.optional(),
  fetch: fn.optional(),
  transformInput: fn.optional(),
  transformOutput: fn.optional(),
  onBatch: fn.optional(),
  onError: fn.optional(),
  batchSize: positiveInt.max(2048).optional(),
  concurrency: positiveInt.max(64).optional(),
  normalize: z.boolean().optional(),
  logger: z
    .custom(
      (value) =>
        typeof value === 'object' &&
        value !== null &&
        typeof (value as Record<string, unknown>).debug === 'function' &&
        typeof (value as Record<string, unknown>).warn === 'function',
      { error: 'Expected a logger with debug and warn methods' },
    )
    .optional(),
  env: z.custom((value) => typeof value === 'object' && value !== null, { error: 'Expected an object' }).optional(),
});

/** Zod schema for Fastify plugin options. */
export const pluginOptionsSchema = embedderOptionsSchema.extend({
  name: z.string().min(1).optional(),
  route: z
    .union([
      z.boolean(),
      z.strictObject({
        path: z.string().min(1).optional(),
        bodyLimit: positiveInt.optional(),
      }),
    ])
    .optional(),
  eager: z.boolean().optional(),
});

/**
 * Validate `value` against `schema` and throw a `TypeError` that names the
 * first bad option.
 *
 * @example
 * ```ts
 * validateOptions(embedderOptionsSchema, { batchSize: 0 });
 * // TypeError: fastify-embeddings: invalid option "batchSize": Too small: expected number to be >0
 * ```
 */
export function validateOptions(schema: z.ZodType, value: unknown, label = 'option'): void {
  const result = schema.safeParse(value ?? {});
  if (result.success) return;
  const issue = result.error.issues[0];
  const path = issue?.path.map(String).join('.') ?? '';
  throw new TypeError(`fastify-embeddings: invalid ${label} "${path}": ${issue?.message ?? 'invalid value'}`);
}

/** Read one env var, treating empty strings as unset. */
function readEnv(env: Record<string, string | undefined>, key: string): string | undefined {
  const value = env[key]?.trim();
  return value === undefined || value === '' ? undefined : value;
}

/** Env vars read in zero config mode, validated with the same rules as options. */
const envSchema = z.object({
  EMBEDDINGS_BASE_URL: embedderOptionsSchema.shape.baseURL,
  EMBEDDINGS_MODEL: embedderOptionsSchema.shape.model,
  EMBEDDINGS_FORMAT: embedderOptionsSchema.shape.format,
  EMBEDDINGS_API_KEY: embedderOptionsSchema.shape.apiKey,
});

/**
 * Build the provider described by `options`, filling gaps from env vars and
 * the preset. A provider object passed as `options.provider` is used as is.
 *
 * @example
 * ```ts
 * const provider = resolveProvider({ baseURL: 'http://localhost:11434', format: 'ollama', preset: 'nomic' });
 * provider.name; // 'http'
 * ```
 */
export function resolveProvider(options: EmbedderOptions): EmbeddingProvider {
  if (typeof options.provider === 'object') return options.provider;

  const env = options.env ?? process.env;
  const fromEnv = {
    EMBEDDINGS_BASE_URL: readEnv(env, 'EMBEDDINGS_BASE_URL'),
    EMBEDDINGS_MODEL: readEnv(env, 'EMBEDDINGS_MODEL'),
    EMBEDDINGS_FORMAT: readEnv(env, 'EMBEDDINGS_FORMAT'),
    EMBEDDINGS_API_KEY: readEnv(env, 'EMBEDDINGS_API_KEY'),
  };
  validateOptions(envSchema, fromEnv, 'env var');

  const preset = typeof options.preset === 'string' ? presets[options.preset] : options.preset;
  const baseURL = options.baseURL ?? fromEnv.EMBEDDINGS_BASE_URL;
  const model = options.model ?? fromEnv.EMBEDDINGS_MODEL;
  const kind = options.provider ?? (baseURL === undefined ? 'local' : 'http');

  if (kind === 'mock') {
    return mockProvider({ ...(options.dimensions === undefined ? {} : { dimensions: options.dimensions }), ...(model === undefined ? {} : { model }) });
  }
  if (kind === 'local') {
    return localProvider({
      ...(model === undefined ? {} : { model }),
      ...(options.dimensions === undefined ? {} : { dimensions: options.dimensions }),
      ...(options.prefixes === undefined ? {} : { prefixes: options.prefixes }),
      ...(preset === undefined ? {} : { preset }),
    });
  }
  if (baseURL === undefined) {
    throw new TypeError('fastify-embeddings: invalid option "baseURL": required when using the http provider (or set EMBEDDINGS_BASE_URL)');
  }
  const format = options.format ?? (fromEnv.EMBEDDINGS_FORMAT as HttpFormat | undefined);
  const apiKey = options.apiKey ?? fromEnv.EMBEDDINGS_API_KEY;
  return httpProvider({
    baseURL,
    ...(model === undefined ? {} : { model }),
    ...(format === undefined ? {} : { format }),
    ...(apiKey === undefined ? {} : { apiKey }),
    ...(preset === undefined ? {} : { preset }),
    ...pick(options, ['headers', 'dimensions', 'timeoutMs', 'retries', 'retryBaseDelayMs', 'prefixes', 'buildRequest', 'parseResponse', 'fetch']),
  });
}

/** Copy the listed keys that are defined. */
function pick<T extends object, K extends keyof T>(source: T, keys: readonly K[]): Partial<Pick<T, K>> {
  const out: Partial<Pick<T, K>> = {};
  for (const key of keys) {
    if (source[key] !== undefined) out[key] = source[key];
  }
  return out;
}
