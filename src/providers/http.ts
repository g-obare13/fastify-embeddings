/**
 * The http provider: talks to any embeddings API with `fetch` only (no SDKs).
 * Speaks the OpenAI and Ollama formats out of the box and accepts
 * `buildRequest` and `parseResponse` to wrap any other API. Retries transient
 * failures with exponential backoff and jitter, honoring `Retry-After`.
 */
import { EmbeddingError } from '#src/errors.js';
import { withRetry } from '#src/helpers/retry.js';
import { applyPrefixes } from '#src/presets.js';
import type { Prefixes, Preset } from '#src/presets.js';
import type { EmbeddingProvider, ProviderOutput } from '#src/types.js';
import { formats } from './formats.js';
import type { BuildRequest, HttpFormat, ParseResponse, RequestContext, RequestSpec } from './formats.js';

/** A `fetch`-compatible function. Defaults to the global `fetch`. */
export type FetchLike = (url: string, init: RequestInit) => Promise<Response>;

/** Options for `httpProvider`. */
export interface HttpProviderOptions {
  /** API root, for example `'https://api.openai.com/v1'` or `'http://localhost:11434'`. */
  baseURL: string;
  /** Model id. Falls back to `preset.model`. Required one way or the other. */
  model?: string;
  /** Wire format. Defaults to `'openai'`. Ignored for parts overridden by `buildRequest` or `parseResponse`. */
  format?: HttpFormat;
  /** Sent as `Authorization: Bearer <apiKey>`. Never logged. */
  apiKey?: string;
  /** Extra headers, merged over the defaults. */
  headers?: Record<string, string>;
  /** Expected vector size. Sent to OpenAI-format APIs as `dimensions`. */
  dimensions?: number;
  /** Per-attempt timeout in milliseconds. Defaults to 30000. */
  timeoutMs?: number;
  /** Retries after the first attempt for 408, 429, 5xx, timeouts, and network errors. Defaults to 2. */
  retries?: number;
  /** Base delay for exponential backoff, in milliseconds. Defaults to 250. */
  retryBaseDelayMs?: number;
  /** Query and document prefixes. Merged over `preset.prefixes`. */
  prefixes?: Prefixes;
  /** Model family defaults. */
  preset?: Preset;
  /** Replace the built-in request builder. */
  buildRequest?: BuildRequest;
  /** Replace the built-in response parser. */
  parseResponse?: ParseResponse;
  /** Custom `fetch` (for proxies, undici agents, or tests). */
  fetch?: FetchLike;
}

/** Join a relative path onto a base URL. Absolute URLs pass through. */
function joinUrl(baseURL: string, url: string): string {
  if (/^https?:\/\//i.test(url)) return url;
  return `${baseURL.replace(/\/+$/, '')}/${url.replace(/^\/+/, '')}`;
}

/** Parse a `Retry-After` header (seconds or HTTP date) into milliseconds. */
function parseRetryAfter(header: string | null): number | undefined {
  if (header === null || header.trim() === '') return undefined;
  const seconds = Number(header);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(header);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}

/** Map a non-2xx response to an `EmbeddingError`. */
function errorForStatus(response: Response): EmbeddingError {
  const status = response.status;
  const retryAfterMs = parseRetryAfter(response.headers.get('retry-after'));
  const details = { upstreamStatus: status, retryAfterMs };
  if (status === 429) {
    return new EmbeddingError('EMBEDDING_RATE_LIMITED', 'Embeddings provider rate limit exceeded (status 429)', details);
  }
  if (status === 408) {
    return new EmbeddingError('EMBEDDING_TIMEOUT', 'Embeddings provider timed out (status 408)', details);
  }
  return new EmbeddingError('EMBEDDING_PROVIDER_ERROR', `Embeddings provider responded with status ${String(status)}`, {
    ...details,
    retryable: status >= 500,
  });
}

/**
 * Create the http provider. Usually built for you from plugin options, but
 * handy to wrap (for example to add a cache).
 *
 * @example
 * ```ts
 * const provider = httpProvider({
 *   baseURL: 'http://localhost:11434',
 *   format: 'ollama',
 *   preset: presets.nomic,
 * });
 * const embedder = createEmbedder({ provider });
 * ```
 */
export function httpProvider(options: HttpProviderOptions): EmbeddingProvider {
  const model = options.model ?? options.preset?.model;
  if (model === undefined || model === '') {
    throw new TypeError('fastify-embeddings: invalid option "model": required when using the http provider');
  }
  const format = formats[options.format ?? 'openai'];
  const buildRequest = options.buildRequest ?? format.buildRequest;
  const parseResponse = options.parseResponse ?? format.parseResponse;
  const prefixes: Prefixes = { ...options.preset?.prefixes, ...options.prefixes };
  const timeoutMs = options.timeoutMs ?? 30_000;
  const fetchImpl: FetchLike = options.fetch ?? ((url, init) => fetch(url, init));
  const { baseURL, apiKey, dimensions } = options;
  const origin = URL.canParse(baseURL) ? new URL(baseURL).origin : 'the configured baseURL';

  const headersFor = (spec: RequestSpec): Headers => {
    const headers = new Headers({ 'content-type': 'application/json', accept: 'application/json' });
    if (apiKey !== undefined && apiKey !== '') headers.set('authorization', `Bearer ${apiKey}`);
    for (const extra of [options.headers, spec.headers]) {
      for (const [key, value] of Object.entries(extra ?? {})) headers.set(key, value);
    }
    return headers;
  };

  const send = async (spec: RequestSpec, context: RequestContext, signal?: AbortSignal): Promise<ProviderOutput> => {
    const timeout = AbortSignal.timeout(timeoutMs);
    const combined = signal === undefined ? timeout : AbortSignal.any([signal, timeout]);
    const fail = (error: unknown, fallback: EmbeddingError): never => {
      if (signal?.aborted) throw signal.reason;
      if (timeout.aborted) {
        throw new EmbeddingError('EMBEDDING_TIMEOUT', `Embeddings request timed out after ${String(timeoutMs)} ms`, { cause: error });
      }
      throw fallback;
    };

    let response: Response;
    try {
      response = await fetchImpl(joinUrl(baseURL, spec.url), {
        method: spec.method ?? 'POST',
        headers: headersFor(spec),
        signal: combined,
        ...(spec.body === undefined ? {} : { body: typeof spec.body === 'string' ? spec.body : JSON.stringify(spec.body) }),
      });
    } catch (error) {
      return fail(
        error,
        new EmbeddingError('EMBEDDING_PROVIDER_UNREACHABLE', `Could not reach embeddings provider at ${origin}`, { cause: error }),
      );
    }

    if (!response.ok) {
      await response.body?.cancel().catch(() => undefined);
      throw errorForStatus(response);
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch (error) {
      return fail(
        error,
        new EmbeddingError('EMBEDDING_PROVIDER_ERROR', 'Embeddings provider returned invalid JSON', { cause: error }),
      );
    }
    return parseResponse(body, context);
  };

  return {
    name: 'http',
    model,
    ...(dimensions === undefined ? {} : { dimensions }),
    async embed(texts, inputType, signal) {
      const context: RequestContext = {
        texts: applyPrefixes(texts, inputType, prefixes),
        inputType,
        model,
        dimensions,
        baseURL,
      };
      const spec = await buildRequest(context);
      return withRetry(() => send(spec, context, signal), {
        retries: options.retries ?? 2,
        baseDelayMs: options.retryBaseDelayMs ?? 250,
        shouldRetry: (error) => error instanceof EmbeddingError && error.retryable,
        ...(signal === undefined ? {} : { signal }),
      });
    },
  };
}
