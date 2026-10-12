/**
 * Request builders and response parsers for the built-in HTTP formats
 * (OpenAI-compatible and Ollama). Also defines the shapes used by the
 * `buildRequest` and `parseResponse` escape hatches.
 */
import { z } from 'zod';
import { EmbeddingError } from '#src/errors.js';
import type { InputType, ProviderOutput } from '#src/types.js';

/** Wire formats supported out of the box. */
export type HttpFormat = 'openai' | 'ollama';

/** Everything a request builder needs. `texts` already include prefixes. */
export interface RequestContext {
  /** Texts for this batch, with query or document prefixes applied. */
  texts: string[];
  /** Input type of the batch. */
  inputType: InputType;
  /** Configured model id. */
  model: string;
  /** Configured dimensions, if any. */
  dimensions: number | undefined;
  /** Configured base URL. */
  baseURL: string;
}

/** An HTTP request description. `body` is JSON-encoded unless it is a string. */
export interface RequestSpec {
  /** Absolute URL, or a path joined onto `baseURL`. */
  url: string;
  /** Defaults to `'POST'`. */
  method?: string;
  /** Merged over the default `content-type` and `authorization` headers. */
  headers?: Record<string, string>;
  /** Request payload. */
  body?: unknown;
}

/** Build the HTTP request for one batch. */
export type BuildRequest = (context: RequestContext) => RequestSpec | Promise<RequestSpec>;

/** Turn a parsed JSON response body into vectors (and optional usage and model). */
export type ParseResponse = (body: unknown, context: RequestContext) => ProviderOutput | Promise<ProviderOutput>;

/** A request builder paired with its response parser. */
export interface HttpFormatDefinition {
  /** Builds the request. */
  buildRequest: BuildRequest;
  /** Parses the response. */
  parseResponse: ParseResponse;
}

const openAIResponse = z.object({
  data: z.array(z.object({ embedding: z.array(z.number()), index: z.number().int().nonnegative() })),
  model: z.string().optional(),
  usage: z.object({ prompt_tokens: z.number(), total_tokens: z.number() }).optional(),
});

const ollamaResponse = z.object({
  embeddings: z.array(z.array(z.number())),
  model: z.string().optional(),
  prompt_eval_count: z.number().optional(),
});

/** Throw a provider error that names the format but never echoes the body. */
function malformed(format: string): EmbeddingError {
  return new EmbeddingError('EMBEDDING_PROVIDER_ERROR', `Malformed ${format} response from embeddings provider`);
}

/**
 * Built-in formats. Reuse them inside your own `parseResponse` when an API is
 * almost, but not quite, compatible.
 *
 * @example
 * ```ts
 * const parseResponse: ParseResponse = (body, ctx) => formats.openai.parseResponse((body as { result: unknown }).result, ctx);
 * ```
 */
export const formats: Readonly<Record<HttpFormat, HttpFormatDefinition>> = Object.freeze({
  openai: {
    buildRequest: ({ texts, model, dimensions }) => ({
      url: '/embeddings',
      body: dimensions === undefined ? { model, input: texts } : { model, input: texts, dimensions },
    }),
    parseResponse: (body) => {
      const parsed = openAIResponse.safeParse(body);
      if (!parsed.success) throw malformed('openai');
      const { data, model, usage } = parsed.data;
      const vectors = [...data].sort((a, b) => a.index - b.index).map((item) => item.embedding);
      return {
        vectors,
        ...(model === undefined ? {} : { model }),
        ...(usage === undefined ? {} : { usage: { promptTokens: usage.prompt_tokens, totalTokens: usage.total_tokens } }),
      };
    },
  },
  ollama: {
    buildRequest: ({ texts, model }) => ({ url: '/api/embed', body: { model, input: texts } }),
    parseResponse: (body) => {
      const parsed = ollamaResponse.safeParse(body);
      if (!parsed.success) throw malformed('ollama');
      const { embeddings, model, prompt_eval_count: tokens } = parsed.data;
      return {
        vectors: embeddings,
        ...(model === undefined ? {} : { model }),
        ...(tokens === undefined ? {} : { usage: { promptTokens: tokens, totalTokens: tokens } }),
      };
    },
  },
});
