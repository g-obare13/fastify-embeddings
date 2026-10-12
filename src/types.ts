/**
 * Core public types: the provider contract, the result of `embed`, hooks, and
 * the logger shape. Everything a user needs to type their own code lives here.
 */
import type { EmbeddingError } from './errors.js';

/** Whether the text is a search query or a document being indexed. */
export type InputType = 'query' | 'document';

/** A single embedding vector. */
export type Vector = number[];

/** Token usage, when the provider reports it. */
export interface Usage {
  /** Tokens consumed by the input text. */
  promptTokens: number;
  /** Total tokens billed for the request. */
  totalTokens: number;
}

/** What a provider returns: vectors and optional request metadata. */
export interface ProviderOutput {
  /** One vector per input text, in input order. */
  vectors: Vector[];
  /** Token usage, if known. */
  usage?: Usage;
  /** Model that actually served the request, if it differs from the configured one. */
  model?: string;
}

/**
 * The one extension point. Anything that can turn strings into vectors.
 * Build one with `defineProvider` to get type checking.
 */
export interface EmbeddingProvider {
  /** Short identifier used in logs and `embedder.provider`, for example `'http'`. */
  readonly name: string;
  /** Model identifier reported in results. */
  readonly model: string;
  /** Expected vector size. When set, responses of a different size are rejected. */
  readonly dimensions?: number;
  /**
   * Embed one batch of texts. Must return `{ vectors }` with exactly one vector
   * per text, in order. Usage and the actual model may also be reported.
   * Receives at most `batchSize` texts per call.
   */
  embed(texts: string[], inputType: InputType, signal?: AbortSignal): Promise<ProviderOutput>;
  /** Release sockets, models, or other resources. Called once on shutdown. */
  close?(): Promise<void> | void;
}

/** The data returned by every `embed` call, for single strings and arrays alike. */
export interface EmbedResult {
  /** One vector per input, in input order. */
  vectors: Vector[];
  /** Model that produced the vectors. */
  model: string;
  /** Length of each vector. */
  dimensions: number;
  /** Summed token usage across batches, when the provider reports it. */
  usage?: Usage;
}

/** Per-call options for `embed`. */
export interface EmbedOptions {
  /** Defaults to `'document'`. Controls query or document prefixes. */
  inputType?: InputType;
  /** Cancels in-flight provider requests. */
  signal?: AbortSignal;
}

/** Metadata passed to `onBatch`. Contains no text and no vectors. */
export interface BatchInfo {
  /** Number of texts in the batch. */
  count: number;
  /** Wall time of the provider call, in milliseconds. */
  durationMs: number;
  /** Model that served the batch. */
  model: string;
  /** Input type of the batch. */
  inputType: InputType;
}

/** Context passed to `transformOutput`. */
export interface TransformOutputContext {
  /** Model that produced the vectors. */
  model: string;
  /** Input type of the call. */
  inputType: InputType;
}

/** Optional functions for small changes without writing a provider. */
export interface EmbedderHooks {
  /** Rewrite texts before they reach the provider. Must return one string per input. */
  transformInput?: (texts: string[], inputType: InputType) => string[] | Promise<string[]>;
  /** Rewrite vectors after normalization. Must return one vector per input. */
  transformOutput?: (vectors: Vector[], context: TransformOutputContext) => Vector[] | Promise<Vector[]>;
  /** Called after each provider batch succeeds. */
  onBatch?: (info: BatchInfo) => void;
  /** Called for every `EmbeddingError` thrown by `embed`. */
  onError?: (error: EmbeddingError) => void;
}

/** Minimal logger shape. Fastify's `app.log` (pino) satisfies it. */
export interface EmbeddingsLogger {
  /** Log debug details (counts, model, duration). */
  debug(details: object, message: string): void;
  /** Log a warning (error code, model, duration). */
  warn(details: object, message: string): void;
}

/** Result of `healthCheck`. Never throws. */
export type HealthStatus =
  | {
      /** The provider answered with valid vectors. */
      ok: true;
      /** Model reported by the provider. */
      model: string;
      /** Vector size reported by the provider. */
      dimensions: number;
      /** Round trip time, in milliseconds. */
      durationMs: number;
    }
  | {
      /** The provider failed. */
      ok: false;
      /** Why it failed. */
      error: EmbeddingError;
      /** Time until failure, in milliseconds. */
      durationMs: number;
    };

/** Options for configuring the optional POST /embeddings route. */
export interface RouteOptions {
  /** Route path. Defaults to `'/embeddings'`. */
  path?: string;
  /** Maximum request body size in bytes. Defaults to 1048576 (1MB). */
  bodyLimit?: number;
}
