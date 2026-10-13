/**
 * Typed errors with stable codes. Every failure that `embed` can produce is an
 * `EmbeddingError`, so callers can branch on `error.code` instead of parsing
 * messages. Error messages never contain input text, vectors, or secrets.
 */

/** Stable error codes. Safe to match on and to expose to API clients. */
export type EmbeddingErrorCode =
  | 'EMBEDDING_INPUT_INVALID'
  | 'EMBEDDING_PROVIDER_UNREACHABLE'
  | 'EMBEDDING_PROVIDER_ERROR'
  | 'EMBEDDING_RATE_LIMITED'
  | 'EMBEDDING_TIMEOUT'
  | 'EMBEDDING_DIMENSION_MISMATCH'
  | 'EMBEDDING_DOCUMENT_UNSUPPORTED'
  | 'EMBEDDING_DOCUMENT_TOO_LARGE'
  | 'EMBEDDING_DOCUMENT_INVALID';

/** HTTP status used when an error reaches the optional route or a Fastify error handler. */
const STATUS_BY_CODE: Record<EmbeddingErrorCode, number> = {
  EMBEDDING_INPUT_INVALID: 400,
  EMBEDDING_PROVIDER_UNREACHABLE: 503,
  EMBEDDING_PROVIDER_ERROR: 502,
  EMBEDDING_RATE_LIMITED: 429,
  EMBEDDING_TIMEOUT: 504,
  EMBEDDING_DIMENSION_MISMATCH: 500,
  EMBEDDING_DOCUMENT_UNSUPPORTED: 415,
  EMBEDDING_DOCUMENT_TOO_LARGE: 413,
  EMBEDDING_DOCUMENT_INVALID: 400,
};

/** Codes that are retried by default because they are usually transient. */
const RETRYABLE_BY_DEFAULT: ReadonlySet<EmbeddingErrorCode> = new Set<EmbeddingErrorCode>([
  'EMBEDDING_PROVIDER_UNREACHABLE',
  'EMBEDDING_RATE_LIMITED',
  'EMBEDDING_TIMEOUT',
]);

/** Extra details for an `EmbeddingError`. */
export interface EmbeddingErrorOptions {
  /** The underlying error, if any. */
  cause?: unknown;
  /** Whether `withRetry` should try again. Defaults depend on the code. */
  retryable?: boolean;
  /** Delay requested by the provider (for example from a `Retry-After` header). */
  retryAfterMs?: number;
  /** Upstream HTTP status, when the error came from an HTTP response. */
  upstreamStatus?: number;
}

/**
 * The single error class thrown by this library.
 *
 * @example
 * ```ts
 * try {
 *   await app.embeddings.embed(text);
 * } catch (error) {
 *   if (error instanceof EmbeddingError && error.code === 'EMBEDDING_RATE_LIMITED') {
 *     // back off and try later
 *   }
 * }
 * ```
 */
export class EmbeddingError extends Error {
  /** Stable machine-readable code. */
  readonly code: EmbeddingErrorCode;
  /** HTTP status Fastify uses when this error is returned from a route. */
  readonly statusCode: number;
  /** Whether retrying the same request may succeed. */
  readonly retryable: boolean;
  /** Provider-requested delay before retrying, in milliseconds. */
  readonly retryAfterMs: number | undefined;
  /** Upstream HTTP status, when known. */
  readonly upstreamStatus: number | undefined;

  /**
   * @example
   * ```ts
   * throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'input must not be empty');
   * ```
   */
  constructor(code: EmbeddingErrorCode, message: string, options: EmbeddingErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'EmbeddingError';
    this.code = code;
    this.statusCode = STATUS_BY_CODE[code];
    this.retryable = options.retryable ?? RETRYABLE_BY_DEFAULT.has(code);
    this.retryAfterMs = options.retryAfterMs;
    this.upstreamStatus = options.upstreamStatus;
  }
}

/**
 * Type guard for `EmbeddingError`, optionally narrowed to one code.
 *
 * @example
 * ```ts
 * if (isEmbeddingError(error, 'EMBEDDING_TIMEOUT')) retryLater();
 * ```
 */
export function isEmbeddingError(error: unknown, code?: EmbeddingErrorCode): error is EmbeddingError {
  return error instanceof EmbeddingError && (code === undefined || error.code === code);
}
