/**
 * OpenTelemetry instrumentation for embedding providers.
 *
 * `@opentelemetry/api` returns a no-op tracer when the application has not
 * registered an SDK, so importing this module never requires telemetry to be
 * configured and never adds overhead when it is absent. Instrumentation only
 * becomes visible once the host application registers a `TracerProvider`.
 */
import { SpanKind, SpanStatusCode, trace } from '@opentelemetry/api';
import type { Attributes, Span, Tracer } from '@opentelemetry/api';
import { isEmbeddingError } from './errors.js';
import type { EmbeddingProvider, ProviderOutput } from './types.js';

/** Instrumentation scope name used when `tracerName` is not provided. */
const DEFAULT_TRACER_NAME = 'fastify-embeddings';

/** Options for `withTelemetry`. */
export interface TelemetryOptions {
  /** Use a specific tracer instead of the global one. Useful in tests or with several SDKs. */
  tracer?: Tracer;
  /** Instrumentation scope name reported to the provider. Defaults to `'fastify-embeddings'`. */
  tracerName?: string;
  /** Instrumentation scope version reported to the provider. */
  tracerVersion?: string;
}

/** Classify an error for the `error.type` attribute. */
function errorType(error: unknown): string {
  if (isEmbeddingError(error)) return error.code;
  if (error instanceof Error && error.name !== '') return error.name;
  return 'Error';
}

/** Record a failure on the span without altering the thrown value. */
function recordError(span: Span, error: unknown): void {
  span.setAttribute('error.type', errorType(error));
  span.recordException(error instanceof Error ? error : new Error(String(error)));
  span.setStatus({
    code: SpanStatusCode.ERROR,
    message: error instanceof Error ? error.message : String(error),
  });
}

/**
 * Wrap a provider so that every batch is reported as one OpenTelemetry span.
 *
 * A provider call is one batch, so each span carries the batch size, the number
 * of vectors returned, their dimensions, the model, and token usage when the
 * provider reports it. Failures mark the span as an error, set `error.type` to
 * the `EmbeddingError` code, and record the exception before it is rethrown.
 *
 * Span names follow the OpenTelemetry generative AI conventions, `embeddings
 * <model>`. Note that `gen_ai.system` carries this library's provider name
 * (`http`, `local`, `mock`, or your own) rather than a vendor system
 * identifier, because a provider can front any API.
 *
 * @example
 * ```ts
 * import { createEmbedder, httpProvider } from 'fastify-embeddings';
 * import { withTelemetry } from 'fastify-embeddings/otel';
 *
 * const provider = withTelemetry(httpProvider({ baseURL: 'http://localhost:11434', format: 'ollama', preset: 'nomic' }));
 * const embedder = createEmbedder({ provider, batchSize: 8, concurrency: 1 });
 * await embedder.embed(['one document', 'another document']); // one span per batch
 * ```
 */
export function withTelemetry(provider: EmbeddingProvider, options: TelemetryOptions = {}): EmbeddingProvider {
  const tracer = options.tracer ?? trace.getTracer(options.tracerName ?? DEFAULT_TRACER_NAME, options.tracerVersion);
  const spanName = `embeddings ${provider.model}`;

  return {
    name: provider.name,
    model: provider.model,
    ...(provider.dimensions === undefined ? {} : { dimensions: provider.dimensions }),
    embed(texts, inputType, signal): Promise<ProviderOutput> {
      const attributes: Attributes = {
        'gen_ai.operation.name': 'embeddings',
        'gen_ai.system': provider.name,
        'gen_ai.request.model': provider.model,
        'embeddings.batch.size': texts.length,
        'embeddings.input_type': inputType,
      };

      return tracer.startActiveSpan(spanName, { kind: SpanKind.CLIENT, attributes }, async (span) => {
        try {
          const output = await provider.embed(texts, inputType, signal);
          span.setAttribute('embeddings.vector.count', output.vectors.length);
          const first = output.vectors[0];
          if (first !== undefined) span.setAttribute('embeddings.dimensions', first.length);
          if (output.model !== undefined && output.model !== '') {
            span.setAttribute('gen_ai.response.model', output.model);
          }
          if (output.usage !== undefined) {
            span.setAttribute('gen_ai.usage.input_tokens', output.usage.promptTokens);
          }
          span.setStatus({ code: SpanStatusCode.OK });
          return output;
        } catch (error) {
          recordError(span, error);
          throw error;
        } finally {
          span.end();
        }
      });
    },
    close: () => provider.close?.(),
  };
}
