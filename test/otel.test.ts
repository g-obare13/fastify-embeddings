import { afterAll, beforeEach, describe, expect, it } from 'vitest';
import { SpanKind, SpanStatusCode, trace } from '@opentelemetry/api';
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { createEmbedder, defineProvider, EmbeddingError, mockProvider } from 'fastify-embeddings';
import { withTelemetry } from 'fastify-embeddings/otel';

const exporter = new InMemorySpanExporter();
const tracerProvider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
const finished = () => exporter.getFinishedSpans();

beforeEach(() => {
  trace.setGlobalTracerProvider(tracerProvider);
  exporter.reset();
});

afterAll(() => {
  trace.disable();
});

describe('withTelemetry', () => {
  it('emits one span per batch carrying batch size, dimensions, model, and usage', async () => {
    const provider = withTelemetry(mockProvider({ dimensions: 8, model: 'mock-embedding' }));
    const output = await provider.embed(['first', 'second'], 'document');

    expect(output.vectors).toHaveLength(2);
    const [span] = finished();
    expect(span?.name).toBe('embeddings mock-embedding');
    expect(span?.kind).toBe(SpanKind.CLIENT);
    expect(span?.status.code).toBe(SpanStatusCode.OK);
    expect(span?.attributes['gen_ai.operation.name']).toBe('embeddings');
    expect(span?.attributes['gen_ai.system']).toBe('mock');
    expect(span?.attributes['gen_ai.request.model']).toBe('mock-embedding');
    expect(span?.attributes['embeddings.input_type']).toBe('document');
    expect(span?.attributes['embeddings.batch.size']).toBe(2);
    expect(span?.attributes['embeddings.vector.count']).toBe(2);
    expect(span?.attributes['embeddings.dimensions']).toBe(8);
    expect(span?.attributes['gen_ai.usage.input_tokens']).toBeGreaterThan(0);
  });

  it('produces one span per batch when driven through createEmbedder', async () => {
    const embedder = createEmbedder({
      provider: withTelemetry(mockProvider({ dimensions: 4, model: 'batch-model' })),
      batchSize: 2,
      concurrency: 1,
    });

    const result = await embedder.embed(['a', 'b', 'c', 'd', 'e']);
    expect(result.vectors).toHaveLength(5);
    expect(finished().map((span) => span.attributes['embeddings.batch.size'])).toEqual([2, 2, 1]);
    await embedder.close();
  });

  it('marks the span as an error and records the EmbeddingError code', async () => {
    const failing = defineProvider({
      name: 'failing',
      model: 'fail-model',
      async embed() {
        throw new EmbeddingError('EMBEDDING_RATE_LIMITED', 'provider said slow down');
      },
    });

    const provider = withTelemetry(failing);
    await expect(provider.embed(['x'], 'document')).rejects.toThrow('provider said slow down');

    const [span] = finished();
    expect(span?.status.code).toBe(SpanStatusCode.ERROR);
    expect(span?.attributes['error.type']).toBe('EMBEDDING_RATE_LIMITED');
    expect(span?.events.some((event) => event.name === 'exception')).toBe(true);
  });

  it('uses the error name for a plain Error and tolerates a thrown non-Error', async () => {
    const named = defineProvider({
      name: 'named',
      model: 'named-model',
      async embed() {
        const error = new Error('boom');
        error.name = 'CustomFailure';
        throw error;
      },
    });
    await expect(withTelemetry(named).embed(['x'], 'document')).rejects.toThrow('boom');
    expect(finished()[0]?.attributes['error.type']).toBe('CustomFailure');

    exporter.reset();
    const odd = defineProvider({
      name: 'odd',
      model: 'odd-model',
      embed() {
        return Promise.reject('just a string');
      },
    });
    await expect(withTelemetry(odd).embed(['x'], 'document')).rejects.toBe('just a string');
    expect(finished()[0]?.attributes['error.type']).toBe('Error');
  });

  it('forwards provider identity and close()', async () => {
    let closed = false;
    const inner = defineProvider({
      name: 'inner',
      model: 'inner-model',
      dimensions: 3,
      async embed() {
        return { vectors: [[1, 2, 3]] };
      },
      close() {
        closed = true;
      },
    });

    const provider = withTelemetry(inner);
    expect(provider.name).toBe('inner');
    expect(provider.model).toBe('inner-model');
    expect(provider.dimensions).toBe(3);
    await provider.close?.();
    expect(closed).toBe(true);
  });

  it('tolerates a provider without dimensions, close(), or reported metadata', async () => {
    const bare = defineProvider({
      name: 'bare',
      model: 'bare-model',
      async embed() {
        return { vectors: [] };
      },
    });

    const provider = withTelemetry(bare);
    expect(provider.dimensions).toBeUndefined();
    await provider.close?.();

    const output = await provider.embed(['x'], 'document');
    expect(output.vectors).toEqual([]);

    const [span] = finished();
    expect(span?.attributes['embeddings.vector.count']).toBe(0);
    expect(span?.attributes['embeddings.dimensions']).toBeUndefined();
    expect(span?.attributes['gen_ai.usage.input_tokens']).toBeUndefined();
    expect(span?.attributes['gen_ai.response.model']).toBeUndefined();
  });

  it('records the model the provider reports when it differs from the configured one', async () => {
    const provider = withTelemetry(
      defineProvider({
        name: 'override',
        model: 'configured-model',
        async embed() {
          return { vectors: [[1]], model: 'served-model' };
        },
      }),
    );

    await provider.embed(['x'], 'query');
    const [span] = finished();
    expect(span?.attributes['gen_ai.request.model']).toBe('configured-model');
    expect(span?.attributes['gen_ai.response.model']).toBe('served-model');
    expect(span?.attributes['embeddings.input_type']).toBe('query');
  });

  it('accepts a custom tracer or a custom instrumentation scope', async () => {
    const custom = withTelemetry(mockProvider({ dimensions: 4 }), {
      tracer: tracerProvider.getTracer('custom-scope', '9.9.9'),
    });
    await custom.embed(['x'], 'document');
    expect(finished()[0]?.instrumentationScope.name).toBe('custom-scope');

    exporter.reset();
    const named = withTelemetry(mockProvider({ dimensions: 4 }), { tracerName: 'named-scope', tracerVersion: '1.2.3' });
    await named.embed(['x'], 'document');
    expect(finished()[0]?.instrumentationScope.name).toBe('named-scope');
  });

  it('is a silent no-op when the application has not registered an SDK', async () => {
    trace.disable();

    const provider = withTelemetry(mockProvider({ dimensions: 4, model: 'no-sdk' }));
    const output = await provider.embed(['x'], 'document');

    expect(output.vectors).toHaveLength(1);
    expect(output.vectors[0]).toHaveLength(4);
    expect(finished()).toHaveLength(0);
  });
});
