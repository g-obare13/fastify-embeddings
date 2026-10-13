# fastify-embeddings

Fastify 5 plugin and standalone TypeScript library for generating text embeddings. It batches requests, preserves input order, validates vector dimensions, and returns model and usage metadata when available.

## What this library does

It turns text into vectors. That is the whole job. The call below is a real one, not pseudocode: register the plugin, then hand `embed` one string or an array of strings.

<!-- docs-snippet: embed-call.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'http://localhost:11434',
  format: 'ollama',
  preset: 'nomic',
});

const { vectors, model, dimensions, usage } = await app.embeddings.embed(['some text', 'more text']);
app.log.info(
  { model, dimensions, vectors: vectors.length, tokens: usage?.totalTokens },
  'embedded two texts in one batch',
);
await app.close();
```

Everything else in this repository is scaffolding around that call: either one HTTP request to an embeddings API, or one in-process model call.

## What it buys you

The engine is a few hundred lines, and it earns its keep in these places:

1. **Batching with bounded concurrency.** Hand it 10,000 strings and it splits them into chunks (`batchSize`, default 32) and runs a limited number at once (`concurrency`, default 2).
2. **Order preservation.** `vectors[0]` is always the vector for `input[0]`, however the batches happen to complete. This is where hand-written versions quietly break.
3. **One result shape**, whether the provider is OpenAI, Ollama, a local model, or an object you wrote yourself.
4. **A dimension guard**, so a provider silently changing its vector size becomes an error instead of a corrupted index.
5. **L2 normalization by default**, because cosine search expects unit vectors.
6. **HTTP resilience**: retries with backoff and jitter, `Retry-After` honoured, a per-attempt timeout, and abort support.
7. **Typed errors** such as `EMBEDDING_RATE_LIMITED`, `EMBEDDING_TIMEOUT`, and `EMBEDDING_PROVIDER_UNREACHABLE`, each carrying an HTTP status.
8. **Fastify wiring**: `app.embeddings`, cleanup on `onClose`, an optional route, and named embedders.
9. **Log hygiene**: it never logs your input text, your vectors, or your API key.
10. **A mock provider**, so your tests never need a network.

## What it does not do

It is **not** a vector database. It does not store vectors, search them, chunk documents, rerank results, or manage collections. Qdrant, pgvector, and Pinecone do that. The app in `examples/ollama-qdrant` demonstrates the handoff between this library and a database, and is not part of the library itself.

If your project embeds a handful of strings in one place, a short `fetch` call is simpler and you should skip this library. It pays off when you have many texts, more than one provider, or you need retries, ordering, and observability.

## TODOs

- **Document chunking and parsing.** Split long documents, and formats other than plain strings, into embedding sized pieces before they reach `embed`. Today that is the caller's job, and it is the most common cause of poor retrieval quality in practice.

## Install and requirements

- Node.js **20.3 or newer**.
- Fastify 5 when using the plugin.
- The optional `@huggingface/transformers` peer dependency when using the in-process local provider.
- An HTTP embedding API (for example Ollama or an OpenAI-compatible API) when using the HTTP provider.

```sh
pnpm add fastify-embeddings fastify
```

The library defaults to the local `Xenova/all-MiniLM-L6-v2` model when no HTTP base URL is configured. Install the optional provider dependency to use that default:

```sh
pnpm add @huggingface/transformers
```

On the first local embedding call, `@huggingface/transformers` downloads the model. For the default `Xenova/all-MiniLM-L6-v2` model that is roughly 86 MB, and a cold first call took about 80 seconds in testing.

Model files are cached inside the installed package, at `node_modules/@huggingface/transformers/.cache`, and the loaded model stays in memory for the life of the process. `HF_HOME` has no effect here, because that variable belongs to the Python `huggingface_hub` library and this package never reads it. To relocate the cache, set `cacheDir` on the Transformers `env` object before the first embedding call: `import { env } from '@huggingface/transformers'` then `env.cacheDir = '/path/to/cache'`. The provider imports the same module instance, so the change applies to embeddings made through this library. To avoid a model download entirely, configure an HTTP provider with `EMBEDDINGS_BASE_URL` or explicit options.

## Quick start

<!-- docs-snippet: quick-start.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings);
const result = await app.embeddings.embed(['Hello vector search']);
app.log.info(
  { model: result.model, dimensions: result.dimensions, count: result.vectors.length },
  'generated embeddings',
);
await app.close();
```

## Embed result and provider contract

`embed()` always returns an `EmbedResult`, whether its input is a string or an array:

<!-- docs-snippet: embed-result.ts -->
```ts
import type { EmbedResult } from 'fastify-embeddings';

const result: EmbedResult = {
  vectors: [[0.1, 0.2, 0.3]],
  model: 'example-model',
  dimensions: 3,
  usage: { promptTokens: 4, totalTokens: 4 },
};
void result;
```

Persist the returned `model` and `dimensions` alongside stored vectors. Vectors from different models or dimensions should not be compared or mixed in the same index without an explicit migration strategy.

The library has a single provider result contract: every provider returns an object with a required `vectors` array and optional `usage` and `model`. A custom provider must return the same shape. The optional model identifies the model that actually served the request when it differs from the configured model.

## Configure a provider

### Zero-config environment setup

Without `EMBEDDINGS_BASE_URL`, the default is the local MiniLM provider. When the base URL is configured, the HTTP provider is selected. Only these environment variables are read:

| Variable | Purpose | Default |
| --- | --- | --- |
| `EMBEDDINGS_BASE_URL` | HTTP API root; selects the HTTP provider when set | unset |
| `EMBEDDINGS_MODEL` | Model identifier | local MiniLM, or the selected preset's model |
| `EMBEDDINGS_FORMAT` | HTTP wire format: `openai` or `ollama` | `openai` |
| `EMBEDDINGS_API_KEY` | HTTP bearer token | unset |

Explicit options take precedence over environment variables, which take precedence over preset defaults. Set `env: {}` when calling `createEmbedder()` to ignore the process environment.

### Options

| Option | Purpose | Default / notes |
| --- | --- | --- |
| `provider` | `http`, `local`, `mock`, or a custom provider object | `http` if a base URL exists, otherwise `local` |
| `baseURL` | HTTP API root | `EMBEDDINGS_BASE_URL` |
| `model` | Model identifier | `EMBEDDINGS_MODEL`, then preset/default model |
| `format` | Built-in HTTP format (`openai` or `ollama`) | `openai` |
| `apiKey` | HTTP bearer token | `EMBEDDINGS_API_KEY` |
| `dimensions` | Expected vector size | For OpenAI format it is sent in the request body and also validated; for Ollama and local it is validation-only; mock uses it to generate vectors. A custom HTTP `buildRequest` decides whether to send it. For a custom provider object, set `provider.dimensions`; this option does not override the provider object's dimensions. |
| `prefixes` | Query/document prefixes | Merged over preset prefixes |
| `preset` | Preset name or `{ model, prefixes }` object | none |
| `batchSize` | Maximum texts per provider call | `32` |
| `concurrency` | Maximum concurrent provider calls | `2` |
| `normalize` | L2-normalize returned vectors | `true` |
| `timeoutMs` | HTTP timeout per attempt | `30000` |
| `retries` | HTTP retries after the first attempt | `2` |
| `retryBaseDelayMs` | HTTP exponential-backoff base delay | `250` ms |
| `headers` | Extra HTTP headers | none |
| `buildRequest` / `parseResponse` | Customize the HTTP request and response | built-in format handlers |
| `transformInput` / `transformOutput` | Transform texts or vectors | none |
| `onBatch` / `onError` | Per-batch and per-error hooks | none |
| `logger` | Structured logger for counts, model, durations, and error codes | silent in standalone use; Fastify `app.log` in the plugin |
| `env` | Environment map used for resolution | `process.env` |
| `name` | Named Fastify embedder | default embedder |
| `eager` | Run health check during Fastify startup | `false` |
| `route` | Enable the optional HTTP route or configure `path` and `bodyLimit` | disabled |

### Ollama and the Nomic preset

The preset supplies the Nomic model name and distinct query/document prefixes; no explicit `model` is required:

<!-- docs-snippet: nomic-preset.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'http://localhost:11434',
  format: 'ollama',
  preset: 'nomic',
});
const result = await app.embeddings.embed('What is vector search?', { inputType: 'query' });
void result;
await app.close();
```

Ollama must be running and have the selected model pulled before the first request.

### OpenAI-compatible APIs

<!-- docs-snippet: openai.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'https://api.openai.com/v1',
  model: 'text-embedding-3-small',
  apiKey: process.env.OPENAI_API_KEY,
  dimensions: 1536,
});
await app.close();
```

### Custom HTTP API

`buildRequest` receives the prefixed texts, input type, model, dimensions, and base URL. `parseResponse` must return the provider result object; include usage when the API reports it.

<!-- docs-snippet: custom-http.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'https://api.mycompany.internal',
  model: 'enterprise-v1',
  buildRequest: ({ texts, inputType }) => ({
    url: '/vectors/generate',
    method: 'POST',
    body: { passages: texts, mode: inputType === 'query' ? 'search' : 'index' },
  }),
  parseResponse: (body) => {
    const data = body as { items: Array<{ embedding: number[] }> };
    return { vectors: data.items.map((item) => item.embedding) };
  },
});
await app.close();
```

### Custom provider

<!-- docs-snippet: custom-provider.ts -->
```ts
import Fastify from 'fastify';
import { defineProvider, fastifyEmbeddings } from 'fastify-embeddings';

const customProvider = defineProvider({
  name: 'my-custom-engine',
  model: 'custom-model',
  dimensions: 256,
  async embed(texts, inputType, signal) {
    const response = await fetch('https://custom-service/embed', {
      method: 'POST',
      body: JSON.stringify({ texts, inputType }),
      signal,
    });
    const json = (await response.json()) as { embeddings: number[][] };
    return { vectors: json.embeddings };
  },
  async close() {
    // Release connections or resources.
  },
});

const app = Fastify();
await app.register(fastifyEmbeddings, { provider: customProvider });
await app.close();
```

## Self-hosted with Ollama

Ollama serves embeddings from `POST /api/embed`, which the `ollama` format already speaks, so a self-hosted server needs no special configuration.

### Environment variables and the .env workflow

| Variable | Purpose | Placeholder in `.env.example` |
| --- | --- | --- |
| `EMBEDDINGS_BASE_URL` | Ollama API root. Setting it selects the HTTP provider. | `http://ollama.example.internal:11434` |
| `EMBEDDINGS_FORMAT` | Wire format. Use `ollama`. | `ollama` |
| `EMBEDDINGS_MODEL` | Model to send. The `nomic` preset expects `nomic-embed-text`. | `nomic-embed-text` |

Copy `.env.example` to `.env` and replace the placeholders with your own values. `.env` and `.env.*` are git-ignored, so only the placeholder template is ever committed. Shell variables take priority over `.env` values, which means CI and production can inject real values with no file at all.

<!-- docs-snippet: ollama-self-hosted.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  preset: 'nomic',
  batchSize: 8,
  concurrency: 1,
  timeoutMs: 120_000,
});
const result = await app.embeddings.embed('What is vector search?', { inputType: 'query' });
app.log.info({ model: result.model, dimensions: result.dimensions }, 'embedded with self-hosted Ollama');
await app.close();
```

### What the nomic preset does

The `nomic` preset sets the model id to `nomic-embed-text` and applies the two asymmetric prefixes that model was trained with: `search_query: ` for queries and `search_document: ` for documents. Because the prefixes differ, the same text embedded as a query and as a document deliberately produces different vectors. Pass `inputType` so the correct prefix is applied, and keep queries and documents on separate code paths.

### Tuning for a small, single-parallel server

A small Ollama instance handles one request at a time, so raising concurrency only queues work and grows memory use.

- `batchSize: 8` keeps each request small and predictable. Larger batches raise peak memory on the server without improving throughput on a single-parallel host.
- `concurrency: 1` matches the server's actual parallelism. Raise it only when you know the host serves parallel requests.
- `timeoutMs: 120000` covers a cold model load. The 30 second default is often too short for the first request after a restart.
- `retries` can stay at its default for transient failures. Set it to `0` when you want a failure to surface immediately instead of after a retry.

<!-- docs-snippet: ollama-tuning.ts -->
```ts
import { createEmbedder } from 'fastify-embeddings';

const embedder = createEmbedder({
  baseURL: process.env.EMBEDDINGS_BASE_URL,
  format: 'ollama',
  model: process.env.EMBEDDINGS_MODEL,
  preset: 'nomic',
  batchSize: 8,
  concurrency: 1,
  timeoutMs: 120_000,
});

const health = await embedder.healthCheck();
if (health.ok) {
  process.stdout.write(`healthy: ${String(health.dimensions)} dimensions\n`);
}
await embedder.close();
```

### `OLLAMA_KEEP_ALIVE` and cold starts

By default Ollama unloads a model after a few minutes of inactivity, and the next request pays for the load. That first call can take tens of seconds, which is easily enough to exceed a short `timeoutMs`. Either raise `timeoutMs` as shown above, or keep the model resident by setting `OLLAMA_KEEP_ALIVE` on the server, for example `30m`, or `-1` to keep it loaded until the server stops. Expect the first call of a run to be the slowest.

### Context length

Ollama truncates input that exceeds the model's context window, and a truncated document silently loses everything past the limit. Embed chunks that fit comfortably inside the window instead of whole documents, and check the model's context size before increasing `batchSize`, because a batch of long inputs multiplies the tokens in a single request.

### Real server smoke test

`pnpm smoke:real` exercises a live Ollama server end to end. It reads `EMBEDDINGS_BASE_URL`, `EMBEDDINGS_FORMAT` and `EMBEDDINGS_MODEL` from your shell or from a local `.env`, names any variable that is unset, and prints `PASS` or `FAIL` for each check:

- `healthCheck` succeeds and reports 768 dimensions and the configured model
- the same text embedded as a query and as a document returns different vectors with the `nomic` preset
- similar sentences score higher than unrelated ones
- 50 texts with `batchSize: 8` and `concurrency: 1` come back in input order with the expected count and batch sizes
- a dead local port returns `EMBEDDING_PROVIDER_UNREACHABLE` quickly, without contacting the real server
- an unknown model against the real server returns `EMBEDDING_PROVIDER_ERROR`
- captured logs contain no input text, no vectors and no API key

The script is written for a server that may be shared with other projects. It only calls the embedding endpoint, never pulls or deletes models, never calls an admin endpoint, and prints a host class rather than the server address.

### Security

Ollama has no authentication. Anything that can reach the port can use every model on the host and can pull or delete models. Do not expose `11434` to the internet, and do not commit a real server address, which is why `.env.example` ships with placeholders only.

Two ways to keep it private:

- **SSH tunnel.** Bind Ollama to loopback on the server and forward the port from your machine. `EMBEDDINGS_BASE_URL` then points at a local port and the service is never publicly reachable.
- **Reverse proxy with authentication.** Put a proxy in front of Ollama that terminates TLS and requires a shared secret, then send that secret from the `headers` option so the library adds it to every request.

Both approaches keep the address and any credential out of source control: store them in `.env` locally and in your platform's secret store everywhere else.

## Cache recipe

This optional wrapper uses an LRU cache capped at 500 entries by default. Change the limit with `withCache(provider, maxEntries)`. Keys include the configured model, `inputType`, and text; texts longer than 1,024 characters are represented by a SHA-256 digest. Vectors are copied on storage and return. A full cache hit has no usage to report; partial misses preserve the provider's usage for the texts it computed.

<!-- docs-snippet: cache-recipe.ts -->
```ts
import { createHash } from 'node:crypto';
import type { EmbeddingProvider, ProviderOutput, Vector } from 'fastify-embeddings';

const LONG_TEXT_THRESHOLD = 1024;

export function withCache(baseProvider: EmbeddingProvider, maxEntries = 500): EmbeddingProvider {
  if (!Number.isInteger(maxEntries) || maxEntries < 1) {
    throw new RangeError('maxEntries must be a positive integer');
  }

  const cache = new Map<string, Vector>();

  function cacheKey(text: string, inputType: 'query' | 'document'): string {
    const cacheText =
      text.length > LONG_TEXT_THRESHOLD
        ? `sha256:${createHash('sha256').update(text).digest('hex')}`
        : `text:${text}`;
    return JSON.stringify([baseProvider.model, inputType, cacheText]);
  }

  function store(key: string, vector: Vector): void {
    cache.delete(key);
    cache.set(key, [...vector]);
    if (cache.size > maxEntries) {
      const oldestKey = cache.keys().next().value;
      if (oldestKey !== undefined) cache.delete(oldestKey);
    }
  }

  return {
    name: `cached-${baseProvider.name}`,
    model: baseProvider.model,
    ...(baseProvider.dimensions === undefined ? {} : { dimensions: baseProvider.dimensions }),
    async embed(texts, inputType, signal): Promise<ProviderOutput> {
      const results: (Vector | undefined)[] = new Array(texts.length);
      const missingTexts: string[] = [];
      const missingIndices: number[] = [];
      const missingKeys: string[] = [];

      texts.forEach((text, index) => {
        const key = cacheKey(text, inputType);
        const cached = cache.get(key);
        if (cached !== undefined) {
          cache.delete(key);
          cache.set(key, cached);
          results[index] = [...cached];
        } else {
          missingTexts.push(text);
          missingIndices.push(index);
          missingKeys.push(key);
        }
      });

      let output: ProviderOutput | undefined;
      if (missingTexts.length > 0) {
        output = await baseProvider.embed(missingTexts, inputType, signal);
        if (output.vectors.length !== missingTexts.length) {
          throw new Error(`Cached provider returned ${String(output.vectors.length)} vectors for ${String(missingTexts.length)} inputs`);
        }
        output.vectors.forEach((vector, index) => {
          const originalIndex = missingIndices[index];
          const key = missingKeys[index];
          if (originalIndex === undefined || key === undefined) {
            throw new Error('Cached provider result could not be matched to its input');
          }
          store(key, vector);
          results[originalIndex] = [...vector];
        });
      }

      const vectors = results.map((vector, index) => {
        if (vector === undefined) throw new Error(`Missing cached vector for input ${String(index)}`);
        return [...vector];
      });
      return {
        vectors,
        ...(output?.usage === undefined ? {} : { usage: output.usage }),
        ...(output?.model === undefined ? {} : { model: output.model }),
      };
    },
    close: () => baseProvider.close?.(),
  };
}
```

## Telemetry (OpenTelemetry)

`fastify-embeddings/otel` wraps any provider so that every batch becomes one OpenTelemetry span. It is a separate subpath, so an application that does not import it never loads the OpenTelemetry API.

<!-- docs-snippet: otel.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings, httpProvider, presets } from 'fastify-embeddings';
import { withTelemetry } from 'fastify-embeddings/otel';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  provider: withTelemetry(
    httpProvider({ baseURL: 'http://localhost:11434', format: 'ollama', preset: presets.nomic }),
  ),
  batchSize: 8,
  concurrency: 1,
});

const result = await app.embeddings.embed(['one document', 'another document']);
app.log.info({ vectors: result.vectors.length }, 'each provider batch produced one span');
await app.close();
```

A provider call is one batch, so the span covers the batch rather than the whole `embed` call. The span name is `embeddings <model>` and the span kind is `CLIENT`.

| Attribute | Meaning |
| --- | --- |
| `gen_ai.operation.name` | Always `embeddings` |
| `gen_ai.system` | This library's provider name: `http`, `local`, `mock`, or your own |
| `gen_ai.request.model` | The configured model |
| `gen_ai.response.model` | The model the provider reported, when it differs from the configured one |
| `gen_ai.usage.input_tokens` | `usage.promptTokens`, when the provider reports usage |
| `embeddings.batch.size` | Texts in this provider call |
| `embeddings.vector.count` | Vectors returned |
| `embeddings.dimensions` | Length of the returned vectors |
| `embeddings.input_type` | `query` or `document` |

On failure the span is marked as an error, `error.type` carries the `EmbeddingError` code, and the exception is recorded before the original error is rethrown.

Zero configuration is the point: `@opentelemetry/api` returns a no-op tracer until your application registers a `TracerProvider`, so the wrapper is inert until you set one up and the OpenTelemetry SDK is never a dependency of this library. Register an SDK as you normally would, for example `@opentelemetry/sdk-trace-base` with a span processor, and spans appear.

The wrapper returns a plain provider: `name`, `model`, `dimensions`, `embed`, and `close`. Extra properties on the provider you wrap are not copied onto it, so if you wrap a mock provider, read `calls` from the original object rather than from the wrapper.

Two caveats worth knowing:

- The generative AI attributes are still marked *development* in the OpenTelemetry specification, so those names may change upstream.
- `recordException` stores the error message on the span. Provider messages never contain your input text or API key, but `EMBEDDING_PROVIDER_UNREACHABLE` does include the provider origin, so spans can carry your internal hostname.

## Output transforms and hooks

`transformOutput` runs after the provider response has passed its dimension check and after optional normalization. The returned result's `dimensions` is computed from the transformed vectors, so truncation reports the shorter size.

<!-- docs-snippet: truncate.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings, normalize, truncate } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  baseURL: 'https://api.openai.com/v1',
  model: 'text-embedding-3-small',
  apiKey: process.env.OPENAI_API_KEY,
  transformOutput: (vectors) => vectors.map((vector) => normalize(truncate(vector, 256))),
});
const result = await app.embeddings.embed('A Matryoshka embedding example');
app.log.info({ dimensions: result.dimensions }, 'transformed dimensions');
await app.close();
```

- `transformInput(texts, inputType)` rewrites the input strings and must return one string per input.
- `transformOutput(vectors, context)` rewrites vectors and must return one vector per input.
- `onBatch(info)` receives count, duration, model, and input type, not input text or vectors.
- `onError(error)` receives the structured `EmbeddingError`.

<!-- docs-snippet: hooks.ts -->
```ts
import { createEmbedder, mockProvider, normalize, truncate } from 'fastify-embeddings';

const embedder = createEmbedder({
  provider: mockProvider({ dimensions: 8 }),
  transformInput: (texts, inputType) =>
    texts.map((text) => `${inputType === 'query' ? 'search: ' : 'document: '}${text}`),
  transformOutput: (vectors) => vectors.map((vector) => normalize(truncate(vector, 4))),
  onBatch: ({ count, durationMs, model, inputType }) => {
    process.stdout.write(`${model}: ${String(count)} ${inputType}s in ${String(durationMs)}ms\n`);
  },
  onError: (error) => {
    process.stderr.write(`${error.code}: ${error.message}\n`);
  },
});

await embedder.embed(['search this']);
await embedder.close();
```

## Fastify integration

### Named embedders

Register a named instance with `name` and retrieve it with `app.embeddings.get(name)`.

<!-- docs-snippet: named-embedders.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, { provider: mockProvider(), name: 'default' });
await app.register(fastifyEmbeddings, {
  provider: mockProvider({ model: 'documentation-model' }),
  name: 'docs',
});
await app.ready();
const docs = app.embeddings.get('docs');
if (docs !== undefined) await docs.embed('A page to index');
await app.close();
```

### Optional route and health check

The `POST /embeddings` route is disabled by default. Set `route: true` to use the default path, or pass a path and body limit. `healthCheck()` returns an `ok` result with model, dimensions, and duration, or a structured error; it does not throw.

<!-- docs-snippet: route-health.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, {
  provider: mockProvider({ dimensions: 8 }),
  route: { path: '/api/embeddings', bodyLimit: 16_384 },
});
await app.ready();
const health = await app.embeddings.healthCheck();
if (!health.ok) throw health.error;
const response = await app.inject({
  method: 'POST',
  url: '/api/embeddings',
  payload: { input: 'example text', inputType: 'query' },
});
void response;
await app.close();
```

The route accepts `{ input, inputType? }` and returns the normal `EmbedResult`. Fastify enforces `bodyLimit` before the handler runs.

## Presets

Presets are defaults, not opaque behavior: each supplies a model name and optional query/document prefixes. The HTTP and local built-in providers apply the matching prefix before embedding. Explicit `model` and prefix options override preset values; environment model values also take precedence over preset defaults.

| Preset | Default model | Query prefix | Document prefix |
| --- | --- | --- | --- |
| `nomic` | `nomic-embed-text` | `search_query: ` | `search_document: ` |
| `e5` | `intfloat/e5-base-v2` | `query: ` | `passage: ` |
| `bge` | `BAAI/bge-small-en-v1.5` | `Represent this sentence for searching relevant passages: ` | empty |
| `minilm` | `Xenova/all-MiniLM-L6-v2` | empty | empty |

## Errors

All library errors are `EmbeddingError` instances with stable `code` and `statusCode` values.

| Code | HTTP status | Meaning |
| --- | ---: | --- |
| `EMBEDDING_INPUT_INVALID` | 400 | Invalid or empty input, input type, or transform output |
| `EMBEDDING_PROVIDER_UNREACHABLE` | 503 | Provider could not be reached or local model could not be loaded |
| `EMBEDDING_PROVIDER_ERROR` | 502 | Provider failed or returned malformed data |
| `EMBEDDING_RATE_LIMITED` | 429 | Provider rate limit remains after retries |
| `EMBEDDING_TIMEOUT` | 504 | Provider request timed out |
| `EMBEDDING_DIMENSION_MISMATCH` | 500 | A vector's dimensions differed from the expected dimensions |

<!-- docs-snippet: errors.ts -->
```ts
import { createEmbedder, EmbeddingError, mockProvider } from 'fastify-embeddings';

const embedder = createEmbedder({ provider: mockProvider() });
try {
  await embedder.embed('');
} catch (error) {
  if (error instanceof EmbeddingError && error.code === 'EMBEDDING_INPUT_INVALID') {
    process.stderr.write(error.message);
  } else {
    throw error;
  }
}
await embedder.close();
```

## Standalone helpers

- `batch(items, size)` splits items into ordered chunks.
- `withRetry(fn, options)` retries with exponential backoff and jitter, honors provider `Retry-After` delays, and supports a retry predicate and abort signal. Options are `retries`, `baseDelayMs`, `maxDelayMs`, `shouldRetry`, and `signal`.
- `normalize(vector)` returns an L2-normalized copy; a zero vector remains unchanged.
- `truncate(vector, dimensions)` returns a shorter copy and throws if the requested size is invalid or larger than the source.
- `cosineSimilarity(a, b)` returns a value from -1 to 1, returns 0 for zero vectors, and throws `EmbeddingError` with `EMBEDDING_DIMENSION_MISMATCH` when vector lengths differ.

<!-- docs-snippet: helpers.ts -->
```ts
import { cosineSimilarity, normalize, truncate, withRetry } from 'fastify-embeddings';

const similarity = cosineSimilarity(normalize([3, 4]), normalize([3, 4]));
const shortened = truncate([0.1, 0.2, 0.3, 0.4], 2);
const data = await withRetry(async () => ({ ok: true }), {
  retries: 3,
  baseDelayMs: 100,
  maxDelayMs: 2_000,
  shouldRetry: (_error, attempt) => attempt < 2,
});
void similarity;
void shortened;
void data;
```

## Mock provider

The deterministic mock provider is useful for tests, examples, and offline development. It is exported from `fastify-embeddings/testing`.

<!-- docs-snippet: mock-provider.ts -->
<!-- docs-snippet: mock-provider.ts -->
```ts
import Fastify from 'fastify';
import { fastifyEmbeddings } from 'fastify-embeddings';
import { mockProvider } from 'fastify-embeddings/testing';

const app = Fastify();
const mock = mockProvider({ dimensions: 8 });
await app.register(fastifyEmbeddings, { provider: mock });
await app.ready();
await app.embeddings.embed(['Document A', 'Document B']);
process.stdout.write(JSON.stringify(mock.calls));
await app.close();
```

## Runnable examples

Example scripts live in the source repository and run from a clone; they are not included as runnable commands in a package-only install:

```sh
pnpm example:basic
pnpm example:custom-provider
pnpm example:custom-http
```

## License

MIT
