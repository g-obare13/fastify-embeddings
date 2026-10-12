/**
 * Standalone demo: embed documents with a self-hosted Ollama server and store
 * the vectors in Qdrant.
 *
 * This app is not part of the library build and is not covered by the library
 * test suite. It talks to a shared server, so it only ever touches a collection
 * whose name starts with `fe-test-`, and it only deletes that exact collection
 * when started with `--cleanup`.
 */
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { QdrantClient } from '@qdrant/js-client-rest';
import { config as loadDotenv } from 'dotenv';
import { createEmbedder } from 'fastify-embeddings';

/** Collection names created by this demo must carry this prefix. */
const COLLECTION_PREFIX = 'fe-test-';
const COLLECTION = process.env.QDRANT_COLLECTION ?? `${COLLECTION_PREFIX}ollama-qdrant-demo`;

/** The demo targets a 768 dimension model such as nomic-embed-text. */
const EXPECTED_DIMENSIONS = 768;

const appDir = dirname(fileURLToPath(import.meta.url));
const envFile = join(appDir, '.env');

if (existsSync(envFile)) {
  // dotenv never overwrites variables that are already set, so the shell wins.
  loadDotenv({ path: envFile, quiet: true });
}

const REQUIRED = [
  'EMBEDDINGS_BASE_URL',
  'EMBEDDINGS_FORMAT',
  'EMBEDDINGS_MODEL',
  'QDRANT_URL',
  'QDRANT_API_KEY',
] as const;

const missing = REQUIRED.filter((name) => (process.env[name] ?? '').trim() === '');
if (missing.length > 0) {
  process.stderr.write(
    [
      `Unset environment variable${missing.length === 1 ? '' : 's'}:`,
      ...missing.map((name) => `  - ${name}`),
      '',
      'Copy .env.example to .env in this directory and fill in your own values,',
      'or export the variables in your shell. Shell values take priority.',
      '',
    ].join('\n'),
  );
  process.exit(1);
}

const embeddingsBaseURL = process.env.EMBEDDINGS_BASE_URL ?? '';
const embeddingsFormat = process.env.EMBEDDINGS_FORMAT ?? '';
const embeddingsModel = process.env.EMBEDDINGS_MODEL ?? '';
const qdrantUrl = process.env.QDRANT_URL ?? '';
const qdrantApiKey = process.env.QDRANT_API_KEY ?? '';

if (embeddingsFormat !== 'ollama') {
  process.stderr.write(`This demo targets Ollama, but EMBEDDINGS_FORMAT is "${embeddingsFormat}". Set it to "ollama".\n`);
  process.exit(1);
}

const SAMPLE_DOCUMENTS = [
  'Fastify is a web framework for Node.js focused on low overhead and a plugin based architecture.',
  'Qdrant stores vectors alongside JSON payloads and supports cosine, dot, and euclidean distance.',
  'Cosine similarity ignores vector magnitude, which makes it a good fit for normalized embeddings.',
  'Ollama runs open language models locally and exposes them over a small HTTP API.',
  'Matryoshka embeddings let a single vector be truncated to a shorter length with modest quality loss.',
  'Approximate nearest neighbour indexes trade recall for speed on large collections.',
  'A reranker can improve precision after a first pass of vector retrieval.',
  'Chunking strategy affects retrieval quality more than most embedding model choices.',
  'Keeping the model name and vector size with each stored vector makes later migrations possible.',
  'Hybrid search combines keyword scoring with dense vector similarity.',
];

const secrets = [embeddingsBaseURL, qdrantUrl, qdrantApiKey].filter((value) => value !== '');

/** Strip anything that could identify a server or leak a key before printing. */
function redact(text: string): string {
  let out = text;
  for (const secret of secrets) out = out.split(secret).join('[redacted]');
  out = out.replace(/[a-z][a-z0-9+.-]*:\/\/[^\s"']+/gi, '[redacted-url]');
  return out.replace(/\b\d{1,3}(?:\.\d{1,3}){3}\b/g, '[redacted-host]');
}

/** Describe an address without revealing it. */
function hostClass(rawUrl: string): 'local' | 'private' | 'public' {
  let host: string;
  try {
    host = new URL(rawUrl).hostname.replace(/^\[|\]$/g, '');
  } catch {
    return 'public';
  }
  if (host === 'localhost' || host === '::1' || host === '0.0.0.0' || host.startsWith('127.')) return 'local';
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (v4 !== null) {
    const first = Number(v4[1]);
    const second = Number(v4[2]);
    if (first === 10 || (first === 172 && second >= 16 && second <= 31) || (first === 192 && second === 168) || (first === 169 && second === 254)) {
      return 'private';
    }
    return 'public';
  }
  if (/^(fc|fd|fe80)/i.test(host)) return 'private';
  return 'public';
}

/** Read a vector size out of a Qdrant collection description, if it is present. */
function recordedDimensions(description: unknown): number | undefined {
  const config = (description as { config?: { params?: { vectors?: unknown } } }).config;
  const vectors = config?.params?.vectors;
  if (typeof vectors !== 'object' || vectors === null || !('size' in vectors)) return undefined;
  const size = (vectors as { size?: unknown }).size;
  return typeof size === 'number' ? size : undefined;
}

const client = new QdrantClient({ url: qdrantUrl, apiKey: qdrantApiKey });
const cleanupOnly = process.argv.includes('--cleanup');

async function collectionExists(name: string): Promise<boolean> {
  const result = (await client.collectionExists(name)) as unknown as boolean | { exists?: boolean };
  return typeof result === 'boolean' ? result : result.exists === true;
}

async function main(): Promise<void> {
  if (!COLLECTION.startsWith(COLLECTION_PREFIX)) {
    throw new Error(
      `Refusing to run: the collection name must start with "${COLLECTION_PREFIX}" but is "${COLLECTION}". This guard keeps the demo away from other projects' collections.`,
    );
  }

  process.stdout.write(`Collection: ${COLLECTION}\n`);
  process.stdout.write(`Ollama host class: ${hostClass(embeddingsBaseURL)}, Qdrant host class: ${hostClass(qdrantUrl)}\n`);

  if (cleanupOnly) {
    if (await collectionExists(COLLECTION)) {
      await client.deleteCollection(COLLECTION);
      process.stdout.write(`Deleted the collection this demo created: ${COLLECTION}\n`);
    } else {
      process.stdout.write(`Nothing to delete: ${COLLECTION} does not exist.\n`);
    }
    return;
  }

  const embedder = createEmbedder({
    baseURL: embeddingsBaseURL,
    format: 'ollama',
    model: embeddingsModel,
    preset: 'nomic',
    batchSize: 8,
    concurrency: 1,
    timeoutMs: 120_000,
  });

  try {
    const health = await embedder.healthCheck();
    if (!health.ok) {
      throw new Error(`The embeddings server is not usable: ${health.error.code}`);
    }
    if (health.dimensions !== EXPECTED_DIMENSIONS) {
      throw new Error(
        `This demo expects a ${String(EXPECTED_DIMENSIONS)} dimension model, but the server reported ${String(health.dimensions)}.`,
      );
    }
    process.stdout.write(`Embeddings model "${health.model}" reports ${String(health.dimensions)} dimensions.\n`);

    if (await collectionExists(COLLECTION)) {
      throw new Error(
        `Refusing to run: collection "${COLLECTION}" already exists. Inspect it, then run "pnpm cleanup" to delete only that collection.`,
      );
    }

    const documents = await embedder.embed(SAMPLE_DOCUMENTS);
    const { model, dimensions, vectors } = documents;

    await client.createCollection(COLLECTION, { vectors: { size: dimensions, distance: 'Cosine' } });

    const description = await client.getCollection(COLLECTION);
    const recorded = recordedDimensions(description);
    if (recorded !== dimensions) {
      throw new Error(
        `Refusing to upsert: collection "${COLLECTION}" records ${String(recorded)} dimensions but the embedder produced ${String(dimensions)}.`,
      );
    }

    const points = SAMPLE_DOCUMENTS.map((text, index) => {
      const vector = vectors[index];
      if (vector === undefined) throw new Error(`Missing vector for sample document ${String(index)}`);
      return {
        id: index + 1,
        vector,
        payload: { text, model, dimensions },
      };
    });

    await client.upsert(COLLECTION, { wait: true, points });
    process.stdout.write(`Upserted ${String(points.length)} documents with model and dimensions in the payload.\n`);

    const query = await embedder.embed('How should I split documents before embedding them?', { inputType: 'query' });
    const queryVector = query.vectors[0];
    if (queryVector === undefined) throw new Error('The query produced no vector');

    const found = await client.query(COLLECTION, { query: queryVector, limit: 3, with_payload: true });
    process.stdout.write('\nTop matches:\n');
    for (const hit of found.points) {
      const payload = hit.payload as { text?: string; model?: string; dimensions?: number } | null;
      const text = payload?.text ?? '(no text)';
      const preview = text.length > 72 ? `${text.slice(0, 72)}...` : text;
      const modelName = String(payload?.model ?? 'unknown');
      const dims = String(payload?.dimensions ?? 'unknown');
      process.stdout.write(`  ${String(hit.score.toFixed(4))}  ${preview}  [${modelName}, ${dims}d]\n`);
    }

    process.stdout.write(`\nThe collection was left in place. Run "pnpm cleanup" to delete only "${COLLECTION}".\n`);
  } finally {
    await embedder.close();
  }
}

main().catch((error: unknown) => {
  process.stderr.write(`${redact(error instanceof Error ? error.message : String(error))}\n`);
  process.exit(1);
});
