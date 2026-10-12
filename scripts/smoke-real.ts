/**
 * Real server smoke test for a self-hosted Ollama instance.
 *
 * Run `pnpm smoke:real` after copying `.env.example` to `.env`, or with the
 * `EMBEDDINGS_*` variables already exported in the shell. Shell values always
 * win over `.env` values.
 *
 * The server may be shared with other projects, so this script only ever calls
 * the embedding endpoint. It never pulls or deletes models, never calls an
 * admin endpoint, uses batch size 8 with concurrency 1, and prints a host class
 * rather than the server address.
 */
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { config as loadDotenv } from 'dotenv';
import { createEmbedder, EmbeddingError } from 'fastify-embeddings';
import type { EmbeddingsLogger } from 'fastify-embeddings';

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const envFile = join(rootDir, '.env');

if (existsSync(envFile)) {
  // dotenv never overwrites variables that are already set, so the shell wins.
  loadDotenv({ path: envFile, quiet: true });
}

const REQUIRED = ['EMBEDDINGS_BASE_URL', 'EMBEDDINGS_FORMAT', 'EMBEDDINGS_MODEL'] as const;

const missing = REQUIRED.filter((name) => (process.env[name] ?? '').trim() === '');
if (missing.length > 0) {
  process.stderr.write(
    [
      `smoke:real cannot start. Unset environment variable${missing.length === 1 ? '' : 's'}:`,
      ...missing.map((name) => `  - ${name}`),
      '',
      'Copy .env.example to .env and fill in your own server details, or export',
      'the variables in your shell. Shell values take priority over .env.',
      '',
    ].join('\n'),
  );
  process.exit(1);
}

const baseURL = process.env.EMBEDDINGS_BASE_URL ?? '';
const format = process.env.EMBEDDINGS_FORMAT ?? '';
const model = process.env.EMBEDDINGS_MODEL ?? '';
const apiKey = process.env.EMBEDDINGS_API_KEY ?? '';

if (format !== 'ollama') {
  process.stderr.write(`smoke:real targets Ollama, but EMBEDDINGS_FORMAT is "${format}". Set it to "ollama".\n`);
  process.exit(1);
}

const DIMENSIONS = 768;
const TIMEOUT_MS = 120_000;

const secrets = [baseURL, apiKey].filter((value) => value !== '');

/** Strip anything that could identify the server before printing. */
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

interface CheckResult {
  name: string;
  ok: boolean;
}

const results: CheckResult[] = [];

function describe(error: unknown): string {
  if (error instanceof EmbeddingError) return `${error.code}: ${error.message}`;
  if (error instanceof Error) return error.message;
  return String(error);
}

async function run(name: string, body: () => Promise<string>): Promise<void> {
  try {
    const detail = await body();
    results.push({ name, ok: true });
    process.stdout.write(`PASS  ${name}\n      ${redact(detail)}\n`);
  } catch (error) {
    results.push({ name, ok: false });
    process.stdout.write(`FAIL  ${name}\n      ${redact(describe(error))}\n`);
  }
}

function maxDelta(a: readonly number[], b: readonly number[]): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  }
  return max;
}

function isClose(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && maxDelta(a, b) <= 1e-9;
}

/** Ollama reports canonical ids such as `nomic-embed-text:latest`. */
function sameModel(reported: string, configured: string): boolean {
  const base = (value: string): string => value.split(':')[0] ?? value;
  return reported === configured || base(reported) === base(configured);
}

async function main(): Promise<void> {
  process.stdout.write(`Target: ollama host class = ${hostClass(baseURL)}\n`);
  process.stdout.write(`Model: ${model}, batch size 8, concurrency 1, timeout ${String(TIMEOUT_MS)}ms\n\n`);

  const embedder = createEmbedder({
    baseURL,
    format: 'ollama',
    model,
    preset: 'nomic',
    batchSize: 8,
    concurrency: 1,
    timeoutMs: TIMEOUT_MS,
  });

  try {
    await run('healthCheck succeeds with 768 dimensions and the configured model', async () => {
      const health = await embedder.healthCheck();
      if (!health.ok) throw new Error(`healthCheck failed with ${health.error.code}`);
      if (health.dimensions !== DIMENSIONS) {
        throw new Error(`expected ${String(DIMENSIONS)} dimensions, received ${String(health.dimensions)}`);
      }
      if (!sameModel(health.model, model)) {
        throw new Error(`expected model "${model}", server reported "${health.model}"`);
      }
      return `${String(health.dimensions)} dimensions from "${health.model}" in ${String(health.durationMs)}ms`;
    });

    await run('query and document inputs give different vectors with the nomic preset', async () => {
      const text = 'Vector databases index embeddings for similarity search.';
      const asQuery = await embedder.embed(text, { inputType: 'query' });
      const asDocument = await embedder.embed(text, { inputType: 'document' });
      const query = asQuery.vectors[0];
      const document = asDocument.vectors[0];
      if (query === undefined || document === undefined) throw new Error('embed returned no vectors');
      const delta = maxDelta(query, document);
      if (delta < 1e-6) {
        throw new Error(`expected different vectors, largest difference was ${delta.toExponential(2)}`);
      }
      return `largest per dimension difference ${delta.toExponential(2)}`;
    });

    await run('similar sentences score higher than unrelated ones', async () => {
      const reference = await embedder.embed('The canine barked loudly in the yard.', { inputType: 'document' });
      const similar = await embedder.embed('A dog was making loud barking sounds outside.', { inputType: 'query' });
      const unrelated = await embedder.embed('Quantum computing relies on superposition and entanglement.', { inputType: 'query' });
      const ref = reference.vectors[0];
      const near = similar.vectors[0];
      const far = unrelated.vectors[0];
      if (ref === undefined || near === undefined || far === undefined) throw new Error('embed returned no vectors');
      const nearScore = embedder.cosineSimilarity(ref, near);
      const farScore = embedder.cosineSimilarity(ref, far);
      if (!(nearScore > farScore)) {
        throw new Error(`similar scored ${nearScore.toFixed(4)}, unrelated scored ${farScore.toFixed(4)}`);
      }
      return `similar ${nearScore.toFixed(4)} > unrelated ${farScore.toFixed(4)}`;
    });

    await run('50 texts with batchSize 8 and concurrency 1 return in order', async () => {
      const topics = ['erosion', 'compilers', 'tides', 'yeast', 'orbits'];
      const texts = Array.from(
        { length: 50 },
        (_unused, index) => `Sample document ${String(index + 1)} discusses ${topics[index % topics.length] ?? 'science'} in depth.`,
      );
      const batchSizes: number[] = [];
      const ordered = createEmbedder({
        baseURL,
        format: 'ollama',
        model,
        preset: 'nomic',
        batchSize: 8,
        concurrency: 1,
        timeoutMs: TIMEOUT_MS,
        onBatch: (info) => {
          batchSizes.push(info.count);
        },
      });
      try {
        const result = await ordered.embed(texts);
        if (result.vectors.length !== 50) {
          throw new Error(`expected 50 vectors, received ${String(result.vectors.length)}`);
        }
        const wrongSize = result.vectors.find((vector) => vector.length !== DIMENSIONS);
        if (wrongSize !== undefined) {
          throw new Error(`expected ${String(DIMENSIONS)} dimensions, received ${String(wrongSize.length)}`);
        }
        const expected = '8,8,8,8,8,8,2';
        if (batchSizes.join(',') !== expected) {
          throw new Error(`expected batch sizes ${expected}, received ${batchSizes.join(',')}`);
        }
        const probes = [0, 17, 49];
        const probed = await embedder.embed(probes.map((index) => texts[index] ?? ''));
        probes.forEach((index, position) => {
          const fromBatch = result.vectors[index];
          const fromProbe = probed.vectors[position];
          if (fromBatch === undefined || fromProbe === undefined) {
            throw new Error(`missing vector at index ${String(index)}`);
          }
          if (!isClose(fromBatch, fromProbe)) {
            throw new Error(`vector ${String(index)} does not match that text embedded alone, so order may not be preserved`);
          }
        });
        return `50 vectors in ${batchSizes.length} batches (${batchSizes.join('/')}), spot checked ${probes.join(', ')}`;
      } finally {
        await ordered.close();
      }
    });

    await run('a dead local port returns EMBEDDING_PROVIDER_UNREACHABLE quickly', async () => {
      const dead = createEmbedder({
        baseURL: 'http://127.0.0.1:1',
        format: 'ollama',
        model,
        batchSize: 8,
        concurrency: 1,
        timeoutMs: 2_000,
        retries: 0,
      });
      try {
        const started = Date.now();
        const health = await dead.healthCheck();
        const elapsed = Date.now() - started;
        if (health.ok) throw new Error('expected a dead port to fail, but the probe succeeded');
        if (health.error.code !== 'EMBEDDING_PROVIDER_UNREACHABLE') {
          throw new Error(`expected EMBEDDING_PROVIDER_UNREACHABLE, received ${health.error.code}`);
        }
        if (elapsed > 5_000) throw new Error(`took ${String(elapsed)}ms, expected a fast failure`);
        return `${health.error.code} in ${String(elapsed)}ms against a local dead port`;
      } finally {
        await dead.close();
      }
    });

    await run('an unknown model against the real server returns EMBEDDING_PROVIDER_ERROR', async () => {
      const wrong = createEmbedder({
        baseURL,
        format: 'ollama',
        model: `${model}-does-not-exist`,
        batchSize: 8,
        concurrency: 1,
        timeoutMs: 30_000,
        retries: 0,
      });
      try {
        const health = await wrong.healthCheck();
        if (health.ok) throw new Error('expected an unknown model to fail');
        if (health.error.code !== 'EMBEDDING_PROVIDER_ERROR') {
          throw new Error(`expected EMBEDDING_PROVIDER_ERROR, received ${health.error.code}`);
        }
        return `${health.error.code} with upstream status ${String(health.error.upstreamStatus ?? 'unknown')}`;
      } finally {
        await wrong.close();
      }
    });

    await run('captured logs contain no input text, no vectors, and no API key', async () => {
      const captured: string[] = [];
      const logger: EmbeddingsLogger = {
        debug: (details, message) => {
          captured.push(JSON.stringify([details, message]));
        },
        warn: (details, message) => {
          captured.push(JSON.stringify([details, message]));
        },
      };
      const sentinel = `smoke-sentinel-${randomUUID()}`;
      const noisy = createEmbedder({
        baseURL,
        format: 'ollama',
        model,
        preset: 'nomic',
        batchSize: 8,
        concurrency: 1,
        timeoutMs: TIMEOUT_MS,
        retries: 0,
        logger,
      });
      const vectorProbes: string[] = [];
      try {
        const result = await noisy.embed(sentinel);
        vectorProbes.push(String(result.vectors[0]?.[0] ?? ''));
      } finally {
        await noisy.close();
      }
      const failing = createEmbedder({
        baseURL: 'http://127.0.0.1:1',
        format: 'ollama',
        model,
        batchSize: 8,
        concurrency: 1,
        timeoutMs: 2_000,
        retries: 0,
        logger,
      });
      try {
        await failing.healthCheck();
      } finally {
        await failing.close();
      }
      const blob = captured.join('\n');
      if (blob === '') throw new Error('expected the library to log at least one entry');
      if (blob.includes(sentinel)) throw new Error('logs contain the input text');
      if (apiKey !== '' && blob.includes(apiKey)) throw new Error('logs contain the API key');
      if (/https?:\/\//i.test(blob)) throw new Error('logs contain a URL');
      const firstComponent = vectorProbes[0];
      if (firstComponent !== undefined && firstComponent !== '' && blob.includes(firstComponent)) {
        throw new Error('logs contain a vector value');
      }
      return `${String(captured.length)} log entries captured and inspected`;
    });
  } finally {
    await embedder.close();
  }

  const failed = results.filter((result) => !result.ok).length;
  process.stdout.write(`\n${String(results.length - failed)}/${String(results.length)} checks passed\n`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((error: unknown) => {
  process.stderr.write(`smoke:real failed unexpectedly: ${redact(describe(error))}\n`);
  process.exit(1);
});
