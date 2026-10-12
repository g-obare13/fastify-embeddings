/**
 * The mock provider: deterministic, hash-based vectors with no network and no
 * model download. Each word maps to a fixed pseudo-random vector and a text's
 * vector is the sum of its words, so texts that share words score higher.
 * Also exported from `fastify-embeddings/testing`.
 */
import type { EmbeddingProvider, InputType, Vector } from '#src/types.js';

/** Options for `mockProvider`. */
export interface MockProviderOptions {
  /** Vector size. Defaults to 384. */
  dimensions?: number;
  /** Reported model id. Defaults to `'mock-embedding'`. */
  model?: string;
  /** Artificial latency per batch, in milliseconds. Defaults to 0. */
  latencyMs?: number;
}

/** One recorded batch call. Holds counts only, never text. */
export interface MockCall {
  /** Texts in the batch. */
  count: number;
  /** Input type of the batch. */
  inputType: InputType;
}

/** A mock provider that also records its calls. */
export interface MockProvider extends EmbeddingProvider {
  /** Vector size produced by this provider. */
  readonly dimensions: number;
  /** Every batch call made so far, in order. */
  readonly calls: MockCall[];
  /** True after `close()` has been called. */
  readonly closed: boolean;
}

/** 32-bit FNV-1a hash. */
function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** Deterministic vector for one token (mulberry32 PRNG seeded by the hash). */
function tokenVector(token: string, dimensions: number): Vector {
  let seed = fnv1a(token);
  const out = new Array<number>(dimensions);
  for (let i = 0; i < dimensions; i++) {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

/** Split text into lowercase word tokens. */
function tokenize(text: string): string[] {
  const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu);
  return words ?? [text];
}

/**
 * Deterministic vector for a text, independent of input type.
 *
 * @example
 * ```ts
 * mockVector('hello world', 8).length; // 8
 * ```
 */
export function mockVector(text: string, dimensions = 384): Vector {
  const sum = new Array<number>(dimensions).fill(0);
  for (const token of tokenize(text)) {
    const v = tokenVector(token, dimensions);
    for (let i = 0; i < dimensions; i++) sum[i] = (sum[i] ?? 0) + (v[i] ?? 0);
  }
  return sum;
}

/**
 * Create a mock provider for tests, demos, and offline development.
 *
 * @example
 * ```ts
 * import { mockProvider } from 'fastify-embeddings/testing';
 *
 * const provider = mockProvider({ dimensions: 8 });
 * await app.register(embeddings, { provider });
 * await app.embeddings.embed(['a', 'b']);
 * provider.calls; // [{ count: 2, inputType: 'document' }]
 * ```
 */
export function mockProvider(options: MockProviderOptions = {}): MockProvider {
  const dimensions = options.dimensions ?? 384;
  const latencyMs = options.latencyMs ?? 0;
  const calls: MockCall[] = [];
  let closed = false;

  return {
    name: 'mock',
    model: options.model ?? 'mock-embedding',
    dimensions,
    calls,
    get closed() {
      return closed;
    },
    async embed(texts, inputType) {
      calls.push({ count: texts.length, inputType });
      if (latencyMs > 0) await new Promise((resolve) => setTimeout(resolve, latencyMs));
      const tokens = texts.reduce((total, text) => total + tokenize(text).length, 0);
      return {
        vectors: texts.map((text) => mockVector(text, dimensions)),
        usage: { promptTokens: tokens, totalTokens: tokens },
      };
    },
    close() {
      closed = true;
    },
  };
}
