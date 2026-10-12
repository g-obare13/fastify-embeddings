/**
 * Per-worker warm-up. The first Fastify boot lazily initializes several hundred
 * milliseconds of framework internals, and the first test that boots Fastify in
 * a file used to pay that cost inside its own timeout budget. Warming the path
 * here keeps the one-time cost out of the per-test budget so the suite does not
 * depend on how contended or slow the machine is.
 */
import Fastify from 'fastify';
import { fastifyEmbeddings, mockProvider } from 'fastify-embeddings';

const app = Fastify();
await app.register(fastifyEmbeddings, { provider: mockProvider({ dimensions: 4 }) });
await app.ready();
await app.close();
