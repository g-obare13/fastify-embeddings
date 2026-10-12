/**
 * Custom provider example: demonstrates defineProvider and createEmbedder.
 * Run with: `pnpm example:custom-provider`
 */
import Fastify from 'fastify';
import { createEmbedder, defineProvider, fastifyEmbeddings } from 'fastify-embeddings';

// 1. Define a custom provider (e.g. lightweight bag-of-words hash)
const customHashProvider = defineProvider({
  name: 'hash-embedder',
  model: 'hash-v1',
  dimensions: 4,
  async embed(texts, _inputType, _signal) {
    return { vectors: texts.map((text) => {
      // Deterministic 4D vector from character frequencies
      const v = [0, 0, 0, 0];
      for (let i = 0; i < text.length; i++) {
        v[i % 4] = (v[i % 4] ?? 0) + text.charCodeAt(i);
      }
      return v;
    }) };
  },
  close() {
    // Release sockets or memory if necessary
  },
});

async function main(): Promise<void> {
  // A. Standalone usage (workers, scripts, CLI)
  const standalone = createEmbedder({ provider: customHashProvider });
  const standaloneResult = await standalone.embed('Standalone embedding test');
  process.stdout.write(
    `[Standalone] Provider: ${standalone.provider}, Dimensions: ${String(standaloneResult.dimensions)}\n`,
  );
  await standalone.close();

  // B. Fastify plugin usage
  const app = Fastify({ logger: true });
  await app.register(fastifyEmbeddings, { provider: customHashProvider });
  await app.ready();

  const res = await app.embeddings.embed(['Document A', 'Document B']);
  app.log.info(
    { provider: app.embeddings.provider, dimensions: res.dimensions, count: res.vectors.length },
    'Custom provider embedded documents successfully',
  );

  await app.close();
}

main().catch((err: unknown) => {
  process.stderr.write(String(err) + '\n');
  process.exit(1);
});
