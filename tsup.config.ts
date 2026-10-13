/**
 * Build configuration: bundles the main and testing entries into ESM and CJS
 * with type declarations. The `#src` alias is resolved at bundle time, so no
 * alias survives into `dist` (checked by `scripts/verify-dist.mjs`).
 */
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    testing: 'src/testing.ts',
    otel: 'src/otel.ts',
    documents: 'src/documents/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  target: 'node20',
  splitting: false,
  treeshake: true,
  external: ['fastify', 'fastify-plugin', 'zod', '@huggingface/transformers', '@opentelemetry/api'],
});
