/**
 * Test configuration: maps the `#src` alias to source files and enforces the
 * 85 percent coverage threshold across the library source.
 */
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const src = fileURLToPath(new URL('./src/', import.meta.url));

export default defineConfig({
  resolve: {
    alias: [
      { find: /^#src\/(.*)\.js$/, replacement: `${src}$1.ts` },
      { find: /^fastify-embeddings\/otel$/, replacement: `${src}otel.ts` },
      { find: /^fastify-embeddings\/documents$/, replacement: `${src}documents/index.ts` },
      { find: /^fastify-embeddings\/testing$/, replacement: `${src}testing.ts` },
      { find: /^fastify-embeddings$/, replacement: `${src}index.ts` },
    ],
  },
  test: {
    include: ['test/**/*.test.ts'],
    setupFiles: ['./test/setup.ts'],
    environment: 'node',
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/types.ts', 'src/index.ts', 'src/testing.ts'],
      reporter: ['text', 'html'],
      thresholds: {
        lines: 85,
        functions: 85,
        branches: 85,
        statements: 85,
      },
    },
  },
});
