/**
 * Fastify 5 plugin wrapper: decorates `app.embeddings`, mounts optional routes,
 * manages resource cleanup on shutdown, supports named embedders, and supports
 * eager readiness probes.
 */
import type { FastifyInstance, FastifyPluginAsync } from 'fastify';
import fp from 'fastify-plugin';
import { createEmbedder } from './embedder.js';
import type { Embedder } from './embedder.js';
import { EmbeddingError } from './errors.js';
import { cosineSimilarity } from '#src/helpers/vector.js';
import { pluginOptionsSchema, validateOptions } from './options.js';
import type { FastifyEmbeddingsOptions } from './options.js';
import { registerRoute } from './route.js';
import type { EmbeddingsLogger } from './types.js';

declare module 'fastify' {
  interface FastifyInstance {
    /**
     * Embeddings instance providing text embedding generation,
     * similarity calculations, and health checks.
     */
    embeddings: EmbeddingsDecorator;
  }
}

/**
 * Fastify instance decorator accessible at `app.embeddings`.
 */
export interface EmbeddingsDecorator extends Embedder {
  /**
   * Retrieve a named embedder registered via the `name` option.
   *
   * @example
   * ```ts
   * const docsEmbedder = app.embeddings.get('docs');
   * ```
   */
  get(name: string): Embedder | undefined;
}

const REGISTRY_SYMBOL = Symbol('fastify-embeddings.registry');
const SET_DEFAULT_SYMBOL = Symbol('fastify-embeddings.setDefault');

interface InternalDecorator extends EmbeddingsDecorator {
  [REGISTRY_SYMBOL]: Map<string, Embedder>;
  [SET_DEFAULT_SYMBOL]: (embedder: Embedder) => void;
}

/**
 * Underlying plugin function registered on the Fastify instance.
 */
const fastifyEmbeddingsAsync: FastifyPluginAsync<FastifyEmbeddingsOptions> = async (
  fastify: FastifyInstance,
  options: FastifyEmbeddingsOptions = {},
): Promise<void> => {
  validateOptions(pluginOptionsSchema, options);

  const logger: EmbeddingsLogger = options.logger ?? {
    debug: (details: object, message: string) => {
      fastify.log.debug(details, message);
    },
    warn: (details: object, message: string) => {
      fastify.log.warn(details, message);
    },
  };

  const embedder = createEmbedder({
    ...options,
    logger,
  });

  fastify.addHook('onClose', async () => {
    await embedder.close();
  });

  if (options.eager) {
    fastify.addHook('onReady', async () => {
      const health = await embedder.healthCheck();
      if (!health.ok) {
        throw health.error;
      }
    });
  }

  if (options.route) {
    const routeOptions = typeof options.route === 'object' ? options.route : {};
    await registerRoute(fastify, embedder, routeOptions);
  }

  if (!fastify.hasDecorator('embeddings')) {
    let defaultEmbedder: Embedder | undefined = options.name === undefined ? embedder : undefined;
    const registry = new Map<string, Embedder>();
    if (options.name !== undefined) {
      registry.set(options.name, embedder);
    }

    const decorator: InternalDecorator = {
      [REGISTRY_SYMBOL]: registry,
      [SET_DEFAULT_SYMBOL]: (newDefault: Embedder) => {
        defaultEmbedder = newDefault;
      },
      async embed(input, embedOptions) {
        if (!defaultEmbedder) {
          throw new EmbeddingError('EMBEDDING_PROVIDER_ERROR', 'No default embedder registered on app.embeddings');
        }
        return defaultEmbedder.embed(input, embedOptions);
      },
      get model() {
        return defaultEmbedder?.model ?? '';
      },
      get dimensions() {
        return defaultEmbedder?.dimensions;
      },
      get provider() {
        return defaultEmbedder?.provider ?? '';
      },
      async healthCheck(healthOptions) {
        if (!defaultEmbedder) {
          throw new EmbeddingError('EMBEDDING_PROVIDER_ERROR', 'No default embedder registered on app.embeddings');
        }
        return defaultEmbedder.healthCheck(healthOptions);
      },
      cosineSimilarity(a, b) {
        if (defaultEmbedder) {
          return defaultEmbedder.cosineSimilarity(a, b);
        }
        return cosineSimilarity(a, b);
      },
      close() {
        if (!defaultEmbedder) {
          return Promise.resolve();
        }
        return defaultEmbedder.close();
      },
      get(name: string) {
        if (name === 'default') {
          return defaultEmbedder;
        }
        return registry.get(name);
      },
    };

    fastify.decorate('embeddings', decorator);
  } else {
    const existing = fastify.embeddings as unknown as InternalDecorator;
    if (options.name !== undefined) {
      existing[REGISTRY_SYMBOL].set(options.name, embedder);
    } else {
      existing[SET_DEFAULT_SYMBOL](embedder);
    }
  }
};

/**
 * Fastify plugin for text embeddings generation and similarity calculation.
 *
 * @example
 * ```ts
 * import { fastifyEmbeddings } from 'fastify-embeddings';
 *
 * await app.register(fastifyEmbeddings, {
 *   baseURL: 'http://localhost:11434',
 *   format: 'ollama',
 *   preset: 'nomic',
 * });
 *
 * const { vectors } = await app.embeddings.embed('sample text');
 * ```
 */
export const fastifyEmbeddings = fp(fastifyEmbeddingsAsync, {
  fastify: '^5.0.0',
  name: 'fastify-embeddings',
});

export default fastifyEmbeddings;
