/**
 * Optional HTTP route for Fastify: exposes `POST /embeddings` (or custom path)
 * backed by the registered embedder. Validated with Zod, enforces a body size
 * limit, and returns the standard EmbedResult shape.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { EmbeddingError } from './errors.js';
import type { Embedder } from './embedder.js';
import type { EmbedResult, InputType, RouteOptions } from './types.js';

export type { RouteOptions };

const inputTypeSchema = z.enum(['query', 'document']).optional();

const routeBodySchema = z.object({
  input: z.union([
    z.string().min(1, 'input string cannot be empty'),
    z.array(z.string().min(1, 'input element cannot be empty')).min(1, 'input array cannot be empty'),
  ]),
  inputType: inputTypeSchema,
});

/**
 * Register the embeddings POST endpoint on a Fastify instance.
 *
 * @example
 * ```ts
 * await registerRoute(fastify, embedder, { path: '/v1/embeddings' });
 * ```
 */
export const registerRoute = async (
  fastify: FastifyInstance,
  embedder: Embedder,
  options: RouteOptions = {},
): Promise<void> => {
  const path = options.path ?? '/embeddings';
  const bodyLimit = options.bodyLimit ?? 1_048_576;

  fastify.post<{ Body: unknown; Reply: EmbedResult }>(
    path,
    { bodyLimit },
    async (request, reply) => {
      const parsed = routeBodySchema.safeParse(request.body);
      if (!parsed.success) {
        const issue = parsed.error.issues[0];
        const msg = issue ? `${issue.path.join('.')}: ${issue.message}` : 'Invalid request body';
        throw new EmbeddingError('EMBEDDING_INPUT_INVALID', msg);
      }

      const { input, inputType } = parsed.data;
      const result = await embedder.embed(input, {
        inputType: inputType as InputType | undefined,
      });

      return reply.code(200).send(result);
    },
  );
};
