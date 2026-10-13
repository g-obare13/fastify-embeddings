/**
 * Chunking and embedding in one step. The contract that matters: every chunk is
 * paired with the vector computed from its text, in the same order.
 */
import { EmbeddingError } from '#src/errors.js';
import type { Embedder } from '#src/embedder.js';
import type { InputType, Usage, Vector } from '#src/types.js';
import { chunkText } from './chunk.js';
import type { ChunkOptions, DocumentChunk } from './chunk.js';
import { loadDocument } from './load.js';
import type { DocumentSource, LoadDocumentOptions } from './load.js';

/** Options shared by `chunkDocuments` and `embedDocuments`. */
export interface DocumentOptions extends ChunkOptions, LoadDocumentOptions {
  /** Input type passed to the embedder. Defaults to `'document'`. */
  inputType?: InputType;
}

/** Chunks paired with the vectors computed for them. */
export interface DocumentEmbedResult {
  /** Every chunk, in document order and then chunk order. `chunks[i]` is the text behind `vectors[i]`. */
  chunks: DocumentChunk[];
  /** One vector per chunk, in the same order. */
  vectors: Vector[];
  /** Model that produced the vectors. */
  model: string;
  /** Length of each vector. Falls back to the embedder's configured size, then zero, when there was nothing to embed. */
  dimensions: number;
  /** Summed token usage, when the provider reports it. */
  usage?: Usage;
}

/**
 * Read every source and split each one into chunks, without embedding.
 *
 * @example
 * ```ts
 * import { chunkDocuments } from 'fastify-embeddings/documents';
 *
 * const chunks = await chunkDocuments([{ path: 'handbook.md' }], { chunkSize: 800 });
 * ```
 */
export async function chunkDocuments(sources: readonly DocumentSource[], options: DocumentOptions = {}): Promise<DocumentChunk[]> {
  const chunks: DocumentChunk[] = [];
  for (const source of sources) {
    const document = await loadDocument(source, options);
    for (const chunk of chunkText(document.text, options)) {
      chunks.push({
        ...chunk,
        ...(document.source === undefined ? {} : { source: document.source }),
        ...(document.metadata === undefined ? {} : { metadata: document.metadata }),
      });
    }
  }
  return chunks;
}

/**
 * Read, chunk, and embed a set of documents.
 *
 * The embedder batches and preserves order, so `chunks[i]` is always the text
 * behind `vectors[i]`, across documents. Token usage is summed over every batch.
 *
 * @example
 * ```ts
 * import { createEmbedder } from 'fastify-embeddings';
 * import { embedDocuments } from 'fastify-embeddings/documents';
 *
 * const embedder = createEmbedder({ baseURL: 'http://localhost:11434', format: 'ollama', preset: 'nomic' });
 * const { chunks, vectors, dimensions } = await embedDocuments(
 *   [{ path: 'handbook.md' }, { text: 'inline notes', source: 'notes' }],
 *   embedder,
 *   { chunkSize: 800, chunkOverlap: 100 },
 * );
 * await embedder.close();
 * ```
 */
export async function embedDocuments(
  sources: readonly DocumentSource[],
  embedder: Embedder,
  options: DocumentOptions = {},
): Promise<DocumentEmbedResult> {
  const chunks = await chunkDocuments(sources, options);
  const inputType = options.inputType ?? 'document';

  if (chunks.length === 0) {
    return { chunks, vectors: [], model: embedder.model, dimensions: embedder.dimensions ?? 0 };
  }

  const result = await embedder.embed(
    chunks.map((chunk) => chunk.text),
    { inputType, ...(options.signal === undefined ? {} : { signal: options.signal }) },
  );

  if (result.vectors.length !== chunks.length) {
    throw new EmbeddingError(
      'EMBEDDING_PROVIDER_ERROR',
      `Expected one vector per chunk, but the embedder returned ${String(result.vectors.length)} vectors for ${String(chunks.length)} chunks`,
    );
  }

  return {
    chunks,
    vectors: result.vectors,
    model: result.model,
    dimensions: result.dimensions,
    ...(result.usage === undefined ? {} : { usage: result.usage }),
  };
}
