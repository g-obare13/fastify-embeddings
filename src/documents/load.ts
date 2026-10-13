/**
 * Reading documents from a path, a buffer, or literal text. Path reading is the
 * only part that needs the file system, and it is imported lazily so the buffer
 * and text sources keep working on runtimes without `node:fs`.
 */
import { EmbeddingError } from '#src/errors.js';
import type * as NodeFs from 'node:fs/promises';
import { extensionOf, extractorByName, extractorFor, supportedExtensions } from './extract.js';
import type { ExtractResult, Extractor } from './extract.js';

/**
 * Where a document comes from.
 *
 * A path is an arbitrary file read, so never build one from untrusted input,
 * such as a request body. `buffer` carries its own filename so the format can
 * be determined without guessing.
 */
export type DocumentSource =
  | { path: string }
  | { buffer: Uint8Array; filename: string }
  | { text: string; source?: string };

/** A document that has been read and converted to text. */
export interface LoadedDocument {
  /** Extracted plain text. */
  text: string;
  /** Path or filename the text came from, when known. */
  source?: string;
  /** Name of the extractor that produced the text, when one was used. */
  extractor?: string;
  /** Metadata the extractor reported alongside the text. */
  metadata?: Record<string, unknown>;
}

/** Options for `loadDocument`. */
export interface LoadDocumentOptions {
  /** Refuse documents larger than this many bytes. Defaults to 10485760 (10 MB). */
  maxBytes?: number;
  /** Use a specific extractor, by name or instance, instead of choosing by extension. */
  extractor?: string | Extractor;
  /** Cancels reading and extraction. */
  signal?: AbortSignal;
}

const DEFAULT_MAX_BYTES = 10 * 1024 * 1024;

/** The final path segment of a path, handling both separators. */
function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] ?? path;
}

/** Resolve which extractor should handle a filename. */
function selectExtractor(options: LoadDocumentOptions, filename: string): Extractor {
  const requested = options.extractor;
  if (typeof requested === 'object') return requested;
  if (typeof requested === 'string') {
    const named = extractorByName(requested);
    if (named === undefined) {
      throw new EmbeddingError('EMBEDDING_DOCUMENT_UNSUPPORTED', `No extractor named "${requested}" is registered`);
    }
    return named;
  }
  const chosen = extractorFor(filename);
  if (chosen === undefined) {
    const extension = extensionOf(filename) === '' ? 'a file without an extension' : `"${extensionOf(filename)}" files`;
    throw new EmbeddingError(
      'EMBEDDING_DOCUMENT_UNSUPPORTED',
      `No extractor handles ${extension}. Register one with registerExtractor, or name one with the extractor option. Supported extensions: ${supportedExtensions().join(', ')}`,
    );
  }
  return chosen;
}

/** Run an extractor and normalise both its result and its failures. */
async function runExtractor(
  extractor: Extractor,
  input: { buffer: Uint8Array; filename: string; signal?: AbortSignal },
): Promise<{ text: string; metadata?: Record<string, unknown> }> {
  let result: ExtractResult;
  try {
    result = await extractor.extract(input);
  } catch (error) {
    if (error instanceof EmbeddingError) throw error;
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `Extractor "${extractor.name}" failed to read ${input.filename}`, {
      cause: error,
      retryable: false,
    });
  }
  if (typeof result === 'string') return { text: result };
  if (typeof result.text !== 'string') {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `Extractor "${extractor.name}" returned no text`);
  }
  return { text: result.text, ...(result.metadata === undefined ? {} : { metadata: result.metadata }) };
}

/** Read a document from disk through `node:fs`, loaded lazily. */
async function loadFromPath(path: string, maxBytes: number, options: LoadDocumentOptions): Promise<LoadedDocument> {
  let fs: typeof NodeFs;
  try {
    fs = await import('node:fs/promises');
  } catch (error) {
    throw new EmbeddingError(
      'EMBEDDING_DOCUMENT_INVALID',
      'Reading a document from a path needs Node.js file system access. Pass a buffer or text instead',
      { cause: error, retryable: false },
    );
  }

  const filename = baseName(path);
  const extractor = selectExtractor(options, filename);

  const info = await fs.stat(path);
  if (info.size > maxBytes) {
    throw new EmbeddingError(
      'EMBEDDING_DOCUMENT_TOO_LARGE',
      `${filename} is ${String(info.size)} bytes, which exceeds the ${String(maxBytes)} byte limit`,
    );
  }

  const buffer = await fs.readFile(path, options.signal === undefined ? undefined : { signal: options.signal });
  const extracted = await runExtractor(extractor, {
    buffer,
    filename,
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
  return {
    text: extracted.text,
    source: path,
    extractor: extractor.name,
    ...(extracted.metadata === undefined ? {} : { metadata: extracted.metadata }),
  };
}

/**
 * Read one document and return its text.
 *
 * @example
 * ```ts
 * import { loadDocument } from 'fastify-embeddings/documents';
 *
 * const document = await loadDocument({ path: 'notes/handbook.md' });
 * const fromMemory = await loadDocument({ buffer: bytes, filename: 'handbook.md' });
 * const alreadyHave = await loadDocument({ text: 'plain text', source: 'inline' });
 * ```
 */
export async function loadDocument(source: DocumentSource, options: LoadDocumentOptions = {}): Promise<LoadedDocument> {
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
  if (!Number.isInteger(maxBytes) || maxBytes < 1) {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `maxBytes must be a positive integer, received ${String(maxBytes)}`);
  }

  if ('text' in source) {
    if (typeof source.text !== 'string') {
      throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', 'source.text must be a string');
    }
    return { text: source.text, ...(source.source === undefined ? {} : { source: source.source }) };
  }

  if ('path' in source) {
    return loadFromPath(source.path, maxBytes, options);
  }

  if (source.buffer.byteLength > maxBytes) {
    throw new EmbeddingError(
      'EMBEDDING_DOCUMENT_TOO_LARGE',
      `${source.filename} is ${String(source.buffer.byteLength)} bytes, which exceeds the ${String(maxBytes)} byte limit`,
    );
  }

  const extractor = selectExtractor(options, source.filename);
  const extracted = await runExtractor(extractor, {
    buffer: source.buffer,
    filename: source.filename,
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
  return {
    text: extracted.text,
    source: source.filename,
    extractor: extractor.name,
    ...(extracted.metadata === undefined ? {} : { metadata: extracted.metadata }),
  };
}
