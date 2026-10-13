/**
 * Extractor registry. Formats are pluggable so this library can stay free of
 * parser dependencies: plain text is built in, and everything else is yours to
 * register. Registering an extractor for, say, `.pdf` makes PDF sources work
 * everywhere a source is accepted.
 */
import { EmbeddingError } from '#src/errors.js';

/** Bytes handed to an extractor, with the information it needs to read them. */
export interface ExtractInput {
  /** File contents. */
  buffer: Uint8Array;
  /** Original filename, used to choose and to report. */
  filename: string;
  /** Cancels a long extraction. */
  signal?: AbortSignal;
}

/** What an extractor returns: text, or text plus metadata such as page numbers. */
export type ExtractResult = string | { text: string; metadata?: Record<string, unknown> };

/** Turns file bytes into text. */
export interface Extractor {
  /** Identifier used in results and error messages, for example `'text'`. */
  name: string;
  /** Lowercase extensions this handles, each including the dot, for example `['.md']`. */
  extensions: readonly string[];
  /** Return the document text, optionally with metadata. */
  extract(input: ExtractInput): ExtractResult | Promise<ExtractResult>;
}

/** Extensions handled by the built-in text extractor. */
export const TEXT_EXTENSIONS: readonly string[] = [
  '.txt',
  '.md',
  '.markdown',
  '.json',
  '.jsonl',
  '.csv',
  '.tsv',
  '.log',
  '.yaml',
  '.yml',
];

const byExtension = new Map<string, Extractor>();
const byName = new Map<string, Extractor>();

/** Lowercase the extension of a filename, including the dot, or `''`. */
export function extensionOf(filename: string): string {
  const at = filename.lastIndexOf('.');
  return at === -1 ? '' : filename.slice(at).toLowerCase();
}

/**
 * Register an extractor, replacing any previous one for the same extensions.
 *
 * Registered extractors are global to the process and take effect for every
 * later `loadDocument` call.
 *
 * @example
 * ```ts
 * registerExtractor({
 *   name: 'pdf',
 *   extensions: ['.pdf'],
 *   async extract({ buffer }) {
 *     return extractTextWithYourPdfLibrary(buffer);
 *   },
 * });
 * ```
 */
export function registerExtractor(extractor: Extractor): void {
  if (extractor.name === '') {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', 'An extractor must have a name');
  }
  if (extractor.extensions.length === 0) {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `Extractor "${extractor.name}" must declare at least one extension`);
  }
  byName.set(extractor.name, extractor);
  for (const extension of extractor.extensions) {
    byExtension.set(extension.toLowerCase(), extractor);
  }
}

/** The extractor registered for a filename, or `undefined`. */
export function extractorFor(filename: string): Extractor | undefined {
  return byExtension.get(extensionOf(filename));
}

/** The extractor registered under a name, or `undefined`. */
export function extractorByName(name: string): Extractor | undefined {
  return byName.get(name);
}

/** Every extension that currently has an extractor, sorted. */
export function supportedExtensions(): string[] {
  return [...byExtension.keys()].sort();
}

/**
 * Built-in extractor for text formats. Decodes UTF-8, strips a byte order mark,
 * and refuses content that looks binary rather than decoding it into noise.
 */
export const textExtractor: Extractor = {
  name: 'text',
  extensions: TEXT_EXTENSIONS,
  extract({ buffer }: ExtractInput): string {
    const decoded = new TextDecoder('utf-8').decode(buffer);
    const text = decoded.charCodeAt(0) === 0xfeff ? decoded.slice(1) : decoded;
    if (text.includes('\u0000')) {
      throw new EmbeddingError(
        'EMBEDDING_DOCUMENT_INVALID',
        'This file looks binary. Register an extractor for its format with registerExtractor instead of reading it as text',
      );
    }
    return text;
  },
};

registerExtractor(textExtractor);
