/**
 * Text chunking. Pure string work with exact character offsets, so every chunk
 * points back at the span of the document it came from.
 */
import { EmbeddingError } from '#src/errors.js';

/** A piece of a document, sized for embedding. */
export interface DocumentChunk {
  /** The chunk text, ready to embed. */
  text: string;
  /** Zero-based position of this chunk within its document. */
  index: number;
  /** Character offset where the chunk starts in the source text. */
  start: number;
  /** Character offset where the chunk ends in the source text. */
  end: number;
  /** Identifier of the document this chunk came from, when known. */
  source?: string;
  /** Metadata copied from the source document. */
  metadata?: Record<string, unknown>;
}

/** How to split a document into chunks. */
export interface ChunkOptions {
  /**
   * Largest chunk, in units of `measure`. Defaults to 1000, which with the
   * default measure means 1000 characters.
   */
  chunkSize?: number;
  /** How much text neighbouring chunks share, in units of `measure`. Defaults to 200. */
  chunkOverlap?: number;
  /**
   * Preferred split points, most preferred first. A chunk grows to `chunkSize`
   * and then backs up to the last preferred boundary inside that window, so
   * chunks tend to end on a paragraph or sentence rather than mid-word.
   * Defaults to paragraph, line, sentence, then word boundaries.
   */
  separators?: readonly string[];
  /**
   * Measure a piece of text. Defaults to its character count. Supply a token
   * counter to size chunks in tokens instead, which is the accurate way to keep
   * a chunk inside a model's context window.
   */
  measure?: (text: string) => number;
}

const DEFAULT_CHUNK_SIZE = 1000;
const DEFAULT_CHUNK_OVERLAP = 200;
const DEFAULT_SEPARATORS: readonly string[] = ['\n\n', '\n', '. ', ' '];

interface ResolvedChunkOptions {
  chunkSize: number;
  chunkOverlap: number;
  separators: readonly string[];
  measure: (text: string) => number;
}

/** The default measure: number of characters. */
function characterCount(text: string): number {
  return text.length;
}

/** Apply defaults and reject options that could not make progress. */
function resolveOptions(options: ChunkOptions): ResolvedChunkOptions {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const chunkOverlap = options.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP;
  if (!Number.isInteger(chunkSize) || chunkSize < 1) {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `chunkSize must be a positive integer, received ${String(chunkSize)}`);
  }
  if (!Number.isInteger(chunkOverlap) || chunkOverlap < 0) {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `chunkOverlap must be zero or a positive integer, received ${String(chunkOverlap)}`);
  }
  if (chunkOverlap >= chunkSize) {
    throw new EmbeddingError('EMBEDDING_DOCUMENT_INVALID', `chunkOverlap must be smaller than chunkSize, received ${String(chunkOverlap)} and ${String(chunkSize)}`);
  }
  return {
    chunkSize,
    chunkOverlap,
    separators: options.separators ?? DEFAULT_SEPARATORS,
    measure: options.measure ?? characterCount,
  };
}

/**
 * Split text into overlapping chunks, aligned to `separators` where that helps.
 *
 * `chunkSize` and `chunkOverlap` are expressed in whatever unit `measure`
 * returns, so passing a token counter sizes chunks in tokens. Character offsets
 * are always reported, whichever measure is used.
 *
 * @example
 * ```ts
 * chunkText('One paragraph.\n\nAnother paragraph.', { chunkSize: 20, chunkOverlap: 0 });
 * // [{ text: 'One paragraph.\n\n', index: 0, start: 0, end: 16 }, ...]
 * ```
 */
export function chunkText(text: string, options: ChunkOptions = {}): DocumentChunk[] {
  const { chunkSize, chunkOverlap, separators, measure } = resolveOptions(options);
  if (text.length === 0) return [];

  const measureRange = (from: number, to: number): number => measure(text.slice(from, to));

  /** Largest offset `to` such that the slice `[from, to)` measures at most `limit`. */
  const largestEnd = (from: number, limit: number): number => {
    if (measureRange(from, from + 1) > limit) return from;
    let low = from + 1;
    let high = text.length;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (measureRange(from, mid) <= limit) low = mid;
      else high = mid - 1;
    }
    return low;
  };

  /** Smallest offset `from` such that the slice `[from, to)` measures at most `limit`. */
  const smallestStart = (to: number, limit: number): number => {
    let low = 0;
    let high = to;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (measureRange(mid, to) <= limit) high = mid;
      else low = mid + 1;
    }
    return low;
  };

  /** Last preferred split point inside `(from, to]`, if any. */
  const lastBoundary = (from: number, to: number): number | undefined => {
    let best: number | undefined;
    for (const separator of separators) {
      if (separator === '') continue;
      let at = text.indexOf(separator, from);
      while (at !== -1 && at < to) {
        const candidate = at + separator.length;
        if (candidate <= to && candidate > from && (best === undefined || candidate > best)) best = candidate;
        at = text.indexOf(separator, at + 1);
      }
    }
    return best;
  };

  const chunks: DocumentChunk[] = [];
  let start = 0;

  while (start < text.length) {
    const end = largestEnd(start, chunkSize);
    if (end <= start) {
      throw new EmbeddingError(
        'EMBEDDING_DOCUMENT_INVALID',
        `chunkSize ${String(chunkSize)} is too small to hold even one character of this document`,
      );
    }

    // When the rest of the document fits in one chunk, take all of it. Only a
    // document with more to come is worth aligning to a separator.
    let stop = end;
    if (end < text.length) {
      const boundary = lastBoundary(start, end);
      // Ignore a boundary that would leave the chunk less than half full: it
      // makes for a worse chunk and can stall the overlap step below.
      if (boundary !== undefined && measureRange(start, boundary) >= chunkSize / 2) stop = boundary;
    }

    // A whitespace-only run carries no meaning, so it is not worth a chunk.
    if (text.slice(start, stop).trim() !== '') {
      chunks.push({ text: text.slice(start, stop), index: chunks.length, start, end: stop });
    }
    if (stop >= text.length) break;
    start = chunkOverlap === 0 ? stop : Math.max(smallestStart(stop, chunkOverlap), start + 1);
  }

  return chunks;
}
