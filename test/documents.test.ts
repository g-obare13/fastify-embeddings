import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createEmbedder, defineProvider, EmbeddingError, mockProvider } from 'fastify-embeddings';
import type { Embedder } from 'fastify-embeddings';
import {
  chunkDocuments,
  chunkText,
  embedDocuments,
  extractorByName,
  extractorFor,
  extensionOf,
  loadDocument,
  registerExtractor,
  supportedExtensions,
  textExtractor,
} from 'fastify-embeddings/documents';
import type { DocumentChunk, Extractor } from 'fastify-embeddings/documents';

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);

/** The invariant that makes chunks useful: each one is a real slice of its source. */
function expectWellFormed(chunks: DocumentChunk[], text: string): void {
  chunks.forEach((chunk, position) => {
    expect(chunk.index).toBe(position);
    expect(chunk.start).toBeLessThan(chunk.end);
    expect(chunk.text).toBe(text.slice(chunk.start, chunk.end));
  });
}

describe('chunkText', () => {
  it('keeps every chunk pointing at the slice of text it came from', () => {
    const text = 'First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.';
    const chunks = chunkText(text, { chunkSize: 30, chunkOverlap: 10 });

    expect(chunks.length).toBeGreaterThan(1);
    expectWellFormed(chunks, text);
  });

  it('overlaps neighbouring chunks by the requested amount', () => {
    const text = 'a'.repeat(50);
    const chunks = chunkText(text, { chunkSize: 20, chunkOverlap: 5 });

    expect(chunks.map((chunk) => [chunk.start, chunk.end])).toEqual([
      [0, 20],
      [15, 35],
      [30, 50],
    ]);
    expectWellFormed(chunks, text);
  });

  it('ends a chunk on a paragraph boundary when one fits', () => {
    const text = 'alpha beta\n\ngamma delta';
    const chunks = chunkText(text, { chunkSize: 12, chunkOverlap: 0 });

    expect(chunks[0]?.text).toBe('alpha beta\n\n');
    expect(chunks[0]?.end).toBe(12);
    expectWellFormed(chunks, text);
  });

  it('hard splits text that offers no separators', () => {
    const text = 'x'.repeat(25);
    const chunks = chunkText(text, { chunkSize: 10, chunkOverlap: 0, separators: [] });

    expect(chunks.map((chunk) => chunk.text.length)).toEqual([10, 10, 5]);
    expectWellFormed(chunks, text);
  });

  it('sizes chunks with a custom measure', () => {
    const text = 'one two three four five six seven eight nine ten';
    const words = (value: string): number => (value.trim() === '' ? 0 : value.trim().split(/\s+/).length);
    const chunks = chunkText(text, { chunkSize: 3, chunkOverlap: 0, measure: words });

    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) expect(words(chunk.text)).toBeLessThanOrEqual(3);
    // With no overlap, chunks never repeat text: each starts at or after the
    // end of the one before it.
    for (const [position, chunk] of chunks.entries()) {
      const next = chunks[position + 1];
      if (next !== undefined) expect(next.start).toBeGreaterThanOrEqual(chunk.end);
    }
    // Joining them therefore covers the whole text. Whitespace widths are
    // compared loosely since a boundary can absorb a separator run.
    expect(chunks.map((chunk) => chunk.text).join('').replace(/\s+/g, ' ')).toBe(text.replace(/\s+/g, ' '));
    expectWellFormed(chunks, text);
  });

  it('keeps a document that fits in one chunk whole', () => {
    const text = 'A short note that fits well inside the default chunk size.';
    const chunks = chunkText(text);

    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({ index: 0, start: 0, end: text.length, text });
  });

  it('ignores a separator that would leave the chunk less than half full', () => {
    const text = `aaaa\n\n${'b'.repeat(40)}`;
    const chunks = chunkText(text, { chunkSize: 20, chunkOverlap: 0 });

    expect(chunks[0]?.end).toBe(20);
    expect(chunks[0]?.text).toContain('b');
    expectWellFormed(chunks, text);
  });

  it('returns nothing for empty or whitespace only text', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('\n\n\n')).toEqual([]);
  });

  it('does not emit a leading whitespace run as a chunk', () => {
    const text = '\n\n\nReal content follows here.';
    const chunks = chunkText(text, { chunkSize: 12, chunkOverlap: 0 });

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0]?.text.trim()).not.toBe('');
    expectWellFormed(chunks, text);
  });

  it('rejects options that could never make progress', () => {
    const invalid = (options: Parameters<typeof chunkText>[1]): void => {
      expect(() => chunkText('some text', options)).toThrowError(
        expect.objectContaining({ code: 'EMBEDDING_DOCUMENT_INVALID' }),
      );
    };

    invalid({ chunkSize: 0 });
    invalid({ chunkSize: 10, chunkOverlap: -1 });
    invalid({ chunkSize: 10, chunkOverlap: 10 });
    invalid({ chunkSize: 1.5 });
  });

  it('rejects a measure too coarse to fit a single character', () => {
    expect(() => chunkText('some text', { chunkSize: 5, chunkOverlap: 0, measure: () => 100 })).toThrow(
      'is too small to hold even one character',
    );
  });
});

describe('extractors', () => {
  it('resolves extensions case insensitively and reports the unknown case', () => {
    expect(extensionOf('docs/HANDBOOK.MD')).toBe('.md');
    expect(extensionOf('Makefile')).toBe('');
  });

  it('registers the built-in text extractor', () => {
    expect(extractorFor('notes.md')).toBe(textExtractor);
    expect(extractorByName('text')).toBe(textExtractor);
    expect(supportedExtensions()).toContain('.md');
    expect(supportedExtensions()).toContain('.csv');
    expect(supportedExtensions()).not.toContain('.pdf');
  });

  it('strips a byte order mark and refuses binary content', () => {
    expect(textExtractor.extract({ buffer: bytes('\uFEFFhello'), filename: 'a.txt' })).toBe('hello');
    expect(() => textExtractor.extract({ buffer: new Uint8Array([0x50, 0x00, 0x4b]), filename: 'a.txt' })).toThrow(
      'looks binary',
    );
  });

  it('rejects an extractor without a name or extensions', () => {
    expect(() => registerExtractor({ name: '', extensions: ['.x'], extract: () => '' })).toThrow('must have a name');
    expect(() => registerExtractor({ name: 'nameless', extensions: [], extract: () => '' })).toThrow(
      'at least one extension',
    );
  });
});

describe('loadDocument', () => {
  let dir = '';

  beforeAll(async () => {
    dir = await mkdtemp(join(tmpdir(), 'fe-documents-'));
    await writeFile(join(dir, 'notes.md'), '# Title\n\nFirst paragraph.', 'utf8');
  });

  afterAll(async () => {
    if (dir !== '') await rm(dir, { recursive: true, force: true });
  });

  it('accepts literal text', async () => {
    await expect(loadDocument({ text: 'inline', source: 'memory' })).resolves.toEqual({ text: 'inline', source: 'memory' });
  });

  it('rejects a non-string text source', async () => {
    await expect(loadDocument({ text: 5 as unknown as string })).rejects.toThrowError(
      expect.objectContaining({ code: 'EMBEDDING_DOCUMENT_INVALID' }),
    );
  });

  it('reads a buffer and reports the extractor it used', async () => {
    const document = await loadDocument({ buffer: bytes('# Heading'), filename: 'page.md' });
    expect(document).toEqual({ text: '# Heading', source: 'page.md', extractor: 'text' });
  });

  it('reads a path through the file system', async () => {
    const path = join(dir, 'notes.md');
    const document = await loadDocument({ path });

    expect(document.text).toContain('First paragraph.');
    expect(document.source).toBe(path);
    expect(document.extractor).toBe('text');
  });

  it('refuses a format no extractor handles', async () => {
    await expect(loadDocument({ buffer: bytes('%PDF'), filename: 'report.pdf' })).rejects.toThrow(
      'No extractor handles ".pdf"',
    );
    await expect(loadDocument({ buffer: bytes('x'), filename: 'Makefile' })).rejects.toThrow(
      'a file without an extension',
    );
  });

  it('accepts an extractor chosen by name or by instance', async () => {
    const byName = await loadDocument({ buffer: bytes('named'), filename: 'no-extension' }, { extractor: 'text' });
    expect(byName.text).toBe('named');

    const custom: Extractor = { name: 'shout', extensions: ['.shout'], extract: ({ buffer }) => new TextDecoder().decode(buffer).toUpperCase() };
    const byInstance = await loadDocument({ buffer: bytes('quiet'), filename: 'ignored.bin' }, { extractor: custom });
    expect(byInstance.text).toBe('QUIET');
    expect(byInstance.extractor).toBe('shout');
  });

  it('rejects an extractor name that is not registered', async () => {
    await expect(loadDocument({ buffer: bytes('x'), filename: 'a.txt' }, { extractor: 'nope' })).rejects.toThrow(
      'No extractor named "nope"',
    );
  });

  it('enforces the byte limit for buffers and for paths', async () => {
    await expect(
      loadDocument({ buffer: bytes('much longer than the limit'), filename: 'a.txt' }, { maxBytes: 4 }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'EMBEDDING_DOCUMENT_TOO_LARGE' }));

    await expect(loadDocument({ path: join(dir, 'notes.md') }, { maxBytes: 4 })).rejects.toThrowError(
      expect.objectContaining({ code: 'EMBEDDING_DOCUMENT_TOO_LARGE' }),
    );
  });

  it('rejects an invalid byte limit', async () => {
    await expect(loadDocument({ text: 'x' }, { maxBytes: 0 })).rejects.toThrowError(
      expect.objectContaining({ code: 'EMBEDDING_DOCUMENT_INVALID' }),
    );
  });

  it('normalises extractor failures and passes EmbeddingErrors through', async () => {
    registerExtractor({
      name: 'boom',
      extensions: ['.boom'],
      extract() {
        throw new Error('parser exploded');
      },
    });
    await expect(loadDocument({ buffer: bytes('x'), filename: 'a.boom' })).rejects.toThrow(
      'Extractor "boom" failed to read a.boom',
    );

    registerExtractor({
      name: 'typed',
      extensions: ['.typed'],
      extract() {
        throw new EmbeddingError('EMBEDDING_DOCUMENT_TOO_LARGE', 'reported by the extractor');
      },
    });
    await expect(loadDocument({ buffer: bytes('x'), filename: 'a.typed' })).rejects.toThrow('reported by the extractor');
  });

  it('carries metadata reported by an extractor', async () => {
    registerExtractor({
      name: 'paged',
      extensions: ['.paged'],
      extract: ({ buffer }) => ({ text: new TextDecoder().decode(buffer), metadata: { pages: 7 } }),
    });

    const document = await loadDocument({ buffer: bytes('paged text'), filename: 'a.paged' });
    expect(document.metadata).toEqual({ pages: 7 });
  });

  it('rejects an extractor that returns no text', async () => {
    registerExtractor({
      name: 'wordless',
      extensions: ['.wordless'],
      extract: () => ({ text: undefined as unknown as string }),
    });
    await expect(loadDocument({ buffer: bytes('x'), filename: 'a.wordless' })).rejects.toThrow(
      'Extractor "wordless" returned no text',
    );
  });
});

describe('chunkDocuments and embedDocuments', () => {
  it('chunks documents in order and labels each chunk with its source', async () => {
    const chunks = await chunkDocuments(
      [
        { text: 'alpha one. alpha two.', source: 'a' },
        { text: 'beta one.', source: 'b' },
      ],
      { chunkSize: 10, chunkOverlap: 0 },
    );

    expect(chunks.length).toBeGreaterThan(2);
    expect(new Set(chunks.map((chunk) => chunk.source))).toEqual(new Set(['a', 'b']));
    const firstAlpha = chunks.findIndex((chunk) => chunk.text.includes('alpha'));
    expect(firstAlpha).toBeGreaterThanOrEqual(0);
    expect(chunks.findIndex((chunk) => chunk.text.includes('beta'))).toBeGreaterThan(firstAlpha);
  });

  it('pairs every chunk with the vector computed from its own text', async () => {
    const options = { chunkSize: 12, chunkOverlap: 0 };
    const embedder = createEmbedder({ provider: mockProvider({ dimensions: 6, model: 'doc-model' }), batchSize: 2, concurrency: 1 });

    const result = await embedDocuments(
      [
        { text: 'alpha one. alpha two. alpha three.', source: 'a' },
        { text: 'beta one. beta two.', source: 'b' },
      ],
      embedder,
      options,
    );

    expect(result.chunks.length).toBeGreaterThan(3);
    expect(result.vectors).toHaveLength(result.chunks.length);
    expect(result.model).toBe('doc-model');
    expect(result.dimensions).toBe(6);
    expect(result.usage).toBeDefined();

    // Independent check: embedding the chunk texts on their own must agree.
    const verifier = createEmbedder({ provider: mockProvider({ dimensions: 6, model: 'doc-model' }) });
    const independent = await verifier.embed(result.chunks.map((chunk) => chunk.text));
    expect(result.vectors).toEqual(independent.vectors);

    await embedder.close();
    await verifier.close();
  });

  it('passes the input type through, defaulting to document', async () => {
    const provider = mockProvider({ dimensions: 4 });
    const embedder = createEmbedder({ provider });

    await embedDocuments([{ text: 'one two three' }], embedder, { chunkSize: 50, chunkOverlap: 0 });
    expect(provider.calls.map((call) => call.inputType)).toEqual(['document']);

    await embedDocuments([{ text: 'one two three' }], embedder, { chunkSize: 50, chunkOverlap: 0, inputType: 'query' });
    expect(provider.calls.map((call) => call.inputType)).toEqual(['document', 'query']);

    await embedder.close();
  });

  it('carries extractor metadata onto every chunk', async () => {
    registerExtractor({
      name: 'meta-doc',
      extensions: ['.meta-doc'],
      extract: ({ buffer }) => ({ text: new TextDecoder().decode(buffer), metadata: { pages: 2 } }),
    });

    const chunks = await chunkDocuments([{ buffer: bytes('one two three four five six'), filename: 'x.meta-doc' }], {
      chunkSize: 8,
      chunkOverlap: 0,
    });

    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) expect(chunk.metadata).toEqual({ pages: 2 });
  });

  it('returns an empty result when there is nothing to embed', async () => {
    const embedder = createEmbedder({ provider: mockProvider({ dimensions: 8, model: 'm' }) });
    const result = await embedDocuments([{ text: '' }], embedder);

    expect(result.chunks).toEqual([]);
    expect(result.vectors).toEqual([]);
    expect(result.dimensions).toBe(8);
    expect(result.model).toBe('m');
    expect(result.usage).toBeUndefined();
    await embedder.close();
  });

  it('reports zero dimensions when the embedder does not declare any', async () => {
    const embedder = createEmbedder({
      provider: defineProvider({ name: 'silent', model: 'silent-1', async embed() { return { vectors: [] }; } }),
    });
    const result = await embedDocuments([{ text: '   ' }], embedder);

    expect(result.chunks).toEqual([]);
    expect(result.dimensions).toBe(0);
    await embedder.close();
  });

  it('refuses a result whose vector count does not match the chunk count', async () => {
    const real = createEmbedder({ provider: mockProvider({ dimensions: 4 }) });
    const mismatched: Embedder = { ...real, embed: async () => ({ vectors: [], model: 'mismatched', dimensions: 4 }) };

    await expect(embedDocuments([{ text: 'some chunkable text' }], mismatched)).rejects.toThrow(
      'Expected one vector per chunk',
    );
    await real.close();
  });
});
