/**
 * Document processing: read files, split them into embedding sized chunks, and
 * embed the chunks while keeping every vector paired with its source text.
 *
 * This is a separate entry point so the core library stays free of parser
 * dependencies and of file system access.
 */
export { chunkText } from './chunk.js';
export type { ChunkOptions, DocumentChunk } from './chunk.js';

export { extractorByName, extractorFor, extensionOf, registerExtractor, supportedExtensions, textExtractor, TEXT_EXTENSIONS } from './extract.js';
export type { ExtractInput, Extractor } from './extract.js';

export { loadDocument } from './load.js';
export type { DocumentSource, LoadDocumentOptions, LoadedDocument } from './load.js';

export { chunkDocuments, embedDocuments } from './embed.js';
export type { DocumentEmbedResult, DocumentOptions } from './embed.js';
