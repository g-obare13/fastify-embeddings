import { createEmbedder, mockProvider } from 'fastify-embeddings';
import { chunkText, embedDocuments, loadDocument } from 'fastify-embeddings/documents';

const document = await loadDocument({ buffer: new Uint8Array(), filename: 'handbook.md' });
const chunks = chunkText(document.text, { chunkSize: 800, chunkOverlap: 100 });

const embedder = createEmbedder({ provider: mockProvider({ dimensions: 768 }) });
const result = await embedDocuments(
  [{ path: 'handbook.md' }, { text: 'An inline note', source: 'notes' }],
  embedder,
  { chunkSize: 800, chunkOverlap: 100, inputType: 'document' },
);

void chunks;
void result.chunks;
void result.vectors;
void result.model;
void result.dimensions;
await embedder.close();
