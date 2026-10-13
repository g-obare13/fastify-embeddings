import { registerExtractor } from 'fastify-embeddings/documents';

registerExtractor({
  name: 'markdown-with-frontmatter',
  extensions: ['.mdown'],
  extract: (input) => new TextDecoder().decode(input.buffer),
});
