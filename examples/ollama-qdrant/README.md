# ollama-qdrant example

A standalone app that embeds ten sample documents with a self-hosted Ollama
server and stores the vectors in Qdrant. It is not part of the library build and
is not covered by the library test suite.

## Safety rules this app follows

- It only ever touches a collection whose name starts with `fe-test-`.
- It refuses to run if that collection already exists, so it never mixes its
  data with an existing collection.
- It only deletes the single collection it created, and only when you pass
  `--cleanup`. It never lists collections and deletes, and never modifies any
  other collection.
- It refuses to upsert when the collection's recorded vector size differs from
  the embedder's dimensions.
- It never prints the Qdrant API key or a full server address, only a host class
  (`local`, `private`, or `public`).

## Setup

Build the library first, because this app consumes the package rather than the
source files:

```bash
pnpm install
pnpm build
```

Then install this app's own dependencies and create its environment file:

```bash
cd examples/ollama-qdrant
pnpm install
cp .env.example .env
```

Fill in `.env` with your own values. Shell variables take priority over `.env`,
so you can also export them instead of writing a file.

If you prefer to consume a packed tarball rather than the repository directory,
run `pnpm pack` in the repository root and point the `fastify-embeddings`
dependency at the resulting `fastify-embeddings-<version>.tgz`.

## Run

```bash
pnpm start
```

The app checks the embeddings server, embeds ten documents, creates the
`fe-test-ollama-qdrant-demo` collection with cosine distance, upserts the
documents with the model name and vector size stored in each payload, and then
runs one search. It leaves the collection in place.

## Clean up

```bash
pnpm cleanup
```

That deletes only the `fe-test-` collection this app created. Nothing else is
touched.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `EMBEDDINGS_BASE_URL` | Ollama API root |
| `EMBEDDINGS_FORMAT` | Must be `ollama` |
| `EMBEDDINGS_MODEL` | Embedding model, for example `nomic-embed-text` |
| `QDRANT_URL` | Qdrant API root |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Optional. Overrides the collection name and must start with `fe-test-` |
