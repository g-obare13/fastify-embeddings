/**
 * Presets: plain objects of defaults for popular embedding model families.
 * Spread and override them freely, for example
 * `{ ...presets.nomic, model: 'nomic-embed-text:v1.5' }`.
 */

/** Text added before each input, by input type. */
export interface Prefixes {
  /** Added to `'query'` inputs. */
  query?: string;
  /** Added to `'document'` inputs. */
  document?: string;
}

/** Defaults for a model family. Explicit options and env vars take priority. */
export interface Preset {
  /** Default model id. */
  model?: string;
  /** Default query and document prefixes. */
  prefixes?: Prefixes;
}

/** Names of the built-in presets. */
export type PresetName = 'nomic' | 'e5' | 'bge' | 'minilm';

/**
 * Built-in presets.
 *
 * @example
 * ```ts
 * await app.register(embeddings, { baseURL: 'http://localhost:11434', format: 'ollama', preset: presets.nomic });
 * // or by name
 * await app.register(embeddings, { baseURL: 'http://localhost:11434', format: 'ollama', preset: 'nomic' });
 * ```
 */
export const presets: Readonly<Record<PresetName, Preset>> = Object.freeze({
  nomic: {
    model: 'nomic-embed-text',
    prefixes: { query: 'search_query: ', document: 'search_document: ' },
  },
  e5: {
    model: 'intfloat/e5-base-v2',
    prefixes: { query: 'query: ', document: 'passage: ' },
  },
  bge: {
    model: 'BAAI/bge-small-en-v1.5',
    prefixes: { query: 'Represent this sentence for searching relevant passages: ', document: '' },
  },
  minilm: {
    model: 'Xenova/all-MiniLM-L6-v2',
    prefixes: { query: '', document: '' },
  },
});

/**
 * Add the prefix for `inputType` to each text. Used by the built-in providers.
 *
 * @example
 * ```ts
 * applyPrefixes(['hello'], 'query', presets.nomic.prefixes); // ['search_query: hello']
 * ```
 */
export function applyPrefixes(texts: readonly string[], inputType: 'query' | 'document', prefixes?: Prefixes): string[] {
  const prefix = prefixes?.[inputType] ?? '';
  return texts.map((text) => `${prefix}${text}`);
}
