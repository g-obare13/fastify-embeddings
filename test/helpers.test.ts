import { describe, expect, it } from 'vitest';
import { batch, cosineSimilarity, normalize, truncate, withRetry } from 'fastify-embeddings';
import { EmbeddingError } from 'fastify-embeddings';

describe('batch helper', () => {
  it('splits items into fixed-size chunks preserving order', () => {
    const input = [1, 2, 3, 4, 5];
    const result = batch(input, 2);
    expect(result).toEqual([[1, 2], [3, 4], [5]]);
  });

  it('handles empty arrays', () => {
    expect(batch([], 3)).toEqual([]);
  });

  it('handles batch size larger than array length', () => {
    expect(batch(['a', 'b'], 10)).toEqual([['a', 'b']]);
  });

  it('throws RangeError on non-positive or non-integer size', () => {
    expect(() => batch([1, 2], 0)).toThrow(RangeError);
    expect(() => batch([1, 2], -1)).toThrow(RangeError);
    expect(() => batch([1, 2], 1.5)).toThrow(RangeError);
  });
});

describe('normalize helper', () => {
  it('normalizes a vector to unit length', () => {
    const vector = [3, 4];
    const unit = normalize(vector);
    expect(unit).toEqual([0.6, 0.8]);
    const length = Math.sqrt(unit[0]! ** 2 + unit[1]! ** 2);
    expect(length).toBeCloseTo(1, 5);
  });

  it('does not mutate input array', () => {
    const original = [3, 4];
    const copy = [...original];
    normalize(original);
    expect(original).toEqual(copy);
  });

  it('leaves zero vector unchanged', () => {
    const zero = [0, 0, 0];
    expect(normalize(zero)).toEqual([0, 0, 0]);
  });
});

describe('truncate helper', () => {
  it('keeps first N dimensions', () => {
    const vector = [1, 2, 3, 4, 5];
    expect(truncate(vector, 3)).toEqual([1, 2, 3]);
  });

  it('does not mutate input array', () => {
    const original = [1, 2, 3, 4];
    const result = truncate(original, 2);
    expect(result).toEqual([1, 2]);
    expect(original).toEqual([1, 2, 3, 4]);
  });

  it('throws RangeError on invalid dimensions', () => {
    expect(() => truncate([1, 2], 0)).toThrow(RangeError);
    expect(() => truncate([1, 2], -2)).toThrow(RangeError);
    expect(() => truncate([1, 2], 3)).toThrow(RangeError);
    expect(() => truncate([1, 2], 1.5)).toThrow(RangeError);
  });
});

describe('cosineSimilarity helper', () => {
  it('returns 1 for identical vectors', () => {
    const v = [1, 2, 3];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
  });

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 5);
  });

  it('returns 0 when either vector is all zeros', () => {
    expect(cosineSimilarity([0, 0], [1, 2])).toBe(0);
    expect(cosineSimilarity([1, 2], [0, 0])).toBe(0);
  });

  it('throws EMBEDDING_DIMENSION_MISMATCH on length mismatch', () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrowError(
      expect.objectContaining({
        code: 'EMBEDDING_DIMENSION_MISMATCH',
      }),
    );
  });
});

describe('withRetry helper', () => {
  it('succeeds on first attempt without retrying', async () => {
    let calls = 0;
    const result = await withRetry(async (attempt) => {
      calls++;
      return `ok-${attempt}`;
    });
    expect(result).toBe('ok-0');
    expect(calls).toBe(1);
  });

  it('retries on retryable errors up to limit and succeeds', async () => {
    let calls = 0;
    const result = await withRetry(
      async (attempt) => {
        calls++;
        if (attempt < 2) {
          throw new EmbeddingError('EMBEDDING_RATE_LIMITED', 'Rate limited', { retryable: true });
        }
        return 'success';
      },
      { retries: 2, baseDelayMs: 1 },
    );
    expect(result).toBe('success');
    expect(calls).toBe(3);
  });

  it('fails immediately on non-retryable error', async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls++;
          throw new EmbeddingError('EMBEDDING_INPUT_INVALID', 'Invalid input', { retryable: false });
        },
        { retries: 3, baseDelayMs: 1 },
      ),
    ).rejects.toThrow('Invalid input');
    expect(calls).toBe(1);
  });

  it('re-throws the last error when retry attempts are exhausted', async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls++;
          throw new EmbeddingError('EMBEDDING_TIMEOUT', 'Timeout', { retryable: true });
        },
        { retries: 2, baseDelayMs: 1 },
      ),
    ).rejects.toThrow('Timeout');
    expect(calls).toBe(3);
  });

  it('stops waiting and retrying when signal is aborted', async () => {
    const controller = new AbortController();
    const promise = withRetry(
      async () => {
        controller.abort(new Error('User aborted'));
        throw new EmbeddingError('EMBEDDING_RATE_LIMITED', 'Rate limit', { retryable: true });
      },
      { retries: 3, baseDelayMs: 50, signal: controller.signal },
    );
    await expect(promise).rejects.toThrow('User aborted');
  });

  it('honors retryAfterMs when present on error', async () => {
    let calls = 0;
    const start = Date.now();
    await withRetry(
      async (attempt) => {
        calls++;
        if (attempt === 0) {
          throw new EmbeddingError('EMBEDDING_RATE_LIMITED', 'Rate limit', {
            retryable: true,
            retryAfterMs: 30,
          });
        }
        return 'done';
      },
      { retries: 1, baseDelayMs: 1 },
    );
    const elapsed = Date.now() - start;
    expect(calls).toBe(2);
    expect(elapsed).toBeGreaterThanOrEqual(25);
  });
});
