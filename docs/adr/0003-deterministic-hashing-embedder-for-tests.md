# 3. Deterministic hashing embedder for tests

Status: Accepted

## Context

Retrieval and memory-store logic (ranking, hybrid scoring, recall/NDCG
metrics) needs *some* embedding to operate on, but the correctness of that
logic doesn't depend on the embeddings coming from a real semantic model. A
real sentence/image encoder would require model downloads and add real
latency to the test suite, which conflicts with the project's "tests run
without GPUs or large downloads" goal (see [0001](0001-light-base-install-optional-extras.md)).

## Decision

`grounded_memory.memory.embeddings.HashingTextEmbedder` implements a
deterministic bag-of-tokens hashing vectorizer (MD5-hash each token into a
fixed-size vector, then L2-normalize) behind the same `Embedder` protocol a
real encoder would implement. It's explicitly documented as "NOT a semantic
model" — good enough to exercise retrieval logic and tests, not to produce
meaningful semantic rankings. `tests/conftest.py`'s `embedder` fixture and
`toy_store` fixture are built on it.

## Consequences

- The full test suite runs with no network access and no model weights.
- Tests that assert on retrieval *ranking* rely on token overlap (e.g. shared
  words like "backpack") rather than true semantic similarity — acceptable
  for this fixture's toy sentences, but this embedder must never be used to
  evaluate real retrieval quality; that requires swapping in a real encoder
  via the same `Embedder` protocol.
