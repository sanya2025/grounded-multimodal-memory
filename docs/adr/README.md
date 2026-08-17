# Architecture Decision Records

Short records of decisions that shaped this codebase — what was decided, why, and
what it costs. New ADRs go here as `NNNN-short-title.md`, numbered sequentially,
and are never edited after acceptance; a later decision that changes course adds
a new ADR that supersedes the old one (and says so in the old one's Status line).

| # | Title | Status |
|---|-------|--------|
| [0001](0001-light-base-install-optional-extras.md) | Light base install, heavy deps as optional extras | Accepted |
| [0002](0002-drop-faiss-exact-numpy-retrieval.md) | Drop FAISS; exact NumPy cosine retrieval | Accepted |
| [0003](0003-deterministic-hashing-embedder-for-tests.md) | Deterministic hashing embedder for tests | Accepted |
| [0004](0004-separate-shared-docs-from-personal-notes.md) | Separate shared `docs/` from personal `notes/` | Accepted |
| [0005](0005-module-level-pydantic-models-in-fastapi-app.md) | Module-level Pydantic models in the FastAPI app | Accepted |
