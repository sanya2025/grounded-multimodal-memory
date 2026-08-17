# 2. Drop FAISS; exact NumPy cosine retrieval

Status: Accepted

## Context

`MemoryStore.semantic_search` originally tried to build a `faiss.IndexFlatIP`
when `faiss` was importable, falling back to exact NumPy cosine similarity
otherwise. On macOS, `faiss-cpu` and `torch` each bundle their own copy of
`libomp.dylib`. Once a process imports both (e.g. a VLM adapter pulling in
`torch`, then retrieval calling into `faiss`), LLVM's OpenMP runtime detects
the second copy and calls `abort()` — a `SIGABRT`, not a catchable Python
exception, so the try/except around FAISS index construction couldn't save
it. This reproduced reliably running the full test suite (`pytest -q`)
whenever a torch-importing test ran before a faiss-search test, with the
crash log showing `OMP: Error #15: Initializing libomp.dylib, but found
libomp.dylib already initialized.`

The proper fix (conda-forge builds of both packages sharing one `libomp`) or
the workaround (`KMP_DUPLICATE_LIB_OK=TRUE`) both add environment complexity.
Given this store's target scale (a handful to low thousands of memory
events, not a large-scale vector index), FAISS's speed advantage over exact
NumPy search isn't needed yet.

## Decision

Remove the `faiss-cpu` dependency and the FAISS code path entirely.
`MemoryStore.semantic_search` always does exact cosine similarity via a
NumPy matrix multiply + `argsort`. The `retrieval` extra was removed from
`pyproject.toml`.

## Consequences

- No more OpenMP crash risk from mixing `torch` and `faiss` in one process.
- One code path instead of two (simpler `store.py`, no faiss/no-faiss test
  split).
- Retrieval is exact but `O(n)` per query. If the store's event count grows
  large enough for this to matter, revisit with an ANN index — ideally one
  installed in a way that doesn't reintroduce the OpenMP conflict (e.g. both
  `torch` and the index library from conda-forge, sharing `llvm-openmp`).
