# 1. Light base install, heavy deps as optional extras

Status: Accepted

## Context

This project spans schema/metrics/retrieval logic that should run anywhere
(CI, a laptop, no GPU) and heavy model-inference code (VLMs via `torch` +
`transformers`, optionally quantized via `bitsandbytes`) that only makes sense
on a machine with real hardware. If everything were a single flat dependency
list, installing the repo just to run the test suite or the retrieval/eval
code would force multi-GB downloads (`torch`, CUDA wheels) that have nothing
to do with what's being tested.

## Decision

`pyproject.toml` keeps `dependencies` intentionally small — `numpy`, `pandas`,
`pydantic`, `PyYAML`, `scipy`, `scikit-learn`, `Pillow` — and pushes everything
hardware- or model-specific into `[project.optional-dependencies]` extras:
`models` (torch+transformers+accelerate), `api` (fastapi+uvicorn), `viz`
(matplotlib), `quant` (bitsandbytes, Linux-only), `tracking` (mlflow),
`notebooks`, and `dev`. An `all` meta-extra bundles them for a full local
research environment. `pip install -e ".[dev]"` is enough to run tests.

## Consequences

- Tests and retrieval/eval code never require a GPU or a model download.
- Contributors pick the install that matches what they're doing instead of
  always paying for the heaviest one.
- Any new heavy dependency should default to living in an extra, not
  `dependencies`, unless the base install genuinely can't function without it.
