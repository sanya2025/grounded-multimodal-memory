# Grounded Multimodal Scene Memory

[![CI](https://github.com/sanya2025/grounded-multimodal-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/sanya2025/grounded-multimodal-memory/actions/workflows/ci.yml)

**Visual Evidence, Temporal Retrieval, and Efficient Multimodal Reasoning**

Can a multimodal AI system distinguish visual evidence from plausible
hallucination, remember what it has seen, retrieve the right past experience, and
reason over time under practical inference constraints?

We evaluate:

1. standard vs grounded multimodal reasoning,
2. real-world scene-graph grounding,
3. full context vs retrieval-based memory,
4. semantic vs temporal/entity-aware retrieval,
5. abstention when visual evidence is insufficient,
6. accuracy-latency-memory tradeoffs.

> **Strongest result figure — placeholder.** E1's 720 raw predictions are frozen
> and structured-condition metrics (object/attribute/relation/spatial PRF1,
> tri-state hallucination, evidence-support, bootstrap CIs) are computed — see
> `results/tables/`. Standard/grounded-condition metrics, E2, E3, and final
> figures are still pending. This slot will hold the strongest final figure
> (e.g. the E3.5 accuracy-latency-memory frontier) once the full experiment
> matrix is frozen. **No headline results are reported yet.**

---

## Architecture

```
Image / Frame -> Vision-Language Model -> Structured Scene Representation
   (objects, attributes, relations, actions, spatial, evidence, uncertainty)
      -> Multimodal Memory (embedding, timestamp, entities, description, source)
      -> Retrieval (semantic | temporal | hybrid)
      -> VLM / LLM Reasoning
      -> Grounded Answer + Evidence + Confidence / Abstention
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment matrix and
[BLOG_SERIES.md](BLOG_SERIES.md) for the three-part write-up.

## Benchmark design

Frozen **120-image GQA manifest**, six buckets of 20: multi-object, attributes,
spatial, human-object interaction, complex relational, and hard/ambiguous. Plus a
~30-example CLEVR controlled diagnostic and optional COCO generalization. Scene
graphs are treated as references, **not** omniscient ground truth (evidence is
scored tri-state: supported / contradicted / not-verifiable).

## Experiment matrix (one-to-one with the blog series)

| Experiment | Focus | Primary run |
|-----------|-------|-------------|
| E1 Grounded Scene Understanding | grounding, hallucination, abstention | 120 × 2 VLMs × 3 prompts = **720 predictions** |
| E2 Multimodal Memory | temporal retrieval, memory representation | ~100–150 temporal questions |
| E3 Efficiency & Production | latency, quantization, tradeoffs | model/quant/top-k sweeps |

## Status checklist (planned vs implemented)

Setup pass (this scaffold):

- [x] Repository scaffold, packaging, configs, `.env.example`, `.gitignore`
- [x] Core schemas (`scene/schema.py`) + robust structured-output parser
- [x] Common `VisionLanguageModel` interface + Qwen/LLaVA adapter skeletons + `MockVLM`
- [x] GQA/VG/CLEVR loaders + 120-image manifest-selection framework
- [x] Prompt definitions (3 conditions, versioned)
- [x] Evaluation metrics (objects/attrs/relations/spatial/hallucination/abstention/retrieval/QA/latency/stats) **with tests**
- [x] Episodic memory (events/store/embeddings/temporal) + FAISS abstraction w/ NumPy fallback
- [x] Semantic / temporal / entity-aware / hybrid retrieval
- [x] FastAPI service (`/health`, `/observe`, `/query`) + provenance-aware answers
- [x] Experiment tracking (provenance-complete records, no-clobber JSONL)
- [x] 14 companion notebook skeletons + runnable scripts
- [x] Notebooks 03 & 04 upgraded to real (non-mock), both-model
      (`qwen3_vl_8b_ollama` + `llava_ollama`) smoke tests on a handful of
      real GQA images — see [Notebooks](#notebooks)
- [x] CI (GitHub Actions: ruff + pytest on push/PR, Python 3.11 & 3.12)
- [x] Download GQA and freeze the 120-image manifest v1
- [x] Run E1 720 predictions with real VLMs *(local Ollama: qwen3-vl:8b + llava:7b)*
- [x] Structured-condition E1 metrics computed (object/attribute/relation/
      spatial PRF1, tri-state hallucination, evidence-support, bootstrap CIs)
- [x] Real CONTRADICTED detection (attribute color/material conflicts,
      spatial-opposite conflicts) — `hallucination_rate` is non-zero and
      real now, not structurally 0.0. Non-spatial relations (holding,
      wearing, ...) still have no configured opposite, so a wrong claim
      there still reads as `not_verifiable` rather than `contradicted`.
- [ ] Standard/grounded-condition metrics *(no free-text extraction path yet)*
- [ ] Build the temporal dataset and run E2 *(requires data collection)*
- [ ] Profile E3 latency/quantization *(requires GPU)*
- [ ] Freeze result tables, generate figures, write blogs

Nothing above claims experiments were run. Placeholders are clearly labeled.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 or 3.12
pip install -e ".[dev]"          # light: schemas, metrics, retrieval logic, tests
# pip install -e ".[all]"        # full local research env (adds torch/fastapi/viz)
cp .env.example .env             # then edit paths / tokens
```

Optional extras: `models` (torch+transformers),
`api` (fastapi+uvicorn), `viz` (matplotlib), `quant` (bitsandbytes), `notebooks`.

## Dataset setup

```bash
python scripts/download_gqa.py          # prints steps + target paths (no auto-download)
# ... download GQA, then:
python scripts/select_manifest.py       # builds the CANDIDATE manifest (needs manual review)
```

Do not commit raw datasets or model weights (see `.gitignore`). Only the frozen
manifest CSV is versioned.

## Run the MVP experiment

```bash
# Offline smoke test with the deterministic MockVLM (no downloads):
python scripts/run_scene_experiment.py --mock --limit 5

# Real E1 run via local Ollama models (how the actual 720-prediction run was
# done — no GPU/HF download required; `ollama serve` + models pulled first):
python scripts/run_scene_experiment.py --model qwen3_vl_8b_ollama --images-dir data/raw/gqa/images
python scripts/run_scene_experiment.py --model llava_ollama --images-dir data/raw/gqa/images
python scripts/evaluate_predictions.py

# Real E1 run via Hugging Face weights instead (requires .[models] + GPU):
python scripts/run_scene_experiment.py --images-dir /path/to/gqa/images

# E2 memory/retrieval on the toy sequence (offline):
python scripts/run_memory_experiment.py --sequences-dir data/sequences

# API (requires .[api]):
uvicorn grounded_memory.api.main:app --reload
```

## Notebooks

Fourteen companion notebooks in `notebooks/` (`00_…` – `13_…`). They are thin,
executable companions: reusable logic lives in `src/grounded_memory/`, notebooks
call it, save artifacts to `results/`, use relative paths, and end with a
Findings section. They do **not** fabricate performance numbers.

`03_vlm_baseline_inference` and `04_grounded_prompting_experiment` already run
real (non-mock) inference against both E1 models (`qwen3_vl_8b_ollama`,
`llava_ollama`) on a handful of real GQA images — a fast sanity check of the
pipeline and adapters, distinct from the frozen 720-prediction E1 run. Each
writes a distinctly-named `nb03_baseline_preview__<run_id>.jsonl` /
`nb04_bucket_preview__<run_id>.jsonl` that can never collide with the frozen
`e1_*.jsonl` results.

## Reproducibility

Freeze the manifest · pin model IDs+revisions · save prompts verbatim · set seeds
· record versions + hardware · separate raw predictions from derived metrics ·
never overwrite prior results · generate figures from scripts. See
[EXPERIMENTS.md](EXPERIMENTS.md).

## Tests

```bash
pytest -q                 # runs without large models/datasets (mocks + fixtures)
pytest -m "not gpu"       # explicitly skip anything GPU-bound
```

Note: `.[dev]`/`.[api]` extras must be installed for the full `pytest` suite. A
dependency-light validation of the core logic is available in the setup notes.

CI (`.github/workflows/ci.yml`) runs `ruff check src scripts tests` and
`pytest -q` on every push/PR to `main`, on Python 3.11 and 3.12. It's scoped
to `src`/`scripts`/`tests` (matching `pyproject.toml`'s own `[tool.ruff] src`
config) — the notebooks aren't linted, since they're exploratory companions,
not package code.

## References

See [REFERENCES.md](REFERENCES.md). Built as an applied-research portfolio project
aligned with Apple's SIML · ISE Generative AI Applied Scientist role.

## License

MIT (code only). Datasets and model weights keep their own licenses. See
[LICENSE](LICENSE).
