# Experiment Matrix

Three top-level experiments — used consistently across code, notebooks, result
directories, and the blog series. Evaluation metrics, failure analysis, human
evaluation, statistical analysis, and API scaffolding are **supporting methods**,
NOT separate experiments.

```
Experiment 1 -> Blog 1: Grounded Visual Evidence
Experiment 2 -> Blog 2: Temporal Multimodal Memory
Experiment 3 -> Blog 3: Production and Efficiency
```

## Experiment 1 — Grounded Scene Understanding
| ID | Name | What it varies |
|----|------|----------------|
| E1.1 | Standard vs grounded prompting | prompt condition A vs B |
| E1.2 | Structured vs natural-language grounded output | condition B vs C |
| E1.3 | Counterfactual & abstention probes | positive vs counterfactual probes |
| E1.4 | CLEVR controlled diagnostic | real (GQA) vs synthetic (CLEVR) |

Primary run: **120 GQA images × 2 VLMs × 3 conditions = 720 predictions.**
Supporting: object/attribute/relation/spatial metrics, tri-state hallucination &
evidence-support, abstention, bootstrap CIs, per-bucket breakdown, failure
taxonomy, human evaluation.

## Experiment 2 — Multimodal Memory & Temporal Reasoning
| ID | Name | What it varies |
|----|------|----------------|
| E2.1 | No memory vs full history vs retrieval | context strategy |
| E2.2 | Semantic vs temporal vs hybrid retrieval | retrieval strategy |
| E2.3 | Text-only vs image-only vs hybrid memory | memory representation |
| E2.4 | Retrieval-signal ablation | semantic (+time)(+entities) |
| E2.5 | Top-k context sweep | k ∈ {1,3,5,10,20} |

Dataset: ~20–30 sequences, 5–20 events each, ~100–150 temporal questions.
Supporting: Recall@K, MRR, nDCG, end-to-end QA (scored **separately** from
retrieval), hallucination, evidence support, temporal question categories.

## Experiment 3 — Efficiency & Production Tradeoffs
| ID | Name | What it varies |
|----|------|----------------|
| E3.1 | Model-size / inference comparison | larger vs smaller VLM |
| E3.2 | Quantization | none / int8 / int4 (hardware-guarded) |
| E3.3 | Retrieval overhead | VLM-only vs VLM+retrieval |
| E3.4 | Latency decomposition | per-component timing |
| E3.5 | Accuracy-latency-memory tradeoff | the frontier plot |

Supporting: context/token usage, throughput, peak memory, API behavior,
reproducibility.

## Research questions
- **RQ1** Does grounded prompting improve factual reliability / reduce unsupported claims? (E1)
- **RQ2** Does external memory beat no-memory / full-history for temporal reasoning? (E2)
- **RQ3** Which memory representation + retrieval strategy retrieves the right evidence? (E2)
- **RQ4** Can the system correctly abstain when evidence is insufficient? (E1, E2)
- **RQ5** What accuracy / grounding / latency / memory tradeoffs arise? (E3)

## Execution order (outcome-driven)
1. environment + tests → 2. GQA metadata → 3. dataset exploration →
4. candidate manifest → 5. manual inspection → 6. freeze manifest v1 →
7–8. smoke-test both VLMs on 3–5 images → 9. validate serialization →
10. run 120×2×3 → 11. freeze raw predictions → 12. deterministic metrics →
13. audit failures → 14. Blog 1 figures → 15. *then* temporal memory.

## Reproducibility rules (summary)
Freeze the manifest; pin model IDs+revisions; save every prompt verbatim; set
seeds; record package versions + hardware; separate raw predictions from derived
metrics; never overwrite prior results; generate figures from scripts; keep
notebooks as companions with logic in `src/`.
