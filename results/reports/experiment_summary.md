# Experiment Summary

> **Status: no experiments run yet.** This file is a template. Fill it in only
> from frozen result files — never with invented numbers.

## Provenance
- Manifest version: _pending_ (freeze as v1 after manual review)
- Model IDs / revisions: _pending_ (pin in configs/models.yaml)
- Git commit(s): _pending_
- Hardware: _pending_

## Experiment 1 — Grounded Scene Understanding
| Model | Prompt | Object F1 | Relation Acc | Spatial Acc | Hallucination ↓ | Evidence Support ↑ | Abstention ↑ |
|-------|--------|----------:|------------:|------------:|----------------:|-------------------:|-------------:|
| Qwen  | Standard   | — | — | — | — | — | — |
| Qwen  | Grounded   | — | — | — | — | — | — |
| Qwen  | Structured | — | — | — | — | — | — |
| LLaVA | Standard   | — | — | — | — | — | — |
| LLaVA | Grounded   | — | — | — | — | — | — |
| LLaVA | Structured | — | — | — | — | — | — |

Parser-failure accounting (structured): _pending_. Per-bucket breakdown: _pending_.

## Experiment 2 — Multimodal Memory
| Method | QA Acc | Recall@5 | MRR | nDCG@5 | Hallucination ↓ | Context Tokens | Latency |
|--------|-------:|--------:|----:|-------:|----------------:|---------------:|--------:|
| No memory    | — | — | — | — | — | — | — |
| Full history | — | — | — | — | — | — | — |
| Semantic     | — | — | — | — | — | — | — |
| Hybrid       | — | — | — | — | — | — | — |

## Experiment 3 — Efficiency & Production
Latency decomposition, model-size comparison, quantization, top-k sweep,
accuracy-latency-memory frontier: _pending_.

## Key findings
_3–5 defensible findings with bootstrap CIs once results exist._
