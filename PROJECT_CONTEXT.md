# PROJECT_CONTEXT.md

# Grounded Multimodal Scene Memory

## 1. Purpose of This File

This file is the high-level source of truth for AI coding assistants and collaborators working on this repository.

Before modifying code, generating documentation, analyzing experiments, or writing blog content:

1. read this file,
2. inspect the actual repository,
3. distinguish implemented functionality from planned functionality,
4. do not invent experimental results,
5. do not assume the implementation matches the research plan.

When this document and the code disagree about implementation status, **the repository is authoritative about what currently exists**, while this document describes the intended research direction.

---

# 2. Project Overview

**Grounded Multimodal Scene Memory** is an applied multimodal AI research project investigating how vision-language systems can become more reliable and useful by combining:

- visual scene understanding,
- evidence-grounded generation,
- structured scene representations,
- hallucination-aware evaluation,
- multimodal episodic memory,
- semantic and temporal retrieval,
- temporal reasoning,
- uncertainty and abstention,
- efficient inference,
- production-oriented system design.

The project is intended to function simultaneously as:

1. an applied research project,
2. a reproducible experimental benchmark,
3. a technical portfolio project,
4. an interview case study for senior multimodal AI / Applied Scientist / ML roles,
5. the experimental foundation for a three-part technical blog series.

The central progression is:

```text
Visual perception
      ↓
Grounded scene understanding
      ↓
Structured observations
      ↓
Multimodal episodic memory
      ↓
Retrieval
      ↓
Temporal reasoning
      ↓
Grounded answer + evidence
      ↓
Efficiency / production tradeoffs
```

---

# 3. Core Research Question

The overarching research question is:

> Can a multimodal AI system improve reliability and temporal reasoning by explicitly grounding visual claims in evidence, storing structured multimodal observations as external memory, and retrieving only the evidence relevant to a query?

Three related questions organize the project:

### RQ1 — Grounding

Can grounded prompting and structured evidence reduce unsupported visual claims while preserving useful scene understanding?

### RQ2 — Memory

Can an external multimodal memory improve reasoning across observations compared with no memory or placing the entire history in the model context?

### RQ3 — Efficiency

What accuracy, grounding, latency, context-size, and memory tradeoffs arise when moving from a research prototype toward a deployable multimodal system?

---

# 4. Experiment Numbering

Use exactly three top-level experiments throughout:

- source code,
- notebooks,
- documentation,
- result directories,
- plots,
- tables,
- blog posts,
- interview material.

Do not introduce additional top-level experiment numbers without updating this document.

---

# 5. Experiment 1 — Grounded Scene Understanding

## Research Goal

Evaluate whether explicitly requiring visual evidence and structured grounding changes VLM reliability.

## E1.1 — Standard vs Grounded Prompting

Compare a conventional VLM prompt with a prompt requiring claims to be tied explicitly to observable evidence.

Primary questions:

- Does grounding reduce unsupported claims?
- Does grounding preserve useful information?
- Does it make the model excessively conservative?
- Which error categories improve?

---

## E1.2 — Structured vs Natural-Language Grounded Output

Compare free-form grounded answers with structured output.

Possible structured fields include:

```json
{
  "answer": "...",
  "observations": [],
  "evidence": [],
  "uncertainty": "...",
  "abstain": false
}
```

Investigate whether output structure:

- improves evidence attribution,
- improves evaluation reliability,
- changes model behavior,
- introduces parsing/schema failures.

---

## E1.3 — Counterfactual and Abstention Probes

Test whether the model can reject unsupported premises and abstain when visual evidence is insufficient.

Evaluate:

- correct abstention,
- false-answer rate,
- unsupported assertions,
- overconfidence,
- counterfactual acceptance.

---

## E1.4 — CLEVR Controlled Diagnostic

Use CLEVR as a controlled diagnostic for compositional and spatial reasoning.

CLEVR is a **supplementary diagnostic**, not the primary real-world benchmark.

Its purpose is to help separate reasoning failures from ambiguity and annotation limitations present in real-world imagery.

---

# 6. Experiment 1 — Primary Dataset

## GQA

GQA is the primary benchmark candidate for grounded real-world visual reasoning.

Relevant properties include:

- real-world images,
- scene graphs,
- objects,
- attributes,
- relations,
- compositional questions,
- functional programs.

The project should use a **frozen benchmark manifest** rather than repeatedly sampling images.

The benchmark subset should intentionally cover important visual reasoning categories rather than relying solely on random sampling.

Candidate categories include:

1. multi-object scenes,
2. attributes,
3. spatial relationships,
4. human-object interactions,
5. relational reasoning,
6. hard or ambiguous scenes.

The exact benchmark size and composition must be recorded in the experiment configuration and manifest.

Do not silently change the evaluation subset between runs.

---

# 7. Visual Genome and Annotation Limitations

GQA builds on visual annotations related to Visual Genome.

Scene graphs may contain:

```text
objects
attributes
relationships
bounding boxes
```

Example:

```text
person --holding--> umbrella
umbrella --color--> black
person --left_of--> bicycle
```

A critical methodological rule is:

```text
claim absent from annotation
        ≠
claim necessarily false
```

Scene-graph annotations can be incomplete.

Therefore, evaluation should distinguish at least:

### SUPPORTED

Available evidence supports the claim.

### CONTRADICTED

Available evidence explicitly conflicts with the claim.

### NOT VERIFIABLE

Available annotations are insufficient to determine whether the claim is true or false.

Do not automatically classify every unannotated model claim as a hallucination.

Ambiguous cases may require human review.

---

# 8. Hallucination and Grounding

The project should distinguish direct observation from inference.

Example:

> A person is standing next to an oven.

may be visually observable.

But:

> The person is cooking dinner.

may introduce an inferred activity or intention not sufficiently supported by the image.

Potential failure categories include:

- object hallucination,
- attribute hallucination,
- relation error,
- spatial error,
- action-inference error,
- identity assumption,
- intent inference,
- unsupported causal inference,
- overconfidence,
- failure to abstain,
- evidence mismatch,
- correct conclusion for the wrong reason.

The final taxonomy should be frozen before final evaluation whenever practical.

---

# 9. Experiment 1 Metrics

Candidate metrics include:

- precision,
- recall,
- F1,
- object accuracy,
- attribute accuracy,
- relation accuracy,
- spatial accuracy,
- hallucination rate,
- evidence-support rate,
- correct abstention rate,
- false-answer rate.

When comparing two prompting strategies on the same images, treat the observations as paired where statistically appropriate.

Report uncertainty rather than only point estimates.

Possible statistical methods include:

- bootstrap confidence intervals,
- paired bootstrap comparisons.

Do not claim statistical significance without an appropriate test and sufficient data.

---

# 10. Experiment 2 — Multimodal Memory and Temporal Reasoning

## Research Goal

Investigate how external memory and retrieval affect reasoning across sequences of multimodal observations.

Conceptual pipeline:

```text
scene understanding
       ↓
event representation
       ↓
episodic memory
       ↓
retrieval
       ↓
temporal reasoning
```

A memory event may contain:

```text
event_id
timestamp
textual observation
entities
attributes
relationships
image reference / image embedding
text embedding
provenance
confidence or evidence metadata
```

The exact schema should reflect the actual implementation.

---

# 11. E2.1 — No Memory vs Full History vs Retrieval

Compare at least conceptually:

### No memory

The model receives only the current observation/query.

### Full history

Previous observations are inserted into the model context.

### Retrieval-based memory

Relevant observations are retrieved from an external memory.

Questions include:

- Which approach produces better temporal answers?
- How does context length affect performance?
- How much irrelevant information enters the prompt?
- What is the latency cost?
- Where do retrieval failures occur?

---

# 12. E2.2 — Retrieval Strategies

Compare retrieval approaches where implemented.

Candidates include:

- semantic retrieval,
- temporal retrieval,
- entity-aware retrieval,
- hybrid retrieval.

A conceptual hybrid score may take the form:

```text
score =
alpha * semantic_similarity
+ beta * temporal_relevance
+ gamma * entity_overlap
```

Important:

The weights are experimental configuration parameters.

Do not describe arbitrary starting weights as theoretically optimal.

Document:

- score normalization,
- weight-selection procedure,
- development data,
- test-data separation.

---

# 13. E2.3 — Memory Representation

Possible comparisons include:

- text-only memory,
- image-only memory,
- hybrid multimodal memory.

The exact experiment should depend on what the repository actually implements.

---

# 14. E2.4 — Retrieval-Signal Ablation

Where implemented, evaluate the contribution of individual retrieval signals.

Examples:

```text
semantic only
semantic + temporal
semantic + entity
semantic + temporal + entity
```

Ablations should help answer:

> Which component actually contributes to retrieval and downstream reasoning performance?

---

# 15. E2.5 — Top-K Context Sweep

Evaluate the effect of retrieval depth.

Possible values:

```text
K = 1, 3, 5, 10
```

Do not assume larger K is better.

More retrieved context may introduce:

- distractors,
- redundant observations,
- contradictory historical states,
- increased prompt length,
- increased latency.

---

# 16. Temporal Reasoning

The project should support or study questions requiring different temporal operations.

Example event history:

```text
09:15 — keys are on the desk
10:30 — keys are inside a backpack
11:45 — backpack is placed in the car
```

Possible queries:

> Where were the keys at 09:15?

> Where were the keys last observed?

> Where were the keys before they entered the backpack?

> What changed between 09:15 and 10:30?

Different questions require different combinations of retrieval and temporal reasoning.

Semantic similarity alone may not be sufficient.

---

# 17. Experiment 2 Retrieval Metrics

Candidate retrieval metrics include:

- Recall@1,
- Recall@3,
- Recall@5,
- Precision@K where appropriate,
- Mean Reciprocal Rank (MRR),
- DCG,
- nDCG@K.

Always distinguish:

```text
retrieval quality
        ≠
end-to-end answer quality
```

Possible cases include:

```text
correct retrieval + incorrect reasoning
incorrect retrieval + lucky correct answer
```

Evaluate retrieval and downstream QA separately when possible.

---

# 18. Embeddings and Vector Search

The memory system may use text, image, or multimodal embeddings.

Relevant concepts include:

- vector representations,
- cosine similarity,
- nearest-neighbor search,
- exact search,
- approximate nearest-neighbor search.

FAISS is a candidate/local vector-search implementation.

The project should document:

- embedding model,
- vector dimension,
- normalization,
- similarity function,
- index type,
- indexing strategy.

Do not assume FAISS is inherently the correct production solution. Explain why it is appropriate for the current experimental scale and what alternatives would be considered at larger scale.

---

# 19. Long Context vs Retrieval

A major design comparison is:

```text
full interaction history
          vs
external retrieval-based memory
```

Relevant tradeoffs include:

- token consumption,
- latency,
- irrelevant context,
- context utilization,
- retrieval errors,
- information loss,
- scaling,
- provenance.

Important principle:

> More context is not necessarily better context.

The project should experimentally investigate this rather than assume retrieval is superior.

---

# 20. Experiment 3 — Efficiency and Production Tradeoffs

## Research Goal

Evaluate what happens when the multimodal research pipeline is considered as an ML system rather than only a benchmark experiment.

---

# 21. E3.1 — Model and Inference Comparison

Where resources permit, compare relevant model/inference configurations.

Record:

- model,
- parameter scale,
- precision,
- device,
- input resolution,
- context length,
- decoding parameters,
- software versions.

Comparisons must be fair and reproducible.

---

# 22. E3.2 — Quantization

Where supported by the hardware and inference stack, investigate quantization.

Relevant formats may include:

- FP32,
- FP16,
- BF16,
- INT8,
- INT4.

Measure both resource savings and potential quality changes.

Do not assume quantization is free.

---

# 23. E3.3 — Retrieval Overhead

Measure the cost introduced by the memory system.

Potential components include:

```text
query embedding
vector search
metadata filtering
reranking
context construction
```

---

# 24. E3.4 — Latency Decomposition

Where practical, separately measure:

```text
image preprocessing
vision encoding / VLM inference
memory embedding
retrieval
prompt construction
generation
post-processing
```

Report appropriate latency distributions rather than only a single average when enough measurements are available.

Potential system metrics include:

- p50 latency,
- p95 latency,
- p99 latency,
- throughput,
- peak memory.

---

# 25. E3.5 — Accuracy–Latency–Memory Tradeoff

The goal is not simply to identify the most accurate configuration.

Analyze tradeoffs among:

```text
quality
grounding
latency
memory footprint
context size
retrieval cost
```

Use Pareto-style reasoning where appropriate.

---

# 26. Production/API Direction

The project may expose a FastAPI interface.

Candidate endpoints include:

```text
POST /observe
POST /query
GET /health
```

A conceptual lifecycle is:

```text
observation
    ↓
scene extraction
    ↓
event construction
    ↓
embedding
    ↓
memory index

query
    ↓
query representation
    ↓
retrieval
    ↓
context construction
    ↓
VLM / reasoning
    ↓
answer + evidence + provenance
```

The implementation must be inspected before claiming that all components exist.

---

# 27. Reproducibility Requirements

Every experiment should record enough configuration to reproduce it.

Where applicable record:

- dataset version,
- benchmark manifest,
- model identifier,
- model revision,
- prompt version,
- random seed,
- decoding parameters,
- embedding model,
- retrieval configuration,
- top-k,
- hybrid weights,
- hardware,
- software/library versions,
- timestamp,
- Git commit hash once Git is used.

Prefer configuration files over hidden notebook constants.

---

# 28. Experiment Artifact Structure

A recommended structure is:

```text
experiments/
|
|-- e1_grounded_scene_understanding/
|   |-- e1_1_prompting/
|   |-- e1_2_structured_output/
|   |-- e1_3_counterfactual_abstention/
|   `-- e1_4_clevr_diagnostic/
|
|-- e2_multimodal_memory/
|   |-- e2_1_context_vs_retrieval/
|   |-- e2_2_retrieval_strategies/
|   |-- e2_3_memory_representation/
|   |-- e2_4_retrieval_ablation/
|   `-- e2_5_topk/
|
`-- e3_efficiency_production/
    |-- e3_1_model_inference/
    |-- e3_2_quantization/
    |-- e3_3_retrieval_overhead/
    |-- e3_4_latency/
    `-- e3_5_tradeoff/
```

Do not restructure an existing repository automatically just to match this example.

First inspect the repository and propose changes.

---

# 29. Blog Series

The experiments support a three-part technical blog series.

## Blog 1

**Beyond Image Captioning: Grounding Multimodal Models in Visual Evidence**

Primary topics:

- VLM scene understanding,
- grounding,
- scene graphs,
- hallucination,
- evidence,
- abstention,
- Experiment 1.

---

## Blog 2

**Giving Multimodal AI a Memory: Temporal Retrieval Across Visual Experiences**

Primary topics:

- episodic multimodal memory,
- embeddings,
- retrieval,
- semantic vs temporal retrieval,
- hybrid retrieval,
- temporal reasoning,
- Experiment 2.

---

## Blog 3

**From Multimodal Prototype to Production: Accuracy, Memory, and Latency Tradeoffs**

Primary topics:

- model/inference choices,
- quantization,
- retrieval overhead,
- latency,
- memory footprint,
- production architecture,
- Experiment 3.

---

# 30. Blog Evidence Policy

Never invent results for the blogs.

Before finalizing an empirical claim, verify it against experiment artifacts.

When evidence is missing, use explicit placeholders:

```text
[RESULT PENDING — E1.1]

[FIGURE PENDING — E1.1]

[CLAIM REQUIRES HUMAN AUDIT]

[RESULT PENDING — E2.2]

[RESULT PENDING — E3.5]
```

Blog conclusions should be written only after the relevant experiment results are frozen.

---

# 31. Documentation / Interview Companion

The project should eventually support documentation such as:

```text
docs/
|
|-- PROJECT_IMPLEMENTATION_STATUS.md
|-- PROJECT_MASTERY_GUIDE.md
|-- EXPERIMENT_COMPANION.md
|-- INTERVIEW_DEFENSE_GUIDE.md
|-- PROJECT_PITCH.md
|-- WHITEBOARD_GUIDE.md
|-- RESEARCH_CRITIQUE.md
|-- LITERATURE_MAP.md
|-- INTERVIEW_FLASHCARDS.md
`-- MOCK_INTERVIEW.md
```

These are documentation goals, not proof that the files currently exist.

---

# 32. Interview Preparation Goals

The project owner should eventually be able to explain without notes:

### Problem

What problem does the project solve?

### Architecture

How does information flow through the system?

### Grounding

What counts as visual evidence?

### Hallucination

How is an unsupported claim defined and measured?

### Dataset

Why GQA?

Why Visual Genome?

Why CLEVR only as a diagnostic?

### Evaluation

Why these metrics?

What do they fail to measure?

### Memory

Why external memory?

### Retrieval

Why semantic, temporal, entity-aware, or hybrid retrieval?

### Temporal Reasoning

How are historical states represented and queried?

### Systems

Where does latency come from?

### Tradeoffs

How do accuracy, grounding, latency, context size, and memory interact?

### Limitations

What are the major methodological weaknesses?

### Future Work

What would be changed with more compute, data, or engineering time?

---

# 33. High-Priority Concepts for Mastery

The most important areas for deep understanding are:

1. GQA and scene-graph annotation limitations,
2. hallucination and evidence-grounded generation,
3. VLM architecture and multimodal representation,
4. precision, recall, F1, and grounding metrics,
5. embeddings and cosine similarity,
6. FAISS/vector retrieval,
7. Recall@K, MRR, DCG, and nDCG,
8. episodic multimodal memory,
9. temporal reasoning,
10. long context vs retrieval,
11. abstention and uncertainty,
12. model efficiency and quantization,
13. latency decomposition,
14. accuracy–latency–memory tradeoffs,
15. experimental design and statistical validity.

---

# 34. References Policy

Prefer primary sources.

Reference priority:

1. original research paper,
2. official technical report,
3. official dataset documentation,
4. official model documentation,
5. official repository,
6. high-quality secondary source only when needed.

Important reference areas include:

- GQA,
- Visual Genome,
- CLEVR,
- VLM architectures,
- Qwen-VL/Qwen2.5-VL if used,
- LLaVA/LLaVA-NeXT if used,
- FAISS,
- multimodal hallucination evaluation,
- multimodal memory,
- retrieval-augmented systems,
- long-context reasoning,
- temporal reasoning,
- calibration,
- selective prediction,
- efficient inference.

Do not cite papers that have not been verified.

---

# 35. Rules for AI Coding Assistants

## Rule 1 — Inspect Before Claiming

Do not infer implementation from this plan.

Inspect the repository.

---

## Rule 2 — Never Fabricate Results

Never create plausible-looking experiment values.

If results do not exist, say:

```text
RESULT NOT AVAILABLE
```

---

## Rule 3 — Separate Status Clearly

Use:

```text
IMPLEMENTED
PARTIALLY IMPLEMENTED
PLANNED
PROPOSED IMPROVEMENT
```

---

## Rule 4 — Preserve Experiment Numbering

Use only:

```text
Experiment 1 / E1.x
Experiment 2 / E2.x
Experiment 3 / E3.x
```

for the experimental hierarchy described above.

Metrics, statistical analyses, failure analyses, and human evaluations are supporting analyses, not new experiments.

---

## Rule 5 — Do Not Silently Change Experimental Methodology

If proposing a change to:

- benchmark composition,
- prompt,
- metric,
- model,
- retrieval algorithm,
- thresholds,
- scoring weights,
- human-evaluation protocol,

explain why and record the change.

---

## Rule 6 — Protect Test Integrity

Do not tune hyperparameters, prompts, retrieval weights, or thresholds using the final test results.

Use development/validation data where tuning is required.

---

## Rule 7 — Favor Reproducibility

Prefer:

```text
config
+
manifest
+
script
+
saved raw output
+
evaluation script
```

over manual notebook-only procedures.

---

## Rule 8 — Preserve Raw Outputs

Do not overwrite raw model responses with parsed or cleaned versions.

Prefer:

```text
raw output
      ↓
parsed output
      ↓
evaluation
```

so evaluation can be audited.

---

## Rule 9 — Separate Model Failure From Evaluation Failure

Possible sources of error include:

```text
model
dataset annotation
parser
retriever
metric
human annotation
experiment configuration
```

Do not automatically attribute every error to the VLM.

---

## Rule 10 — Avoid Unnecessary Complexity

This is an applied research project.

Do not introduce infrastructure merely because it is common in production systems.

Every major dependency should have a reason.

---

# 36. Rules for Generated Tutorial Material

Educational documentation should not be generic.

For every major concept answer:

1. What is it?
2. Why does this project need it?
3. Where is it implemented?
4. How does it work mathematically?
5. What alternatives exist?
6. What tradeoffs does it introduce?
7. How can it fail?
8. How should it be explained in an interview?
9. What difficult follow-up might an interviewer ask?
10. What primary reference supports it?

---

# 37. Rules for Interview Practice

When conducting interactive interview practice:

1. ask one question at a time,
2. do not reveal the answer first,
3. wait for the candidate's answer,
4. evaluate correctness,
5. evaluate technical depth,
6. evaluate clarity and concision,
7. identify missing ideas,
8. propose a stronger answer,
9. ask a harder follow-up.

Do not turn interview practice into passive flashcard reading.

---

# 38. Security and Repository Hygiene

Never commit or expose:

- `.env`,
- API keys,
- access tokens,
- passwords,
- private credentials,
- private user data,
- proprietary datasets without permission,
- unnecessary large model weights,
- generated caches.

Before GitHub publication, review:

```text
.gitignore
README
LICENSE
environment / dependency files
data licensing
model licensing
dataset licensing
```

Do not assume every dataset or model artifact can be redistributed.

---

# 39. Current Development Stage

This project is currently in an **initial development stage**.

Therefore:

- repository structure may change,
- some experiments may be only scaffolds,
- some notebooks may be incomplete,
- results may not exist yet,
- documentation may describe intended rather than implemented behavior.

AI assistants should prioritize:

```text
understand
   ↓
audit
   ↓
test
   ↓
implement
   ↓
run experiments
   ↓
analyze
   ↓
document
   ↓
publish
```

rather than prematurely polishing final conclusions.

---

# 40. Recommended Near-Term Workflow

```text
1. Audit current repository
        ↓
2. Run existing tests/notebooks
        ↓
3. Identify IMPLEMENTED vs PLANNED functionality
        ↓
4. Fix baseline reproducibility
        ↓
5. Freeze initial benchmark manifest
        ↓
6. Complete Experiment 1 pipeline
        ↓
7. Run E1 experiments
        ↓
8. Analyze + human-audit ambiguous cases
        ↓
9. Draft/finalize Blog 1
        ↓
10. Complete Experiment 2
        ↓
11. Draft/finalize Blog 2
        ↓
12. Complete Experiment 3
        ↓
13. Draft/finalize Blog 3
```

Git versioning should be introduced early enough that experimental changes and results can be associated with specific commits.

---

# 41. Definition of Project Success

The project is successful when it provides more than a working demo.

It should make it possible to answer:

> Does grounding actually improve reliability?

> What kinds of hallucinations remain?

> How trustworthy are the annotations used to evaluate grounding?

> When does external memory outperform full context?

> Which retrieval strategy works best for which temporal question?

> Where does retrieval fail?

> Where does reasoning fail despite correct retrieval?

> What does multimodal memory cost in latency and memory?

> What configuration provides the best practical tradeoff?

And, from an interview perspective:

> Can the project owner explain, defend, critique, and redesign the system without relying on the code or prepared notes?

That is the standard this project should aim for.
