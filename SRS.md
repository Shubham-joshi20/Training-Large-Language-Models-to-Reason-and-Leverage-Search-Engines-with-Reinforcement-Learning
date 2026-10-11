# Software Requirements Specification

## 1. Document Purpose

This SRS defines the requirements for a lightweight reinforcement-learning system that enables a language model to use a search index while answering Natural Questions. It is an adaptation of Search-R1 for student-project hardware and time constraints.

## 2. Scope

### In scope

- Natural Questions preprocessing and answer normalization.
- A local Wikipedia passage corpus and reproducible retriever.
- Structured multi-turn reasoning and search rollouts.
- Qwen0.5B baseline inference and parameter-efficient RL training.
- Outcome-based reward computation.
- Evaluation against no-search and fixed-RAG baselines.
- Kaggle training and checkpoint export/import.

### Out of scope

- Reproducing the paper's Qwen2.5-3B/7B multi-GPU results.
- A live commercial search API as a required dependency.
- Training a retriever from scratch.
- Claims of real-time web freshness when using a static Wikipedia index.
- Full NQ-scale training before the smaller pipeline is validated.

## 3. Users and Stakeholders

- **Student developers:** implement, debug, and document the system.
- **Project evaluator:** assess the four deliverables, experimental rigor, and results.
- **End user/researcher:** submit a question and inspect the answer, searches, evidence, and metrics.

## 4. System Overview

```text
NQ data -> preprocessing -> train/validation/test manifests
Wikipedia -> passage chunking -> BM25/dense index
Question -> policy rollout -> search parser -> retriever -> evidence insertion
                                      |
                              answer + trajectory
                                      |
                       reward -> GRPO/parameter-efficient update
```

The policy model produces text. The environment, not the model, performs retrieval and inserts `<information>` blocks. A rollout ends at a valid `<answer>` block or a configured token/search limit.

## 5. Functional Requirements

| ID | Requirement | Priority | Acceptance evidence |
|---|---|---|---|
| FR-01 | The system shall load NQ records and preserve question IDs, questions, answer annotations, and split information. | Must | Loader test and manifest |
| FR-02 | The system shall normalize short answers for case, punctuation, articles, and whitespace during scoring. | Must | Unit tests with equivalent answers |
| FR-03 | The system shall build or load a versioned Wikipedia passage index. | Must | Index manifest and retrieval smoke test |
| FR-04 | The retriever shall return a configurable top-k set of passages for a query. | Must | Deterministic top-k test |
| FR-05 | The environment shall detect valid `<search>query</search>` calls and reject or truncate malformed calls safely. | Must | Parser tests |
| FR-06 | The environment shall insert retrieved evidence inside `<information>` tags and continue generation. | Must | End-to-end rollout trace |
| FR-07 | The environment shall stop on `<answer>` or configured maximum tokens/search turns. | Must | Termination tests |
| FR-08 | The system shall record question, generated trajectory, search queries, retrieved passage IDs, answer, reward, and failure reason. | Must | JSONL trajectory artifact |
| FR-09 | The training pipeline shall support a no-RL baseline and a fixed-RAG baseline. | Must | Baseline metric report |
| FR-10 | The RL pipeline shall update the Qwen0.5B policy using a reproducible GRPO or equivalent parameter-efficient method. | Must | Checkpoint and config |
| FR-11 | Retrieved evidence tokens shall not receive policy loss when the selected training method supports token masking. | Should | Loss-mask inspection or documented limitation |
| FR-12 | The evaluator shall report EM, token F1, valid-search rate, search count, response length, and latency. | Must | Machine-readable metrics |
| FR-13 | The system shall export a checkpoint, tokenizer, config, metrics, and manifest for Kaggle-to-local transfer. | Must | Downloaded artifact loads locally |

## 6. Non-Functional Requirements

- **Reproducibility:** configurations, dataset manifests, seeds, package versions, and checkpoint steps shall be recorded.
- **Resource efficiency:** the first experiment shall run on a single modest GPU or Kaggle GPU using a documented subset; CPU inference should remain available for smoke tests.
- **Robustness:** missing fields, empty retrieval results, malformed model output, and early termination shall produce logged failures rather than crashes.
- **Observability:** every evaluation run shall produce JSON/CSV metrics and a small sample of readable trajectories.
- **Maintainability:** data processing, retrieval, rollout, reward, training, and evaluation shall be separable modules.
- **Security and data handling:** credentials and large artifacts shall not be committed; external dataset terms shall be respected.

## 7. Data Requirements

Each processed example should contain `id`, `question`, `answers`, and `split`. The loader may retain long-answer annotations and document metadata for analysis, but scoring initially uses short answers. The split assignment must not change between runs. The initial experiment should use a fixed subset with a recorded seed.

## 8. Reward and Evaluation Requirements

The initial reward shall be simple and auditable:

```text
reward = answer_correctness - optional_format_penalty - optional_search_cost
```

The answer correctness component should be the normalized EM score, with token F1 used for analysis. Search-cost penalties must be small and reported separately so the model is not rewarded for avoiding useful searches. A later extension may add evidence support or multi-reward components, but it must not replace the first reproducible outcome-based experiment.

## 9. Constraints and Assumptions

- Qwen0.5B is assumed to fit the selected inference/training setup; quantization or LoRA may be required.
- Static Wikipedia retrieval is an approximation of a search engine and does not provide live web access.
- NQ mirrors may differ in schema and licensing; the dataset loader must be adapted after inspection.
- Full RL training may exceed laptop resources and should be performed on Kaggle when necessary.
- A small subset is acceptable for pipeline validation, but final claims must state the subset size and limitations.

## 10. Acceptance Criteria

The project is acceptable when:

1. A question can pass through preprocessing, retrieval, rollout, answer extraction, and scoring without manual intervention.
2. The same seed and configuration reproduce the reported baseline metrics within a documented tolerance.
3. The RL system produces loadable artifacts and at least one trajectory containing a valid search call.
4. Results compare all three systems: no-search, fixed RAG, and search-RL.
5. The report includes failure cases, resource usage, limitations, and the distinction between this adaptation and the original Search-R1 scale.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Qwen0.5B cannot learn reliable structured tags | Start with parser tests, prompt/template constraints, and a small supervised formatting warm-up if needed. |
| RL is unstable or too expensive | Use LoRA/QLoRA, short rollouts, GRPO groups of small size, checkpoint frequently, and keep fixed-RAG as a valid result. |
| Retrieval corpus is too large for local storage | Start with a smaller Wikipedia subset or prebuilt passages; record the corpus limitation. |
| NQ mirror schema is inconsistent | Inspect files, write a schema adapter, and save a processed manifest. |
| Reward hacking through malformed or repeated searches | Validate output, cap search turns, log trajectories, and report search cost. |
