# Four-Part Implementation Plan

The project is divided into four equal parts. Each part represents 25% of the project grade/work and ends with a concrete deliverable that can be demonstrated independently.

## Part 1 - Data, Environment, and Reproducible Baseline (25%)

### Goals

- Finalize the NQ source and inspect its schema.
- Build the preprocessing script, answer normalization, split manifest, and small fixed subset.
- Prepare a Wikipedia passage corpus and BM25 retriever.
- Load Qwen0.5B and verify local/Kaggle inference.
- Implement no-search and fixed one-shot RAG baselines.

### Deliverables

- Dataset loader and processed NQ files.
- Retriever index and retrieval smoke-test report.
- Baseline inference script and initial EM/F1 results.
- Environment file, seed/config conventions, and data manifest.

### Exit criteria

A question can be loaded, retrieved, answered, normalized, and scored end to end. Baseline outputs are saved in JSONL with configuration metadata.

## Part 2 - Search Environment and Rollout Logging (25%)

### Goals

- Define the structured generation template.
- Implement parsing for `<think>`, `<search>`, `<information>`, and `<answer>`.
- Add multi-turn retrieval with maximum search turns and token limits.
- Record complete trajectories, retrieved passage IDs, termination reason, and latency.
- Add parser, termination, and failure-mode tests.

### Deliverables

- Search environment/rollout module.
- End-to-end trajectory examples, including valid and malformed outputs.
- Automated tests for parser and retriever integration.
- Search usage statistics for the baseline policy.

### Exit criteria

The model can issue a search call during generation, receive inserted evidence, continue generation, and produce a scored answer without crashing.

## Part 3 - Reward and Reinforcement Learning (25%)

### Goals

- Implement normalized EM/token-F1 scoring and the initial outcome reward.
- Add optional format and search-cost penalties with separate logging.
- Choose and configure GRPO or the smallest stable available RL trainer.
- Use LoRA/QLoRA or another parameter-efficient method if full fine-tuning is not feasible.
- Exclude retrieved environment tokens from the policy loss where supported.
- Train on the fixed subset first, then scale only if stable.

### Deliverables

- Reward module with unit tests.
- RL training script and saved configuration.
- Checkpoints and training curves.
- Before/after trajectory examples and resource report.

### Exit criteria

Training completes for a reproducible smoke run, produces a loadable checkpoint, and shows recorded reward/search statistics. A negative result is acceptable if the experiment is valid and analyzed.

## Part 4 - Evaluation, Kaggle Transfer, and Final Report (25%)

### Goals

- Evaluate no-search, fixed RAG, and search-RL systems on the same held-out subset.
- Report EM, token F1, valid-search rate, average searches, response length, and latency.
- Perform at least one ablation: search-turn limit, retriever type, RL versus no RL, or subset size.
- Export the best checkpoint and metadata from Kaggle and load it locally.
- Analyze successes, failures, hallucinations, retrieval errors, and compute limits.

### Deliverables

- Unified evaluation script and machine-readable metrics.
- Comparison tables and plots.
- Kaggle-to-local artifact transfer demonstration.
- Final technical report, presentation update, and limitations/future-work section.

### Exit criteria

All three systems are compared under a documented protocol, artifacts are reproducible, and the final report states exactly what was trained, on which data, with which compute, and what the results do and do not establish.

## Suggested Initial Milestones

| Milestone | Target outcome |
|---|---|
| M1 | NQ schema confirmed and 5,000/500 train-validation subset created |
| M2 | BM25 index and fixed-RAG baseline working |
| M3 | Structured search rollout and trajectory logs working |
| M4 | Reward tests and RL smoke run complete |
| M5 | Main comparison and one ablation complete |
| M6 | Kaggle artifact imported locally and final report prepared |

## Recommended Repository Layout

```text
project/
  data/raw/                 # downloaded data, ignored by git
  data/processed/           # normalized subset and manifests
  index/                    # retriever artifacts, ignored by git
  src/
    data/
    retrieval/
    environment/
    rewards/
    training/
    evaluation/
  configs/
  scripts/
  notebooks/
  artifacts/                # checkpoints and metrics, ignored by git
  tests/
  README.md
  SRS.md
  IMPLEMENTATION_PLAN.md
```