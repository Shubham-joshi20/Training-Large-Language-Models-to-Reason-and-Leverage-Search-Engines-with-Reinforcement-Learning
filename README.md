# Search-RL: Training a Lightweight LLM to Reason and Use Search

## Project Summary

This project implements a small-scale version of the Search-R1 idea: a language model learns to interleave reasoning, search-engine calls, evidence use, and answer generation through reinforcement learning. The project is based on the paper [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) and the accompanying project presentation.

The original paper trains Qwen2.5-3B/7B models with large-scale multi-GPU resources. Our implementation is an educational, resource-aware adaptation using a Qwen0.5B model, the Natural Questions (NQ) dataset, a local Wikipedia-derived search index, and a smaller training subset that can run on Kaggle or a laptop.

## Problem and Objectives

An ordinary LLM may answer from stale or incomplete parametric memory and may hallucinate when it does not know an answer. A fixed RAG pipeline always retrieves before generation, while a tool-using model may not know when to search or how to formulate a useful query.

Our objectives are to:

1. Build a search-augmented open-domain question-answering system.
2. Allow the model to decide whether and how to search during reasoning.
3. Use reinforcement learning to improve answer correctness and search behavior.
4. Compare the RL system with a no-search and a non-RL RAG baseline.
5. Measure the trade-off between answer quality, search usage, response length, and compute cost.

## Adapted System Design

The model generates a structured response using the following markers:

```text
<think>reasoning and decision to search</think>
<search>search query</search>
<information>retrieved passages inserted by the environment</information>
<think>reasoning using evidence</think>
<answer>final answer</answer>
```

The rollout environment detects `<search>...</search>`, queries the local retriever, inserts the top passages as `<information>...</information>`, and lets generation continue. Retrieved tokens are treated as environment output and are excluded from the policy loss. The initial reward is outcome-based: normalized answer correctness, with an optional small penalty for malformed output, excessive searches, or unsupported answers. GRPO is the primary RL algorithm because it avoids training a separate value model; PPO is a possible extension.

### Components

- **Policy model:** Qwen0.5B (base or instruction-tuned variant, selected after a smoke test).
- **Dataset:** Natural Questions, initially a reproducible subset of train and validation examples.
- **Knowledge source:** Wikipedia passages prepared locally or on Kaggle.
- **Retriever:** A lightweight BM25 baseline first; a dense E5-style retriever is an optional comparison if compute permits.
- **Environment:** Parses search tags, executes retrieval, injects evidence, limits search turns, and records trajectories.
- **Reward/evaluation:** Exact Match and token-level F1 against NQ short answers, plus valid-search rate, average searches, answer length, and latency.
- **Baselines:** Qwen0.5B without search, fixed one-shot RAG, and the RL-trained search policy.

## Dataset and Kaggle Workflow

### Recommended initial scale

Do not begin with all NQ examples. Start with a fixed, documented subset such as 5,000 training examples and 500 validation examples. Increase the subset only after the data pipeline, retriever, rollout parser, and baseline are working. Keep the official NQ test set untouched for final evaluation where licensing and storage allow.

### Option A: Download from Kaggle

1. Create or sign in to a Kaggle account and accept any dataset rules required by the selected NQ dataset page.
2. In Kaggle, open **Settings > API > Create New Token**. This downloads `kaggle.json`.
3. Store the file in the Kaggle notebook secrets/environment, or locally at `%USERPROFILE%\\.kaggle\\kaggle.json`. Do not commit it.
4. Identify the exact dataset slug from the Kaggle URL, then run:

```bash
pip install -q kaggle
kaggle datasets download -d <owner>/<dataset-slug> -p data/raw --unzip
```

5. Inspect the extracted schema before writing the loader. NQ mirrors use different filenames and may provide JSONL, JSON, or preprocessed CSV files.
6. Convert the selected records to the project schema:

```json
{"id": "...", "question": "...", "answers": ["..."], "split": "train"}
```

7. Save the processed files under `data/processed/` and record the source URL, download date, number of examples, and preprocessing version in a manifest.

### Option B: Download in Kaggle, train in Kaggle, evaluate here

1. Create a Kaggle Notebook and turn on **GPU** in the notebook settings.
2. Add the NQ Kaggle Dataset as an input. Do not copy the raw dataset into the Git repository.
3. Clone or upload this project into the notebook, then install dependencies:

```bash
!pip install -q -r requirements.txt
```

4. Inspect the attached dataset before choosing the input file:

```python
from pathlib import Path
for path in Path('/kaggle/input').rglob('*'):
	if path.is_file():
		print(path)
```

5. Prepare the fixed subsets and passages. Replace the input path with the actual file found in Step 4:

```bash
!python scripts/prepare_nq.py --input /kaggle/input/<dataset>/<train-file> \
	--output data/processed/train.jsonl --split train --limit 5000
!python scripts/prepare_nq.py --input /kaggle/input/<dataset>/<validation-file> \
	--output data/processed/validation.jsonl --split validation --limit 500
!python scripts/prepare_passages.py --input /kaggle/input/<dataset>/<train-file> \
	--output data/processed/passages.jsonl --max-records 5000
```

If the Kaggle mirror does not contain `document_text`, attach a Wikipedia passage dataset instead and convert it to JSONL records with `id` and `text`. NQ questions and a separate Wikipedia corpus are then used together.

6. Run the dependency-free rollout test:

```bash
!python scripts/run_rollout_smoke.py --passages data/processed/passages.jsonl
```

7. Run Qwen0.5B inference on a small validation sample. The first run downloads the model into Kaggle's cache:

```bash
!python scripts/run_qwen_baseline.py \
	--questions data/processed/validation.jsonl \
	--passages data/processed/passages.jsonl \
	--output /kaggle/working/artifacts/qwen05b_rollouts.jsonl \
	--limit 100
```

8. Save model checkpoints, tokenizer files, configuration, metrics, trajectories, the processed subset, and the index as a Kaggle notebook output or Dataset version. Parts 1 and 2 are complete when preprocessing, retrieval, smoke rollout, and trajectory logging all work.

9. Part 3 will add RL training. Do not call the Qwen baseline output an RL-trained model; it is the pre-RL comparison required for the experiment.
4. Download the output archive from the Kaggle **Output** tab, or use the Kaggle API:

```bash
kaggle kernels output <owner>/<notebook-slug> -p artifacts/kaggle_output
```

5. Extract the archive locally and place files using this convention:

```text
artifacts/
	qwen05b-search-rl/
		adapter_config.json
		adapter_model.safetensors
		tokenizer.json
		training_config.json
	metrics/
	manifest.json
```

6. Load the adapter with the matching base model and tokenizer. Record the base-model revision, library versions, GPU type, training subset, random seed, and checkpoint step before comparing results.

Never upload Kaggle credentials, private notebook outputs, or unlicensed data to this repository. Large datasets, indexes, checkpoints, and generated outputs should be excluded by `.gitignore` and transferred through Kaggle or another artifact store.

## Expected Experiments

| System | Search | RL | Purpose |
|---|---:|---:|---|
| Parametric baseline | No | No | Measure the model's existing knowledge |
| Fixed RAG baseline | Always/one-shot | No | Measure retrieval without learned search decisions |
| Search-RL model | Learned multi-turn | Yes | Test whether RL improves search and QA |

At minimum, report Exact Match, token F1, valid-search rate, average number of searches, average response length, and inference latency. Include examples of successful and failed trajectories. The main claim is comparative and bounded: whether RL improves this lightweight system on the selected NQ evaluation subset, not whether it matches the original paper's large-scale results.

## Repository Documents

- [Software Requirements Specification](SRS.md)
- [Four-Part Implementation Plan](IMPLEMENTATION_PLAN.md)

## Current Status

Parts 1 and 2 are implemented in `src/`, `scripts/`, and `tests/`. The next step is to run the Kaggle preprocessing and baseline commands, inspect the actual NQ schema, and save the first trajectory artifacts. Parts 3 and 4 cover RL training and final comparison/evaluation.

## References

- Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, COLM 2025: https://arxiv.org/abs/2503.09516
- Natural Questions dataset: https://ai.google.com/research/NaturalQuestions
- Project presentation: `DL_Project.pdf`