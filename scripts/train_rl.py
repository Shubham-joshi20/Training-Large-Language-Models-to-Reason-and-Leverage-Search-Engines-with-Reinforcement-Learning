"""Run a small LoRA/GRPO reinforcement-learning smoke experiment.

This first trainer supplies top-k retrieved context in the prompt so that the
reward and LoRA/GRPO integration can be tested cheaply. It is a fixed-context
RL smoke test, not yet the paper's fully interleaved multi-turn environment.

Example:
python scripts/train_rl.py --train-data data/processed/train.jsonl \
  --passages data/processed/passages.jsonl \
  --output artifacts/qwen05b-search-rl --limit 100 --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import read_records
from search_rl.retrieval import BM25Retriever
from search_rl.rewards import score_trajectory

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SEARCH_PATTERN = re.compile(r"<search>.*?</search>", re.IGNORECASE | re.DOTALL)


def passage_context(retriever: BM25Retriever, question: str, top_k: int = 3) -> str:
    passages = retriever.search(question, top_k=top_k)
    evidence = "\n".join(f"[{item['id']}] {item['text']}" for item in passages)
    return f"<information>\n{evidence}\n</information>"


def build_prompt(question: str, evidence: str) -> str:
    return (
        "Answer the question using the available evidence. Think briefly. "
        "You may write a search query using <search>query</search>. "
        "Always finish with only the short answer inside "
        "<answer>short answer</answer>.\n\n"
        f"Question: {question}\n{evidence}"
    )


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return "".join(str(item) for item in completion)
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    return str(completion)


def reward_function(completions: list[Any], answers: list[list[str]] | None = None, **_: Any) -> list[float]:
    answers = answers or [[] for _ in completions]
    rewards: list[float] = []
    for completion, gold_answers in zip(completions, answers):
        text = completion_text(completion)
        answer_match = ANSWER_PATTERN.search(text)
        prediction = answer_match.group(1).strip() if answer_match else text.strip()
        termination = "answer" if answer_match else "no_valid_action"
        score = score_trajectory(
            prediction,
            gold_answers,
            termination,
            len(SEARCH_PATTERN.findall(text)),
        )
        rewards.append(float(score["reward"]))
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.max_steps < 1:
        raise ValueError("--limit and --max-steps must be positive")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise SystemExit(
            "RL dependencies are missing. Run `pip install -r requirements.txt` first."
        ) from error

    records = [record for record in read_records(args.train_data) if record.get("answers")][:args.limit]
    if not records:
        raise ValueError("No usable records with answers were found in --train-data")
    retriever = BM25Retriever.from_jsonl(args.passages)
    dataset = Dataset.from_list([
        {
            "prompt": build_prompt(
                str(record["question"]),
                passage_context(retriever, str(record["question"]), args.top_k),
            ),
            "answers": record["answers"],
        }
        for record in records
    ])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto",
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    training_args = GRPOConfig(
        output_dir=str(output_path),
        max_steps=args.max_steps,
        # GRPO requires the effective batch size to be divisible by num_generations.
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=128,
        temperature=0.8,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    (output_path / "training_manifest.json").write_text(
        json.dumps({
            "model": args.model,
            "records": len(records),
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "top_k": args.top_k,
            "mode": "fixed-context GRPO smoke test",
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
