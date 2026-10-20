"""Teach the model the literal Search-R1 output tags before GRPO.

This is a short supervised formatting warm-up, not the final QA training.
It teaches the model to emit <think>, <search>, and <answer> instead of
plain-text Thought/Search/Answer labels.

Example:
python scripts/train_format_warmup.py --train-data data/processed/train.jsonl \
  --output artifacts/qwen05b-format-warmup --limit 500 --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import read_records


def make_example(record: dict, eos_token: str) -> dict[str, str]:
    question = str(record["question"]).strip()
    answer = str(record["answers"][0]).strip()
    return {
        "prompt": (
            "Question: " + question + "\n"
            "Respond using only this protocol: <think>reasoning</think>, "
            "optional <search>query</search>, then <answer>short answer</answer>.\n"
        ),
        "completion": (
            "<think>I need factual evidence for this question.</think>"
            "<search>" + question + "</search>"
            "<answer>" + answer + "</answer>" + eos_token
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit("Run `pip install -r requirements.txt` first.") from error

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = [record for record in read_records(args.train_data) if record.get("answers")][:args.limit]
    if not records:
        raise ValueError("No usable training records with answers were found")
    dataset = Dataset.from_list([make_example(record, tokenizer.eos_token) for record in records])
    cuda_available = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability()[0] if cuda_available else 0
    use_bfloat16 = cuda_available and capability >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16 if cuda_available else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_path),
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=args.max_steps,
        report_to="none",
        completion_only_loss=True,
        max_length=256,
        fp16=cuda_available and not use_bfloat16,
        bf16=use_bfloat16,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    (output_path / "warmup_manifest.json").write_text(
        json.dumps({"records": len(records), "max_steps": args.max_steps, "mode": "format warm-up"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
