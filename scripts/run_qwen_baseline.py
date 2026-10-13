"""Run Qwen0.5B search-aware inference and save JSONL trajectories.

Example:
python scripts/run_qwen_baseline.py --questions data/processed/validation.jsonl \
  --passages data/processed/passages.jsonl --output artifacts/baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import answer_exact_match, answer_token_f1
from search_rl.retrieval import BM25Retriever
from search_rl.rollout import SearchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--output", default="artifacts/qwen05b_rollouts.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    retriever = BM25Retriever.from_jsonl(args.passages)
    environment = SearchEnvironment(retriever, max_searches=2, top_k=3)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.questions).open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for index, line in enumerate(source):
            if index >= args.limit:
                break
            item = json.loads(line)

            def generate(prompt: str) -> str:
                encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[0][encoded["input_ids"].shape[1]:]
                return tokenizer.decode(new_tokens, skip_special_tokens=True)

            trajectory = environment.run(item["question"], generate)
            result = trajectory.to_dict()
            result["id"] = item.get("id", str(index))
            result["exact_match"] = answer_exact_match(trajectory.answer, item.get("answers", []))
            result["token_f1"] = answer_token_f1(trajectory.answer, item.get("answers", []))
            target.write(json.dumps(result, ensure_ascii=True) + "\n")
            print(f"{index + 1}: EM={result['exact_match']:.0f} F1={result['token_f1']:.3f}")


if __name__ == "__main__":
    main()
