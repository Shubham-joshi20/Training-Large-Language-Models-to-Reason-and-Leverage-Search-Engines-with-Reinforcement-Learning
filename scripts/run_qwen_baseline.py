"""Run Qwen0.5B search-aware inference and save JSONL trajectories.

Examples:
python scripts/run_qwen_baseline.py --questions data/processed/validation.jsonl \
  --passages data/processed/passages.jsonl --output artifacts/baseline.jsonl
python scripts/run_qwen_baseline.py --questions data/processed/validation.jsonl \
    --passages data/processed/passages.jsonl --adapter artifacts/qwen05b-search-rl \
    --output artifacts/rl_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import answer_exact_match, answer_token_f1
from search_rl.retrieval import BM25Retriever
from search_rl.prompts import format_chat_prompt
from search_rl.rollout import SEARCH_R1_SPECIAL_TOKENS, SearchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--output", default="artifacts/qwen05b_rollouts.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--plain-output", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"additional_special_tokens": SEARCH_R1_SPECIAL_TOKENS})
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
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
                encoded = tokenizer(format_chat_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[0][encoded["input_ids"].shape[1]:]
                return tokenizer.decode(new_tokens, skip_special_tokens=True)

            if args.plain_output:
                passages = retriever.search(item["question"], top_k=3)
                evidence = "\n".join(
                    f"[{passage.get('id')}] {passage.get('text', '')}"
                    for passage in passages
                )
                plain_prompt = (
                    "Answer the question using the evidence below. Return only the "
                    "shortest correct answer, with no explanation or special tags.\n\n"
                    f"Question: {item['question']}\n<information>\n{evidence}\n</information>"
                )
                generated_answer = generate(plain_prompt).strip()
                result = {
                    "question": item["question"],
                    "text": generated_answer,
                    "answer": generated_answer,
                    "mode": "plain",
                    "searches": [],
                    "termination": "answer",
                    "latency_seconds": 0.0,
                }
            else:
                trajectory = environment.run(item["question"], generate)
                result = trajectory.to_dict()
            result["id"] = item.get("id", str(index))
            result["answers"] = item.get("answers", [])
            result["exact_match"] = answer_exact_match(result["answer"], result["answers"])
            result["token_f1"] = answer_token_f1(result["answer"], result["answers"])
            target.write(json.dumps(result, ensure_ascii=True) + "\n")
            print(f"{index + 1}: EM={result['exact_match']:.0f} F1={result['token_f1']:.3f}")


if __name__ == "__main__":
    main()
