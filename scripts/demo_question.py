"""Show one end-user Search-RL trajectory, including model and tool steps.

Examples:
python scripts/demo_question.py --question "Who wrote Hamlet?" \
  --passages data/processed/passages.jsonl
python scripts/demo_question.py --question "Who wrote Hamlet?" \
  --passages data/processed/passages.jsonl \
  --adapter artifacts/qwen05b-search-rl-think-100-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.retrieval import BM25Retriever
from search_rl.rollout import SEARCH_R1_SPECIAL_TOKENS, SearchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"additional_special_tokens": SEARCH_R1_SPECIAL_TOKENS})
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    environment = SearchEnvironment(BM25Retriever.from_jsonl(args.passages), max_searches=2, top_k=3)

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

    trajectory = environment.run(args.question, generate)
    print("QUESTION:")
    print(args.question)
    print("\nMODEL OUTPUT:")
    print(trajectory.text)
    print("\nSEARCH EVENTS:")
    if not trajectory.searches:
        print("No search call was made.")
    for index, event in enumerate(trajectory.searches, start=1):
        print(f"\nSearch {index}: {event.query}")
        for passage in event.passages:
            print(f"  <information id={passage.get('id')}> {passage.get('text')} </information>")
    print("\nFINAL ANSWER:")
    print(trajectory.answer)
    print("\nTERMINATION:", trajectory.termination)


if __name__ == "__main__":
    main()
