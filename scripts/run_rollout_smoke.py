"""Run a local rollout smoke test with a deterministic mock generator.

Replace mock_generate with the Qwen generator after installing requirements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.retrieval import BM25Retriever
from search_rl.rollout import SearchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", default="data/processed/passages.jsonl")
    parser.add_argument("--question", default="Who wrote Hamlet?")
    args = parser.parse_args()
    retriever = BM25Retriever.from_jsonl(args.passages)
    environment = SearchEnvironment(retriever)
    calls = 0

    def mock_generate(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "<think>I need evidence.</think><search>author of Hamlet</search>"
        return "<think>The evidence identifies the author.</think><answer>William Shakespeare</answer>"

    trajectory = environment.run(args.question, mock_generate)
    print(json.dumps(trajectory.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
