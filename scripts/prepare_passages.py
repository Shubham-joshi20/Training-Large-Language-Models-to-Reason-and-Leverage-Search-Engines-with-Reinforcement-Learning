"""Create BM25 passages from document text in an NQ export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import extract_passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/passages.jsonl")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--words-per-passage", type=int, default=120)
    args = parser.parse_args()
    count = extract_passages(args.input, args.output, args.max_records, args.words_per_passage)
    print(f"Wrote {count} passages to {args.output}")


if __name__ == "__main__":
    main()
