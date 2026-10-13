"""Convert one Kaggle NQ export into the project JSONL schema.

Example:
python scripts/prepare_nq.py --input data/raw/train.jsonl --output data/processed/train.jsonl --limit 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import prepare_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_subset(args.input, args.output, args.split, args.limit, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
