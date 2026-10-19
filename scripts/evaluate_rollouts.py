"""Score saved rollout JSONL files and write a metrics JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.rewards import summarize_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/metrics.json")
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        results = [json.loads(line) for line in handle if line.strip()]
    metrics = summarize_results(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
