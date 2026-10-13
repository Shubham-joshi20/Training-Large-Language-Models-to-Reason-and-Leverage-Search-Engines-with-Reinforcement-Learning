"""A dependency-light BM25 retriever for a local passage collection."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, passages: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if not passages:
            raise ValueError("At least one passage is required")
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(str(item.get("text", ""))) for item in passages]
        self.lengths = [len(item) for item in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for document in self.tokens:
            self.document_frequency.update(set(document))

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        scored: list[tuple[float, int]] = []
        total_documents = len(self.passages)
        for index, document in enumerate(self.tokens):
            counts = Counter(document)
            score = 0.0
            for term in query_tokens:
                if term not in counts:
                    continue
                document_frequency = self.document_frequency[term]
                idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
                frequency = counts[term]
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                score += idf * frequency * (self.k1 + 1) / denominator
            scored.append((score, index))
        scored.sort(reverse=True)
        results = []
        for score, index in scored[:top_k]:
            result = dict(self.passages[index])
            result["score"] = round(score, 6)
            results.append(result)
        return results

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Retriever":
        with Path(path).open(encoding="utf-8") as handle:
            passages = [json.loads(line) for line in handle if line.strip()]
        return cls(passages)


def save_passages(passages: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, passage in enumerate(passages):
            item = {"id": str(passage.get("id", index)), "text": str(passage["text"])}
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
