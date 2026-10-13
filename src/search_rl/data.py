"""Dataset loading, normalization, and small deterministic subset creation."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


def normalize_answer(text: str) -> str:
    """Apply the standard lightweight QA normalization used for NQ scoring."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_exact_match(prediction: str, answers: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(answer) for answer in answers))


def answer_token_f1(prediction: str, answers: Iterable[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        overlap = len(set(prediction_tokens) & set(answer_tokens))
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _first_value(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if record.get(name) not in (None, ""):
            return record[name]
    return None


def _extract_answers(record: dict[str, Any]) -> list[str]:
    value = _first_value(record, ("answers", "answer", "short_answers", "short_answer"))
    if value is None:
        annotations = record.get("annotations", [])
        if annotations and isinstance(annotations[0], dict):
            value = annotations[0].get("short_answers", [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = value.get("text", value.get("answer", []))
    if isinstance(value, list):
        answers: list[str] = []
        for item in value:
            if isinstance(item, str):
                answers.append(item)
            elif isinstance(item, dict) and item.get("text"):
                answers.append(str(item["text"]))
        return answers
    return []


def normalize_record(record: dict[str, Any], index: int, split: str) -> dict[str, Any] | None:
    question = _first_value(record, ("question", "question_text", "query"))
    if isinstance(question, dict):
        question = question.get("text")
    if not question:
        return None
    answers = _extract_answers(record)
    return {
        "id": str(_first_value(record, ("id", "example_id", "question_id")) or index),
        "question": str(question).strip(),
        "answers": answers,
        "split": str(record.get("split", split)),
    }


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, JSON, or CSV records from a Kaggle dataset export."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "records", "train", "validation", "test"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Could not find a list of records in {path}")


def prepare_subset(input_path: str | Path, output_path: str | Path, split: str,
                   limit: int, seed: int = 42) -> dict[str, int]:
    records = read_records(input_path)
    normalized = [item for i, record in enumerate(records)
                  if (item := normalize_record(record, i, split)) is not None]
    normalized = [item for item in normalized if item["answers"]]
    random.Random(seed).shuffle(normalized)
    selected = normalized[:limit]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    return {"input_records": len(records), "usable_records": len(normalized), "written_records": len(selected)}


def extract_passages(input_path: str | Path, output_path: str | Path,
                     max_records: int = 5000, words_per_passage: int = 120) -> int:
    """Extract searchable chunks from common NQ document-text fields."""
    passages: list[dict[str, str]] = []
    for index, record in enumerate(read_records(input_path)):
        if index >= max_records:
            break
        document = _first_value(record, ("document_text", "context", "passage", "text"))
        if isinstance(document, list):
            document = " ".join(str(item) for item in document)
        if not document:
            continue
        words = str(document).split()
        for offset in range(0, len(words), words_per_passage):
            chunk = " ".join(words[offset:offset + words_per_passage]).strip()
            if chunk:
                passages.append({"id": f"nq-{index}-{offset}", "text": chunk})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(passage, ensure_ascii=True) + "\n")
    return len(passages)
