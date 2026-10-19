"""Auditable outcome rewards and aggregate trajectory metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .data import answer_exact_match, answer_token_f1


@dataclass(frozen=True)
class RewardConfig:
    format_penalty: float = 0.10
    repeated_search_penalty: float = 0.02


def score_trajectory(
    prediction: str,
    answers: Iterable[str],
    termination: str,
    search_count: int,
    config: RewardConfig | None = None,
) -> dict[str, float | bool]:
    """Return correctness, penalties, and the final scalar reward."""
    config = config or RewardConfig()
    exact_match = answer_exact_match(prediction, answers)
    token_f1 = answer_token_f1(prediction, answers)
    correctness = 0.5 * exact_match + 0.5 * token_f1
    valid_format = termination == "answer"
    format_penalty = config.format_penalty if not valid_format else 0.0
    repeated_searches = max(0, search_count - 1)
    search_penalty = repeated_searches * config.repeated_search_penalty
    reward = correctness - format_penalty - search_penalty
    return {
        "exact_match": exact_match,
        "token_f1": token_f1,
        "correctness": correctness,
        "valid_format": valid_format,
        "format_penalty": format_penalty,
        "search_penalty": search_penalty,
        "reward": reward,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate saved rollout records, including reward statistics."""
    if not results:
        raise ValueError("At least one rollout result is required")
    scored = []
    for result in results:
        if result.get("answers"):
            scored.append(
                score_trajectory(
                    result.get("answer", ""),
                    result["answers"],
                    result.get("termination", "unknown"),
                    len(result.get("searches", [])),
                )
            )
            continue
        # Older rollout files already contain these metrics but not gold answers.
        exact_match = float(result.get("exact_match", 0.0))
        token_f1 = float(result.get("token_f1", 0.0))
        valid_format = result.get("termination") == "answer"
        format_penalty = 0.0 if valid_format else RewardConfig().format_penalty
        search_penalty = max(0, len(result.get("searches", [])) - 1) * RewardConfig().repeated_search_penalty
        scored.append({
            "exact_match": exact_match,
            "token_f1": token_f1,
            "correctness": 0.5 * exact_match + 0.5 * token_f1,
            "valid_format": valid_format,
            "format_penalty": format_penalty,
            "search_penalty": search_penalty,
            "reward": 0.5 * exact_match + 0.5 * token_f1 - format_penalty - search_penalty,
        })
    return {
        "examples": len(results),
        "exact_match": sum(item["exact_match"] for item in scored) / len(scored),
        "token_f1": sum(item["token_f1"] for item in scored) / len(scored),
        "reward": sum(item["reward"] for item in scored) / len(scored),
        "valid_format_rate": sum(item["valid_format"] for item in scored) / len(scored),
        "search_rate": sum(bool(result.get("searches")) for result in results) / len(results),
        "average_searches": sum(len(result.get("searches", [])) for result in results) / len(results),
    }
