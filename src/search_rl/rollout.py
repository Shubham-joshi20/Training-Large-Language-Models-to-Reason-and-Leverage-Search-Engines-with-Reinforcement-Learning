"""Search-aware generation loop and trajectory logging."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass
class SearchEvent:
    query: str
    passages: list[dict]


@dataclass
class Trajectory:
    question: str
    text: str
    answer: str
    searches: list[SearchEvent] = field(default_factory=list)
    termination: str = "unknown"
    latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SearchEnvironment:
    def __init__(self, retriever, max_searches: int = 2, top_k: int = 3):
        self.retriever = retriever
        self.max_searches = max_searches
        self.top_k = top_k

    def build_prompt(self, question: str, evidence: str = "") -> str:
        return (
            "Follow this exact output protocol and do not use any other tags: "
            "first write <think>brief reasoning</think>, optionally write "
            "<search>search query</search>, then finish with "
            "<answer>short answer</answer>. Never use <solution> and never "
            "write the answer outside <answer>...</answer>. Search only when useful.\n\n"
            f"Question: {question}\n{evidence}"
        )

    def _evidence_block(self, passages: list[dict]) -> str:
        text = "\n".join(f"[{item.get('id')}] {item.get('text', '')}" for item in passages)
        return f"<information>\n{text}\n</information>\n"

    def run(self, question: str, generate: Callable[[str], str]) -> Trajectory:
        started = time.perf_counter()
        prompt = self.build_prompt(question)
        full_text = ""
        events: list[SearchEvent] = []
        for _ in range(self.max_searches + 1):
            generated = generate(prompt)
            full_text += generated
            answer_match = ANSWER_PATTERN.search(full_text)
            if answer_match:
                termination = "answer"
                answer = answer_match.group(1).strip()
                break
            matches = SEARCH_PATTERN.findall(full_text)
            if len(matches) <= len(events):
                termination = "no_valid_action"
                answer = full_text.strip()
                break
            query = matches[-1].strip()
            passages = self.retriever.search(query, top_k=self.top_k)
            events.append(SearchEvent(query=query, passages=passages))
            prompt = self.build_prompt(question, self._evidence_block(passages))
        else:
            termination = "search_limit"
            answer = full_text.strip()
        return Trajectory(
            question=question,
            text=full_text,
            answer=answer,
            searches=events,
            termination=termination,
            latency_seconds=time.perf_counter() - started,
        )
