import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_rl.data import answer_exact_match, answer_token_f1
from search_rl.retrieval import BM25Retriever
from search_rl.rollout import SearchEnvironment


class CoreTests(unittest.TestCase):
    def test_answer_metrics_normalize_articles_and_case(self):
        self.assertEqual(answer_exact_match("The Eiffel Tower", ["eiffel tower"]), 1.0)
        self.assertGreater(answer_token_f1("Paris, France", ["Paris"]), 0.5)


    def test_bm25_returns_relevant_passage_first(self):
        retriever = BM25Retriever([
            {"id": "a", "text": "Hamlet was written by William Shakespeare."},
            {"id": "b", "text": "The Pacific Ocean is very large."},
        ])
        self.assertEqual(retriever.search("who wrote Hamlet", top_k=1)[0]["id"], "a")


    def test_rollout_inserts_evidence_and_stops_at_answer(self):
        retriever = BM25Retriever([{"id": "a", "text": "Shakespeare wrote Hamlet."}])
        environment = SearchEnvironment(retriever, max_searches=1)
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return "<search>Hamlet author</search>" if len(calls) == 1 else "<answer>Shakespeare</answer>"

        trajectory = environment.run("Who wrote Hamlet?", generate)
        self.assertEqual(trajectory.termination, "answer")
        self.assertEqual(trajectory.answer, "Shakespeare")
        self.assertEqual(len(trajectory.searches), 1)
        self.assertIn("<information>", calls[1])


if __name__ == "__main__":
    unittest.main()
