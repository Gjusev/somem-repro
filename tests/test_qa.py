"""End-to-end QA on the tiny fixture: accuracy + evidence-read metrics, offline.

Uses the stub Writer/Reader and the scripted containment judge — no network,
no API key. This validates the MECHANISM and the metric plumbing, NOT the
paper's numbers (those need real LLM runs; see ROADMAP.md item 1).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_eval import judge_offline, run  # noqa: E402
from somem_repro.longmemeval import load_fixture  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "tiny_longmemeval.json"


class TestFixture(unittest.TestCase):
    def test_fixture_has_ten_questions(self):
        items = load_fixture(FIXTURE)
        self.assertEqual(len(items), 10)
        for item in items:
            for key in ("qid", "question", "answer", "sessions"):
                self.assertIn(key, item, f"{item.get('qid')} missing {key}")


class TestOfflineQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = load_fixture(FIXTURE)
        cls.results = run(cls.items, use_llm=False, limit=None, quiet=True)

    def test_all_ten_questions_answered(self):
        self.assertEqual(self.results["questions"], 10)
        self.assertEqual(len(self.results["per_question"]), 10)

    def test_stub_accuracy_is_full_on_fixture(self):
        # The fixture is built so the extractive stub can answer every question
        # from linked evidence; a value below 1.0 means a mechanism regression.
        self.assertEqual(self.results["accuracy"], 1.0, self.results["per_question"])

    def test_evidence_reads_are_counted_per_question(self):
        self.assertGreater(self.results["mean_evidence_reads"], 0)
        for entry in self.results["per_question"]:
            self.assertGreaterEqual(entry["evidence_reads"], 1)

    def test_mode_label_is_offline(self):
        self.assertEqual(self.results["mode"], "offline-stub")

    def test_judge_offline_containment(self):
        self.assertTrue(judge_offline("The answer is Jasmine Tea.", "jasmine tea"))
        self.assertFalse(judge_offline("unrelated", "jasmine tea"))


if __name__ == "__main__":
    unittest.main()
