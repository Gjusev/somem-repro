"""Agent-layer tests: stubs, owner-rule gates on ZaiConfig, build_memory modes.

No network is touched in this module: ZaiConfig is validated before any
request is made, and the client's ``chat`` method is never called here.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from somem_repro.agent import BillingError, StubReader, StubSummarizer, ZaiConfig, build_memory
from somem_repro.organizer import SourceInteraction


class TestStubWriter(unittest.TestCase):
    def test_stub_links_evidence_and_upgrades_to_update(self):
        stub = StubSummarizer()
        ops1 = stub("user: I like tea.", [{"source_id": "s-1", "time": "2025-01-01"}])
        ops2 = stub("user: I like tea.", [
            {"topic_id": "topic-s0", "title": "Session s0", "lines": ["- first [[source:s-1]]"]},
            {"source_id": "s-2", "time": "2025-01-02"},
        ])
        self.assertEqual(ops1[0]["op"], "create")
        self.assertIn("[[source:s-1]]", ops1[0]["lines"][0])
        self.assertEqual(ops2[0]["op"], "update")
        self.assertTrue(any("[[source:s-2]]" in line for line in ops2[0]["lines"]))


class TestStubReader(unittest.TestCase):
    def test_extracts_best_overlapping_sentence(self):
        reader = StubReader()
        answer = reader.answer(
            "Where did the user move?",
            "user: I moved to Rotterdam for work. assistant: Congrats.\n\n"
            "user: I drink jasmine tea. assistant: Noted.",
        )
        self.assertIn("Rotterdam", answer)

    def test_insufficient_evidence_fallback(self):
        self.assertEqual(StubReader().answer("???", ""), "insufficient evidence")


class TestOwnerRules(unittest.TestCase):
    """Z.ai Coding Plan gates: pinned model, key STOP, no silent fallback."""

    def test_missing_key_is_stop(self):
        env = {k: v for k, v in os.environ.items() if k != "SOMEM_ZAI_KEY"}
        old = dict(os.environ)
        os.environ.clear()
        os.environ.update(env)
        try:
            with self.assertRaises(BillingError):
                ZaiConfig()
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_alternative_model_refused(self):
        with self.assertRaises(BillingError):
            ZaiConfig(api_key="dummy", model="glm-4.7-air")

    def test_pinned_endpoint_and_model(self):
        cfg = ZaiConfig(api_key="dummy")
        self.assertEqual(cfg.base_url, "https://api.z.ai/api/coding/paas/v4")
        self.assertEqual(cfg.model, "glm-5.3-flash")


class TestBuildMemoryModes(unittest.TestCase):
    def test_sources_only_mode_writes_no_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = build_memory(
                [SourceInteraction("s-1", "sess-1", "2025-01-01", "user: hello")],
                Path(tmp) / "memory",
                llm=None,
            )
            self.assertEqual(stats["mode"], "sources-only")
            ws_root = Path(tmp) / "memory"
            topics = sorted(p.name for p in (ws_root / "topics").iterdir())
            self.assertEqual(topics, [])
            self.assertEqual(len(list((ws_root / "sources").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
