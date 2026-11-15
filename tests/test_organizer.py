"""Validated-edit and Organizer tests (offline, deterministic)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from somem_repro.agent import StubSummarizer
from somem_repro.organizer import (
    EditError,
    MemoryWorkspace,
    Organizer,
    SourceInteraction,
)


def make_ws(tmp: str) -> MemoryWorkspace:
    return MemoryWorkspace(Path(tmp) / "memory")


def make_interaction(sid: str = "s-1", session: str = "sess-1", ts: str = "2025-01-01", text: str = "user: I drink jasmine tea every morning.\nassistant: Noted.") -> SourceInteraction:
    return SourceInteraction(
        source_id=sid,
        session_id=session,
        timestamp=ts,
        text=text,
    )


class TestWorkspaceValidation(unittest.TestCase):
    def test_store_source_is_idempotent_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ws.store_source(make_interaction())
            ws.store_source(make_interaction())  # identical rewrite is a no-op
            with self.assertRaises(EditError):
                ws.store_source(make_interaction(text="user: changed facts"))
            self.assertEqual(ws.source_ids(), ["s-1"])

    def test_create_requires_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            with self.assertRaises(EditError):
                ws.apply_create("drinks", "Drinks", ["- tea [[source:missing]]"])
            ws.store_source(make_interaction())
            ws.apply_create("drinks", "Drinks", ["- jasmine tea [[source:s-1]]"])
            self.assertEqual(ws.topic_ids(), ["drinks"])

    def test_create_requires_at_least_one_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ws.store_source(make_interaction())
            with self.assertRaises(EditError):
                ws.apply_create("drinks", "Drinks", ["- jasmine tea, no link"])

    def test_update_on_missing_topic_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ws.store_source(make_interaction())
            with self.assertRaises(EditError):
                ws.apply_update("ghost", "Ghost", ["- x [[source:s-1]]"])

    def test_create_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ws.store_source(make_interaction())
            ws.apply_create("drinks", "Drinks", ["- tea [[source:s-1]]"])
            with self.assertRaises(EditError):
                ws.apply_create("drinks", "Drinks", ["- tea [[source:s-1]]"])

    def test_invalid_topic_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ws.store_source(make_interaction())
            with self.assertRaises(EditError):
                ws.apply_create("../escape", "Evil", ["- x [[source:s-1]]"])


class TestOrganizerLoop(unittest.TestCase):
    def test_stub_ingest_creates_topic_and_core_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            organizer = Organizer(ws, StubSummarizer())
            stats = organizer.ingest([make_interaction("s-1"), make_interaction("s-2", "sess-1", "2025-01-02")])
            self.assertEqual(stats["sources"], 2)
            self.assertEqual(stats["creates"], 1)  # single topic-s0 in one workspace
            self.assertEqual(stats["updates"], 1)
            self.assertEqual(stats["rejected"], 0)
            core = (ws.topics_dir / "core.md").read_text(encoding="utf-8")
            self.assertIn("topics/topic-s0.md", core)
            doc = ws.load_topic("topic-s0")
            self.assertEqual(doc.evidence_ids(), ["s-1", "s-2"])

    def test_malformed_op_dict_is_counted_not_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ws.store_source(make_interaction())
            organizer = Organizer(ws, lambda text, existing: [{"op": "create", "topic_id": "bad"}])
            with self.assertRaises(KeyError):
                organizer.ingest([make_interaction("s-2")])


if __name__ == "__main__":
    unittest.main()
