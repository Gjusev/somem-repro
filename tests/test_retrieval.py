"""Retrieval tests: BM25 ranking, evidence reads, link-following context."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from somem_repro.agent import StubSummarizer
from somem_repro.organizer import MemoryWorkspace, Organizer, SourceInteraction
from somem_repro.retrieval import EvidenceReader, search, tokenize


def build(tmp: str) -> MemoryWorkspace:
    ws = MemoryWorkspace(Path(tmp) / "memory")
    interactions = [
        SourceInteraction("s-tea", "sess-1", "2025-01-01", "user: I drink jasmine tea every morning."),
        SourceInteraction("s-move", "sess-2", "2025-02-01", "user: I moved to Rotterdam for work."),
    ]
    Organizer(ws, StubSummarizer()).ingest(interactions)
    return ws


class TestSearch(unittest.TestCase):
    def test_relevant_sources_rank_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            hits = search(ws, "Where did the user move?", top_k=3)
            self.assertTrue(hits)
            source_hits = [h for h in hits if h[1] == "source"]
            self.assertEqual(source_hits[0][0], "s-move")

    def test_empty_query_yields_no_positive_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            self.assertEqual(search(ws, "zzz qqq"), [])

    def test_tokenize_lowercases(self):
        self.assertEqual(tokenize("Jasmine TEA!"), ["jasmine", "tea"])


class TestEvidenceReader(unittest.TestCase):
    def test_read_source_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            er = EvidenceReader(ws)
            er.read_source("s-tea")
            er.read_source("s-tea")  # deliberate double read still counts
            er.read_source("s-move")
            self.assertEqual(er.evidence_reads, 3)
            self.assertEqual(len(er.read_log), 3)

    def test_collect_context_follows_topic_links_and_counts_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            er = EvidenceReader(ws)
            context = er.collect_context("jasmine tea preferences")
            self.assertIn("jasmine tea", context)
            self.assertGreater(er.evidence_reads, 0)

    def test_nearest_neighbor_excludes_nothing_but_scores_positive_only(self):
        # Regression guard (skill pitfall): a query matching everything must
        # still rank the truly relevant doc on top, not itself first.
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            hits = search(ws, "the user moved", top_k=10)
            source_hits = [h for h in hits if h[1] == "source"]
            self.assertEqual(source_hits[0][0], "s-move")


if __name__ == "__main__":
    unittest.main()
