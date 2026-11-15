"""Derived-view tests: timeline, relationship graph, recent memory."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from somem_repro.agent import StubSummarizer
from somem_repro.organizer import MemoryWorkspace, Organizer, SourceInteraction
from somem_repro.views import recent_view, relationship_graph, timeline_view


def build(tmp: str):
    ws = MemoryWorkspace(Path(tmp) / "memory")
    interactions = [
        SourceInteraction("s-b", "sess-b", "2025-02-01", "user: Moved to Rotterdam."),
        SourceInteraction("s-a", "sess-a", "2025-01-01", "user: Lived in Utrecht."),
        SourceInteraction("s-c", "sess-c", "2025-03-01", "user: New job started."),
    ]
    Organizer(ws, StubSummarizer()).ingest(interactions)
    return ws


class TestTimeline(unittest.TestCase):
    def test_sorted_by_timestamp_with_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            tl = timeline_view(ws)
            self.assertEqual([e["source_id"] for e in tl], ["s-a", "s-b", "s-c"])
            self.assertTrue(all("cited_by_topics" in e for e in tl))
            cited = {c for e in tl for c in e["cited_by_topics"]}
            self.assertEqual(cited, {"topic-s0"})


class TestGraph(unittest.TestCase):
    def test_bipartite_adjacency(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            graph = relationship_graph(ws)
            self.assertEqual(graph["topic-s0"], {"s-a", "s-b", "s-c"})

    def test_empty_workspace_graph_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = MemoryWorkspace(Path(tmp) / "memory")
            self.assertEqual(relationship_graph(ws), {})


class TestRecent(unittest.TestCase):
    def test_recent_k_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = build(tmp)
            recent = recent_view(ws, k=2)
            self.assertEqual([e["source_id"] for e in recent], ["s-b", "s-c"])
            self.assertEqual(len(recent_view(ws, k=10)), 3)


if __name__ == "__main__":
    unittest.main()
