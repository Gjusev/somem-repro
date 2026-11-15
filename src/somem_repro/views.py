"""Derived views the runtime computes from the organized content.

Paper mechanism: the runtime derives temporal (timeline), relational
(relationship graph), and recent-memory views from the organized documents —
they are never written by the agent itself.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .organizer import EVIDENCE_LINK, MemoryWorkspace


def timeline_view(ws: MemoryWorkspace) -> List[dict]:
    """Interactions ordered by timestamp, with the topics that cite them."""
    entries: List[dict] = []
    for sid in ws.source_ids():
        text = ws.load_source(sid)
        meta: Dict[str, str] = {}
        for line in text.splitlines():
            if line.startswith("session: "):
                meta["session_id"] = line.split("session: ", 1)[1].strip()
            elif line.startswith("time: "):
                meta["timestamp"] = line.split("time: ", 1)[1].strip()
        citing = [
            tid
            for tid in ws.topic_ids()
            if sid in (ws.load_topic(tid).evidence_ids() if ws.load_topic(tid) else [])
        ]
        entries.append(
            {
                "source_id": sid,
                "session_id": meta.get("session_id", ""),
                "timestamp": meta.get("timestamp", ""),
                "cited_by_topics": citing,
            }
        )
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def relationship_graph(ws: MemoryWorkspace) -> Dict[str, Set[str]]:
    """Adjacency: topic -> set of evidence source ids (bipartite graph)."""
    graph: Dict[str, Set[str]] = defaultdict(set)
    for tid in ws.topic_ids():
        doc = ws.load_topic(tid)
        if doc is not None:
            graph[tid] = set(doc.evidence_ids())
    return dict(graph)


def recent_view(ws: MemoryWorkspace, k: int = 5) -> List[dict]:
    """The k most recent interactions (tail of the timeline view)."""
    return timeline_view(ws)[-k:]
