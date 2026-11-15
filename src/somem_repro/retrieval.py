"""Retrieval over the organized workspace, with evidence-read accounting.

Paper mechanism: at query time the agent chooses what to search and read,
following document links to the original evidence. We implement a stdlib
BM25-lite ranking over topics + sources, plus an ``EvidenceReader`` that
counts every source read — the repo's second reported metric.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .organizer import MemoryWorkspace

_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _bm25_scores(query_tokens: List[str], docs: Dict[str, List[str]], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
    """Classic Okapi BM25 with stdlib-only counting."""
    n_docs = len(docs)
    if n_docs == 0:
        return {}
    df: Dict[str, int] = {}
    for tokens in docs.values():
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    avgdl = sum(len(t) for t in docs.values()) / n_docs
    scores: Dict[str, float] = {}
    for doc_id, tokens in docs.items():
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = math.log((n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
            score += idf * (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * len(tokens) / avgdl))
        scores[doc_id] = score
    return scores


def search(ws: MemoryWorkspace, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """Rank topic and source documents for a query.

    Returns ``[(doc_id, kind, score), ...]`` sorted by score descending where
    ``kind`` is ``"topic"`` or ``"source"``.
    """
    q_tokens = tokenize(query)
    corpus: Dict[str, List[str]] = {}
    for tid in ws.topic_ids():
        doc = ws.load_topic(tid)
        corpus[f"topic:{tid}"] = tokenize(" ".join(doc.lines)) if doc else []
    for sid in ws.source_ids():
        corpus[f"source:{sid}"] = tokenize(ws.load_source(sid))
    scores = _bm25_scores(q_tokens, corpus)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Tuple[str, str, float]] = []
    for doc_id, score in ranked[:top_k]:
        kind, name = doc_id.split(":", 1)
        if score <= 0.0:
            continue
        out.append((name, kind, score))
    return out


@dataclass
class EvidenceReader:
    """Reads documents, following links from topics to their sources.

    Counts every evidence (source) read — this is the repo's second metric,
    standing in for the paper's token/turn cost accounting.
    """

    workspace: MemoryWorkspace
    evidence_reads: int = 0
    read_log: List[str] = field(default_factory=list)

    def read_topic(self, topic_id: str) -> str:
        text = self.workspace.load_topic(topic_id)
        return text.render() if text else ""

    def read_source(self, source_id: str) -> str:
        self.evidence_reads += 1
        self.read_log.append(source_id)
        return self.workspace.load_source(source_id)

    def collect_context(self, query: str, top_k: int = 4, max_sources: int = 4) -> str:
        """Search, read top topics, then follow their evidence links."""
        hits = search(self.workspace, query, top_k=top_k)
        parts: List[str] = []
        # Paper-faithful start: the Reader begins with the Core index and a
        # file inventory before any search hits (not counted as evidence reads).
        core = self.workspace.topics_dir / "core.md"
        if core.exists():
            inventory = "\n".join(
                f"- topics/{tid}.md" for tid in self.workspace.topic_ids()
            ) + "\n" + "\n".join(f"- sources/{sid}.md" for sid in self.workspace.source_ids())
            parts.append(core.read_text(encoding="utf-8") + "\n\nWorkspace inventory:\n" + inventory)
        sources_seen: List[str] = []
        for name, kind, _score in hits:
            if kind == "topic":
                parts.append(self.read_topic(name))
                doc = self.workspace.load_topic(name)
                if doc:
                    for sid in doc.evidence_ids():
                        if sid not in sources_seen and len(sources_seen) < max_sources:
                            sources_seen.append(sid)
            elif kind == "source" and name not in sources_seen and len(sources_seen) < max_sources:
                sources_seen.append(name)
        for sid in sources_seen:
            parts.append(self.read_source(sid))
        return "\n\n---\n\n".join(parts)
