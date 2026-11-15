"""Memory workspace, validated edits, and the Organizer loop.

Mechanism (paper section: self-organized memory):
- Memory is a set of ordinary documents in a workspace directory:
  ``sources/`` hold the ORIGINAL interactions and are immutable evidence;
  ``topics/`` hold agent-written summaries, each summary line carrying an
  evidence link ``[[source:<id>]]`` back to the interaction it summarizes;
  ``topics/core.md`` is a rolling index of topics.
- Every edit goes through the runtime, which validates it BEFORE acceptance:
  evidence links must point at existing sources, sources may never be
  rewritten, and every topic must keep at least one evidence link.

The Organizer is model-agnostic: it drives any callable matching
``Summarizer``. Tests use a deterministic stub; real runs use ``ZaiLLM``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

EVIDENCE_LINK = re.compile(r"\[\[source:([A-Za-z0-9_.-]+)\]\]")


class EditError(ValueError):
    """Raised when the runtime rejects a proposed memory edit."""


@dataclass
class SourceInteraction:
    """One original interaction (e.g. a session turn pair). Immutable once stored."""

    source_id: str
    session_id: str
    timestamp: str  # ISO-8601 date or datetime string
    text: str

    def render(self) -> str:
        header = f"# source:{self.source_id}\n\nsession: {self.session_id}\ntime: {self.timestamp}\n\n"
        return header + self.text


@dataclass
class TopicDoc:
    """An agent-written summary document linking back to evidence."""

    topic_id: str
    title: str
    lines: List[str] = field(default_factory=list)

    def evidence_ids(self) -> List[str]:
        seen: List[str] = []
        for line in self.lines:
            for m in EVIDENCE_LINK.finditer(line):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
        return seen

    def render(self) -> str:
        out = [f"# topic:{self.topic_id} {self.title}", ""]
        out.extend(self.lines)
        return "\n".join(out) + "\n"


Summarizer = Callable[[str, List[dict]], List[dict]]
"""Contract: (new_interaction_text, existing_topic_summaries) -> list of ops.

Each op is a dict:
  {"op": "create", "topic_id": ..., "title": ..., "lines": [...]}
  {"op": "update", "topic_id": ..., "title": ..., "lines": [...]}   # full rewrite
  {"op": "skip", ...}
"""


class MemoryWorkspace:
    """Validated document store on disk (``sources/`` + ``topics/``)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.sources_dir = self.root / "sources"
        self.topics_dir = self.root / "topics"
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        # content hashes of stored sources, to enforce immutability
        self._source_hashes: Dict[str, str] = {}

    # -- reads -------------------------------------------------------------
    def source_ids(self) -> List[str]:
        return sorted(p.stem for p in self.sources_dir.glob("*.md"))

    def topic_ids(self) -> List[str]:
        return sorted(p.stem for p in self.topics_dir.glob("*.md") if p.stem != "core")

    def load_source(self, source_id: str) -> str:
        return (self.sources_dir / f"{source_id}.md").read_text(encoding="utf-8")

    def load_topic(self, topic_id: str) -> Optional[TopicDoc]:
        path = self.topics_dir / f"{topic_id}.md"
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        title = topic_id.replace("-", " ")
        body: List[str] = []
        for line in lines[1:]:
            if line.startswith("# ") and "topic:" in line:
                title = line.split("topic:" + topic_id, 1)[1].strip()
                continue
            body.append(line)
        while body and not body[0].strip():
            body.pop(0)
        return TopicDoc(topic_id=topic_id, title=title, lines=body)

    def topic_summaries_for_prompt(self) -> List[dict]:
        out = []
        for tid in self.topic_ids():
            doc = self.load_topic(tid)
            assert doc is not None
            out.append({"topic_id": tid, "title": doc.title, "lines": list(doc.lines)})
        return out

    # -- validated writes ---------------------------------------------------
    def store_source(self, interaction: SourceInteraction) -> None:
        path = self.sources_dir / f"{interaction.source_id}.md"
        text = interaction.render()
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise EditError(f"source:{interaction.source_id} is immutable evidence; refusing rewrite")
        if path.exists():
            return
        path.write_text(text, encoding="utf-8")

    def _validate_lines(self, topic_id: str, lines: Iterable[str]) -> None:
        lines = list(lines)
        linked = EVIDENCE_LINK.findall("\n".join(lines))
        if not linked:
            raise EditError(f"topic:{topic_id} has no [[source:...]] evidence link")
        known = set(self.source_ids())
        for sid in linked:
            if sid not in known:
                raise EditError(f"topic:{topic_id} links unknown evidence source:{sid}")

    def apply_create(self, topic_id: str, title: str, lines: List[str]) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", topic_id):
            raise EditError(f"invalid topic id: {topic_id!r}")
        if (self.topics_dir / f"{topic_id}.md").exists():
            raise EditError(f"topic:{topic_id} already exists; use update")
        self._validate_lines(topic_id, lines)
        self._write_topic(TopicDoc(topic_id, title, lines))

    def apply_update(self, topic_id: str, title: str, lines: List[str]) -> None:
        if not (self.topics_dir / f"{topic_id}.md").exists():
            raise EditError(f"topic:{topic_id} does not exist; use create")
        self._validate_lines(topic_id, lines)
        self._write_topic(TopicDoc(topic_id, title, lines))

    def _write_topic(self, doc: TopicDoc) -> None:
        (self.topics_dir / f"{doc.topic_id}.md").write_text(doc.render(), encoding="utf-8")
        self._rewrite_core()

    def _rewrite_core(self) -> None:
        rows = ["# core", "", "Topic index (one row per topic document).", ""]
        for tid in self.topic_ids():
            doc = self.load_topic(tid)
            assert doc is not None
            evidences = doc.evidence_ids()
            rows.append(f"- {doc.title} ({len(evidences)} evidence links) -> topics/{tid}.md")
        (self.topics_dir / "core.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # -- persistence of raw interactions between runs -----------------------
    def save_manifest(self, path: Optional[Path] = None) -> None:
        path = path or (self.root / "manifest.json")
        sources = [
            {"source_id": p.stem, "text": p.read_text(encoding="utf-8")}
            for p in self.sources_dir.glob("*.md")
        ]
        path.write_text(json.dumps({"sources": sources}, indent=2), encoding="utf-8")


class Organizer:
    """Drives a Summarizer over a stream of interactions with validated edits."""

    def __init__(self, workspace: MemoryWorkspace, summarizer: Summarizer):
        self.workspace = workspace
        self.summarizer = summarizer

    def ingest(self, interactions: Iterable[SourceInteraction]) -> dict:
        stats = {"sources": 0, "creates": 0, "updates": 0, "skips": 0, "rejected": 0}
        for inter in interactions:
            self.workspace.store_source(inter)
            stats["sources"] += 1
            existing = self.workspace.topic_summaries_for_prompt()
            ops = self.summarizer(
                inter.text,
                existing + [{"source_id": inter.source_id, "time": inter.timestamp}],
            )
            for op in ops:
                try:
                    if op.get("op") == "create":
                        self.workspace.apply_create(
                            op["topic_id"], op.get("title", op["topic_id"]), op["lines"]
                        )
                        stats["creates"] += 1
                    elif op.get("op") == "update":
                        self.workspace.apply_update(
                            op["topic_id"], op.get("title", op["topic_id"]), op["lines"]
                        )
                        stats["updates"] += 1
                    else:
                        stats["skips"] += 1
                except (EditError, KeyError) as exc:
                    stats["rejected"] += 1
                    if isinstance(exc, KeyError):
                        raise
        return stats
