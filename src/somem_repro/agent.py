"""Model-facing layer: stub stand-ins (offline) and the Z.ai client (real runs).

OWNER RULES (baked in, see CONTINUE.md):
- LLM calls go ONLY through the Z.ai Coding Plan:
  base ``https://api.z.ai/api/coding/paas/v4``, model ``glm-5.3-flash`` ONLY,
  key from env ``SOMEM_ZAI_KEY``.
- On a billing/endpoint error this client raises ``BillingError`` and the run
  STOPS. Never fall back to a pay-per-call provider.
- Full-dataset runs cost money (~$20-40 estimate) and are owner-approval-gated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from .organizer import SourceInteraction, Summarizer
from .retrieval import EvidenceReader

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
ZAI_MODEL = "glm-5.3-flash"


class BillingError(RuntimeError):
    """Raised on billing/quota/endpoint errors. The run stops; no fallback."""


@dataclass
class StubSummarizer:
    """Deterministic offline stand-in for the Writer.

    Creates one topic per session id, appending one summary line per
    interaction that quotes the first sentence — enough to exercise the
    organizer, views, retrieval and QA loop without any model.
    """

    session_of: Optional[Callable[[str], str]] = None

    def __call__(self, interaction_text: str, existing: List[dict]) -> List[dict]:
        first_sentence = interaction_text.strip().splitlines()[0][:160]
        session = self.session_of(interaction_text) if self.session_of else "s0"
        topic_id = f"topic-{session}"
        # Organizer appends {"source_id": ..., "time": ...} for the current
        # interaction at the end of `existing`; the stub uses it for evidence links.
        source_id = next(
            (e["source_id"] for e in reversed(existing) if "source_id" in e), "unknown"
        )
        line = f"- {first_sentence} [[source:{source_id}]]"
        is_new = not any(e.get("topic_id") == topic_id for e in existing)
        lines = [line]
        if not is_new:
            previous = next(e for e in existing if e.get("topic_id") == topic_id)
            lines = list(previous.get("lines", [])) + [line]
        return [
            {
                "op": "create" if is_new else "update",
                "topic_id": topic_id,
                "title": f"Session {session}",
                "lines": lines,
            }
        ]


class StubReader:
    """Deterministic offline stand-in for the Reader: extractive QA.

    Scores each candidate sentence by token overlap (regex-normalized,
    same tokenizer family as retrieval) with the question and returns the
    best-scoring sentence — no model, no network, deterministic.
    """

    def answer(self, question: str, context: str) -> str:
        import re

        word = re.compile(r"[a-z0-9]+")
        q_tokens = set(word.findall(question.lower()))
        best_sentence, best_score = "", -1
        for para in context.split("\n\n"):
            for sentence in para.replace("\n", " ").split(". "):
                tokens = set(word.findall(sentence.lower()))
                overlap = len(q_tokens & tokens)
                if overlap > best_score:
                    best_sentence, best_score = sentence.strip(), overlap
        return best_sentence or "insufficient evidence"


class ZaiConfig:
    def __init__(self, api_key: Optional[str] = None, base_url: str = ZAI_BASE_URL, model: str = ZAI_MODEL):
        self.api_key = api_key or os.environ.get("SOMEM_ZAI_KEY", "")
        self.base_url = base_url
        self.model = model
        if self.model != ZAI_MODEL:
            raise BillingError(
                f"owner rule: model is pinned to {ZAI_MODEL}; refusing to run with {self.model!r}"
            )
        if not self.api_key:
            raise BillingError(
                "SOMEM_ZAI_KEY is not set; per owner rules this is a STOP point — "
                "no live runs without the Z.ai Coding Plan key"
            )


class ZaiLLM:
    """OpenAI-compatible client for the Z.ai Coding Plan.

    Network imports (urllib) are lazy and live only in ``chat`` so the
    offline suite never touches the network (PEP 668: no installs allowed).
    """

    def __init__(self, config: Optional[ZaiConfig] = None):
        self.config = config or ZaiConfig()

    def chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (401, 402, 403, 429):
                raise BillingError(
                    f"Z.ai Coding Plan rejected the call (HTTP {exc.code}): {detail}"
                ) from exc
            raise RuntimeError(f"Z.ai HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BillingError(f"Z.ai endpoint unreachable: {exc.reason}") from exc
        return body["choices"][0]["message"]["content"]


def _zai_summarizer_ops(llm: ZaiLLM, interaction_text: str, existing: List[dict]) -> List[dict]:
    import re

    topics = "\n".join(
        f"- {t['topic_id']}: {t['title']} ({len(t.get('lines', []))} lines)" for t in existing if "topic_id" in t
    ) or "(none yet)"
    prompt = (
        "You maintain an agent's long-term memory as linked documents.\n"
        f"Existing topics:\n{topics}\n\n"
        "New interaction:\n" + interaction_text + "\n\n"
        "Return ONLY a JSON array of operations. Each operation is one of:\n"
        ' {"op":"create","topic_id":"<snake-case-id>","title":"...","lines":["- summary text [[source:SOURCE_ID]]"]}\n'
        ' {"op":"update","topic_id":"<existing-id>","title":"...","lines":["- full rewritten lines ... [[source:SOURCE_ID]]"]}\n'
        ' {"op":"skip"}\n'
        "Every summary line MUST link evidence with [[source:SOURCE_ID]] using the id given below.\n"
        "Decide yourself whether to create a new topic, revise an existing one, or skip.\n"
        "SOURCE_ID will be substituted by the harness; write the literal token SOURCE_ID.\n"
        "JSON array only, no prose."
    )
    raw = llm.chat("You are a memory organizer. Output JSON only.", prompt)
    ops = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))  # type: ignore[union-attr]
    return ops


def build_memory(
    interactions: List[SourceInteraction],
    workspace_root,
    llm: Optional[ZaiLLM] = None,
) -> dict:
    """Build memory from interactions. Pass llm=None for a dry structure check."""
    from .organizer import MemoryWorkspace, Organizer

    ws = MemoryWorkspace(workspace_root)
    if llm is None:
        for inter in interactions:
            ws.store_source(inter)
        return {"sources": len(interactions), "creates": 0, "updates": 0, "skips": 0, "rejected": 0, "mode": "sources-only"}
    summarizer: Summarizer = lambda text, existing: _with_source_ids(
        _zai_summarizer_ops(llm, text, existing), existing
    )
    return Organizer(ws, summarizer).ingest(interactions)


def _with_source_ids(ops: List[dict], existing: List[dict]) -> List[dict]:
    """Substitute the SOURCE_ID placeholder with the current interaction's id."""
    current = None
    for entry in existing:
        if "source_id" in entry:
            current = entry["source_id"]
    if current is None:
        return ops
    import json as _json

    blob = _json.dumps(ops).replace("SOURCE_ID", current)
    return _json.loads(blob)


def answer_question(
    workspace_root,
    question: str,
    reader=None,
    llm: Optional[ZaiLLM] = None,
    top_k: int = 4,
) -> dict:
    """Answer one question against a built workspace. Returns answer + metrics."""
    from .organizer import MemoryWorkspace
    from .retrieval import EvidenceReader

    ws = MemoryWorkspace(workspace_root)
    er = EvidenceReader(ws)
    context = er.collect_context(question, top_k=top_k)
    if llm is not None:
        answer = llm.chat(
            "Answer the question using only the context. Be concise.",
            f"Context:\n{context}\n\nQuestion: {question}",
        )
    else:
        reader = reader or StubReader()
        answer = reader.answer(question, context)
    return {
        "question": question,
        "answer": answer,
        "evidence_reads": er.evidence_reads,
        "read_log": list(er.read_log),
    }
