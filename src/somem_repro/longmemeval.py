"""LongMemEval-S data loading.

Two paths:
- ``load_fixture``: a 10-question synthetic fixture under ``data/fixtures/``
  (committed, offline, used by the test suite).
- ``load_longmemeval_s``: the real LongMemEval-S subset (500 questions) from
  the HuggingFace dataset. Network imports are LAZY; downloading costs data
  bandwidth and real runs against it cost LLM budget (~$20-40 estimate,
  owner-approval-gated). Tests never call this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "tiny_longmemeval.json"


def load_fixture(path: Path = FIXTURE_PATH) -> List[dict]:
    """Load the committed 10-question offline fixture (LongMemEval-shaped)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_hf_rows(dataset: str, split: str, token: str) -> Iterator[dict]:
    """Stream rows from the HuggingFace datasets-server (lazy, network)."""
    import urllib.request

    base = f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config=default&split={split}"
    offset = 0
    page = 100
    while True:
        url = f"{base}&offset={offset}&length={page}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            return
        for row in rows:
            yield row["row"]
        offset += len(rows)


def normalize_question(row: dict) -> dict:
    """Map a raw LongMemEval row to the repo's question schema."""
    return {
        "qid": str(row.get("qid", "")),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "sessions": row.get("haystack_sessions", []),
        "session_ids": row.get("haystack_session_ids", []),
        "question_type": row.get("question_type", ""),
    }


def load_longmemeval_s(limit: int = 150, token: str = "") -> List[dict]:
    """Download a subset of LongMemEval-S from HuggingFace.

    MONEY/DATA GATE: callers must obtain owner approval before invoking this.
    Returns at most ``limit`` normalized questions (paper subset: 100-150).
    """
    dataset = os.environ.get("SOMEM_LME_DATASET", "xiaowu0162/LongMemEval")
    token = token or os.environ.get("HF_TOKEN", "")
    out: List[dict] = []
    for row in _iter_hf_rows(dataset, "s", token):
        out.append(normalize_question(row))
        if len(out) >= limit:
            break
    return out
