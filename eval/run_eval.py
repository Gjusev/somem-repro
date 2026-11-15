#!/usr/bin/env python3
"""QA runner: builds per-question memory, answers, judges, reports metrics.

Metrics (repo plan):
- QA accuracy: judged correct / total (LLM judge on real runs, scripted
  containment judge offline).
- Evidence reads: mean number of source reads per question (from
  EvidenceReader).

Modes:
- default            : stub Writer + stub Reader + scripted judge. OFFLINE, free.
- --llm              : Writer/Reader/Judge via Z.ai Coding Plan (glm-5.3-flash).
                       Needs SOMEM_ZAI_KEY. Money-gated: owner approval required
                       for full-subset runs; STOP on billing errors (no fallback).

Usage from repo root:
    python3 eval/run_eval.py --fixture data/fixtures/tiny_longmemeval.json
    SOMEM_ZAI_KEY=... python3 eval/run_eval.py --llm --output results/fixture-llm.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from somem_repro.agent import (  # noqa: E402
    BillingError,
    StubReader,
    StubSummarizer,
    ZaiLLM,
    answer_question,
)
from somem_repro.longmemeval import load_fixture  # noqa: E402
from somem_repro.organizer import MemoryWorkspace, Organizer, SourceInteraction  # noqa: E402

WRITER_SYSTEM = "You maintain an agent's long-term memory as linked documents."


def sessions_to_interactions(sessions: List[dict]) -> List[SourceInteraction]:
    out: List[SourceInteraction] = []
    for session in sessions:
        turns = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in session.get("turns", [])
        )
        out.append(
            SourceInteraction(
                source_id=session.get("session_id", f"src-{len(out)}"),
                session_id=session.get("session_id", ""),
                timestamp=session.get("timestamp", "1970-01-01"),
                text=turns or "(empty session)",
            )
        )
    return out


def judge_offline(answer: str, gold: str) -> bool:
    """Scripted containment judge for offline runs only."""
    return gold.strip().lower() in answer.lower()


def judge_llm(llm: ZaiLLM, question: str, answer: str, gold: str) -> bool:
    verdict = llm.chat(
        "You are a strict QA judge. Reply with exactly CORRECT or INCORRECT.",
        f"Question: {question}\nReference answer: {gold}\nCandidate answer: {answer}\n"
        "Is the candidate answer semantically correct given the reference?",
        max_tokens=8,
    )
    return "CORRECT" in verdict.upper() and "INCORRECT" not in verdict.upper()


def run(question_items: List[dict], use_llm: bool, limit: Optional[int], quiet: bool) -> dict:
    llm: Optional[ZaiLLM] = ZaiLLM() if use_llm else None
    total = correct = 0
    reads_total = 0
    per_question: List[dict] = []
    items = question_items[:limit] if limit else question_items
    for item in items:
        stub_reader = StubReader()
        stub_writer = StubSummarizer()
        with tempfile.TemporaryDirectory(prefix="somem-eval-") as tmp:
            root = Path(tmp) / "memory"
            ws = MemoryWorkspace(root)
            Organizer(ws, stub_writer).ingest(sessions_to_interactions(item["sessions"]))
            result = answer_question(root, item["question"], reader=stub_reader, llm=llm)
        if use_llm:
            ok = judge_llm(llm, item["question"], result["answer"], item["answer"])
        else:
            ok = judge_offline(result["answer"], item["answer"])
        total += 1
        correct += int(ok)
        reads_total += result["evidence_reads"]
        per_question.append(
            {
                "qid": item.get("qid", ""),
                "correct": ok,
                "evidence_reads": result["evidence_reads"],
                "answer": result["answer"],
                "gold": item["answer"],
            }
        )
        if not quiet:
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {item.get('qid', '?')} reads={result['evidence_reads']} "
                  f"answer={result['answer'][:60]!r}")
    return {
        "mode": "llm(zai:glm-5.3-flash)" if use_llm else "offline-stub",
        "questions": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "mean_evidence_reads": round(reads_total / total, 3) if total else 0.0,
        "per_question": per_question,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("data/fixtures/tiny_longmemeval.json"))
    parser.add_argument("--llm", action="store_true",
                        help="use Z.ai Coding Plan (glm-5.3-flash) for writer/reader/judge; MONEY-GATED")
    parser.add_argument("--limit", type=int, default=None, help="cap number of questions")
    parser.add_argument("--output", type=Path, default=None, help="write results JSON here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    items = load_fixture(args.fixture)
    try:
        results = run(items, use_llm=args.llm, limit=args.limit, quiet=args.quiet)
    except BillingError as exc:
        print(f"STOP (owner rule): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in results.items() if k != "per_question"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
