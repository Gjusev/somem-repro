# somem-repro

[![tests](https://github.com/Gjusev/somem-repro/actions/workflows/tests.yml/badge.svg)](https://github.com/Gjusev/somem-repro/actions/workflows/tests.yml)
[![dependencies](https://img.shields.io/badge/dependencies-0-success)]()
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-animated-dark.svg">
    <img src="assets/logo/logo-animated-light.svg" alt="somem-repro: agent memory as linked documents, evidence pulses from source to summary" width="560">
  </picture>
</p>

<p align="center">
  <a href="assets/video/somem-repro.mp4">
    <img src="assets/video/preview.gif" alt="somem-repro: the animated mark beside the headline number, 31 offline tests, zero dependencies" width="720">
  </a>
  <br>
  <sub><a href="assets/video/somem-repro.mp4">Watch the 8-second film (MP4, 0.2 MB)</a>. Every number in it comes from the committed results in this repo.</sub>
</p>

Reproduction/test repo for **SoMem** — *Can LLM Agents Organize Their Own
Memory?* (Fu et al., CityU HK + Tencent Jarvis Lab, [arXiv:2611.11929](https://arxiv.org/abs/2611.11929)).
Official code: [Qi202/SoMem](https://github.com/Qi202/SoMem) (MIT).

**Independent, unofficial reimplementation.** No affiliation with the paper
authors. Per our search (2026-11-15, see NOTES.md) this is the first
third-party repro of the mechanism.

## What the paper does, in one paragraph

Instead of fixed-schema memory records, SoMem keeps agent memory as ordinary
**linked documents**: the agent (Writer) decides what to summarize, which
topics to create, and when to revise; every summary line stays linked to the
original interaction as **immutable evidence**; the runtime validates each
edit and derives timeline / relationship-graph / recent-memory views; at query
time a Reader agent searches, follows links, and inspects evidence before
answering. Gains over the strongest baseline: +2.66 / +3.10 / +2.00 pp on
LoCoMo / BEAM-100K / LongMemEval-S. Notably, memory written by one model was
reused successfully by another — the owner's angle for multi-agent stacks.

## Why this repo exists

The official repo is a full framework (Python 3.12+, PyTorch/sentence-transformers,
two paid model endpoints, full-benchmark downloads, and an explicit disclaimer
that manuscript scores have **not** been rerun with the public recipes). This
repro implements the core mechanism in a **stdlib-only** package that is fully
offline-testable, and wires a single-endpoint LLM path for real runs on a
LongMemEval-S subset. See NOTES.md for the full existing-implementation check.

## Quickstart (offline, free, no API key)

```bash
git clone https://github.com/Gjusev/somem-repro.git
cd somem-repro
python3 -m unittest                      # 31 tests, zero dependencies, no network
python3 eval/run_eval.py --fixture data/fixtures/tiny_longmemeval.json --quiet
```

The eval run answers all 10 fixture questions through the full loop
(organize → validate → derive views → retrieve → read evidence → answer →
judge) using deterministic stubs, and reports:

```json
{"mode": "offline-stub", "questions": 10, "accuracy": 1.0, "mean_evidence_reads": 1.8}
```

This is a **mechanism check with stub stand-ins**, not a benchmark result.

## Real runs (money-gated, Z.ai Coding Plan)

```bash
export SOMEM_ZAI_KEY="<Z.ai Coding Plan key>"   # glm-5.3-flash ONLY
python3 eval/run_eval.py --llm --limit 10 --output results/fixture-llm.json
```

Owner rules, baked into the code: LLM calls go ONLY through
`https://api.z.ai/api/coding/paas/v4` with model `glm-5.3-flash`; any
billing/endpoint error **stops the run** (`BillingError`) — there is no
pay-per-call fallback; full-subset runs (~$20–40 estimated) need explicit
owner approval.

## Layout

```
src/somem_repro/
  organizer.py    MemoryWorkspace (validated edits, immutable sources), Organizer loop
  views.py        derived timeline / relationship graph / recent-memory views
  retrieval.py    stdlib BM25 search + EvidenceReader (counts evidence reads)
  agent.py        StubSummarizer/StubReader, ZaiLLM client (lazy network), build/answer
  longmemeval.py  fixture loader + lazy HuggingFace subset downloader (money-gated)
eval/run_eval.py  end-to-end QA runner: accuracy (judged) + evidence reads
data/fixtures/    tiny_longmemeval.json — 10-question committed fixture
tests/            31 offline tests (python3 -m unittest from repo root)
NOTES.md          existing-implementation check + design decisions
ROADMAP.md        next steps (item 1 = first real-model run)
CONTINUE.md       agent handoff: owner rules, current task, STOP point
```

## Findings so far

- The official public snapshot implements the full mechanism but ships with an
  explicit provenance disclaimer: its historical scores were not rerun with the
  public recipes, and its LongMemEval-S recipe uses a different turn budget
  (120) than the manuscript (50) — reproduction there is still open work.
- No third-party repro existed at check time (2026-11-15); the paper is ~1 week old.
- Lexical-retrieval stub QA exposes the known weakness of extractive readers:
  questions whose wording doesn't lexically overlap the evidence fail even when
  the evidence is present (e.g. "backups" vs "backup"). An LLM Reader removes
  this; the stub is only a mechanism harness.

## Honest status

- ✅ **Mechanism implemented + offline-tested**: validated linked-document
  memory, immutable evidence, derived views, evidence-read counting, QA +
  judging pipeline, 31/31 offline tests green (`python3 -m unittest`).
- ❌ **Paper numbers NOT reproduced.** Nothing here should be read as a
  LoCoMo/BEAM/LongMemEval score. The fixture accuracy above is a stub-vs-fixture
  self-check, by construction. Real-subset reproduction is ROADMAP item 1+ and
  is money-gated (owner approval, Z.ai Coding Plan only).
- ❌ Not yet reproduced either: embedding-fusion retrieval, the paper's full
  validator rule set, and the BEAM rubric-nugget metric.

## License

MIT. The official Qi202/SoMem repo is also MIT; this repo shares no code with it.
