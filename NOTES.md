# NOTES.md — existing-implementation check (Step 1) and design decisions

Date: 2026-11-15. Written before building the repro, per the check-first rule.

## 1. Official code: github.com/Qi202/SoMem (studied via shallow clone, read-only)

Paper: *Can LLM Agents Organize Their Own Memory?* (Fu, Qi, Zhang, Deng, Wang,
Li, Wu, Zhao — CityU HK + Tencent Jarvis Lab, arXiv:2611.11929).

What the official repo is:

- **Language/deps**: Python 3.12+, ~17,000 LOC under `code/src/` + `code/somem/`.
  Heavy dependency stack (`requirements.txt`): `anthropic`, `claude-agent-sdk`,
  `openai`, `httpx`, `jsonschema`, `mcp`, `numpy`, `rank-bm25`,
  `sentence-transformers` (pulls PyTorch), `tiktoken`.
- **Structure**: `src/management/` (memory construction, transactions, patching,
  workspace validation, prompts), `src/markdown/` (document model/parser/writer),
  `src/retrieval/` (BM25 + embedding fusion, evidence ledger/gate, reader agent,
  tool server, views), `src/agent_runtime/` (OpenAI/Anthropic/Claude Code/Codex
  adapters), `src/runtime/` (derived views, billing/capacity/tokenization),
  `somem/` facade (CLI `somem`, `somem-benchmark`, MCP server, protocol data),
  `examples/` (live + offline demos), `docs/` (getting-started, benchmarks,
  results/provenance).
- **Mechanism implemented publicly**: Writer revises Topic documents (incl.
  `topics/core.md`); runtime validates edits (document structure + references)
  and derives Graph, Timeline, Recent views; Reader starts from Core + Recent +
  file inventory, then searches (BM25/embedding), follows links, inspects
  original Sources as evidence. Writer and Reader models are independently
  configurable — the cross-model reuse angle works in the public snapshot.
- **What it deliberately does NOT claim**: the README/docs/results.md state the
  manuscript scores (LoCoMo 88.96, BEAM-100K 54.02, LongMemEval-S 76.50; best
  reported 93.25 / 90.20) **have not been rerun with the public recipes**, the
  CHANGELOG explicitly lists "deferred P1 paper-reproduction/results work", the
  benchmark runner needs separately-downloaded benchmark data, and judging uses
  GPT-4o-mini via OpenRouter (a second paid endpoint). There is also a stated
  turn-budget divergence between public recipes and the manuscript (120 vs 50
  QA turns on LongMemEval-S).

Verdict on running it directly: it is runnable in principle, but not on our
constraints — 3.12+ with PyTorch/sentence-transformers installs, two paid
model endpoints, and full-benchmark downloads before producing any number.

## 2. Third-party repro search

Two web searches (`SoMem github reproduction`, `"Qi202/SoMem" OR SoMem
reproduction third-party`) found **no third-party reproduction repo**: only the
official repo, awesome-lists (tfatykhov, AgentMemoryWorld) and coverage posts
(AGI Hunt, FourWeekMBA, Fathom, Inference Report). The paper appeared on
arXiv 2026-11-09, days before this check — too fresh for repros.

Consequence: this repo is (to our knowledge) the **first independent,
minimal reimplementation** — stated in the README without affiliation claims.

## 3. What this repro adds over the official repo

- **Stdlib-only core** (`src/somem_repro/`): no installs, no PyTorch, no
  embedding downloads — runs on a locked-down system (PEP 668-safe: network
  imports are lazy inside `ZaiLLM.chat` and the HF loader).
- **Offline-first testability**: the whole organizer → validation → derived
  views → retrieval → QA loop runs against deterministic `StubSummarizer` /
  `StubReader` stand-ins; `python3 -m unittest` is green with zero network.
- **Single-endpoint LLM path**: real runs go through one OpenAI-compatible
  endpoint (Z.ai Coding Plan, `glm-5.3-flash` only) for Writer, Reader AND
  judge — no OpenRouter/GPT-4o-mini dependency.
- **Reproducibility on a subset**: a committed 10-question LongMemEval-shaped
  fixture + a runner reporting the two planned metrics (QA accuracy judged,
  evidence reads) end-to-end offline; the real 100–150-question subset is a
  lazy, money-gated download.

## 4. Design decisions / divergences from the paper

- Topics/sources are plain Markdown files with `[[source:<id>]]` evidence
  links; validation rules: link targets must exist, sources immutable, every
  topic keeps ≥1 link. (Paper's validator is stricter; we implement the check
  subset that a stdlib runtime can enforce.)
- Retrieval is BM25-only (no embedding fusion) — that is a known gap vs the
  official stack and is listed in ROADMAP.md.
- Evidence reads (counted by `EvidenceReader`) stand in for the paper's
  token/turn cost accounting; it is our second metric, not the paper's.
- The offline stub judge is substring-containment; the LLM judge on real runs
  is the same pinned model asked for CORRECT/INCORRECT.

## 5. Test command

The suite is verified with exactly: `python3 -m unittest` from the repo root
(31 tests). Do not re-derive a different discovery command when checking.
