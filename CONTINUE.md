# CONTINUE.md — read this first when an agent picks up this repo

## What this repo is

Repro of SoMem (arXiv:2611.11929, official code Qi202/SoMem): agent memory as
agent-organized linked documents with immutable evidence and derived views.
Implemented: stdlib-only organizer/views/retrieval/QA loop, offline-tested
(31 tests), plus a money-gated LLM path on one endpoint. Honest status:
mechanism implemented + offline-tested; **paper numbers NOT reproduced**.
Read README.md and ROADMAP.md fully; NOTES.md has the official-repo study.

## Working rules

- **Offline-first**: the suite stays zero-dependency and green with exactly
  `python3 -m unittest` from the repo root. Network imports stay lazy inside
  the method that uses them (PEP 668 systems may have no pip).
- **Honest claims**: never state benchmark numbers that haven't been run;
  label all stub/fixture output as mechanism checks, never as results.
- Independent implementation of arXiv:2611.11929 — never imply affiliation
  with the authors (Qi202/SoMem). Commits: English, conventional style.
- Before pushing: run the full offline suite; it must be green.

## Models — Z.ai Coding Plan (MANDATORY, no other provider)

```bash
export SOMEM_ZAI_KEY="<Z.ai Coding Plan key>"
# base: https://api.z.ai/api/coding/paas/v4   (already coded in src/somem_repro/agent.py)
# model: glm-5.3-flash ONLY — the code refuses any other model name
```

- Flat subscription. On a billing/quota/endpoint error the code raises
  `BillingError` and the run exits with status 2: **STOP and report** — do NOT
  switch providers or models to "get it working".
- Full-subset runs (~$20–40 estimate) are **owner-approval-gated**: propose,
  wait for explicit OK, then run.

## Current task (do this now)

1. Run ROADMAP item 1 (real-model fixture run) if `SOMEM_ZAI_KEY` is set:
   `SOMEM_ZAI_KEY=... python3 eval/run_eval.py --llm --limit 10 --output results/fixture-llm.json`,
   then commit the results file plus a `results/README.md` quoting the command.
2. Compare stub vs LLM per-question verdicts and note disagreements in NOTES.md.
3. If the owner approves budget: item 2 (LongMemEval-S subset, `--limit 150`),
   results committed the same way with a subset-vs-paper caveat.

**STOP point**: if no `SOMEM_ZAI_KEY` is available, stop after step 0 —
commit any doc/mechanism changes with the offline suite green, leave
`results/` empty with a "not yet run" note, and report. Do not attempt any
other model or endpoint.

## After that

Follow ROADMAP.md in order. One item per commit series.
