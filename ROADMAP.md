# ROADMAP.md — ordered next steps

1. **First real-model run (MONEY-GATED — needs owner approval).**
   `SOMEM_ZAI_KEY=... python3 eval/run_eval.py --llm --limit 10 --output results/fixture-llm.json`
   on the committed fixture. Acceptance: results JSON with judged accuracy +
   evidence reads per question, committed under `results/` with the exact
   command in a `results/README.md`. STOP on any billing error (no fallback).
2. **LongMemEval-S subset run (100–150 questions), owner-approved.**
   Load via `somem_repro.longmemeval.load_longmemeval_s(limit=150)` (lazy HF
   download). Budget estimate $20–40 with glm-5.3-flash. Acceptance: accuracy +
   mean evidence reads reported next to the paper's 76.50 (S, 500 questions)
   with an explicit subset-vs-full-set caveat — never compare subset numbers to
   the paper's full-set scores as if equivalent.
3. **Writer/Reader split run (owner's multi-agent angle).** Run Writer and
   Reader with separate `ZaiLLM` configs on the same workspace to replicate the
   paper's cross-model reuse table at subset scale; report the transfer gap.
4. **Retrieval upgrade.** Optional embedding fusion behind a lazy import;
   keep BM25 default so offline tests stay dependency-free.
5. **Validator parity.** Port stricter checks from the official runtime
   (document structure rules beyond link existence).
6. **Evidence-read cost model.** Report tokens per question alongside read
   counts once real runs exist.

Standing rule for every item: offline suite green (`python3 -m unittest`)
before pushing; any step that spends API credits ships gated and waits for
explicit owner approval.
