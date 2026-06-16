# Resume Here — Gap-Aware Agentic RAG

Session paused 2026-06-17. Working on **M5 (portfolio-grade eval)** to make the
repo Nvidia-interview-ready: real corpus + hard numbers + plots + results-first
README.

> **NEW THIS SESSION — Groq backend.** Added a hosted LLM path so the slow
> verification calls don't crawl on the local 7b. Set `GAPRAG_LLM_BACKEND=groq`
> (needs `GROQ_API_KEY`, already in env) and `chat()` routes to Groq's
> OpenAI-compatible API. Model map in `gaprag/config.py`: `qwen2.5:7b →
> llama-3.3-70b-versatile`, `qwen2.5:3b → llama-3.1-8b-instant`. Default backend
> is still local `ollama` (reproducible). Smoke-tested: ~0.1–0.2s/call vs 420s
> local timeout. `_chat_groq` now backs off on 429 (free tier TPM cap).

## Current state

**Done and working (M1–M3):** full local gap-aware RAG — instrumented retriever,
calibrated gap detector, agent loop with LLM verification. See README + the two
demos (`scripts/demo_gap.py`, `scripts/demo_agent.py`). Validated on the toy
corpus (CEO answered, CFO abstained, etc.).

**M5 in progress:**
- ✅ Real corpus built: `corpus_squad/` (110 Wikipedia paragraphs → 540 chunks;
  topics: Harvard, EU law, Immune system, Sky UK) from SQuAD 2.0.
- ✅ Labeled question set: `eval/questions.json` (160 answerable + 160 unanswerable).
- ✅ Calibrated index saved: `index_store_squad/` (regenerate with
  `scripts/build_index_squad.py`, ~12 min).
- ✅ Tier A separation eval **RE-RUN with correct schema** (`separation.json`):
  **ROC-AUC 0.662 (top1 sim only), 0.647 combined-signal 5-fold CV** — embeddings
  barely separate adversarial unanswerables (this MOTIVATES the verification
  layer). GAP verdict flags 25% of unanswerables / wrongly flags 16% of
  answerables (those get rescued later by LLM verification). DONE — don't re-run.
- ✅ Eval harness + plots written (`eval/eval_separation.py`, `eval/eval_endtoend.py`,
  `eval/make_plots.py`, `eval/metrics.py`).
- 🔶 Tier B (end-to-end Gap-Aware vs naive RAG) **STARTED via Groq, 14/100 done**
  and cached in `eval/results/endtoend.jsonl` (RESUMABLE — skips cached questions).
  Stopped mid-run on a Groq 429 before the backoff fix landed; now fixed. Just
  re-run the same command tomorrow and it picks up from #14.

## Next steps (in order)

1. **Finish Tier B** (RESUMABLE — skips the 14 already cached). Use Groq, it's
   fast and the 429 backoff is now in place:
   ```
   GAPRAG_LLM_BACKEND=groq uv run python eval/eval_endtoend.py 50
   ```
   (50/label = 100 questions. Drop the number if you want a quicker pass.)
2. **Make plots:** `uv run python eval/make_plots.py`
3. **Rewrite README to lead with results** (task #4): headline numbers + plots
   at the top, then architecture, then the 8 lessons.

## Hard-won gotchas (don't relearn these)

- **Never run two Ollama-using scripts at once** on this 8GB box — they contend
  and time out. Run Tier A, then Tier B, sequentially.
- **Verification needs qwen2.5:7b**, not 3b (3b confuses similar facts / flip-flops).
- **Warm up 7b** before a run; it cold-starts slowly. LLM timeout is 420s w/ retry.
- Tier B writes incrementally to `eval/results/endtoend.jsonl` → crash-safe/resumable.
- If the naive baseline abstains too naturally (weak contrast), make its prompt a
  stricter always-answer RAG to represent a true no-gap-awareness baseline.
- **Groq free tier is TPM-limited**, so a long eval WILL hit 429s — the agent
  fires several token-heavy verify calls per question. `_chat_groq` now honors
  `Retry-After` + exponential backoff (8 attempts), so it self-throttles; just
  let it run. `warm_up()` is skipped on the Groq backend (no local 7b to warm).
- **Embeddings stay local** (`nomic-embed-text` via Ollama) even on the Groq
  backend — only `chat()`/verification is hosted. Ollama must still be running.

## Stretch (after M5 numbers are in)

- Bigger eval N; bump corpus size; M4 (sentence-level citations + contradiction
  check); push to GitHub for the portfolio; LinkedIn writeup.
