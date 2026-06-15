# Resume Here — Gap-Aware Agentic RAG

Session paused 2026-06-16. Working on **M5 (portfolio-grade eval)** to make the
repo Nvidia-interview-ready: real corpus + hard numbers + plots + results-first
README.

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
- ✅ Tier A separation eval ran once: **AUC 0.66 (single signal)** — embeddings
  barely separate adversarial unanswerables (this MOTIVATES the verification layer).
  NOTE: `eval/results/separation.json` is OLD-SCHEMA from the first run; re-run to
  get the combined-signal CV-AUC and correct keys.
- ✅ Eval harness + plots written (`eval/eval_separation.py`, `eval/eval_endtoend.py`,
  `eval/make_plots.py`, `eval/metrics.py`).
- ❌ Tier B (end-to-end Gap-Aware vs naive RAG) NOT yet run successfully — only
  warm-up reached before pausing. `eval/results/endtoend.jsonl` is empty.

## Next steps (in order)

1. **Re-run Tier A** (fast, ~1 min, regenerates correct separation.json):
   ```
   uv run python eval/eval_separation.py
   ```
2. **Run Tier B** (slow, ~30–60 min on this 8GB CPU box; RESUMABLE — it skips
   questions already in endtoend.jsonl). Start small if impatient:
   ```
   uv run python eval/eval_endtoend.py 12
   ```
3. **Make plots:** `uv run python eval/make_plots.py`
4. **Rewrite README to lead with results** (task #4): headline numbers + plots
   at the top, then architecture, then the 8 lessons.

## Hard-won gotchas (don't relearn these)

- **Never run two Ollama-using scripts at once** on this 8GB box — they contend
  and time out. Run Tier A, then Tier B, sequentially.
- **Verification needs qwen2.5:7b**, not 3b (3b confuses similar facts / flip-flops).
- **Warm up 7b** before a run; it cold-starts slowly. LLM timeout is 420s w/ retry.
- Tier B writes incrementally to `eval/results/endtoend.jsonl` → crash-safe/resumable.
- If the naive baseline abstains too naturally (weak contrast), make its prompt a
  stricter always-answer RAG to represent a true no-gap-awareness baseline.

## Stretch (after M5 numbers are in)

- Bigger eval N; bump corpus size; M4 (sentence-level citations + contradiction
  check); push to GitHub for the portfolio; LinkedIn writeup.
