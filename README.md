# Gap-Aware Agentic RAG

A retrieval-augmented system that models **the boundary of its own knowledge**.
Instead of always answering, it estimates whether a question is actually
answerable from its corpus — and when it isn't, it **abstains and names the
gap** instead of hallucinating.

Fully local: Ollama (`nomic-embed-text` for embeddings, `qwen2.5` for
reasoning). No cloud, no paid APIs.

## Why this is different from normal RAG

Normal RAG retrieves the top-k chunks and feeds them to an LLM, which answers
*no matter what* — even when the corpus contains nothing relevant. Gap-Aware RAG
adds two ideas:

1. **Calibrated gap detection.** Retrieval is instrumented with *coverage
   signals*, and an answerability verdict is computed by comparing the query's
   best match against *what an answerable query looks like in this corpus*
   (not a hard-coded threshold).
2. **Two-layer gap reasoning.** The embedding layer catches *topical* gaps; an
   agentic verification layer (in progress) catches *fine-grained factual* gaps
   that embeddings cannot see.

## Architecture

```
Question
   │  decompose (planner, qwen2.5:7b)              [M3 - next]
   v
INSTRUMENTED RETRIEVER  -- coverage signals --> GAP DETECTOR (calibrated)
   top1 / score_gap / density / concentration      ANSWERABLE / PARTIAL / GAP
   │                                                      │
   └──────────────────────────────────────────────┐      │
                                                   v      v
            ANSWERABLE -> draft   PARTIAL -> verify/multi-hop   GAP -> abstain + name gap
                                          │
                                          v
                        CONTRADICTION CHECK -> provenance answer   [M4 - next]
```

## Status

- [x] **M1 - Instrumented retriever**: top-k + coverage signals
      (`top1`, `score_gap`, `density`, `concentration`).
- [x] **M2 - Calibrated gap detector**: LLM-pseudo-query calibration;
      ANSWERABLE / PARTIAL / GAP verdicts.
- [x] **M3 - Agent loop**: decompose -> retrieve+assess -> LLM verify -> compose.
      The gap verdict is a *prior*, not a gate; the LLM verifier is the arbiter
      and resolves the factual gaps embeddings can't see.
- [ ] **M4 - Verification + provenance**: contradiction check, sentence-level
      citations, honest confidence.
- [ ] **M5 - Eval harness**: Gap-Aware vs naive-RAG on answerable + gap sets;
      measure hallucination / abstention accuracy.

## Four lessons learned building M1+M2 (the deep part)

1. **nomic-embed-text needs task prefixes.** Documents -> `search_document: `,
   queries -> `search_query: `. Without them, query/document similarity
   collapses and *everything* looks like a gap. (`embeddings.py`)
2. **The calibration yardstick must match the query distribution.** Calibrating
   query->doc scores against doc->doc similarity is apples to oranges -- short
   questions never land as close as two paragraphs do. (`index.py`)
3. **Use realistic pseudo-queries.** Re-embedding whole chunks as queries is
   unrealistically easy. We ask a small LLM to write the short questions each
   chunk answers, and calibrate against *those*. (`calibrate.py`)
4. **Chunk granularity dominates precision.** 700-char multi-topic chunks
   dilute specific facts; ~320-char chunks let a pointed question find its
   pointed answer. One knob, huge effect. (`config.py`)

## The key empirical result (now resolved by M3)

`Who is the CEO?` (answer present) and `Who is the CFO?` (answer absent) embed
almost identically, retrieve the *same* leadership chunk, and both get the
embedding-layer verdict **GAP (p18)**. Embeddings see topical coverage, not
factual coverage. The M3 verification layer reads that chunk and cleanly
separates them:

| Question | Embedding verdict | After LLM verification |
|---|---|---|
| Who is the CEO? | GAP (p18) | **answered:** "Mara Delacroix" |
| Who is the CFO? | GAP (p18) | **abstains** (fact not in passage) |
| How long does the battery last? | PARTIAL (p55) | **answered:** "about 6 hours" (not the 75-min charging-time distractor) |

Tuning thresholds can't separate CEO from CFO; architecture can.

## More lessons from M3 (the agent layer)

5. **Small models over-decompose.** `qwen2.5:3b` rewrites simple questions and
   invents conditions not in the corpus, manufacturing fake gaps. Gate
   decomposition behind a compound-question heuristic and force verbatim
   splitting. (`agent.py:decompose`)
6. **Verification needs a capable model.** 3b confuses similar facts (charging
   vs operating time) and flip-flops on presence; few-shot made it reject
   *everything*. The grounding-verification step uses `qwen2.5:7b`.
7. **One warm model beats mixing sizes.** On an 8GB CPU box, switching 3b/7b
   per call forces a 4.7GB reload from disk every time. Use one model for the
   whole agent and warm it once with `keep_alive`. (`scripts/demo_agent.py`)
8. **The gap verdict is a prior, not a gate.** Hard-abstaining on a GAP verdict
   would kill the very case verification exists for. Abstain cheaply only below
   an absolute relevance floor; otherwise let the LLM verifier decide.
   (`agent.py:answer_subquestion`)

## Run

```bash
uv run python scripts/demo_gap.py
```

## Layout

```
gaprag/
  config.py        # all tunables (models, chunking, gap thresholds)
  embeddings.py    # Ollama embeddings with the required task prefixes
  index.py         # numpy vector store + calibrated answerable-query distribution
  ingest.py        # chunk -> embed -> calibrated Index
  calibrate.py     # LLM-generated realistic pseudo-queries for calibration
  retriever.py     # instrumented retrieval + coverage signals
  gapdetector.py   # coverage signals -> calibrated verdict   <- novel core
  llm.py           # Ollama chat (used by the agent loop)
scripts/demo_gap.py
data/              # fictional Aurelia Robotics corpus (controlled gaps)
```
