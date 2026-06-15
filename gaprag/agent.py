"""M3 — the agentic gap-aware loop.

Pipeline per question:
  1. DECOMPOSE into sub-questions (multi-hop via decomposition).
  2. For each sub-question: retrieve + run the calibrated gap detector.
       - below the absolute relevance floor -> abstain (cheap; no LLM spend).
       - otherwise -> LLM VERIFICATION: read the retrieved passages and confirm
                      the SPECIFIC fact is present. This catches factual gaps the
                      embeddings cannot (CEO present vs CFO absent). The gap
                      detector's verdict is reported as a prior, not a hard gate.
  3. COMPOSE an honest final answer from only the verified facts, explicitly
     naming any gap so the system never hallucinates over missing knowledge.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .index import Index
from .retriever import retrieve
from .gapdetector import assess, Verdict
from .llm import chat
from .config import CHAT_MODEL, CFG, GAP

# Grounding verification needs precision: 3b confuses similar facts (charging vs
# operating time) and flip-flops on presence. 7b is reliable. We use ONE model
# for the whole agent so Ollama keeps a single set of weights warm in RAM —
# mixing 3b/7b on an 8GB box forces a 4.7GB reload from disk on every switch.
AGENT_MODEL = CHAT_MODEL


# ----------------------------- decomposition -----------------------------
# Small models love to OVER-decompose: they rephrase simple questions and
# invent conditions that aren't in the corpus, manufacturing fake gaps. So we
# (a) only invoke the LLM for questions that actually look compound, and
# (b) instruct it to split verbatim, never rephrasing or adding conditions.
_DECOMPOSE_SYS = (
    "Split a COMPOUND question into its standalone parts. Preserve the original "
    "wording of each part as closely as possible. Do NOT rephrase, do NOT add "
    "conditions or assumptions, do NOT invent new questions. Output one "
    "sub-question per line and nothing else.\n"
    "Example:\n"
    "Q: What battery does the F2 use and what is its top speed?\n"
    "What battery does the F2 use?\n"
    "What is the F2's top speed?"
)

_COMPOUND_MARKERS = (" and ", " & ", " as well as ", "; ")


def _looks_compound(question: str) -> bool:
    q = question.lower()
    if question.count("?") > 1:
        return True
    return any(m in q for m in _COMPOUND_MARKERS)


def decompose(question: str) -> list[str]:
    # Simple question -> use verbatim. This alone fixes the bulk of small-model
    # decomposition drift.
    if not _looks_compound(question):
        return [question]
    raw = chat(f"Q: {question}", system=_DECOMPOSE_SYS, model=AGENT_MODEL)
    subs = [ln.strip(" -*\t") for ln in raw.splitlines() if ln.strip()]
    subs = [s for s in subs if len(s) > 3 and "?" in s]
    # Guard: if the model produced nothing usable or just one part, fall back.
    return subs if len(subs) >= 2 else [question]


# ----------------------------- verification ------------------------------
_VERIFY_SYS = (
    "You are a strict grounding checker. Given a QUESTION and PASSAGES, decide "
    "ONLY from the passages whether the specific answer is explicitly stated. "
    "Never use outside knowledge. Read EVERY passage; the answer may be in any "
    "one of them. Match the EXACT thing asked and do not confuse related but "
    "different facts (e.g. charging time vs operating time). Respond in EXACTLY "
    "this format:\n"
    "SUPPORTED: yes|no\n"
    "ANSWER: <the answer, or NONE>\n"
    "EVIDENCE: <a short quote from the passages, or NONE>\n\n"
    "Example 1:\n"
    "QUESTION: Who is the CEO?\n"
    "PASSAGES:\n[P0] The firm has 240 staff. The CEO is Mara Delacroix.\n"
    "SUPPORTED: yes\nANSWER: Mara Delacroix\nEVIDENCE: The CEO is Mara Delacroix\n\n"
    "Example 2:\n"
    "QUESTION: Who is the CFO?\n"
    "PASSAGES:\n[P0] The CEO is Mara Delacroix. It raised a Series A.\n"
    "SUPPORTED: no\nANSWER: NONE\nEVIDENCE: NONE\n\n"
    "Example 3 (do not confuse similar quantities):\n"
    "QUESTION: How long does the battery last on a charge?\n"
    "PASSAGES:\n[P0] A full charge takes about 75 minutes.\n"
    "[P1] Operating time is about 6 hours at -20C.\n"
    "SUPPORTED: yes\nANSWER: about 6 hours\nEVIDENCE: Operating time is about 6 hours"
)


@dataclass
class Verification:
    supported: bool
    answer: str
    evidence: str


def _parse_verify(text: str) -> Verification:
    supported, answer, evidence = False, "NONE", "NONE"
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("supported:"):
            supported = "yes" in low.split(":", 1)[1]
        elif low.startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        elif low.startswith("evidence:"):
            evidence = line.split(":", 1)[1].strip()
    return Verification(supported, answer, evidence)


def verify_fact(question: str, passages: list[str]) -> Verification:
    # Cap passages: a shorter prompt starts generating sooner, which matters a
    # lot on CPU. Top hits are the ones most likely to contain the fact anyway.
    passages = passages[:4]
    ctx = "\n\n".join(f"[P{i}] {p}" for i, p in enumerate(passages))
    prompt = f"QUESTION: {question}\n\nPASSAGES:\n{ctx}"
    out = chat(prompt, system=_VERIFY_SYS, model=AGENT_MODEL)
    return _parse_verify(out)


# ------------------------------- per sub-q -------------------------------
@dataclass
class SubAnswer:
    subq: str
    verdict: Verdict
    top1_percentile: float
    supported: bool
    answer: str
    evidence: str
    source_doc: str
    gap_reason: str = ""

    @property
    def is_gap(self) -> bool:
        return not self.supported


def answer_subquestion(index: Index, subq: str, top_k: int = CFG.top_k) -> SubAnswer:
    res = retrieve(index, subq, top_k=top_k)
    a = assess(index, res)
    src = res.hits[0].chunk.doc

    # Cheap abstain ONLY when retrieval returned nothing topically related.
    # The GAP verdict itself is a prior, NOT a gate: a fact can be present yet
    # score low (e.g. "Who is the CEO?"), so anything above the floor still goes
    # to the LLM verifier, which is the real arbiter.
    if res.hits[0].score < GAP.hard_abstain_floor:
        return SubAnswer(subq, a.verdict, a.top1_percentile, supported=False,
                         answer="", evidence="", source_doc=src,
                         gap_reason="nothing topically related retrieved")

    # Verify the specific fact against the retrieved passages.
    passages = [h.chunk.text for h in res.hits]
    v = verify_fact(subq, passages)
    gap_reason = "" if v.supported else (
        "retrieved passages are topically related but do not state this fact")
    return SubAnswer(subq, a.verdict, a.top1_percentile, supported=v.supported,
                     answer=v.answer, evidence=v.evidence, source_doc=src,
                     gap_reason=gap_reason)


# ------------------------------- compose ---------------------------------
@dataclass
class AgentResult:
    question: str
    sub_answers: list[SubAnswer]
    final_answer: str
    abstained: bool = False
    gaps: list[str] = field(default_factory=list)


def _compose(question: str, verified: list[SubAnswer],
             gaps: list[SubAnswer]) -> str:
    """Deterministic composition: the final answer is built ONLY from verified
    facts plus explicitly named gaps. No LLM call -> no generative freedom ->
    nothing to hallucinate. Each fact carries its source for provenance."""
    if len(verified) == 1 and not gaps:
        s = verified[0]
        return f"{s.answer}  [source: {s.source_doc}]"

    lines = []
    for s in verified:
        lines.append(f"- {s.subq} -> {s.answer}  [source: {s.source_doc}]")
    if gaps:
        named = "; ".join(g.subq for g in gaps)
        lines.append(f"- Not covered by the corpus: {named}")
    return "\n".join(lines)


def run(index: Index, question: str) -> AgentResult:
    subs = decompose(question)
    sub_answers = [answer_subquestion(index, s) for s in subs]

    verified = [s for s in sub_answers if s.supported]
    gaps = [s for s in sub_answers if s.is_gap]

    if not verified:
        named = "; ".join(g.subq for g in gaps)
        return AgentResult(
            question, sub_answers,
            final_answer=f"I can't answer from the corpus. No coverage for: {named}",
            abstained=True, gaps=[g.subq for g in gaps])

    return AgentResult(question, sub_answers,
                       final_answer=_compose(question, verified, gaps),
                       abstained=False, gaps=[g.subq for g in gaps])
