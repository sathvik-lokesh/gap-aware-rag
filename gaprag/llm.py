"""Chat client used by the agent loop. Two backends behind one chat() call:

- "ollama" (default): fully local. Includes keep-alive + retry, because 7b on an
  8GB CPU box can stall past the timeout under memory pressure — a second attempt
  usually lands once weights are warm.
- "groq":  hosted OpenAI-compatible endpoint, ~10x faster for the slow
  verification calls. Selected via GAPRAG_LLM_BACKEND=groq (needs GROQ_API_KEY).

The rest of the codebase keeps passing Ollama model names; for Groq we translate
them through GROQ_MODEL_MAP so callers don't change."""
from __future__ import annotations
import os
import time
import requests

from .config import (OLLAMA_URL, CHAT_MODEL, LLM_BACKEND, GROQ_URL,
                     GROQ_MODEL_MAP)

TIMEOUT = 420
RETRIES = 2


def _chat_ollama(messages: list[dict], model: str, temperature: float) -> str:
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": "10m", "options": {"temperature": temperature}}
    last_err = None
    for _ in range(RETRIES):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last_err = e
    raise last_err


def _chat_groq(messages: list[dict], model: str, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GAPRAG_LLM_BACKEND=groq but GROQ_API_KEY is not set")
    groq_model = GROQ_MODEL_MAP.get(model, model)
    payload = {"model": groq_model, "messages": messages,
               "temperature": temperature}
    headers = {"Authorization": f"Bearer {key}"}
    # The free tier caps tokens-per-minute, so 429s are expected on a long eval.
    # Honor the Retry-After header when present, else exponential backoff. Plenty
    # of attempts because a single 429 just means "wait a few seconds", not fail.
    last_err = None
    for attempt in range(8):
        try:
            r = requests.post(f"{GROQ_URL}/chat/completions", json=payload,
                              headers=headers, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
                time.sleep(min(wait + 1, 30))
                last_err = requests.exceptions.HTTPError("429 rate limited")
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise last_err


def chat(prompt: str, system: str = "", model: str = CHAT_MODEL,
         temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if LLM_BACKEND == "groq":
        return _chat_groq(messages, model, temperature)
    return _chat_ollama(messages, model, temperature)
