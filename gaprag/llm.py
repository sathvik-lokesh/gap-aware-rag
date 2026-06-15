"""Ollama chat client used by the agent loop. Includes a keep-alive and a
retry, because 7b on an 8GB CPU box can occasionally stall past the timeout
under memory pressure — a second attempt usually lands once weights are warm."""
from __future__ import annotations
import requests

from .config import OLLAMA_URL, CHAT_MODEL

TIMEOUT = 420
RETRIES = 2


def chat(prompt: str, system: str = "", model: str = CHAT_MODEL,
         temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
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
