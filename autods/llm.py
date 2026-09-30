"""Provider-agnostic LLM layer for AutoDS Copilot.

Every piece of generated prose in the copilot (grounded Ask answers, the written
insights, the recommendation rationale) routes through this one module. Three
providers are supported, chosen by the ``AUTODS_LLM_PROVIDER`` environment
variable or auto-detected from whichever key is present:

    openai     -> OpenAI or any OpenAI-compatible endpoint (needs OPENAI_API_KEY)
    anthropic  -> Anthropic Claude                          (needs ANTHROPIC_API_KEY)
    ollama     -> a local model served by Ollama            (no key, fully offline)

If no provider is available, :func:`complete` returns ``None`` and every caller
falls back to its rule-based text. So the tool always works offline with no key,
and simply gets richer, grounded prose when a model is wired in.

Nothing here ever prints or logs a key, and provider SDKs are imported lazily so
they stay optional dependencies.
"""
from __future__ import annotations

import json
import os
import urllib.request

from . import config

# Sensible per-provider defaults, overridden by AUTODS_LLM_MODEL.
_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1",
}

# Keep a local Ollama model loaded between calls so steps do not pay the reload
# cost each time. Override with OLLAMA_KEEP_ALIVE (e.g. "30m" or "-1" for forever).
_OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")


# ---------------------------------------------------------------------------
#  Provider resolution
# ---------------------------------------------------------------------------
def provider() -> str:
    """Resolve the active provider: 'openai' | 'anthropic' | 'ollama' | 'none'."""
    choice = (config.LLM_PROVIDER or "auto").strip().lower()
    if choice in ("openai", "anthropic", "ollama", "none"):
        return choice
    # auto-detect
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _ollama_reachable():
        return "ollama"
    return "none"


def available() -> bool:
    """True if a usable LLM provider is configured."""
    return provider() != "none"


def model_name() -> str:
    """The model id to call, honouring AUTODS_LLM_MODEL then a per-provider default."""
    return config.LLM_MODEL or _DEFAULT_MODEL.get(provider(), "")


def describe() -> str:
    """A short human-readable line for logs and the UI, never exposing keys."""
    p = provider()
    return "rule-based (no LLM configured)" if p == "none" else f"{p} · {model_name()}"


# ---------------------------------------------------------------------------
#  Single-shot completion
# ---------------------------------------------------------------------------
def complete(system: str, user: str,
             temperature: float | None = None,
             max_tokens: int | None = None) -> str | None:
    """Run one grounded completion and return the text.

    Returns ``None`` on any failure (no provider, missing SDK, network or API
    error) so the caller can fall back to rule-based output. Never raises.
    """
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    p = provider()
    try:
        if p == "openai":
            return _openai(system, user, temperature, max_tokens)
        if p == "anthropic":
            return _anthropic(system, user, temperature, max_tokens)
        if p == "ollama":
            return _ollama(system, user, temperature, max_tokens)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
#  Provider adapters (SDKs imported lazily so they stay optional)
# ---------------------------------------------------------------------------
def _openai(system: str, user: str, temperature: float, max_tokens: int) -> str | None:
    from openai import OpenAI  # reads OPENAI_API_KEY, and OPENAI_BASE_URL if set
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name(),
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip() or None


def _anthropic(system: str, user: str, temperature: float, max_tokens: int) -> str | None:
    import anthropic  # reads ANTHROPIC_API_KEY
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model_name(),
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [blk.text for blk in resp.content if getattr(blk, "type", "") == "text"]
    return "\n".join(parts).strip() or None


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_reachable() -> bool:
    """Probe the local Ollama server quickly. Skipped unless auto-detecting."""
    try:
        with urllib.request.urlopen(_ollama_host() + "/api/tags", timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama(system: str, user: str, temperature: float, max_tokens: int) -> str | None:
    body = json.dumps({
        "model": model_name(),
        "stream": False,
        "keep_alive": _OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        _ollama_host() + "/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    return (data.get("message", {}).get("content") or "").strip() or None


# ---------------------------------------------------------------------------
#  Streaming completion — yields text chunks as they are generated
# ---------------------------------------------------------------------------
def stream(system: str, user: str,
           temperature: float | None = None,
           max_tokens: int | None = None):
    """Yield the completion in text chunks as the model produces them.

    Yields nothing if no provider is configured, and stops quietly on any error
    after what it has already yielded, so callers can fall back cleanly. Never
    raises.
    """
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    p = provider()
    try:
        if p == "openai":
            yield from _openai_stream(system, user, temperature, max_tokens)
        elif p == "anthropic":
            yield from _anthropic_stream(system, user, temperature, max_tokens)
        elif p == "ollama":
            yield from _ollama_stream(system, user, temperature, max_tokens)
    except Exception:
        return


def _openai_stream(system, user, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name(), temperature=temperature, max_tokens=max_tokens, stream=True,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    for chunk in resp:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        piece = choices[0].delta.content
        if piece:
            yield piece


def _anthropic_stream(system, user, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model_name(), max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    ) as st:
        for text in st.text_stream:
            if text:
                yield text


def _ollama_stream(system, user, temperature, max_tokens):
    body = json.dumps({
        "model": model_name(),
        "stream": True,
        "keep_alive": _OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        _ollama_host() + "/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:  # Ollama streams newline-delimited JSON objects
            line = line.strip()
            if not line:
                continue
            data = json.loads(line.decode())
            piece = data.get("message", {}).get("content")
            if piece:
                yield piece
            if data.get("done"):
                break
