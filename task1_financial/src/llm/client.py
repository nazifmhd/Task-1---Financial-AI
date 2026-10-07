"""Provider-agnostic LLM client that only ever returns validated objects.

* Talks to any OpenAI-compatible endpoint (Groq or OpenRouter free tiers) via
  the ``openai`` SDK, so switching provider is an env-var change, not a code
  change.
* API keys come from Colab Secrets or environment variables / a git-ignored
  ``.env`` - never from source code.
* ``call_json`` = JSON mode -> extract -> Pydantic validate. On a validation
  failure it logs, feeds the error back to the model and retries (bounded).
  After the last attempt it returns ``None``; callers treat ``None`` as
  "excluded" rather than crashing.
* Every attempt is appended to an audit log (``outputs/llm_calls.jsonl``) and
  to ``VALIDATION_EVENTS`` so failures are visible in the notebook.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'OpenAI-compatible Groq /
# OpenRouter client with JSON mode, Pydantic validation, repair-retry loop,
# 429 back-off and audit logging', Date: 2026-10-07
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

import config
from .prompts import PROMPT_VERSION, REPAIR_USER_TMPL

log = logging.getLogger("llm")

T = TypeVar("T", bound=BaseModel)

# Every validation failure in this process, for display in the notebook.
VALIDATION_EVENTS: list[dict] = []
AUDIT_LOG_PATH = config.OUTPUT_DIR / "llm_calls.jsonl"

_KEY_ENV = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


class LLMUnavailable(RuntimeError):
    """Raised when no provider key is configured."""


def _read_secret(name: str) -> Optional[str]:
    """Colab Secrets first, then env vars / .env. Never logs the value."""
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv(config.PROJECT_DIR / ".env")
        load_dotenv()  # also repo root / cwd
    except ImportError:
        pass
    return os.environ.get(name) or None


def resolve_provider() -> tuple[str, str, str]:
    """Return (provider, model, api_key). Auto-detects from available keys."""
    order = [config.LLM_PROVIDER] if config.LLM_PROVIDER else ["groq", "openrouter"]
    for provider in order:
        key = _read_secret(_KEY_ENV[provider])
        if key:
            model = config.LLM_MODEL or config.LLM_DEFAULT_MODELS[provider]
            return provider, model, key
    raise LLMUnavailable(
        "No LLM key found. Set GROQ_API_KEY or OPENROUTER_API_KEY in Colab "
        "Secrets, the environment, or a git-ignored .env file.")


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions endpoint."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, transport=None):
        """``transport`` lets tests inject a fake ``chat.completions`` object."""
        if transport is not None:
            self.provider, self.model = provider or "fake", model or "fake-model"
            self._chat = transport
        else:
            from openai import OpenAI
            if not api_key:
                provider, model, api_key = resolve_provider()
            self.provider, self.model = provider, model
            self._chat = OpenAI(api_key=api_key,
                                base_url=config.LLM_BASE_URLS[provider],
                                timeout=config.LLM_TIMEOUT_S).chat.completions
        self._json_mode = True
        log.info("LLM client ready: provider=%s model=%s", self.provider, self.model)

    # ------------------------------------------------------------------ #
    def _complete(self, messages: list[dict]) -> str:
        """One completion with JSON mode and 429/5xx back-off."""
        kwargs = dict(model=self.model, messages=messages,
                      temperature=config.LLM_TEMPERATURE,
                      max_tokens=config.LLM_MAX_TOKENS)
        if self.model.startswith(config.REASONING_MODEL_PREFIXES):
            kwargs["extra_body"] = {"reasoning_effort": config.LLM_REASONING_EFFORT}
        retries = 0
        while True:
            if self._json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._chat.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # SDK raises provider-specific subclasses
                status = getattr(exc, "status_code", None)
                if status == 400 and self._json_mode:
                    # Some free OpenRouter models reject response_format; drop
                    # it once and rely on the prompt + extract_json instead.
                    log.warning("JSON mode rejected by %s; falling back to "
                                "prompt-only JSON", self.model)
                    self._json_mode = False
                    kwargs.pop("response_format", None)
                    continue
                if status in (429, 500, 502, 503) and retries < config.LLM_MAX_RETRIES:
                    wait = config.LLM_RATE_LIMIT_BACKOFF_S * (2 ** retries)
                    log.warning("HTTP %s from provider; retrying in %.0fs", status, wait)
                    time.sleep(wait)
                    retries += 1
                    continue
                raise

    # ------------------------------------------------------------------ #
    def call_json(self, system: str, user: str, schema: Type[T],
                  task: str = "") -> Optional[T]:
        """Call the LLM and return a validated ``schema`` instance or None."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                raw = self._complete(messages)
            except Exception as exc:
                log.error("[%s] LLM request failed: %s", task, exc)
                _audit(task, self.model, attempt, ok=False, error=f"request: {exc}",
                       latency=time.perf_counter() - started)
                return None
            try:
                obj = schema.model_validate_json(extract_json(raw))
                _audit(task, self.model, attempt, ok=True,
                       latency=time.perf_counter() - started)
                return obj
            except (ValidationError, ValueError) as exc:
                err = _short_error(exc)
                log.warning("[%s] validation failed (attempt %d/%d): %s", task,
                            attempt + 1, config.LLM_MAX_RETRIES + 1, err)
                VALIDATION_EVENTS.append({"task": task, "attempt": attempt + 1,
                                          "error": err, "raw": raw[:300]})
                _audit(task, self.model, attempt, ok=False, error=err,
                       latency=time.perf_counter() - started)
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user",
                              "content": REPAIR_USER_TMPL.format(error=err)}]
        log.error("[%s] giving up after %d attempts - result excluded", task,
                  config.LLM_MAX_RETRIES + 1)
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> str:
    """Pull the JSON object out of a reply that may carry fences or prose.

    Raises ValueError when no object is present so the caller's repair loop
    handles it like any other validation failure.
    """
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    candidate = text[start:end + 1]
    json.loads(candidate)  # raises json.JSONDecodeError (a ValueError)
    return candidate


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}"
                         for e in exc.errors())
    return str(exc)


def _audit(task: str, model: str, attempt: int, ok: bool, latency: float,
           error: str = "") -> None:
    """Append one line per attempt to the JSONL audit log (best effort)."""
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "task": task, "model": model, "prompt_version": PROMPT_VERSION,
                "attempt": attempt + 1, "ok": ok, "latency_s": round(latency, 3),
                "error": error[:300]}) + "\n")
    except OSError as exc:
        log.debug("audit log write failed: %s", exc)


_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Lazily build a shared client (so importing never needs a key)."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def call_json(system: str, user: str, schema: Type[T], task: str = "",
              client: Optional[LLMClient] = None) -> Optional[T]:
    """Module-level convenience wrapper around ``LLMClient.call_json``."""
    return (client or get_client()).call_json(system, user, schema, task)
