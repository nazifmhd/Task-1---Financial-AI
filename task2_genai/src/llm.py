"""Minimal Groq (OpenAI-compatible) JSON client for teacher generation and judging.

Keys are read from Colab Secrets, environment variables, or a git-ignored .env
file - never from source.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Groq JSON-mode client with
# rate-limit back-off and Pydantic validation for data generation', Date: 2026-10-07
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

import config
from src.schemas import extract_json_text

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def read_secret(name: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv(config.PROJECT_DIR / ".env")
        load_dotenv(config.PROJECT_DIR.parent / ".env")
    except ImportError:
        pass
    return os.environ.get(name) or None


class GroqJSON:
    def __init__(self, model: str, temperature: float, reasoning_effort: str,
                 max_tokens: int = config.TEACHER_MAX_TOKENS):
        from openai import OpenAI
        key = read_secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not found (Colab Secrets / env / .env)")
        self.client = OpenAI(api_key=key, base_url=config.TEACHER_PROVIDER_BASE_URL,
                             timeout=120)
        self.model, self.temperature = model, temperature
        self.reasoning_effort, self.max_tokens = reasoning_effort, max_tokens
        self.usage_tokens = 0

    def call(self, system: str, user: str, schema: Type[T]) -> Optional[T]:
        """JSON-mode call validated against ``schema``; None after retries."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning_effort": self.reasoning_effort})
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                wait = config.RATE_LIMIT_BACKOFF_S * (attempt + 1)
                log.warning("API error %s (attempt %d): %s; sleeping %.0fs",
                            status, attempt + 1, str(exc)[:160], wait)
                time.sleep(wait)
                continue
            self.usage_tokens += getattr(r.usage, "total_tokens", 0) or 0
            raw = r.choices[0].message.content or ""
            try:
                return schema.model_validate(json.loads(extract_json_text(raw) or ""))
            except (ValidationError, json.JSONDecodeError) as exc:
                log.warning("validation failed (attempt %d): %s", attempt + 1,
                            str(exc)[:200])
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content":
                              f"That failed validation: {str(exc)[:500]}. "
                              "Return only the corrected JSON object."}]
        return None
