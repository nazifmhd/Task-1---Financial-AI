"""LLM factory (Groq via the OpenAI-compatible API) and key loading.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'ChatOpenAI factory for Groq with
# Colab-secret / env key loading', Date: 2026-10-07
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import config


def read_secret(name: str) -> Optional[str]:
    """Colab Secrets first, then environment / git-ignored .env. Never printed."""
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


@lru_cache(maxsize=4)
def chat_model(model: str = config.AGENT_MODEL, reasoning_effort: Optional[str] = config.AGENT_REASONING_EFFORT):
    from langchain_openai import ChatOpenAI
    key = read_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (Colab Secrets / env / .env)")
    kwargs = dict(model=model, base_url=config.GROQ_BASE_URL, api_key=key,
                  temperature=config.TEMPERATURE, max_retries=config.LLM_MAX_RETRIES,
                  timeout=config.LLM_TIMEOUT_S, max_tokens=config.LLM_MAX_TOKENS)
    if reasoning_effort and model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)
