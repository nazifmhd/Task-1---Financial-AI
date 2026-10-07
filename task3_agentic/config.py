"""Task 3 configuration - every tunable value in one place.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Config for a LangGraph multi-agent
# financial research system on Groq free tier', Date: 2026-10-07
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
CACHE_DIR = PROJECT_DIR / "cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TRACE_PATH = LOG_DIR / "agent_trace.jsonl"

TICKER = os.environ.get("TICKER", "NVDA")
TODAY = date.today()                     # cache key; never a hardcoded date

# --------------------------------------------------------------------------- #
# LLMs (Groq free tier, OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Model choice is driven by Groq free-tier limits (each model has its own buckets):
# * qwen/qwen3.8-27b caps OUTPUT at 1k tokens/minute - too small for report writing;
# * gpt-oss-120b's 200k tokens/day were largely consumed by the Task 2 LLM judge.
# Agents therefore run on gpt-oss-20b (native tool calling + JSON-schema output), and
# the low-volume llm_sentiment tool uses gpt-oss-120b. Both are env-configurable.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-oss-20b")
AGENT_REASONING_EFFORT = "low"           # agents reason in the loop; keep each step cheap
SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "openai/gpt-oss-120b")
LLM_MAX_TOKENS = 4096                    # explicit output cap (reasoning + answer)
TEMPERATURE = 0.0
LLM_MAX_RETRIES = 8                      # openai client honours Retry-After on 429
LLM_TIMEOUT_S = 120

# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
DEFAULT_PRICE_PERIOD = "1y"
VALID_PERIODS = ("3mo", "6mo", "1y", "2y", "5y")
VOL_HISTORY_PERIOD = "2y"                # history used for volatility windows/percentiles
TRADING_DAYS_PER_YEAR = 252
MIN_VOL_WINDOW, MAX_VOL_WINDOW = 5, 252
DEFAULT_NEWS_N, MAX_NEWS_N = 10, 20
DEFAULT_SEARCH_N, MAX_SEARCH_N = 4, 5
SEARCH_SNIPPET_CHARS = 200               # keeps tool messages small (token budget)
SEARCH_BACKENDS = ("auto", "yahoo")       # ddgs backends tried in order
NEWS_HTTP_TIMEOUT_S = 15
SMA_SHORT, SMA_LONG, RSI_PERIOD = 50, 200, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2.0
OHLCV_TAIL_ROWS = 5                      # recent bars returned to the agent

# Fault injection for the error-handling demo: names of tools that should
# simulate an upstream outage (comma-separated env var or set at runtime).
FAULTY_TOOLS: set[str] = {t for t in os.environ.get("FAULTY_TOOLS", "").split(",") if t}

# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
AGENT_RECURSION_LIMIT = 30               # hard stop for any ReAct loop (caught gracefully)
# Context management (Groq free tier: ~7-8k input tokens/minute per model).
AGENT_TOOL_BUDGET = 6                    # after this many tool calls the agent must answer
KEEP_FULL_RECENT_OBSERVATIONS = 2        # newest tool results shown to the model in full
OLD_OBSERVATION_CHARS = 350              # older ones are shortened in the model's view only
TRANSCRIPT_TOOL_CHARS = 900              # per tool result in structured-extraction prompts
HEDGE_HORIZON_DAYS = 90                  # the brief's risk horizon
MAX_CLARIFICATION_ROUNDS = 1             # critique loop: exactly one round-trip
TRACE_OUTPUT_CHARS = 200                 # brief: output truncated to 200 chars
