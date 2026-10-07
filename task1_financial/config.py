"""Central configuration for the Task 1 equity research pipeline.

Every tunable number in the project lives here so that business logic contains
no magic numbers. Values are documented with the reason they were chosen.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement Task 1 config module
# per CDAZZDEV blueprint - all magic numbers centralised', Date: 2026-10-07
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------- #
# Universe / data window
# --------------------------------------------------------------------------- #
# Overridable via env so the same notebook can be re-run for any ticker.
TICKER: str = os.environ.get("TICKER", "NVDA")   # liquid, news-rich large cap
LOOKBACK_YEARS: int = 2                          # brief requires >= 2 years
# Extra calendar days fetched on top of LOOKBACK_YEARS so SMA-200 is already
# warmed up at the start of the 2-year analysis window (200 trading days ~ 290
# calendar days).
WARMUP_CALENDAR_DAYS: int = 300
INTERVAL: str = "1d"                             # daily OHLCV
TODAY: date = date.today()                       # never hardcode a date string

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TRADING_DAYS_PER_YEAR: int = 252                 # used for 52-week fallback

# --------------------------------------------------------------------------- #
# Technical indicator parameters (canonical textbook values, as specified)
# --------------------------------------------------------------------------- #
SMA_SHORT: int = 50
SMA_LONG: int = 200
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_WINDOW: int = 20
BB_STD: float = 2.0

# RSI regime thresholds (Wilder's original 70/30 convention)
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0
RSI_MIDLINE: float = 50.0      # >50 => average gains dominate average losses

# Momentum vote: five boolean bullish conditions are evaluated (see
# indicators.momentum_signal). >= BULLISH_MIN => bullish,
# <= BEARISH_MAX => bearish, otherwise neutral.
MOMENTUM_BULLISH_MIN: int = 4
MOMENTUM_BEARISH_MAX: int = 1
# Minimum number of evaluable (non-NaN) conditions before a label is emitted.
MOMENTUM_MIN_EVALUABLE: int = 3

# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
MIN_HEADLINES: int = 10        # brief requires >= 10
MAX_HEADLINES: int = 15        # cap to bound LLM calls / free-tier rate limits
NEWS_HTTP_TIMEOUT_S: int = 15

# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "")   # "" => auto-detect
LLM_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
LLM_DEFAULT_MODELS = {
    # llama-3.3-70b-versatile was retired from Groq in 2026; gpt-oss-120b is
    # the strongest general model on the free tier and supports JSON mode.
    "groq": "openai/gpt-oss-120b",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}
LLM_MODEL: str = os.environ.get("LLM_MODEL", "")          # "" => provider default
LLM_TEMPERATURE: float = 0.1     # near-deterministic classification/reasoning
# Reasoning models spend completion tokens on hidden reasoning before the JSON,
# so the budget is well above the ~150 tokens the visible answer needs.
LLM_MAX_TOKENS: int = 2000
# Only sent to reasoning-capable models (gpt-oss family); "medium" balances
# depth of indicator reasoning against free-tier latency.
LLM_REASONING_EFFORT: str = "medium"
REASONING_MODEL_PREFIXES = ("openai/gpt-oss",)
LLM_MAX_RETRIES: int = 2         # repair retries after a validation failure
LLM_TIMEOUT_S: int = 60
LLM_RATE_LIMIT_BACKOFF_S: float = 5.0  # base back-off on HTTP 429

# Aggregate-sentiment dead-band: |score| below this => "neutral"
SENTIMENT_NEUTRAL_BAND: float = 0.15

# Signal justification length required by the brief
JUSTIFICATION_MIN_SENTENCES: int = 3
JUSTIFICATION_MAX_SENTENCES: int = 5

# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
PROJECT_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = PROJECT_DIR / "outputs"
CHART_LOOKBACK_DAYS: int = TRADING_DAYS_PER_YEAR   # chart shows last ~1 year
