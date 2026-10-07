"""Reasoned Buy/Hold/Sell signal over the indicator combination.

Two parts:
1. ``build_signal_context`` derives *relational* features from the raw
   indicator frame (distance from each SMA, crossover recency, histogram
   slope, RSI change, bandwidth percentile, volume ratio). Giving the model
   relationships rather than bare levels is what lets it reason about how
   indicators interact instead of echoing them.
2. ``generate_signal`` calls the LLM with that context and validates the
   reply as ``TradeSignal``. If the LLM is unavailable or never returns valid
   JSON, a transparent rule-based fallback is returned and flagged as such,
   so downstream steps (report) still run.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Derive relational indicator
# features (cross recency, histogram slope, bandwidth percentile) and call LLM
# for a validated Buy/Hold/Sell with rule-based fallback', Date: 2026-10-07
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.indicators import COL_RSI, COL_SMA_LONG, COL_SMA_SHORT
from .client import LLMClient, LLMUnavailable, call_json
from .prompts import PROMPT_VERSION, SIGNAL_SYSTEM, SIGNAL_USER_TMPL
from .schemas import SentimentAggregate, TradeSignal

log = logging.getLogger(__name__)

# Lookbacks (trading sessions) for derived context features.
SESSIONS_1M, SESSIONS_3M = 21, 63
RSI_CHANGE_LOOKBACK = 5
HIST_SLOPE_LOOKBACK = 3
VOLUME_AVG_WINDOW = 20


def _fmt(value, spec: str = ".2f", suffix: str = "") -> str:
    """Format a number for the prompt, or 'n/a' if missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:{spec}}{suffix}"


def _last_cross(a: pd.Series, b: pd.Series, up: str, down: str) -> str:
    """Describe the most recent crossing of series a over/under series b."""
    diff = (a - b).dropna()
    if len(diff) < 2:
        return "n/a"
    sign = np.sign(diff)
    changes = sign.ne(sign.shift()).iloc[1:]
    changes = changes[changes]
    if changes.empty:
        return f"none in available history ({'above' if sign.iloc[-1] > 0 else 'below'} throughout)"
    when = changes.index[-1]
    sessions_ago = len(diff.loc[when:]) - 1
    kind = up if sign.loc[when] > 0 else down
    return f"{kind} {sessions_ago} sessions ago ({when.date()})"


def _pct_change(series: pd.Series, sessions: int) -> Optional[float]:
    s = series.dropna()
    if len(s) <= sessions:
        return None
    return (s.iloc[-1] / s.iloc[-1 - sessions] - 1) * 100


def build_signal_context(df: pd.DataFrame, summary: dict,
                         news: Optional[SentimentAggregate]) -> dict:
    """All fields for SIGNAL_USER_TMPL, pre-formatted as strings."""
    last = df.iloc[-1]
    close = df["Close"]
    price = last["Close"]
    sma50, sma200 = last[COL_SMA_SHORT], last[COL_SMA_LONG]

    hist = df["macd_hist"].dropna()
    hist_tail = hist.tail(HIST_SLOPE_LOOKBACK + 1)
    if len(hist_tail) > HIST_SLOPE_LOOKBACK:
        hist_trend = "rising" if hist_tail.diff().iloc[1:].gt(0).all() else \
                     "falling" if hist_tail.diff().iloc[1:].lt(0).all() else "mixed"
    else:
        hist_trend = "n/a"

    bw = df["bb_bandwidth"].dropna().tail(config.TRADING_DAYS_PER_YEAR)
    bw_pctile = (bw.rank(pct=True).iloc[-1] * 100) if len(bw) else None
    vol_avg = df["Volume"].rolling(VOLUME_AVG_WINDOW).mean().iloc[-1]
    rsi_series = df[COL_RSI].dropna()

    if pd.notna(sma50) and pd.notna(sma200):
        regime = ("bullish (SMA-50 above SMA-200)" if sma50 > sma200
                  else "bearish (SMA-50 below SMA-200)")
    else:
        regime = "n/a"

    return dict(
        company=summary.get("company_name") or summary["ticker"],
        ticker=summary["ticker"], sector=summary.get("sector") or "n/a",
        asof=df.index[-1].date().isoformat(),
        price=_fmt(price), sma50=_fmt(sma50), sma200=_fmt(sma200),
        pct_vs_sma50=_fmt((price / sma50 - 1) * 100 if pd.notna(sma50) else None, "+.1f", "%"),
        pct_vs_sma200=_fmt((price / sma200 - 1) * 100 if pd.notna(sma200) else None, "+.1f", "%"),
        sma_regime=regime,
        last_sma_cross=_last_cross(df[COL_SMA_SHORT], df[COL_SMA_LONG],
                                   "golden cross", "death cross"),
        ret_1m=_fmt(_pct_change(close, SESSIONS_1M), "+.1f", "%"),
        ret_3m=_fmt(_pct_change(close, SESSIONS_3M), "+.1f", "%"),
        ytd=_fmt(summary.get("ytd_return_pct"), "+.1f", "%"),
        pct_from_high=_fmt(summary.get("pct_from_52w_high"), "+.1f", "%"),
        rsi=_fmt(last[COL_RSI], ".1f"),
        rsi_5d_ago=_fmt(rsi_series.iloc[-1 - RSI_CHANGE_LOOKBACK]
                        if len(rsi_series) > RSI_CHANGE_LOOKBACK else None, ".1f"),
        macd=_fmt(last["macd"], ".3f"), macd_signal=_fmt(last["macd_signal"], ".3f"),
        macd_hist=_fmt(last["macd_hist"], "+.3f"), hist_trend=hist_trend,
        last_macd_cross=_last_cross(df["macd"], df["macd_signal"],
                                    "bullish cross", "bearish cross"),
        bb_upper=_fmt(last["bb_upper"]), bb_mid=_fmt(last["bb_mid"]),
        bb_lower=_fmt(last["bb_lower"]), pct_b=_fmt(last["bb_pct_b"]),
        bandwidth=_fmt(last["bb_bandwidth"], ".3f"),
        bandwidth_pctile=_fmt(bw_pctile, ".0f"),
        volume_ratio=_fmt(last["Volume"] / vol_avg if vol_avg else None, ".2f"),
        momentum_label=summary.get("momentum_signal", "n/a"),
        momentum_score=(summary.get("momentum_detail") or {}).get("score", "n/a"),
        n_headlines=news.n_scored if news else 0,
        # An empty aggregate is "unavailable", not "neutral": the model must
        # not mistake missing evidence for balanced evidence.
        news_label=news.aggregate_label if news and news.n_scored else "unavailable",
        news_score=_fmt(news.aggregate_score if news and news.n_scored else None, "+.3f"),
    )


def rule_based_signal(summary: dict) -> TradeSignal:
    """Deterministic fallback used only when the LLM path fails."""
    label = summary.get("momentum_signal", "neutral")
    signal = {"bullish": "Buy", "bearish": "Sell"}.get(label, "Hold")
    score = (summary.get("momentum_detail") or {}).get("score", "n/a")
    return TradeSignal(
        signal=signal, conviction="low",
        justification=(
            f"LLM reasoning was unavailable, so this call falls back to the "
            f"rule-based momentum vote, which is {label} ({score} checks). "
            f"The vote counts trend, MACD and RSI conditions equally and does "
            f"not weigh how they interact. Treat this as a placeholder rather "
            f"than an analyst-grade recommendation."),
        key_drivers=[f"momentum vote {label} ({score})",
                     "no LLM interaction analysis"],
        conflicting_signals=[])


def generate_signal(df: pd.DataFrame, summary: dict,
                    news: Optional[SentimentAggregate],
                    client: Optional[LLMClient] = None) -> dict:
    """Return {'signal': TradeSignal, 'source': 'llm'|'rule_based_fallback',
    'prompt': str, 'prompt_version': str}."""
    if df.empty:
        return {"signal": rule_based_signal(summary), "source": "rule_based_fallback",
                "prompt": "", "prompt_version": PROMPT_VERSION}
    user = SIGNAL_USER_TMPL.format(**build_signal_context(df, summary, news))
    try:
        result = call_json(SIGNAL_SYSTEM, user, TradeSignal, task="trade_signal",
                           client=client)
    except LLMUnavailable as exc:
        log.error("%s", exc)
        result = None
    if result is None:
        log.warning("Using rule-based fallback signal")
        return {"signal": rule_based_signal(summary), "source": "rule_based_fallback",
                "prompt": user, "prompt_version": PROMPT_VERSION}
    return {"signal": result, "source": "llm", "prompt": user,
            "prompt_version": PROMPT_VERSION}
