"""Task 1A summary dictionary and end-to-end 1A orchestrator.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build summary dict (price,
# 52w range, P/E, YTD, momentum) with null-safe fallbacks and a run_task1a
# orchestrator', Date: 2026-10-07
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import pandas as pd

import config
from src import data_fetch, indicators, news

log = logging.getLogger(__name__)

WEEKS_PER_YEAR = 52


def _round(value, ndigits: int = 2) -> Optional[float]:
    return None if value is None or pd.isna(value) else round(float(value), ndigits)


def ytd_return_pct(close: pd.Series, today=None) -> Optional[float]:
    """Year-to-date return in percent.

    Base = last close of the previous calendar year (the standard YTD
    convention). If the data does not reach back that far, fall back to the
    first close of the current year. Returns None if neither exists.
    """
    if close.empty:
        return None
    year = (today or config.TODAY).year
    prior = close[close.index.year < year]
    current_year = close[close.index.year == year]
    if not prior.empty:
        base = prior.iloc[-1]
    elif not current_year.empty:
        base = current_year.iloc[0]
    else:
        return None
    if not base or pd.isna(base):
        return None
    return (float(close.iloc[-1]) / float(base) - 1.0) * 100.0


def week52_range(df: pd.DataFrame, fundamentals: dict, today=None):
    """52-week (high, low, source).

    Primary: intraday High/Low over the trailing 52 weeks of the *same*
    adjusted OHLCV used everywhere else, so the range is consistent with the
    price and indicators. Fallback: yfinance ``.info`` values.
    """
    today = pd.Timestamp(today or config.TODAY)
    window = df[df.index >= today - timedelta(weeks=WEEKS_PER_YEAR)]
    if not window.empty and window["High"].notna().any():
        return float(window["High"].max()), float(window["Low"].min()), "ohlcv"
    hi, lo = fundamentals.get("week52_high"), fundamentals.get("week52_low")
    if hi is not None and lo is not None:
        return hi, lo, "yfinance_info"
    return None, None, "unavailable"


def build_summary(df: pd.DataFrame, fundamentals: dict,
                  ticker: str = config.TICKER) -> dict:
    """Clean summary dictionary with every required field.

    Fields that cannot be computed are reported as ``None`` (honest) rather
    than raising or inventing a value.
    """
    if df.empty:
        log.warning("Empty price frame - summary will be mostly null")
        return {"ticker": ticker, "company_name": fundamentals.get("company_name"),
                "as_of": None, "current_price": None, "week52_high": None,
                "week52_low": None, "week52_source": "unavailable",
                "pe_ratio": fundamentals.get("pe_ratio"), "ytd_return_pct": None,
                "momentum_signal": "insufficient_data", "momentum_detail": {}}

    hi, lo, src = week52_range(df, fundamentals)
    momentum = indicators.momentum_signal(df)
    current = float(df["Close"].iloc[-1])
    return {
        "ticker": ticker,
        "company_name": fundamentals.get("company_name"),
        "sector": fundamentals.get("sector"),
        "as_of": df.index[-1].date().isoformat(),
        "current_price": _round(current),
        "week52_high": _round(hi),
        "week52_low": _round(lo),
        "week52_source": src,
        "pct_from_52w_high": _round((current / hi - 1) * 100) if hi else None,
        "pe_ratio": _round(fundamentals.get("pe_ratio")),
        "forward_pe": _round(fundamentals.get("forward_pe")),
        "ytd_return_pct": _round(ytd_return_pct(df["Close"])),
        "momentum_signal": momentum["label"],
        "momentum_detail": momentum,
    }


@dataclass
class Task1AResult:
    """Everything Task 1B and the report need from the data layer."""
    ticker: str
    prices: pd.DataFrame
    fundamentals: dict
    headlines: list[dict]
    summary: dict
    snapshot: dict = field(default_factory=dict)


def run_task1a(ticker: str = config.TICKER) -> Task1AResult:
    """Fetch OHLCV + fundamentals + news, compute indicators, build summary."""
    raw = data_fetch.fetch_ohlcv(ticker)
    prices = indicators.add_all_indicators(raw)
    fundamentals = data_fetch.fetch_fundamentals(ticker)
    headlines = news.get_news(ticker, company=fundamentals.get("company_name"))
    summary = build_summary(prices, fundamentals, ticker)
    return Task1AResult(ticker=ticker, prices=prices, fundamentals=fundamentals,
                        headlines=headlines, summary=summary,
                        snapshot=indicators.latest_snapshot(prices))
