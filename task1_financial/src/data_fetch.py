"""OHLCV and fundamentals ingestion via yfinance.

Design: every external call is wrapped so that a network failure, a bad ticker
or a schema change degrades to an empty frame / ``None`` fields plus a logged
warning - never an unhandled exception.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Robust yfinance OHLCV +
# fundamentals fetch with no hardcoded dates and MultiIndex flattening',
# Date: 2026-10-07
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

import config

log = logging.getLogger(__name__)


def compute_date_range(today=None, years: int = config.LOOKBACK_YEARS,
                       warmup_days: int = config.WARMUP_CALENDAR_DAYS):
    """Return (start, end) dates relative to *today* - no hardcoded strings.

    ``end`` is today + 1 day because yfinance treats ``end`` as exclusive and we
    want the most recent completed session included.
    """
    today = today or config.TODAY
    start = today - relativedelta(years=years) - timedelta(days=warmup_days)
    end = today + timedelta(days=1)
    return start, end


def fetch_ohlcv(ticker: str = config.TICKER) -> pd.DataFrame:
    """Fetch >= LOOKBACK_YEARS of daily OHLCV (plus indicator warm-up).

    Returns a clean frame with columns Open/High/Low/Close/Volume indexed by a
    tz-naive DatetimeIndex, or an empty frame if the download fails.
    """
    start, end = compute_date_range()
    try:
        df = yf.download(ticker, start=start, end=end, interval=config.INTERVAL,
                         auto_adjust=True, progress=False, threads=False)
    except Exception as exc:  # network / yfinance internal errors
        log.warning("OHLCV download failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=config.OHLCV_COLUMNS)

    if df is None or df.empty:
        log.warning("No OHLCV returned for %s", ticker)
        return pd.DataFrame(columns=config.OHLCV_COLUMNS)

    # yfinance >= 0.2.4x returns (field, ticker) MultiIndex columns - flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)

    missing = [c for c in config.OHLCV_COLUMNS if c not in df.columns]
    if missing:
        log.warning("OHLCV for %s missing columns %s", ticker, missing)
        return pd.DataFrame(columns=config.OHLCV_COLUMNS)

    df = df[config.OHLCV_COLUMNS].sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]

    # Robustness: drop rows with no price at all, forward-fill sporadic gaps
    # (e.g. a missing Volume print). Never back-fill: that would leak future
    # data into the past.
    df = df.dropna(how="all")
    df[config.OHLCV_COLUMNS] = df[config.OHLCV_COLUMNS].ffill()
    df = df.dropna(subset=["Close"])

    log.info("Fetched %d daily bars for %s (%s -> %s)", len(df), ticker,
             df.index[0].date(), df.index[-1].date())
    return df


def _safe_float(value) -> Optional[float]:
    """Convert to float, mapping None / NaN / non-numeric / 'Infinity' to None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN or inf
        return None
    return f


def fetch_fundamentals(ticker: str = config.TICKER) -> dict:
    """P/E and 52-week range from ``Ticker.info``.

    ``.info`` is a scraped endpoint: it is slow, rate-limited and fields are
    frequently absent (e.g. no trailing P/E for loss-making companies). Every
    field is therefore optional and validated; consumers fall back to values
    derived from OHLCV where possible.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning(".info lookup failed for %s: %s", ticker, exc)
        info = {}

    return {
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency") or "USD",
        "market_cap": _safe_float(info.get("marketCap")),
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "week52_high": _safe_float(info.get("fiftyTwoWeekHigh")),
        "week52_low": _safe_float(info.get("fiftyTwoWeekLow")),
    }
