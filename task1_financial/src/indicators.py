"""Technical indicators implemented from first principles (no TA-Lib).

Conventions (stated explicitly because they change the numbers):
* SMA            - arithmetic rolling mean; NaN until a full window exists.
* RSI(14)        - J. Welles Wilder (1978): first average gain/loss is a simple
                   mean of the first ``period`` changes, thereafter Wilder's
                   recursive smoothing  avg_t = (avg_{t-1}*(n-1) + x_t) / n,
                   i.e. an EMA with alpha = 1/n. A plain rolling mean of gains
                   (Cutler's RSI) is a different indicator and is NOT used.
* MACD(12,26,9)  - recursive EMAs (``adjust=False``, alpha = 2/(span+1)), the
                   form used by trading platforms. Values are masked to NaN
                   during the EMA warm-up so early, unstable values are never
                   presented as real.
* Bollinger(20,2)- middle = SMA(20); bands = middle +/- 2 * population std
                   (ddof=0), per John Bollinger's definition.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement SMA, Wilder RSI,
# MACD and Bollinger Bands from first principles in pandas without TA-Lib,
# with correct warm-up handling', Date: 2026-10-07
# SOURCE: Formulas cross-checked against Wilder, "New Concepts in Technical
# Trading Systems" (1978) and https://school.stockcharts.com (RSI, MACD,
# Bollinger Bands articles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average = arithmetic mean of the last ``window`` values."""
    return series.rolling(window=window, min_periods=window).mean()


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing with an SMA seed.

    ``values[0]`` is expected to be NaN (first diff). The seed at index
    ``period`` is the mean of values[1 : period + 1]; every later value is
    ``(prev * (period - 1) + x) / period``.
    """
    out = np.full(values.shape, np.nan)
    if len(values) <= period:
        return out
    out[period] = np.nanmean(values[1:period + 1])
    for i in range(period + 1, len(values)):
        x = values[i] if not np.isnan(values[i]) else 0.0  # gap => no change
        out[i] = (out[i - 1] * (period - 1) + x) / period
    return out


def rsi_wilder(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing.

    RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss.
    Edge cases: avg_loss == 0 and avg_gain > 0  -> 100 (all gains);
                avg_gain == avg_loss == 0       -> 50 (flat price, no signal).
    """
    delta = close.diff().to_numpy(dtype=float)
    gains = np.where(np.isnan(delta), np.nan, np.clip(delta, 0, None))
    losses = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0, None))

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)
    rsi = np.where(np.isnan(avg_gain) | np.isnan(avg_loss), np.nan, rsi)
    return pd.Series(rsi, index=close.index, name=f"rsi_{period}")


def ema(series: pd.Series, span: int) -> pd.Series:
    """Recursive exponential moving average, alpha = 2 / (span + 1)."""
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = config.MACD_FAST,
         slow: int = config.MACD_SLOW,
         signal: int = config.MACD_SIGNAL) -> pd.DataFrame:
    """MACD line = EMA(fast) - EMA(slow); signal = EMA(MACD, signal);
    histogram = MACD - signal."""
    macd_line = ema(close, fast) - ema(close, slow)
    # Mask the slow-EMA warm-up before the signal EMA is computed so the
    # signal line is seeded from a meaningful MACD value.
    macd_line = macd_line.where(np.arange(len(close)) >= slow - 1)
    signal_line = macd_line.ewm(span=signal, adjust=False,
                                min_periods=signal).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    }, index=close.index)


def bollinger(close: pd.Series, window: int = config.BB_WINDOW,
              n_std: float = config.BB_STD) -> pd.DataFrame:
    """Bollinger Bands: SMA(window) +/- n_std * population std (ddof=0).

    Also returns %B (position of price inside the band, 0 = lower, 1 = upper)
    and bandwidth (band width / middle) - useful context for the LLM.
    """
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower).replace(0, np.nan)          # flat price guard
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pct_b": (close - lower) / width,
        "bb_bandwidth": (upper - lower) / mid,
    }, index=close.index)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all five indicators attached as columns."""
    out = df.copy()
    if out.empty or "Close" not in out:
        return out
    close = out["Close"].astype(float)
    out[f"sma_{config.SMA_SHORT}"] = sma(close, config.SMA_SHORT)
    out[f"sma_{config.SMA_LONG}"] = sma(close, config.SMA_LONG)
    out[f"rsi_{config.RSI_PERIOD}"] = rsi_wilder(close, config.RSI_PERIOD)
    out = out.join(macd(close))
    out = out.join(bollinger(close))
    return out


# Column-name helpers so downstream code never hardcodes "sma_50" etc.
COL_SMA_SHORT = f"sma_{config.SMA_SHORT}"
COL_SMA_LONG = f"sma_{config.SMA_LONG}"
COL_RSI = f"rsi_{config.RSI_PERIOD}"


def momentum_signal(df: pd.DataFrame) -> dict:
    """Derive one momentum label from a vote over five indicator conditions.

    Conditions (each True = bullish evidence):
      1. trend_short      price above SMA-50
      2. trend_regime     SMA-50 above SMA-200 (golden-cross regime)
      3. macd_cross       MACD line above its signal line
      4. macd_accel       MACD histogram rising vs. previous bar
      5. rsi_momentum     RSI between midline (50) and overbought (70):
                          gains dominate but the move is not yet stretched

    Conditions whose inputs are NaN (warm-up / missing data) are skipped; if
    fewer than three conditions are evaluable the label is
    'insufficient_data' rather than a fabricated call.
    """
    needed = ["Close", COL_SMA_SHORT, COL_SMA_LONG, "macd", "macd_signal",
              "macd_hist", COL_RSI]
    if df.empty or any(c not in df for c in needed) or len(df) < 2:
        return {"label": "insufficient_data", "score": None, "checks": {}}

    last, prev = df.iloc[-1], df.iloc[-2]

    def _cmp(a, b, fn):
        return None if pd.isna(a) or pd.isna(b) else bool(fn(a, b))

    rsi = last[COL_RSI]
    checks = {
        "price_above_sma50": _cmp(last["Close"], last[COL_SMA_SHORT], lambda a, b: a > b),
        "sma50_above_sma200": _cmp(last[COL_SMA_SHORT], last[COL_SMA_LONG], lambda a, b: a > b),
        "macd_above_signal": _cmp(last["macd"], last["macd_signal"], lambda a, b: a > b),
        "macd_hist_rising": _cmp(last["macd_hist"], prev["macd_hist"], lambda a, b: a > b),
        "rsi_bullish_not_overbought": (
            None if pd.isna(rsi)
            else bool(config.RSI_MIDLINE < rsi < config.RSI_OVERBOUGHT)),
    }
    evaluable = [v for v in checks.values() if v is not None]
    if len(evaluable) < config.MOMENTUM_MIN_EVALUABLE:
        return {"label": "insufficient_data", "score": None, "checks": checks}

    bullish = sum(evaluable)
    # Rescale to the 5-check thresholds if some checks were skipped.
    scaled = bullish * len(checks) / len(evaluable)
    if scaled >= config.MOMENTUM_BULLISH_MIN:
        label = "bullish"
    elif scaled <= config.MOMENTUM_BEARISH_MAX:
        label = "bearish"
    else:
        label = "neutral"
    return {"label": label, "score": f"{bullish}/{len(evaluable)}", "checks": checks}


def latest_snapshot(df: pd.DataFrame) -> dict:
    """Latest-bar indicator values as plain floats (None where unavailable)."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    cols = ["Close", COL_SMA_SHORT, COL_SMA_LONG, COL_RSI, "macd",
            "macd_signal", "macd_hist", "bb_mid", "bb_upper", "bb_lower",
            "bb_pct_b", "bb_bandwidth"]
    snap = {}
    for c in cols:
        v = last.get(c, np.nan)
        snap[c] = None if pd.isna(v) else float(v)
    snap["date"] = df.index[-1].date().isoformat()
    return snap
