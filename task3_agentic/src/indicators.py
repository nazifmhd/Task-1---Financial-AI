"""Technical indicators (no TA-Lib), reused from Task 1's first-principles code.

# SOURCE: Adapted from task1_financial/src/indicators.py in this repository
# (my Task 1 implementation): Wilder RSI with SMA seed, recursive-EMA MACD,
# population-std Bollinger Bands.
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Condense Task 1 indicators for
# reuse in agent tools', Date: 2026-10-07
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def rsi_wilder(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff().to_numpy(dtype=float)
    gains, losses = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    ag, al = np.full(len(close), np.nan), np.full(len(close), np.nan)
    if len(close) > period:
        ag[period], al[period] = np.mean(gains[1:period + 1]), np.mean(losses[1:period + 1])
        for i in range(period + 1, len(close)):
            ag[i] = (ag[i - 1] * (period - 1) + gains[i]) / period
            al[i] = (al[i - 1] * (period - 1) + losses[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + ag / al)
    rsi = np.where((al == 0) & (ag > 0), 100.0, rsi)
    rsi = np.where((al == 0) & (ag == 0), 50.0, rsi)
    return pd.Series(rsi, index=close.index)


def macd(close: pd.Series):
    line = close.ewm(span=config.MACD_FAST, adjust=False).mean() - \
        close.ewm(span=config.MACD_SLOW, adjust=False).mean()
    line = line.where(np.arange(len(close)) >= config.MACD_SLOW - 1)
    signal = line.ewm(span=config.MACD_SIGNAL, adjust=False, min_periods=config.MACD_SIGNAL).mean()
    return line, signal, line - signal


def bollinger(close: pd.Series):
    mid = close.rolling(config.BB_WINDOW, min_periods=config.BB_WINDOW).mean()
    std = close.rolling(config.BB_WINDOW, min_periods=config.BB_WINDOW).std(ddof=0)
    return mid - config.BB_STD * std, mid, mid + config.BB_STD * std
