"""Indicator correctness tests against independent, loop-based reference
implementations of the textbook formulas plus analytical edge cases.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest suite validating
# SMA/RSI/MACD/Bollinger against naive reference implementations', Date: 2026-10-07
"""
import math

import numpy as np
import pandas as pd
import pytest

import config
from src import indicators as ind


@pytest.fixture
def close():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-01", periods=400)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)


def ref_sma(x, n):
    return [np.nan if i < n - 1 else sum(x[i - n + 1:i + 1]) / n for i in range(len(x))]


def ref_rsi(x, n=14):
    """Wilder (1978) worked-example procedure, written as a plain loop."""
    out = [np.nan] * len(x)
    gains = [max(x[i] - x[i - 1], 0) for i in range(1, len(x))]
    losses = [max(x[i - 1] - x[i], 0) for i in range(1, len(x))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    out[n] = 100 - 100 / (1 + ag / al)
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al)
    return out


def ref_ema(x, span):
    a, out = 2 / (span + 1), [x[0]]
    for v in x[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out


def test_sma_matches_reference(close):
    x = close.tolist()
    np.testing.assert_allclose(ind.sma(close, 50).to_numpy(), ref_sma(x, 50), equal_nan=True)


def test_rsi_matches_wilder_reference(close):
    got = ind.rsi_wilder(close, 14).to_numpy()
    np.testing.assert_allclose(got, ref_rsi(close.tolist(), 14), equal_nan=True, rtol=1e-10)
    assert np.isnan(got[:14]).all() and not np.isnan(got[14])


def test_rsi_differs_from_naive_rolling_mean(close):
    """Guard against the common wrong implementation (Cutler/SMA RSI)."""
    d = close.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()
    naive = 100 - 100 / (1 + rs)
    assert (ind.rsi_wilder(close) - naive).abs().max() > 1.0


def test_rsi_edge_cases():
    idx = pd.bdate_range("2024-01-01", periods=30)
    up = pd.Series(np.arange(30, dtype=float) + 1, index=idx)
    flat = pd.Series(np.full(30, 5.0), index=idx)
    assert ind.rsi_wilder(up).dropna().eq(100).all()
    assert ind.rsi_wilder(flat).dropna().eq(50).all()


def test_macd_matches_reference(close):
    x = close.tolist()
    line = np.array(ref_ema(x, 12)) - np.array(ref_ema(x, 26))
    m = ind.macd(close)
    valid = slice(config.MACD_SLOW - 1, None)
    np.testing.assert_allclose(m["macd"].to_numpy()[valid], line[valid], rtol=1e-10)
    assert m["macd"].iloc[: config.MACD_SLOW - 1].isna().all()
    sig = ref_ema(line[config.MACD_SLOW - 1:].tolist(), 9)
    got_sig = m["macd_signal"].to_numpy()[config.MACD_SLOW - 1:]
    np.testing.assert_allclose(got_sig[8:], sig[8:], rtol=1e-10)
    np.testing.assert_allclose(m["macd_hist"], m["macd"] - m["macd_signal"], equal_nan=True)


def test_bollinger_matches_reference(close):
    x = close.tolist()
    bb = ind.bollinger(close)
    i = 100
    window = x[i - 19:i + 1]
    mu = sum(window) / 20
    sd = math.sqrt(sum((v - mu) ** 2 for v in window) / 20)   # population std
    assert bb["bb_mid"].iloc[i] == pytest.approx(mu)
    assert bb["bb_upper"].iloc[i] == pytest.approx(mu + 2 * sd)
    assert bb["bb_lower"].iloc[i] == pytest.approx(mu - 2 * sd)


def test_add_all_and_momentum(close):
    df = pd.DataFrame({"Open": close, "High": close, "Low": close,
                       "Close": close, "Volume": 1e6})
    out = ind.add_all_indicators(df)
    for col in ["sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "bb_upper"]:
        assert col in out
    assert out["sma_200"].iloc[:199].isna().all()
    m = ind.momentum_signal(out)
    assert m["label"] in {"bullish", "bearish", "neutral"}


def test_momentum_insufficient_data():
    idx = pd.bdate_range("2024-01-01", periods=10)
    c = pd.Series(np.linspace(1, 2, 10), index=idx)
    df = ind.add_all_indicators(pd.DataFrame({"Close": c, "Volume": 1.0}))
    assert ind.momentum_signal(df)["label"] == "insufficient_data"
    assert ind.momentum_signal(pd.DataFrame())["label"] == "insufficient_data"
