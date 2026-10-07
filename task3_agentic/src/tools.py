"""The five agent tools.

Contract (what makes graceful fallback possible):
* every tool returns its documented type - dict for get_price_data /
  calculate_volatility / llm_sentiment, list[dict] for get_news / web_search;
* no tool raises: failures come back as {"error": ..., "hint": ...} (or a
  one-element list of that), and the hint suggests an alternative so the agent
  can replan;
* every call is logged by @traced to logs/agent_trace.jsonl;
* outputs are compact (summary + short tail) to respect the LLM token budget.

Fault injection (``inject_faults``) simulates an upstream outage for named
tools, so the error-handling and replanning path can be demonstrated on demand.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Five LangChain tools (price data
# with indicators, news with RSS fallback, volatility, LLM sentiment, DuckDuckGo
# search) with structured errors, tracing and fault injection', Date: 2026-10-07
"""
from __future__ import annotations

import contextlib
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool

import config
from src.indicators import bollinger, macd, rsi_wilder
from src.observability.trace import traced
from src.schemas import SentimentResult

_PERIOD_DAYS = {"3mo": 91, "6mo": 182, "1y": 365, "2y": 730, "5y": 1826}
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


@contextlib.contextmanager
def inject_faults(*tool_names: str):
    """Temporarily make the named tools return a simulated outage error."""
    added = set(tool_names) - config.FAULTY_TOOLS
    config.FAULTY_TOOLS.update(added)
    try:
        yield
    finally:
        config.FAULTY_TOOLS.difference_update(added)


def _fault(tool: str) -> Optional[str]:
    return f"simulated upstream outage for {tool} (fault injection)" if tool in config.FAULTY_TOOLS else None


def _bad_ticker(ticker: str) -> Optional[str]:
    return None if _TICKER_RE.match(ticker or "") else f"invalid ticker format: {ticker!r}"


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _history(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


# --------------------------------------------------------------------------- #
# 1. get_price_data
# --------------------------------------------------------------------------- #
@traced
def get_price_data(ticker: str, period: str = config.DEFAULT_PRICE_PERIOD) -> dict:
    """Daily OHLCV for `ticker` over `period` (3mo, 6mo, 1y, 2y, 5y) with computed
    indicators: SMA50, SMA200, RSI14 (Wilder), MACD(12,26,9), Bollinger(20,2),
    30-day change, 52-week range and the last few OHLCV bars."""
    ticker = (ticker or "").strip().upper()
    if err := _fault("get_price_data") or _bad_ticker(ticker):
        return {"error": err, "ticker": ticker, "hint": "check the ticker symbol or retry later"}
    note = None
    if period not in config.VALID_PERIODS:
        note, period = f"period {period!r} not supported; used {config.DEFAULT_PRICE_PERIOD}", config.DEFAULT_PRICE_PERIOD
    try:
        # Fetch at least 2y so SMA200 is defined whatever `period` was asked for.
        df = _history(ticker, "2y" if period in ("3mo", "6mo", "1y") else period)
    except Exception as exc:
        return {"error": f"price download failed: {exc}", "ticker": ticker,
                "hint": "retry later or rely on get_news / web_search for context"}
    if df.empty:
        return {"error": f"no price data for {ticker} (unknown or delisted symbol)", "ticker": ticker,
                "hint": "verify the symbol with web_search"}
    close = df["Close"]
    line, sig, hist = macd(close)
    bb_l, bb_m, bb_u = bollinger(close)
    last_year = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]
    sma50, sma200 = close.rolling(config.SMA_SHORT).mean().iloc[-1], close.rolling(config.SMA_LONG).mean().iloc[-1]
    window = df[df.index >= df.index[-1] - pd.Timedelta(days=_PERIOD_DAYS[period])]
    tail = df.tail(config.OHLCV_TAIL_ROWS)
    out = {
        "ticker": ticker, "as_of": df.index[-1].date().isoformat(), "period": period,
        "bars_in_period": int(len(window)),
        "current_price": _r(close.iloc[-1]),
        "period_return_pct": _r((close.iloc[-1] / window["Close"].iloc[0] - 1) * 100),
        "pct_change_30d": _r(close.pct_change(21).iloc[-1] * 100),
        "high_52w": _r(last_year["High"].max()), "low_52w": _r(last_year["Low"].min()),
        "pct_from_52w_high": _r((close.iloc[-1] / last_year["High"].max() - 1) * 100),
        "sma_50": _r(sma50), "sma_200": _r(sma200),
        "rsi_14": _r(rsi_wilder(close).iloc[-1], 1),
        "macd": _r(line.iloc[-1], 3), "macd_signal": _r(sig.iloc[-1], 3), "macd_histogram": _r(hist.iloc[-1], 3),
        "bollinger_lower": _r(bb_l.iloc[-1]), "bollinger_upper": _r(bb_u.iloc[-1]),
        "avg_volume_20d": int(df["Volume"].tail(20).mean()),
        "ohlcv_last_bars": [{"date": d.date().isoformat(), "open": _r(r.Open), "high": _r(r.High),
                             "low": _r(r.Low), "close": _r(r.Close), "volume": int(r.Volume)}
                            for d, r in tail.iterrows()],
    }
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------- #
# 2. get_news
# --------------------------------------------------------------------------- #
def _parse_date(v) -> str:
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()
        if isinstance(v, str) and v[:4].isdigit():
            return v[:10]
        return parsedate_to_datetime(v).date().isoformat()
    except Exception:
        return ""


def _rss(url: str, source: str) -> list[dict]:
    import feedparser
    import requests
    resp = requests.get(url, timeout=config.NEWS_HTTP_TIMEOUT_S,
                        headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
    resp.raise_for_status()
    items = []
    for e in feedparser.parse(resp.content).entries:
        title, pub = e.get("title", ""), (e.get("source") or {}).get("title", source)
        if title.endswith(f" - {pub}"):
            title = title[: -len(pub) - 3]
        items.append({"title": title.strip(), "publisher": pub,
                      "published": _parse_date(e.get("published")), "source": source})
    return items


@traced
def get_news(ticker: str, n: int = config.DEFAULT_NEWS_N) -> list[dict]:
    """Up to `n` recent headlines for `ticker`, newest first, as a list of
    {title, publisher, published, source}. Sources: yfinance, then Yahoo
    Finance RSS, then Google News RSS (each isolated; failures fall through)."""
    ticker = (ticker or "").strip().upper()
    if err := _fault("get_news") or _bad_ticker(ticker):
        return [{"error": err, "hint": "use web_search for recent news and analyst commentary"}]
    n = max(1, min(int(n), config.MAX_NEWS_N))
    items, errors = [], []
    try:
        import yfinance as yf
        for raw in (yf.Ticker(ticker).news or []):
            c = raw.get("content", raw)
            if c.get("title"):
                items.append({"title": c["title"], "publisher": (c.get("provider") or {}).get("displayName", ""),
                              "published": _parse_date(c.get("pubDate") or raw.get("providerPublishTime")),
                              "source": "yfinance"})
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
    for url, src in ((f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", "yahoo_rss"),
                     (f"https://news.google.com/rss/search?q={quote_plus(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en", "google_news")):
        if len(items) >= n:
            break
        try:
            items += _rss(url, src)
        except Exception as exc:
            errors.append(f"{src}: {exc}")
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: x["published"], reverse=True):
        key = re.sub(r"[^a-z0-9]", "", it["title"].lower())
        if key and key not in seen:
            seen.add(key)
            uniq.append(it)
    if not uniq:
        return [{"error": "no news found" + (f" ({'; '.join(errors)})" if errors else ""),
                 "hint": "use web_search for recent news"}]
    return uniq[:n]


# --------------------------------------------------------------------------- #
# 3. calculate_volatility
# --------------------------------------------------------------------------- #
@traced
def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Annualised historical volatility of daily log returns over the last
    `window` trading days (5-252), plus its percentile versus the past year of
    rolling values, downside volatility and 1-year max drawdown."""
    ticker = (ticker or "").strip().upper()
    if err := _fault("calculate_volatility") or _bad_ticker(ticker):
        return {"error": err, "ticker": ticker, "hint": "retry later"}
    window = int(window)
    if not config.MIN_VOL_WINDOW <= window <= config.MAX_VOL_WINDOW:
        return {"error": f"window must be {config.MIN_VOL_WINDOW}-{config.MAX_VOL_WINDOW} trading days",
                "ticker": ticker, "hint": "use 30 or 90"}
    try:
        df = _history(ticker, config.VOL_HISTORY_PERIOD)
    except Exception as exc:
        return {"error": f"price download failed: {exc}", "ticker": ticker, "hint": "retry later"}
    if len(df) < window + 2:
        return {"error": f"not enough history for {ticker} ({len(df)} bars)", "ticker": ticker,
                "hint": "verify the symbol or use a shorter window"}
    logret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
    rolling = logret.rolling(window).std().dropna() * ann * 100
    recent = logret.tail(window)
    last_year = rolling.tail(config.TRADING_DAYS_PER_YEAR)
    close_1y = df["Close"].tail(config.TRADING_DAYS_PER_YEAR)
    drawdown = (close_1y / close_1y.cummax() - 1).min() * 100
    vol = float(recent.std() * ann * 100)
    return {
        "ticker": ticker, "window": window, "as_of": df.index[-1].date().isoformat(),
        "annualised_vol_pct": _r(vol),
        "daily_vol_pct": _r(recent.std() * 100, 3),
        "downside_vol_pct": _r(recent[recent < 0].std() * ann * 100),
        "vol_percentile_1y": _r((last_year < vol).mean() * 100, 0),
        "vol_1y_min_max_pct": [_r(last_year.min()), _r(last_year.max())],
        "max_drawdown_1y_pct": _r(drawdown),
        "expected_move_over_window_pct": _r(vol * np.sqrt(window / config.TRADING_DAYS_PER_YEAR)),
    }


# --------------------------------------------------------------------------- #
# 4. llm_sentiment
# --------------------------------------------------------------------------- #
SENTIMENT_PROMPT = """You score financial news headlines for their likely impact on the
share price of the company they concern. Return ONLY JSON:
{{"score": <float -1..1>, "label": "positive"|"neutral"|"negative",
  "n_headlines": <int>, "key_drivers": [<up to 4 short phrases>]}}
Weight company-specific news above generic market commentary.

Headlines:
{headlines}"""


@traced
def llm_sentiment(headlines: list[str]) -> dict:
    """Aggregate sentiment of a list of headline strings, scored by an LLM and
    validated: {score in [-1,1], label, n_headlines, key_drivers}."""
    if err := _fault("llm_sentiment"):
        return {"error": err, "hint": "summarise the headlines' tone yourself"}
    heads = [h.strip() for h in (headlines or []) if isinstance(h, str) and h.strip()][:config.MAX_NEWS_N]
    if not heads:
        return {"error": "no headlines provided", "hint": "call get_news or web_search first"}
    from src.llm import chat_model
    try:
        llm = chat_model(config.SENTIMENT_MODEL, "low")
        raw = llm.invoke(SENTIMENT_PROMPT.format(headlines="\n".join(f"- {h}" for h in heads))).content
        block = raw[raw.find("{"): raw.rfind("}") + 1]
        res = SentimentResult.model_validate(json.loads(block))
        res.n_headlines = len(heads)
        return res.model_dump()
    except Exception as exc:
        return {"error": f"sentiment scoring failed: {str(exc)[:150]}",
                "hint": "retry once, or judge the tone of the headlines yourself"}


# --------------------------------------------------------------------------- #
# 5. web_search
# --------------------------------------------------------------------------- #
@traced
def web_search(query: str, n: int = config.DEFAULT_SEARCH_N) -> list[dict]:
    """DuckDuckGo web search (e.g. analyst commentary, price targets, risks).
    Returns up to `n` results as {title, snippet, url}."""
    if err := _fault("web_search"):
        return [{"error": err, "hint": "use get_news instead"}]
    if not (query or "").strip():
        return [{"error": "empty query", "hint": "pass a specific search query"}]
    n = max(1, min(int(n), config.MAX_SEARCH_N))
    from ddgs import DDGS
    res, errors = [], []
    for backend in config.SEARCH_BACKENDS:          # DuckDuckGo's backends fail intermittently
        try:
            res = DDGS().text(query, max_results=n, backend=backend) or []
            if res:
                break
        except Exception as exc:
            errors.append(f"{backend}: {str(exc)[:80]}")
    if errors and not res:
        return [{"error": f"search failed ({'; '.join(errors)})", "hint": "rephrase the query or use get_news"}]
    if not res:
        return [{"error": "no results", "hint": "broaden the query or use get_news"}]
    return [{"title": r.get("title", ""), "snippet": (r.get("body") or "")[:config.SEARCH_SNIPPET_CHARS],
             "url": r.get("href", "")} for r in res]


# --------------------------------------------------------------------------- #
# LangChain tool objects
# --------------------------------------------------------------------------- #
def _as_tool(fn) -> StructuredTool:
    return StructuredTool.from_function(func=fn, name=fn.__name__, description=fn.__doc__.strip())


ALL_TOOLS = {f.__name__: _as_tool(f) for f in
             (get_price_data, get_news, calculate_volatility, llm_sentiment, web_search)}


def tools_for(names: list[str]) -> list[StructuredTool]:
    """Tool subset for an agent - restriction is structural: an agent is built
    only with these objects and cannot call anything else."""
    return [ALL_TOOLS[n] for n in names]
