"""Recent news headlines from free sources, with layered fallbacks.

Source order (first to last):
  1. yfinance ``Ticker.news``       - structured, but its schema changed in 2024
                                      and it frequently returns < 10 (or 0) items.
  2. Yahoo Finance RSS headline feed - ticker-specific, free, no key.
  3. Google News RSS search         - broad coverage, guarantees >= 10 items.

Each source is isolated in its own try/except so one failing source never
prevents the others from contributing. Results are de-duplicated on a
normalised title and ranked (company-specific first, then newest first).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'News retrieval with yfinance
# primary and RSS fallbacks, dedupe and normalisation', Date: 2026-10-07
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus

import feedparser
import requests
import yfinance as yf

import config

log = logging.getLogger(__name__)

YAHOO_RSS_URL = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
                 "?s={ticker}&region=US&lang=en-US")
GOOGLE_NEWS_RSS_URL = ("https://news.google.com/rss/search"
                       "?q={query}&hl=en-US&gl=US&ceid=US:en")
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (equity-research-assistant)"}


def _parse_date(value) -> Optional[datetime]:
    """Parse RFC-822, ISO-8601 or epoch timestamps into aware UTC datetimes."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str) and value[:4].isdigit():
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsedate_to_datetime(value).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _item(title: str, publisher: str, link: str, published, source: str) -> dict:
    dt = _parse_date(published)
    return {
        "title": title.strip(),
        "publisher": publisher or "n/a",
        "link": link or "",
        "published": dt.isoformat() if dt else "",
        "source": source,
    }


def _from_yfinance(ticker: str) -> list[dict]:
    items = []
    try:
        for raw in (yf.Ticker(ticker).news or []):
            content = raw.get("content", raw)   # post-2024 schema nests 'content'
            title = content.get("title")
            if not title:
                continue
            provider = content.get("provider") or {}
            url = (content.get("canonicalUrl") or {}).get("url") or raw.get("link", "")
            items.append(_item(
                title,
                provider.get("displayName") or raw.get("publisher", ""),
                url,
                content.get("pubDate") or raw.get("providerPublishTime"),
                "yfinance"))
    except Exception as exc:
        log.warning("yfinance news failed for %s: %s", ticker, exc)
    return items


def _fetch_feed(url: str):
    """Fetch an RSS feed with an explicit timeout (feedparser has none)."""
    resp = requests.get(url, headers=_HTTP_HEADERS,
                        timeout=config.NEWS_HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _from_yahoo_rss(ticker: str) -> list[dict]:
    try:
        feed = _fetch_feed(YAHOO_RSS_URL.format(ticker=ticker))
        return [_item(e.get("title", ""), "Yahoo Finance", e.get("link", ""),
                      e.get("published"), "yahoo_rss")
                for e in feed.entries if e.get("title")]
    except Exception as exc:
        log.warning("Yahoo RSS failed for %s: %s", ticker, exc)
        return []


def _from_google_rss(ticker: str, company: Optional[str] = None) -> list[dict]:
    query = quote_plus(f"{ticker} stock" + (f" OR \"{company}\"" if company else ""))
    try:
        feed = _fetch_feed(GOOGLE_NEWS_RSS_URL.format(query=query))
    except Exception as exc:
        log.warning("Google News RSS failed for %s: %s", ticker, exc)
        return []
    items = []
    for e in feed.entries:
        title = e.get("title", "")
        publisher = (e.get("source") or {}).get("title", "Google News")
        # Google appends " - Publisher" to every title; strip it.
        suffix = f" - {publisher}"
        if title.endswith(suffix):
            title = title[: -len(suffix)]
        if title:
            items.append(_item(title, publisher, e.get("link", ""),
                               e.get("published"), "google_news_rss"))
    return items


def _normalise(title: str) -> str:
    """Normalised key for de-duplication (case, punctuation, whitespace)."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


_CORPORATE_SUFFIXES = {"inc", "corp", "corporation", "co", "ltd", "plc",
                       "holdings", "group", "company", "the", "class", "a"}


def company_keywords(ticker: str, company: Optional[str]) -> set[str]:
    """Lower-case tokens that mark a headline as being about this company,
    e.g. ('NVDA', 'NVIDIA Corporation') -> {'nvda', 'nvidia'}."""
    keys = {ticker.lower()}
    if company:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9&'-]+", company):
            if tok.lower() not in _CORPORATE_SUFFIXES:
                keys.add(tok.lower())
                break   # first distinctive token ('Apple', 'Alphabet', ...)
    return keys


def is_relevant(title: str, keywords: set[str]) -> bool:
    words = set(re.findall(r"[a-z0-9&'-]+", title.lower()))
    return bool(words & keywords)


# Auto-generated institutional-holdings (13F) headlines, e.g. "X LLC Buys 2,218
# Shares of NVIDIA". They carry no information about the company's prospects
# and would bias sentiment towards "neutral", so they are ranked last.
_PASSIVE_VERBS = ("sold|acquired|bought|purchased|newly purchased|cut|raised|"
                  "boosted|lowered|reduced|increased|trimmed|lifted|decreased|"
                  "grew|added")
_ACTIVE_VERBS = ("buys|sells|acquires|purchases|takes|adds|trims|cuts|boosts|"
                 "raises|lowers|lifts|increases|decreases|reduces|grows")
_HOLDING_NOUNS = "shares|stake|position|holdings|stock position"
_BOILERPLATE_RE = re.compile(
    # "...$NVDA Stock Sold by X LLC", "...Stake Cut by Y", "Shares Acquired by Z"
    r"\b(shares?|stock|stake|holdings|position)\s+(" + _PASSIVE_VERBS + r")\s+by\b"
    # "X LLC Buys 2,218 Shares of", "Purchases New Stake in", "Cuts Stock Position in"
    r"|\b(" + _ACTIVE_VERBS + r")\s+([\d,.$]+\s+|new\s+|its\s+)?(" + _HOLDING_NOUNS + r")\b",
    re.IGNORECASE)


def is_boilerplate(title: str) -> bool:
    return bool(_BOILERPLATE_RE.search(title))


def get_news(ticker: str = config.TICKER, n: int = config.MIN_HEADLINES,
             company: Optional[str] = None,
             max_items: int = config.MAX_HEADLINES) -> list[dict]:
    """Return between ``n`` and ``max_items`` recent, unique headlines.

    Each item: {title, publisher, link, published (ISO-8601 UTC), source,
    relevant, boilerplate}. All sources are queried (each is cheap and
    isolated), merged, de-duplicated, then ranked: substantive headlines naming
    the company first, 13F-filing boilerplate next, generic market news last;
    newest first within each tier. Ticker feeds such as Yahoo RSS often carry
    generic market pieces; ranking keeps the sentiment step focused on
    company-specific news while still guaranteeing >= n items.
    Never raises; may return fewer than ``n`` only if every source is down.
    """
    collected: list[dict] = []
    seen: set[str] = set()
    keywords = company_keywords(ticker, company)

    for batch in (_from_yfinance(ticker), _from_yahoo_rss(ticker),
                  _from_google_rss(ticker, company)):
        for it in batch:
            key = _normalise(it["title"])
            if key and key not in seen:
                seen.add(key)
                it["relevant"] = is_relevant(it["title"], keywords)
                it["boilerplate"] = is_boilerplate(it["title"])
                collected.append(it)

    # Stable sorts: newest first, then rank tiers
    #   0 = company-specific news, 1 = company-specific 13F boilerplate,
    #   2 = generic market news.
    collected.sort(key=lambda it: it["published"] or "", reverse=True)
    collected.sort(key=lambda it: 0 if it["relevant"] and not it["boilerplate"]
                   else 1 if it["relevant"] else 2)
    result = collected[:max_items]
    n_rel = sum(it["relevant"] for it in result)
    if len(result) < n:
        log.warning("Only %d headlines retrieved for %s (wanted %d)",
                    len(result), ticker, n)
    else:
        log.info("Retrieved %d headlines for %s (%d company-specific, from %d "
                 "unique candidates)", len(result), ticker, n_rel, len(collected))
    return result
