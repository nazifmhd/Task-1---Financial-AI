"""End-to-end orchestration: Task 1A data -> Task 1B LLM reasoning.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Orchestrator wiring Task 1A
# outputs into LLM sentiment + signal with graceful degradation', Date: 2026-10-07
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import config
from src.llm.client import LLMClient, LLMUnavailable, get_client
from src.llm.schemas import SentimentAggregate
from src.llm.sentiment import aggregate, score_headlines
from src.llm.signal import generate_signal
from src.summary import Task1AResult, run_task1a

log = logging.getLogger(__name__)


@dataclass
class Task1BResult:
    sentiment: SentimentAggregate
    signal: dict          # {'signal': TradeSignal, 'source', 'prompt', 'prompt_version'}
    llm_model: str


def run_task1b(a: Task1AResult, client: Optional[LLMClient] = None) -> Task1BResult:
    """Score headlines, then produce the reasoned signal. Never raises on LLM
    unavailability: sentiment becomes an empty aggregate and the signal falls
    back to the rule-based vote (both clearly flagged)."""
    try:
        client = client or get_client()
        model = f"{client.provider}/{client.model}"
    except LLMUnavailable as exc:
        log.error("%s", exc)
        client, model = None, "unavailable"

    if client is not None:
        sentiment = score_headlines(a.headlines, a.ticker,
                                    a.summary.get("company_name"), client=client)
    else:
        sentiment = aggregate([], n_total=len(a.headlines))
    signal = generate_signal(a.prices, a.summary, sentiment, client=client) \
        if client is not None else generate_signal(a.prices, a.summary, sentiment,
                                                   client=_NullClient())
    return Task1BResult(sentiment=sentiment, signal=signal, llm_model=model)


class _NullClient:
    """Stand-in that always 'fails', forcing the documented fallback path."""
    provider, model = "none", "none"

    def call_json(self, *args, **kwargs):
        return None


def save_outputs(a: Task1AResult, b: Task1BResult) -> dict:
    """Persist machine-readable outputs next to the report."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": config.OUTPUT_DIR / "summary.json",
        "headlines": config.OUTPUT_DIR / "headlines.json",
        "sentiment": config.OUTPUT_DIR / "sentiment.json",
        "signal": config.OUTPUT_DIR / "signal.json",
        "prices": config.OUTPUT_DIR / "prices_with_indicators.csv",
    }
    paths["summary"].write_text(json.dumps(a.summary, indent=2, default=str), encoding="utf-8")
    paths["headlines"].write_text(json.dumps(a.headlines, indent=2), encoding="utf-8")
    paths["sentiment"].write_text(b.sentiment.model_dump_json(indent=2), encoding="utf-8")
    paths["signal"].write_text(json.dumps({
        **b.signal["signal"].model_dump(), "source": b.signal["source"],
        "prompt_version": b.signal["prompt_version"], "model": b.llm_model},
        indent=2), encoding="utf-8")
    a.prices.to_csv(paths["prices"])
    return {k: str(v) for k, v in paths.items()}


def run_all(ticker: str = config.TICKER):
    a = run_task1a(ticker)
    b = run_task1b(a)
    return a, b
