"""Per-headline LLM sentiment and a confidence-weighted aggregate.

Aggregation design
------------------
score = sum(polarity_i * confidence_i) / sum(confidence_i),  polarity in {-1,0,+1}

* Confidence-weighted, not a majority vote: a 0.55-confidence "negative"
  should not cancel a 0.95-confidence "positive".
* Neutral headlines stay in the denominator, so a feed dominated by routine
  news pulls the score towards 0 instead of letting two positive items read as
  "+1.0 - very positive".
* Headlines whose LLM output never validated are EXCLUDED (and counted in
  ``n_failed``) rather than defaulted to neutral, which would silently bias
  the score.
* A dead-band (config.SENTIMENT_NEUTRAL_BAND) maps small scores to "neutral".

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Per-headline sentiment via
# LLM with Pydantic validation and confidence-weighted aggregation',
# Date: 2026-10-07
"""
from __future__ import annotations

import logging
from typing import Optional

import config
from .client import LLMClient, call_json
from .prompts import SENTIMENT_SYSTEM, SENTIMENT_USER_TMPL
from .schemas import HeadlineSentiment, SentimentAggregate

log = logging.getLogger(__name__)

POLARITY = {"positive": 1, "neutral": 0, "negative": -1}


def score_headline(headline: str, ticker: str, company: str,
                   client: Optional[LLMClient] = None) -> Optional[HeadlineSentiment]:
    """Classify one headline; returns None if the LLM never produced valid JSON."""
    user = SENTIMENT_USER_TMPL.format(company=company, ticker=ticker,
                                      headline=headline)
    out = call_json(SENTIMENT_SYSTEM, user, HeadlineSentiment,
                    task="headline_sentiment", client=client)
    if out is not None and out.headline != headline:
        # The model sometimes paraphrases; keep the source text authoritative.
        out = out.model_copy(update={"headline": headline})
    return out


def aggregate(results: list[HeadlineSentiment], n_total: int) -> SentimentAggregate:
    """Confidence-weighted polarity in [-1, 1] plus a dead-banded label."""
    weight = sum(r.confidence for r in results)
    score = (sum(POLARITY[r.sentiment] * r.confidence for r in results) / weight
             if weight > 0 else 0.0)
    band = config.SENTIMENT_NEUTRAL_BAND
    label = "positive" if score > band else "negative" if score < -band else "neutral"
    counts = {k: sum(r.sentiment == k for r in results) for k in POLARITY}
    return SentimentAggregate(per_headline=results, aggregate_score=round(score, 3),
                              aggregate_label=label, counts=counts,
                              n_scored=len(results), n_total=n_total,
                              n_failed=n_total - len(results))


def score_headlines(headlines: list[dict], ticker: str = config.TICKER,
                    company: Optional[str] = None,
                    client: Optional[LLMClient] = None) -> SentimentAggregate:
    """Score every headline (one LLM call each) and aggregate the valid ones."""
    company = company or ticker
    results: list[HeadlineSentiment] = []
    for item in headlines:
        title = item["title"] if isinstance(item, dict) else str(item)
        res = score_headline(title, ticker, company, client=client)
        if res is None:
            log.warning("Excluded from aggregate (no valid output): %s", title[:80])
        else:
            results.append(res)
    agg = aggregate(results, n_total=len(headlines))
    log.info("Sentiment: %s (%.3f) from %d/%d headlines", agg.aggregate_label,
             agg.aggregate_score, agg.n_scored, agg.n_total)
    return agg
