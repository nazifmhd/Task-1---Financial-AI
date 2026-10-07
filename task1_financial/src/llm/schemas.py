"""Pydantic contracts for every LLM response.

No LLM output is used anywhere in the pipeline until it has passed one of
these models. ``Literal`` enums make out-of-vocabulary answers (e.g. the
model saying "mixed" or "Strong Buy") fail validation loudly instead of
silently corrupting the aggregate.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic v2 schemas for
# headline sentiment and Buy/Hold/Sell signal with sentence-count validation',
# Date: 2026-10-07
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

import config

# A sentence ends with . ! or ? followed by whitespace or end-of-string, so
# decimals such as "RSI 67.3" are not counted as sentence breaks.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def count_sentences(text: str) -> int:
    return len(_SENTENCE_END.findall(text.strip()))


class HeadlineSentiment(BaseModel):
    """Per-headline classification - exactly the four fields the brief asks for."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    headline: str = Field(min_length=1)
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=3, max_length=300)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalise_case(cls, v):
        # Accept "Positive"/"NEUTRAL"; anything outside the enum still fails.
        return v.strip().lower() if isinstance(v, str) else v


class TradeSignal(BaseModel):
    """Reasoned Buy/Hold/Sell call over the indicator combination."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    signal: Literal["Buy", "Hold", "Sell"]
    conviction: Literal["low", "medium", "high"]
    justification: str = Field(min_length=40)
    key_drivers: list[str] = Field(min_length=2, max_length=4)
    conflicting_signals: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("signal", mode="before")
    @classmethod
    def _normalise_signal(cls, v):
        return v.strip().capitalize() if isinstance(v, str) else v

    @field_validator("conviction", mode="before")
    @classmethod
    def _normalise_conviction(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("justification")
    @classmethod
    def _three_to_five_sentences(cls, v: str) -> str:
        n = count_sentences(v)
        lo, hi = config.JUSTIFICATION_MIN_SENTENCES, config.JUSTIFICATION_MAX_SENTENCES
        if not lo <= n <= hi:
            raise ValueError(f"justification must be {lo}-{hi} sentences, got {n}")
        return v


class SentimentAggregate(BaseModel):
    """Aggregate over all successfully validated headlines."""
    per_headline: list[HeadlineSentiment]
    aggregate_score: float = Field(ge=-1.0, le=1.0)
    aggregate_label: Literal["positive", "negative", "neutral"]
    counts: dict[str, int]
    n_scored: int
    n_total: int
    n_failed: int
