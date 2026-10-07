"""Typed contracts: tool outputs, the A->B handoff, the critique loop, the report.

Agents exchange these Pydantic objects (validated by LangGraph's structured
``response_format``), never free strings.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schemas for DataBrief
# handoff, clarification request/response and three-section research report',
# Date: 2026-10-07
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SentimentResult(BaseModel):
    """llm_sentiment output (validated before being returned to the agent)."""
    score: float = Field(ge=-1, le=1, description="-1 very negative ... +1 very positive")
    label: Literal["positive", "neutral", "negative"]
    n_headlines: int = 0
    key_drivers: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("key_drivers", mode="before")
    @classmethod
    def _cap_drivers(cls, v):
        # A fifth driver is not an error worth failing the tool for - keep the first four.
        return v[:4] if isinstance(v, list) else v


# --------------------------------------------------------------------------- #
# Agent A -> Agent B handoff
# --------------------------------------------------------------------------- #
class DataBrief(BaseModel):
    """Quantitative brief produced by the Data Analyst. Every number must come
    from a tool result; fields the analyst could not obtain stay null."""
    model_config = ConfigDict(extra="forbid")

    ticker: str
    as_of: Optional[str] = Field(None, description="date of the last price bar (YYYY-MM-DD)")
    current_price: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_histogram: Optional[float] = None
    pct_change_30d: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    trend_regime: Optional[str] = Field(None, description="e.g. 'uptrend: price > SMA50 > SMA200'")
    volatility_window_days: Optional[int] = None
    annualised_vol_pct: Optional[float] = None
    vol_percentile_1y: Optional[float] = Field(None, description="percentile of current vol vs last year")
    max_drawdown_1y_pct: Optional[float] = None
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[Literal["positive", "neutral", "negative"]] = None
    data_gaps: list[str] = Field(default_factory=list, description="tools that failed / data unavailable")
    analyst_notes: str = Field("", description="2-3 sentence quantitative read-out")


class ClarificationRequest(BaseModel):
    """Agent B -> Agent A: one specific question about missing quantitative data."""
    question: str
    field_needed: str = Field(description="e.g. 'annualised volatility over a 90-day window'")
    reason: str = Field(description="why the report needs it")
    headlines: list[str] = Field(default_factory=list,
                                 description="headline texts B gathered, for A to score with llm_sentiment")


class NamedValue(BaseModel):
    name: str = Field(description="e.g. 'annualised_vol_pct_90d'")
    value: float


class ClarificationResponse(BaseModel):
    """Agent A -> Agent B: the requested data, obtained with A's own tools.
    (``values`` is a list of name/value pairs rather than a dict because strict
    JSON-schema structured output does not allow free-form objects.)"""
    question: str
    answer: str
    values: list[NamedValue] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class BriefReview(BaseModel):
    """Agent B's review of the brief before writing (drives the critique edge)."""
    sufficient: bool
    clarification: Optional[ClarificationRequest] = None


# --------------------------------------------------------------------------- #
# Final report (three required sections)
# --------------------------------------------------------------------------- #
class Risk(BaseModel):
    title: str
    evidence: str = Field(description="specific numbers / headlines / sources that support it")
    source: str = Field(description="tool(s) the evidence came from")


class HedgeStrategy(BaseModel):
    strategy: str = Field(description="e.g. 'buy 90-day 10% OTM protective put'")
    rationale: str = Field(description="why, tied to the data (volatility, RSI, risks)")
    parameters: str = Field(description="strike/tenor/sizing or stop levels, data-derived")
    trade_offs: str


class FinalReport(BaseModel):
    ticker: str
    financial_health_summary: str
    market_sentiment: str
    top_three_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy
    data_limitations: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        risks = "\n".join(f"{i}. **{r.title}** — {r.evidence} _(source: {r.source})_"
                          for i, r in enumerate(self.top_three_risks, 1))
        h = self.hedge_strategy
        lim = "".join(f"\n- {x}" for x in self.data_limitations) or "\n- none reported"
        return (f"# {self.ticker} — Research Report\n\n"
                f"## Financial Health Summary\n{self.financial_health_summary}\n\n"
                f"**Market sentiment:** {self.market_sentiment}\n\n"
                f"## Top Three Risks (next 90 days)\n{risks}\n\n"
                f"## Hedge Strategy Recommendation\n**{h.strategy}**\n\n"
                f"- *Rationale:* {h.rationale}\n- *Parameters:* {h.parameters}\n"
                f"- *Trade-offs:* {h.trade_offs}\n\n"
                f"**Data limitations:**{lim}\n")
