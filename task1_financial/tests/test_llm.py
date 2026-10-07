"""LLM layer tests with a fake transport (no network, no API key).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for Pydantic
# validation, repair-retry loop and sentiment aggregation using a fake
# OpenAI-compatible transport', Date: 2026-10-07
"""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.llm.client import LLMClient, extract_json
from src.llm.schemas import HeadlineSentiment, TradeSignal
from src.llm.sentiment import aggregate, score_headlines


class FakeChat:
    """Returns queued replies; records the messages it was sent."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _hs(s, c, h="h"):
    return json.dumps({"headline": h, "sentiment": s, "confidence": c,
                       "brief_reason": "because reasons"})


def test_extract_json_handles_fences_and_prose():
    assert json.loads(extract_json('Sure!\n```json\n{"a": 1}\n```'))["a"] == 1
    assert json.loads(extract_json('noise {"a": 2} trailing'))["a"] == 2
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_schema_rejects_out_of_enum_and_range():
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate_json(_hs("mixed", 0.5))
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate_json(_hs("positive", 1.4))
    assert HeadlineSentiment.model_validate_json(_hs("Positive", 0.7)).sentiment == "positive"


def test_trade_signal_sentence_count():
    base = {"signal": "buy", "conviction": "High", "key_drivers": ["a", "b"]}
    ok = TradeSignal(**base, justification="RSI is 67.3 here. Trend is up. MACD confirms.")
    assert ok.signal == "Buy" and ok.conviction == "high"
    with pytest.raises(ValidationError):
        TradeSignal(**base, justification="Only one sentence with value 67.3 inside it")


def test_repair_loop_recovers_after_invalid_output():
    fake = FakeChat(["not json at all", _hs("mixed", 0.5), _hs("negative", 0.8)])
    client = LLMClient(transport=fake)
    out = client.call_json("sys", "user", HeadlineSentiment, task="t")
    assert out.sentiment == "negative"
    assert len(fake.calls) == 3
    # The validation error was fed back to the model as a user turn.
    assert "failed validation" in fake.calls[-1][-1]["content"]


def test_returns_none_after_exhausting_retries():
    fake = FakeChat(["{}", "{}", "{}"])
    assert LLMClient(transport=fake).call_json("s", "u", HeadlineSentiment) is None


def test_failed_headlines_are_excluded_not_neutral():
    fake = FakeChat([_hs("positive", 0.9), "{}", "{}", "{}", _hs("negative", 0.3)])
    agg = score_headlines([{"title": "A"}, {"title": "B"}, {"title": "C"}],
                          "XYZ", "XYZ Corp", client=LLMClient(transport=fake))
    assert agg.n_scored == 2 and agg.n_failed == 1
    assert [h.headline for h in agg.per_headline] == ["A", "C"]   # source text kept
    assert agg.aggregate_score == pytest.approx((0.9 - 0.3) / 1.2, abs=1e-3)


def test_aggregate_confidence_weighting_and_deadband():
    mk = lambda s, c: HeadlineSentiment(headline="h", sentiment=s, confidence=c,
                                        brief_reason="xyz")
    agg = aggregate([mk("positive", 0.9), mk("negative", 0.3), mk("neutral", 0.8)], 3)
    assert agg.aggregate_score == pytest.approx(0.6 / 2.0, abs=1e-3)
    assert agg.aggregate_label == "positive"
    assert aggregate([mk("positive", 0.1), mk("neutral", 0.9)], 2).aggregate_label == "neutral"
    assert aggregate([], 5).aggregate_score == 0.0
