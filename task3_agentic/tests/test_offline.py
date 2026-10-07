"""Offline tests (no LLM, no network): tracing, error contract, tool restriction,
context hook, cache, schemas.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Offline pytest for agent tools,
# tracing, tool restriction, context hook and cache', Date: 2026-10-07
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
from src import memory
from src.context import make_pre_model_hook
from src.observability.trace import read_trace, trace_context, traced
from src.schemas import DataBrief, FinalReport, HedgeStrategy, Risk, SentimentResult
from src.tools import ALL_TOOLS, inject_faults, tools_for


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


def test_traced_logs_required_fields_and_truncates():
    @traced
    def echo(x):
        return {"value": "y" * 1000}
    with trace_context(agent="tester", run_id="r1"):
        echo("abc")
    rec = read_trace("r1")[0]
    assert {"tool", "inputs", "output", "duration_ms", "agent", "status"} <= set(rec)
    assert rec["tool"] == "echo" and rec["agent"] == "tester"
    assert len(rec["output"]) <= config.TRACE_OUTPUT_CHARS


def test_traced_logs_and_reraises_exceptions():
    @traced
    def boom():
        raise ValueError("bad")
    with pytest.raises(ValueError):
        boom()
    assert read_trace()[0]["status"] == "exception"


def test_tools_return_structured_errors_not_exceptions():
    assert "error" in ALL_TOOLS["get_price_data"].invoke({"ticker": "bad ticker!"})
    assert "error" in ALL_TOOLS["calculate_volatility"].invoke({"ticker": "NVDA", "window": 1000})
    assert "error" in ALL_TOOLS["llm_sentiment"].invoke({"headlines": []})
    assert "error" in ALL_TOOLS["web_search"].invoke({"query": "  "})[0]
    with inject_faults("get_news"):
        out = ALL_TOOLS["get_news"].invoke({"ticker": "NVDA"})
    assert isinstance(out, list) and "simulated" in out[0]["error"]
    assert "get_news" not in config.FAULTY_TOOLS            # fault removed afterwards
    statuses = {r["tool"]: r["status"] for r in read_trace()}
    assert set(statuses.values()) == {"error"}


def test_tool_restriction_is_structural():
    from langgraph.prebuilt import ToolNode
    node = ToolNode(tools_for(["get_price_data", "calculate_volatility", "llm_sentiment"]))
    call = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    out = node.invoke({"messages": [call]})["messages"][0]
    assert "not a valid tool" in out.content


def test_pre_model_hook_compacts_old_observations_and_enforces_budget():
    msgs = [HumanMessage("q")]
    for i in range(4):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "get_news", "args": {}, "id": str(i)}]))
        msgs.append(ToolMessage(content="x" * 2000, tool_call_id=str(i), name="get_news"))
    view = make_pre_model_hook(tool_budget=4, keep_recent=2, old_chars=100)({"messages": msgs})["llm_input_messages"]
    tool_views = [m for m in view if isinstance(m, ToolMessage)]
    assert [len(m.content) < 300 for m in tool_views] == [True, True, False, False]
    assert "Tool budget reached" in view[-1].content
    assert len(msgs[2].content) == 2000                     # stored state untouched


def _report():
    r = Risk(title="t", evidence="e", source="s")
    return FinalReport(ticker="NVDA", financial_health_summary="ok", market_sentiment="pos",
                       top_three_risks=[r, r, r],
                       hedge_strategy=HedgeStrategy(strategy="put", rationale="r", parameters="p", trade_offs="t"))


def test_cache_roundtrip_and_hit_skips_pipeline(monkeypatch):
    state = {"ticker": "NVDA", "run_id": "3B-x", "brief": DataBrief(ticker="NVDA", current_price=1.0),
             "report": _report(), "handoffs": []}
    path = memory.save(state)
    assert path.name == f"NVDA_{config.TODAY.isoformat()}.json"
    monkeypatch.setattr("src.multi_agent.run_pipeline", lambda *a, **k: pytest.fail("pipeline must not run"))
    hit = memory.research("nvda")
    assert hit["from_cache"] and hit["report"]["ticker"] == "NVDA"
    assert any(r["event"] == "cache_hit" for r in read_trace())
    assert not any(r["event"] == "tool_call" for r in read_trace(hit["run_id"]))


def test_corrupt_cache_is_ignored():
    config.CACHE_DIR.mkdir(parents=True)
    memory.cache_path("NVDA").write_text("{not json")
    assert memory.load("NVDA") is None


def test_schemas_validate_ranges_and_three_risks():
    with pytest.raises(Exception):
        SentimentResult(score=1.5, label="positive")
    with pytest.raises(Exception):
        FinalReport(**{**_report().model_dump(), "top_three_risks": _report().model_dump()["top_three_risks"][:2]})
    assert "## Top Three Risks" in _report().to_markdown()
