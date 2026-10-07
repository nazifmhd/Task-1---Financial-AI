"""Task 3C persistent memory: per-ticker, per-day JSON cache of research briefs.

Short-term memory lives in the LangGraph checkpointer (see single_agent.py);
this module is the cross-run layer. ``research`` checks the cache first: a hit
returns the stored brief + report and skips every agent and tool (logged as a
``cache_hit`` trace event, with zero tool_call events for that run); a miss runs
the full pipeline and writes cache/{TICKER}_{YYYY-MM-DD}.json.
The date in the key makes the cache expire daily, since prices change.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Per-ticker dated JSON cache for
# research briefs with cache-hit tracing', Date: 2026-10-07
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import config
from src.observability.trace import new_run_id, write_event
from src.schemas import ClarificationRequest, ClarificationResponse, DataBrief, FinalReport


def cache_path(ticker: str, day: Optional[date] = None) -> Path:
    return config.CACHE_DIR / f"{ticker.upper()}_{(day or config.TODAY).isoformat()}.json"


def load(ticker: str) -> Optional[dict]:
    p = cache_path(ticker)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        FinalReport.model_validate(data["report"])       # never trust a corrupt cache file
        DataBrief.model_validate(data["brief"])
        return data
    except Exception as exc:
        print(f"[CACHE INVALID] {p.name}: {exc} - ignoring it")
        return None


def save(state: dict) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dump = lambda o: o.model_dump() if o is not None else None
    data = {"ticker": state["ticker"], "date": config.TODAY.isoformat(),
            "source_run_id": state["run_id"],
            "brief": dump(state.get("brief")),
            "clarification": dump(state.get("clarification")),
            "clarification_response": dump(state.get("clarification_response")),
            "report": dump(state.get("report")),
            "handoffs": state.get("handoffs", [])}
    p = cache_path(state["ticker"])
    p.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return p


def research(ticker: str, pipeline=None, use_cache: bool = True) -> dict:
    """Cache-aware entry point used by the notebook's 3C demo."""
    ticker = ticker.upper()
    if use_cache and (hit := load(ticker)):
        run_id = new_run_id("cache")
        print(f"[CACHE HIT] {cache_path(ticker).name} - loaded the saved brief, "
              f"skipping all agents and tool calls")
        write_event({"event": "cache_hit", "tool": None, "agent": "orchestrator", "run_id": run_id,
                     "inputs": {"ticker": ticker}, "output": str(cache_path(ticker).name)})
        return {**hit, "from_cache": True, "run_id": run_id}
    print(f"[CACHE MISS] no brief for {ticker} on {config.TODAY} - running the agents")
    from src.multi_agent import run_pipeline
    state = run_pipeline(ticker, pipeline)
    path = save(state)
    print(f"[CACHE SAVED] {path.name}")
    return {"ticker": ticker, "brief": state["brief"].model_dump(),
            "report": state["report"].model_dump(), "from_cache": False,
            "run_id": state["run_id"], "handoffs": state.get("handoffs", [])}
