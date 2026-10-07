"""@traced: append every tool call to logs/agent_trace.jsonl.

Each line records: tool name, input arguments, output truncated to 200
characters, wall-clock duration, status (ok / error / exception), the calling
agent and the run id (set via ``trace_context``), and a timestamp. The
decorator never swallows the tool's behaviour: tools return structured
errors themselves; if a tool still raises, the exception is logged and then
re-raised so the failure is not hidden.

Agent/run attribution uses contextvars, so the trace shows *which* agent
called a tool - this is how tool restriction is evidenced in the log.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Tracing decorator writing tool
# name, inputs, truncated output, duration, agent and run id to JSONL',
# Date: 2026-10-07
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import config

_agent: contextvars.ContextVar[str] = contextvars.ContextVar("agent", default="-")
_run: contextvars.ContextVar[str] = contextvars.ContextVar("run", default="-")


@contextlib.contextmanager
def trace_context(agent: Optional[str] = None, run_id: Optional[str] = None):
    """Attribute tool calls inside this block to ``agent`` / ``run_id``."""
    tokens = []
    if agent is not None:
        tokens.append((_agent, _agent.set(agent)))
    if run_id is not None:
        tokens.append((_run, _run.set(run_id)))
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _jsonable(v: Any, limit: int = 300) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v if not isinstance(v, str) else v[:limit]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, limit) for x in list(v)[:20]]
    if isinstance(v, dict):
        return {str(k): _jsonable(x, limit) for k, x in v.items()}
    return str(v)[:limit]


def _status(result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        return "error"
    if isinstance(result, list) and result and isinstance(result[0], dict) and "error" in result[0]:
        return "error"
    return "ok"


def write_event(record: dict) -> None:
    """Append one JSON line (used by @traced and for non-tool events)."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    record.setdefault("agent", _agent.get())
    record.setdefault("run_id", _run.get())
    with open(config.TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def traced(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        inputs = {"args": _jsonable(list(args)), "kwargs": _jsonable(kwargs)}
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # logged, then re-raised: never hidden
            write_event({"event": "tool_call", "tool": fn.__name__, "inputs": inputs,
                         "output": f"EXCEPTION {type(exc).__name__}: {exc}"[:config.TRACE_OUTPUT_CHARS],
                         "status": "exception",
                         "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
            raise
        write_event({"event": "tool_call", "tool": fn.__name__, "inputs": inputs,
                     "output": json.dumps(result, default=str, ensure_ascii=False)[:config.TRACE_OUTPUT_CHARS],
                     "status": _status(result),
                     "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return result
    return wrapper


def read_trace(run_id: Optional[str] = None) -> list[dict]:
    if not config.TRACE_PATH.exists():
        return []
    with open(config.TRACE_PATH, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return [r for r in rows if run_id is None or r.get("run_id") == run_id]
