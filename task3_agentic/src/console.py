"""Readable, step-by-step printing of agent message streams for the notebook.

Shows, for every step: the agent's decision (which tool, with which args),
the observation (truncated tool output, flagged if it is an error), and
derived REPLAN markers when a failed observation is followed by a different
action.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pretty-print LangGraph ReAct
# message streams with action/observation/replan markers', Date: 2026-10-07
"""
from __future__ import annotations

import json
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

MAX_OBS_CHARS = 260


def _is_error(content: str) -> bool:
    return '"error"' in content[:200]


def print_message(msg: BaseMessage, label: str = "") -> None:
    p = f"[{label}] " if label else ""
    if isinstance(msg, AIMessage):
        if msg.content and str(msg.content).strip():
            text = str(msg.content).strip().replace("\n", " ")
            print(f"{p}THINK/SAY: {text[:400]}{'...' if len(text) > 400 else ''}")
        for tc in msg.tool_calls or []:
            print(f"{p}ACTION  -> {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:200]})")
    elif isinstance(msg, ToolMessage):
        c = str(msg.content)
        flag = "OBSERVE (ERROR)" if _is_error(c) else "OBSERVE"
        print(f"{p}{flag} <- {msg.name}: {c[:MAX_OBS_CHARS]}{'...' if len(c) > MAX_OBS_CHARS else ''}")


def stream_and_print(agent, inputs: dict, config: dict, label: str) -> list[BaseMessage]:
    """Stream a ReAct agent, printing each new message; return all messages."""
    from langgraph.errors import GraphRecursionError
    messages: list[BaseMessage] = []
    try:
        for update in agent.stream(inputs, config, stream_mode="updates"):
            for node_out in update.values():
                for m in (node_out or {}).get("messages", []) if isinstance(node_out, dict) else []:
                    print_message(m, label)
                    messages.append(m)
    except GraphRecursionError:
        # Hard backstop: keep what was gathered instead of crashing the run.
        print(f"[{label}] step limit reached - continuing with the evidence gathered so far")
    return messages


def tool_sequence(messages: Iterable[BaseMessage]) -> list[str]:
    return [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]


def replan_events(messages: list[BaseMessage]) -> list[str]:
    """Observation -> next-decision links where the observation changed the plan:
    a tool returned an error and the agent's next action used a different tool
    (or different arguments)."""
    events, pending = [], []
    for m in messages:
        if isinstance(m, ToolMessage):
            pending.append(m)
        elif isinstance(m, AIMessage) and pending:
            failed = [t for t in pending if _is_error(str(t.content))]
            nxt = [(tc["name"], tc["args"]) for tc in (m.tool_calls or [])]
            for f in failed:
                if nxt:
                    acts = ", ".join(f"{n}({json.dumps(a, ensure_ascii=False)[:60]})" for n, a in nxt)
                    events.append(f"{f.name} returned an error -> agent replanned: {acts}")
                else:
                    events.append(f"{f.name} returned an error -> agent stopped calling tools and wrote "
                                  "the report from the evidence it had")
            pending = []
    return events
