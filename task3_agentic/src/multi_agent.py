"""Task 3B: two-agent pipeline (LangGraph state machine) with a critique loop.

Roles and ENFORCED tool access (structural: each ReAct agent is constructed
with only its tool objects, so a call to any other tool cannot execute):
  Agent A  Data Analyst     get_price_data, calculate_volatility, llm_sentiment
  Agent B  Research Writer  web_search, get_news

Graph
  analyst_brief --DataBrief--> writer_review --(BriefReview)--+--> analyst_clarify --ClarificationResponse--+
                                                               |                                            |
                                                               +--(sufficient / round used)--> writer_report <+
                                                                                                    |
                                                                                                  END

Why the critique loop is genuine, not forced: A has no news tool, so it cannot
score sentiment on its own; B is told the hedge horizon is 90 days while A's
first brief uses a 30-day volatility window. B reviews the brief against what
the report needs and decides whether to ask; when it asks, it attaches the
headlines it gathered so A can score them. A answers with its own tools, B
incorporates. ``clarification_rounds`` caps the loop at one round-trip.

Handoffs are Pydantic objects held in graph state (DataBrief,
ClarificationRequest, ClarificationResponse, FinalReport), and every handoff
is appended to ``handoffs`` - the visible message trace.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph two-agent pipeline with
# tool-restricted ReAct agents, Pydantic handoffs, one-round critique loop and
# handoff trace', Date: 2026-10-07
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

import config
from src.console import stream_and_print, tool_sequence
from src.context import make_pre_model_hook
from src.llm import chat_model
from src.observability.trace import new_run_id, trace_context, write_event
from src.schemas import (BriefReview, ClarificationRequest, ClarificationResponse, DataBrief,
                         FinalReport)
from src.tools import tools_for

ANALYST_TOOLS = ["get_price_data", "calculate_volatility", "llm_sentiment"]
WRITER_TOOLS = ["web_search", "get_news"]

ANALYST_PROMPT = """You are Agent A, a quantitative Data Analyst. Your ONLY tools are
get_price_data, calculate_volatility and llm_sentiment. You have NO news and NO web access.
Gather the quantitative facts requested using your tools. Use only numbers from tool results.
If you cannot obtain something (e.g. sentiment needs headlines you do not have), leave it
out and say so; never invent values. End with a short quantitative read-out."""

WRITER_PROMPT = f"""You are Agent B, a Research Writer. Your ONLY tools are web_search and
get_news. You have NO price or volatility tools: every number about price, technicals,
volatility or sentiment scores must come from Agent A's DataBrief or its clarification answers.
Research the qualitative picture: recent news, analyst commentary, upcoming catalysts and
risks over the next {config.HEDGE_HORIZON_DAYS} days. Be efficient: about 3-4 tool calls
(news once, then 2-3 focused searches). If a tool fails, read its hint and use the other tool
or a different query. End with concise research notes that cite your sources."""

REVIEW_PROMPT = f"""You are Agent B reviewing Agent A's DataBrief before writing a report that
must (1) summarise financial health and market sentiment, (2) give the top three risks over the
next {config.HEDGE_HORIZON_DAYS} days with evidence, and (3) recommend one data-driven hedge whose
parameters (strike, tenor, sizing) are derived from volatility over the hedge horizon.
Decide whether the brief contains every quantitative input the report needs.
If something essential is missing or measured over the wrong horizon, set sufficient=false and
write ONE specific clarification for Agent A (what to compute, with which window/parameters).
Agent A has no news access: if you need a sentiment score, put the headlines you gathered in the
question so A can score them with its llm_sentiment tool."""

REPORT_PROMPT = """You are Agent B writing the final research report. Use ONLY: the DataBrief and
clarification answer from Agent A (all numbers), and your own research notes (news, analyst views,
sources). Each risk must cite specific evidence and its source tool. The hedge must use A's
horizon-matched volatility to justify strike and tenor; label anything not provided by a tool as
an assumption. Record missing data or tool failures in data_limitations."""


class PipelineState(TypedDict, total=False):
    ticker: str
    run_id: str
    brief: Optional[DataBrief]
    writer_notes: str
    review: Optional[BriefReview]
    clarification: Optional[ClarificationRequest]
    clarification_response: Optional[ClarificationResponse]
    clarification_rounds: int
    report: Optional[FinalReport]
    handoffs: Annotated[list[dict], operator.add]
    tools_used: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------- #
# Agents (tool restriction by construction)
# --------------------------------------------------------------------------- #
def build_agents():
    analyst = create_react_agent(chat_model(), tools=tools_for(ANALYST_TOOLS),
                                 prompt=ANALYST_PROMPT, name="data_analyst",
                                 pre_model_hook=make_pre_model_hook())
    writer = create_react_agent(chat_model(), tools=tools_for(WRITER_TOOLS),
                                prompt=WRITER_PROMPT, name="research_writer",
                                pre_model_hook=make_pre_model_hook(tool_budget=4))
    return analyst, writer


def bound_tool_names(agent) -> list[str]:
    """Tools actually wired into an agent's ToolNode (proof of restriction)."""
    return sorted(agent.get_graph().nodes["tools"].data.tools_by_name)


def _transcript(messages) -> str:
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            out.append(f"TOOL {m.name}: {str(m.content)[:config.TRANSCRIPT_TOOL_CHARS]}")
        elif isinstance(m, AIMessage) and m.content:
            out.append(f"AGENT: {m.content}")
        elif isinstance(m, HumanMessage):
            out.append(f"TASK: {m.content}")
    return "\n\n".join(out)


def _headlines_from(messages, limit: int = config.MAX_NEWS_N) -> list[str]:
    """Headline texts from this agent's own get_news observations."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "get_news":
            try:
                out += [it["title"] for it in json.loads(m.content) if isinstance(it, dict) and it.get("title")]
            except (ValueError, TypeError):
                pass
    return list(dict.fromkeys(out))[:limit]


def _structured(schema, system: str, payload: str):
    """Structured output with method fallback: providers differ in which JSON-schema
    features they accept, so a schema quirk degrades to the next method instead of
    failing the pipeline."""
    msgs = [SystemMessage(system), HumanMessage(payload)]
    last = None
    for method in ("json_schema", "function_calling", "json_mode"):
        try:
            llm = chat_model().with_structured_output(schema, method=method)
            if method == "json_mode":
                msgs = [SystemMessage(system + "\nReturn ONLY JSON matching this schema:\n"
                                      + json.dumps(schema.model_json_schema())), HumanMessage(payload)]
            out = llm.invoke(msgs)
            if out is not None:
                return out
        except Exception as exc:
            last = exc
            print(f"[structured output] {schema.__name__} via {method} failed: {str(exc)[:120]} - trying next method")
    raise RuntimeError(f"could not produce {schema.__name__}: {last}")


def _handoff(frm: str, to: str, kind: str, obj, run_id: str) -> dict:
    rec = {"from": frm, "to": to, "type": kind,
           "payload": obj.model_dump() if hasattr(obj, "model_dump") else obj}
    write_event({"event": "handoff", "tool": None, "agent": frm, "run_id": run_id,
                 "inputs": {"from": frm, "to": to, "type": kind},
                 "output": json.dumps(rec["payload"], default=str)[:config.TRACE_OUTPUT_CHARS]})
    print(f"\n>>> HANDOFF {frm} -> {to}: {kind}\n{json.dumps(rec['payload'], indent=1, default=str)[:1800]}\n")
    return rec


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_pipeline(verbose: bool = True):
    analyst, writer = build_agents()
    rcfg = {"recursion_limit": config.AGENT_RECURSION_LIMIT}

    def run_agent(agent, label, agent_name, task, run_id):
        print(f"\n=== {label} ===")
        with trace_context(agent=agent_name, run_id=run_id):
            msgs = stream_and_print(agent, {"messages": [("user", task)]}, rcfg, label)
        return msgs

    def analyst_brief(s: PipelineState):
        task = (f"Build the quantitative DataBrief for {s['ticker']}: price level, trend (SMA50/200), "
                "RSI, MACD histogram, 30-day change, distance from the 52-week high, and 30-day "
                "annualised volatility with its 1-year percentile and max drawdown.")
        msgs = run_agent(analyst, "Agent A: Data Analyst", "data_analyst", task, s["run_id"])
        brief = _structured(DataBrief, "Fill the DataBrief strictly from these tool results. "
                            "Leave fields null when no tool provided them and list them in data_gaps.",
                            _transcript(msgs))
        brief.ticker = s["ticker"]
        return {"brief": brief, "handoffs": [_handoff("data_analyst", "research_writer", "DataBrief", brief, s["run_id"])],
                "tools_used": [{"agent": "data_analyst", "tools": tool_sequence(msgs)}]}

    def writer_review(s: PipelineState):
        task = (f"Research {s['ticker']} for a {config.HEDGE_HORIZON_DAYS}-day risk report. "
                "Gather recent headlines (get_news) and analyst commentary on risks and catalysts "
                "(web_search).")
        msgs = run_agent(writer, "Agent B: Research Writer (research)", "research_writer", task, s["run_id"])
        notes = _transcript(msgs)
        review = _structured(BriefReview, REVIEW_PROMPT,
                             f"DATABRIEF (from Agent A):\n{s['brief'].model_dump_json(indent=1)}\n\n"
                             f"YOUR RESEARCH NOTES:\n{notes}")
        out = {"writer_notes": notes, "review": review,
               "tools_used": [{"agent": "research_writer", "tools": tool_sequence(msgs)}]}
        if not review.sufficient and review.clarification and s.get("clarification_rounds", 0) < config.MAX_CLARIFICATION_ROUNDS:
            # The data B refers to must travel WITH the request: attach the headlines
            # B observed via get_news (A has no news tool and cannot fetch them).
            if not review.clarification.headlines:
                review.clarification.headlines = _headlines_from(msgs)
            out["clarification"] = review.clarification
            out["handoffs"] = [_handoff("research_writer", "data_analyst", "ClarificationRequest",
                                        review.clarification, s["run_id"])]
        else:
            out["handoffs"] = [_handoff("research_writer", "research_writer", "BriefReview (no clarification needed)", review, s["run_id"])]
        return out

    def analyst_clarify(s: PipelineState):
        req = s["clarification"]
        heads = "".join(f"\n- {h}" for h in req.headlines)
        task = (f"Agent B (Research Writer) asks about {s['ticker']}: {req.question}\n"
                f"Field needed: {req.field_needed}\nWhy: {req.reason}\n"
                + (f"Headlines supplied by Agent B:{heads}\n" if heads else "")
                + "Answer using your own tools only.")
        msgs = run_agent(analyst, "Agent A: Data Analyst (answering clarification)", "data_analyst",
                         task, s["run_id"])
        resp = _structured(ClarificationResponse, "Summarise Agent A's answer strictly from these "
                           "tool results; put each computed number in `values` and list the tools used.",
                           _transcript(msgs))
        resp.question = req.question
        resp.tools_used = tool_sequence(msgs)        # from the trace, not the LLM's claim
        return {"clarification_response": resp,
                "clarification_rounds": s.get("clarification_rounds", 0) + 1,
                "handoffs": [_handoff("data_analyst", "research_writer", "ClarificationResponse", resp, s["run_id"])],
                "tools_used": [{"agent": "data_analyst", "tools": tool_sequence(msgs)}]}

    def writer_report(s: PipelineState):
        print("\n=== Agent B: Research Writer (final report) ===")
        clar = s.get("clarification_response")
        payload = (f"DATABRIEF (Agent A):\n{s['brief'].model_dump_json(indent=1)}\n\n"
                   f"CLARIFICATION ANSWER (Agent A):\n{clar.model_dump_json(indent=1) if clar else 'none'}\n\n"
                   f"RESEARCH NOTES (Agent B):\n{s['writer_notes']}")
        with trace_context(agent="research_writer", run_id=s["run_id"]):
            report = _structured(FinalReport, REPORT_PROMPT, payload)
        report.ticker = s["ticker"]
        return {"report": report,
                "handoffs": [_handoff("research_writer", "user", "FinalReport", report, s["run_id"])]}

    def route_after_review(s: PipelineState) -> str:
        return "analyst_clarify" if s.get("clarification") and not s.get("clarification_response") \
            else "writer_report"

    g = StateGraph(PipelineState)
    g.add_node("analyst_brief", analyst_brief)
    g.add_node("writer_review", writer_review)
    g.add_node("analyst_clarify", analyst_clarify)
    g.add_node("writer_report", writer_report)
    g.add_edge(START, "analyst_brief")
    g.add_edge("analyst_brief", "writer_review")
    g.add_conditional_edges("writer_review", route_after_review,
                            {"analyst_clarify": "analyst_clarify", "writer_report": "writer_report"})
    g.add_edge("analyst_clarify", "writer_report")
    g.add_edge("writer_report", END)
    return g.compile(), analyst, writer


def run_pipeline(ticker: str, pipeline=None) -> PipelineState:
    """One automated end-to-end run: query in, FinalReport out, no manual steps."""
    pipeline = pipeline or build_pipeline()[0]
    run_id = new_run_id("3B")
    write_event({"event": "pipeline_start", "tool": None, "inputs": {"ticker": ticker},
                 "agent": "orchestrator", "run_id": run_id})
    state = pipeline.invoke({"ticker": ticker, "run_id": run_id, "clarification_rounds": 0,
                             "handoffs": [], "tools_used": []},
                            {"recursion_limit": config.AGENT_RECURSION_LIMIT})
    write_event({"event": "pipeline_end", "tool": None, "inputs": {"ticker": ticker},
                 "agent": "orchestrator", "run_id": run_id,
                 "output": f"clarification_rounds={state.get('clarification_rounds', 0)}"})
    return state
