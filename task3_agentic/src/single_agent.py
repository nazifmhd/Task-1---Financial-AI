"""Task 3A: single tool-using research agent (LangGraph ReAct) + short-term memory.

* All five tools are bound; there is NO fixed order - the model picks each next
  tool from what it has observed so far (reason -> act -> observe -> replan).
* Tools return structured errors with hints, and the prompt tells the agent to
  reroute on failure, so a failed tool leads to an alternative, not a crash.
* After the loop, the conversation is distilled into a validated FinalReport
  (Pydantic) with the three required sections.
* A MemorySaver checkpointer keeps the thread's messages, so follow-up
  questions in the same thread are answered from earlier tool results
  (short-term memory, Task 3C).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph ReAct research agent
# with five tools, checkpointer memory and structured three-section report',
# Date: 2026-10-07
"""
from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

import config
from src.console import replan_events, stream_and_print, tool_sequence
from src.context import make_pre_model_hook
from src.llm import chat_model
from src.observability.trace import new_run_id, read_trace, trace_context, write_event
from src.schemas import FinalReport
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = f"""You are a senior equity research agent with five tools:
get_price_data, get_news, calculate_volatility, llm_sentiment, web_search.

How to work:
- Decide each next step from what you have observed so far. There is no fixed
  order and you do not need every tool; choose what the evidence calls for
  (e.g. if news turns negative, check volatility and search for analyst views;
  if volatility is elevated, quantify it over the {config.HEDGE_HORIZON_DAYS}-day horizon).
- Before each tool call, say in one short sentence why you are calling it.
- If a tool returns an "error", read its "hint" and try an alternative tool or
  different arguments. Never give up because one tool failed; if data is truly
  unavailable, say so explicitly in the report.
- Use only numbers that appear in tool results. If you derive a number (e.g. scaling
  volatility to 90 days) show the calculation; if you must estimate something no tool
  provides (e.g. an option premium), label it explicitly as an assumption.

When you have enough evidence, write the final answer with EXACTLY these sections:
## Financial Health Summary
## Top Three Risks (each with specific supporting evidence and its source tool)
## Hedge Strategy Recommendation (one data-driven hedge with parameters derived
   from the data, e.g. strike/tenor from price and volatility)"""

REPORT_EXTRACTION_PROMPT = """Convert the research conversation below into the FinalReport
schema. Keep every number exactly as it appears in the tool results; do not add facts.
List any tool failures or unavailable data under data_limitations."""

QUERY_TMPL = ("Analyse the current financial health and market sentiment of {ticker}. "
              "Identify the top three risks to its share price over the next "
              f"{config.HEDGE_HORIZON_DAYS} days and suggest one data-driven hedge strategy.")


def build_agent(checkpointer: Optional[MemorySaver] = None):
    return create_react_agent(chat_model(), tools=list(ALL_TOOLS.values()),
                              prompt=SYSTEM_PROMPT, checkpointer=checkpointer or MemorySaver(),
                              pre_model_hook=make_pre_model_hook())


def extract_report(messages) -> FinalReport:
    """Distil the agent conversation into a validated FinalReport."""
    transcript = []
    for m in messages:
        if isinstance(m, HumanMessage):
            transcript.append(f"USER: {m.content}")
        elif isinstance(m, ToolMessage):
            transcript.append(f"TOOL {m.name}: {str(m.content)[:config.TRANSCRIPT_TOOL_CHARS]}")
        elif isinstance(m, AIMessage) and m.content:
            transcript.append(f"AGENT: {m.content}")
    from src.multi_agent import _structured   # shared structured-output helper with fallback
    return _structured(FinalReport, REPORT_EXTRACTION_PROMPT, "\n\n".join(transcript))


class ResearchSession:
    """One agent + one memory thread. ``ask`` keeps context across calls."""

    def __init__(self, ticker: str, thread_id: Optional[str] = None):
        self.ticker = ticker
        self.agent = build_agent()
        self.thread_id = thread_id or new_run_id(f"thread-{ticker}")
        self.cfg = {"configurable": {"thread_id": self.thread_id},
                    "recursion_limit": config.AGENT_RECURSION_LIMIT}

    def ask(self, question: str, label: str = "single-agent") -> dict:
        run_id = new_run_id("3A")
        with trace_context(agent="single_agent", run_id=run_id):
            write_event({"event": "agent_start", "tool": None, "inputs": {"question": question[:200]},
                         "thread_id": self.thread_id})
            msgs = stream_and_print(self.agent, {"messages": [("user", question)]}, self.cfg, label)
        calls = [r for r in read_trace(run_id) if r.get("event") == "tool_call"]
        return {"run_id": run_id, "messages": msgs, "answer": msgs[-1].content if msgs else "",
                "tool_sequence": tool_sequence(msgs), "replans": replan_events(msgs),
                "tool_calls_logged": len(calls)}

    def history(self):
        return self.agent.get_state(self.cfg).values.get("messages", [])

    def research(self) -> dict:
        out = self.ask(QUERY_TMPL.format(ticker=self.ticker))
        out["report"] = extract_report(self.history())
        return out
