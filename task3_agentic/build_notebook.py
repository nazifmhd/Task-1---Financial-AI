"""Generates task3_agentic.ipynb (refuses to overwrite an executed copy unless
FORCE_REBUILD=1).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate the Task 3 notebook
# with nbformat', Date: 2026-10-07
"""
import os

import nbformat as nbf

USER, REPO = "nazifmhd", "Task-1---Financial-AI"
NB = "task3_agentic.ipynb"
COLAB = f"https://colab.research.google.com/github/{USER}/{REPO}/blob/main/task3_agentic/{NB}"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# Task 3 — Multi-Agent Financial Research System (LangGraph)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

| Part | What this notebook shows |
|---|---|
| 3A | Five custom tools. A ReAct agent picks tools **autonomously** with visible *action → observation → replan* steps, writes a validated three-section report, and recovers from tool failures (injected outage, fake ticker). |
| 3B | Data Analyst (A) and Research Writer (B) with **structurally enforced** tool access, Pydantic handoffs, a full message trace, and a critique loop (B → A → B). |
| 3C | Short-term memory (follow-up answered with **zero** new tool calls), a persistent per-ticker cache (second run = cache hit), and `logs/agent_trace.jsonl`. |
| Bonus | Streamlit dashboard over the trace (`dashboard.py`), exercised headlessly below. |

LLMs run on the Groq free tier through its OpenAI-compatible API. The agents use `openai/gpt-oss-20b`;
the `llm_sentiment` tool uses `openai/gpt-oss-120b`, a separate model with its own rate limit.
""")
code(f"""
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("/content/{REPO}"):
        subprocess.run(["git", "clone", "-q", "https://github.com/{USER}/{REPO}.git", "/content/{REPO}"], check=True)
    os.chdir("/content/{REPO}/task3_agentic")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
elif os.path.basename(os.getcwd()) != "task3_agentic" and os.path.isdir("task3_agentic"):
    os.chdir("task3_agentic")
sys.path.insert(0, os.getcwd())
import json, time, warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from IPython.display import display, Markdown
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 220)
import config, langgraph, langchain_core
from src.llm import read_secret
print("cwd:", os.getcwd(), "| Colab:", IN_COLAB, "| python:", sys.executable)
print("langgraph", langgraph.__version__ if hasattr(langgraph, "__version__") else "", "| langchain-core", langchain_core.__version__)
print("GROQ key loaded:", bool(read_secret("GROQ_API_KEY")), "(value never printed)")
print("agent model:", config.AGENT_MODEL, "| sentiment model:", config.SENTIMENT_MODEL)
""")
code("""
# Fresh start so logs/agent_trace.jsonl and the cache reflect exactly this notebook run.
from src.memory import cache_path
for p in [config.TRACE_PATH, cache_path("NVDA")]:
    if p.exists():
        p.unlink()
print("trace reset:", config.TRACE_PATH.relative_to(config.PROJECT_DIR), "| cache cleared for NVDA", config.TODAY)
""")

md("""
## 1. The five tools
Each tool is a LangChain `StructuredTool` built from a typed Python function and wrapped by `@traced`.
They return `dict` (price, volatility, sentiment) or `list[dict]` (news, search). **No tool raises.**
Failures return `{"error": ..., "hint": ...}`, and the hint suggests an alternative so the agent can replan.
""")
code("""
from src.tools import ALL_TOOLS, inject_faults
T = ALL_TOOLS
for name, t in T.items():
    print(f"{name}{tuple(t.args)}: {t.description.splitlines()[0]}")
""")
code("""
from src.observability.trace import trace_context, new_run_id
DEMO = new_run_id("demo")
with trace_context(agent="tools_demo", run_id=DEMO):        # attribute these direct calls in the trace
    price = T["get_price_data"].invoke({"ticker": "NVDA", "period": "1y"})
    vol = T["calculate_volatility"].invoke({"ticker": "NVDA", "window": 30})
    news = T["get_news"].invoke({"ticker": "NVDA", "n": 6})
    sent = T["llm_sentiment"].invoke({"headlines": [n["title"] for n in news]})
    search = T["web_search"].invoke({"query": "NVIDIA stock analyst risks outlook", "n": 3})
for name, out in [("get_price_data", price), ("calculate_volatility", vol), ("get_news", news),
                  ("llm_sentiment", sent), ("web_search", search)]:
    print(f"\\n{name} -> {type(out).__name__}" + (f"[{type(out[0]).__name__}] x{len(out)}" if isinstance(out, list) else ""))
    print(json.dumps(out if not isinstance(out, list) else out[:2], ensure_ascii=False, default=str)[:600])
""")
code("""
# Error contract: structured errors, never exceptions
with trace_context(agent="tools_demo", run_id=DEMO):
    with inject_faults("get_news"):
        outage = T["get_news"].invoke({"ticker": "NVDA"})
    cases = {
        "unknown ticker": T["get_price_data"].invoke({"ticker": "FAKETICKER123"}),
        "invalid window": T["calculate_volatility"].invoke({"ticker": "NVDA", "window": 1000}),
        "no headlines": T["llm_sentiment"].invoke({"headlines": []}),
        "injected news outage": outage,
    }
for k, out in cases.items():
    print(f"{k:22s} -> {json.dumps(out, default=str)[:170]}")
""")

md("""
## 2. Task 3A — Single tool-using agent
A LangGraph ReAct agent (`create_react_agent`) with all five tools and **no prescribed order**. Each step below
is either an `ACTION` (the tool and arguments the model chose) or an `OBSERVE` (the tool result it then reasons over).
A `pre_model_hook` keeps prompts small: older observations are shortened in the model's view only, and the full
results stay in memory. After the loop, the conversation is distilled into a Pydantic-validated `FinalReport`.
""")
code("""
from src.single_agent import ResearchSession, QUERY_TMPL
print("QUERY:", QUERY_TMPL.format(ticker="NVDA"), "\\n")
session = ResearchSession("NVDA")
t0 = time.time()
run_a = session.research()
print(f"\\nTool sequence chosen by the agent: {run_a['tool_sequence']}")
print(f"Tool calls logged to the trace for this run: {run_a['tool_calls_logged']} | {time.time()-t0:.0f}s")
print("Replan events:", run_a["replans"] or "none (no tool failed in this run)")
""")
code("""display(Markdown(run_a["report"].to_markdown()))""")

md("""
### 2.1 Observe → replan under failure: injected `get_news` outage (different ticker)
The same agent code runs on **MSFT** while `get_news` simulates an upstream outage. The agent observes the error
(with its hint) and changes its plan, typically switching to `web_search` for news.
""")
code("""
with inject_faults("get_news"):
    run_b = ResearchSession("MSFT").research()
print("\\nTool sequence:", run_b["tool_sequence"])
print("Observe -> replan events:")
for e in run_b["replans"]:
    print("  *", e)
display(Markdown(run_b["report"].to_markdown()))
""")

md("""
### 2.2 Graceful failure: a ticker that does not exist
""")
code("""
run_c = ResearchSession("FAKETICKER123").ask(QUERY_TMPL.format(ticker="FAKETICKER123"), label="fake-ticker")
print("\\nTool sequence:", run_c["tool_sequence"])
print("Replan events:"); [print("  *", e) for e in run_c["replans"]]
print("\\nFinal answer (excerpt):", run_c["answer"][:700])
print("\\nNo unhandled exception: the run completed and the agent explained the data was unavailable.")
""")
code("""
# Autonomy evidence: the same code produced different tool plans depending on what each run observed.
display(pd.DataFrame({"run": ["NVDA (normal)", "MSFT (news outage)", "FAKETICKER123"],
                      "tool sequence chosen by the agent": [" -> ".join(r["tool_sequence"]) for r in (run_a, run_b, run_c)],
                      "replans": [len(r["replans"]) for r in (run_a, run_b, run_c)]}))
""")

md("""
## 3. Task 3C (part 1) — Short-term memory
The agent has a `MemorySaver` checkpointer. A follow-up in the **same thread** is answered from the earlier tool
results, and the trace shows **zero** new tool calls for that turn.
""")
code("""
follow = session.ask("Follow-up: what RSI(14), 30-day annualised volatility and sentiment score did you find "
                     "for NVDA earlier? Answer from what you already retrieved; do not call tools again.",
                     label="follow-up")
print("\\nTool calls made for the follow-up:", follow["tool_calls_logged"], "| tools:", follow["tool_sequence"] or "none")
print("Messages remembered in thread", session.thread_id, ":", len(session.history()))
""")

md("""
## 4. Task 3B — Two agents with enforced roles, structured handoffs and a critique loop
| Agent | Role | Tools (the only ones it is built with) | Output |
|---|---|---|---|
| A — Data Analyst | quantitative | `get_price_data`, `calculate_volatility`, `llm_sentiment` | `DataBrief` (Pydantic) |
| B — Research Writer | qualitative | `web_search`, `get_news` | `BriefReview` → `ClarificationRequest` or `FinalReport` |

**Why the critique loop is genuine:** A has no news tool, so it cannot score sentiment by itself, and A's first brief
uses a 30-day volatility window while the report concerns a 90-day horizon. B reviews the brief, decides what is
missing, and sends **one** `ClarificationRequest`, attaching the headlines it fetched. A answers with its own tools
(`ClarificationResponse`), and B incorporates the answer. A counter in state caps the loop at one round-trip.
""")
code("""
from src.multi_agent import build_pipeline, bound_tool_names
pipeline, analyst, writer = build_pipeline()
print("Agent A (Data Analyst) tools :", bound_tool_names(analyst))
print("Agent B (Research Writer) tools:", bound_tool_names(writer))
# Restriction is structural: A's ToolNode cannot execute a tool it was not built with.
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from src.tools import tools_for
from src.multi_agent import ANALYST_TOOLS
attempt = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "NVDA"}, "id": "x"}])
print("A attempting web_search ->", ToolNode(tools_for(ANALYST_TOOLS)).invoke({"messages": [attempt]})["messages"][0].content)
""")
code("""
from IPython.display import Image
try:
    display(Image(pipeline.get_graph().draw_mermaid_png()))
except Exception:
    print(pipeline.get_graph().draw_ascii())
""")
md("### 4.1 End-to-end run (one call, no manual steps). Persistent cache: first run is a MISS")
code("""
from src.memory import research
t0 = time.time()
res1 = research("NVDA", pipeline=pipeline)
print(f"\\nEnd-to-end time: {time.time()-t0:.0f}s | from cache: {res1['from_cache']}")
""")
code("""
# The message trace: every handoff between agents, with typed payloads
hand = pd.DataFrame([{"#": i + 1, "from": h["from"], "to": h["to"], "type": h["type"],
                      "payload keys": ", ".join(list(h["payload"])[:8])} for i, h in enumerate(res1["handoffs"])])
display(hand)
from src.schemas import DataBrief
brief = DataBrief.model_validate(res1["brief"])           # the A -> B handoff is a validated object
print("DataBrief validated:", type(brief).__name__, "| data_gaps:", brief.data_gaps)
crit = [h for h in res1["handoffs"] if h["type"] in ("ClarificationRequest", "ClarificationResponse")]
print(f"Critique loop round-trips: {len([h for h in crit if h['type'] == 'ClarificationRequest'])}")
for h in crit:
    p = {k: v for k, v in h["payload"].items() if k != "headlines"}
    print(f"  {h['from']} -> {h['to']} [{h['type']}]: {json.dumps(p, ensure_ascii=False)[:400]}")
""")
code("""
# Which agent used which tool in this pipeline run (from the trace): no cross-over.
from src.observability.trace import read_trace
tr = pd.DataFrame([r for r in read_trace(res1["run_id"]) if r["event"] == "tool_call"])
display(pd.crosstab(tr["agent"], tr["tool"]))
""")
code("""
from src.schemas import FinalReport
report = FinalReport.model_validate(res1["report"])
display(Markdown(report.to_markdown()))
""")

md("""
## 5. Task 3C (part 2) — Persistent cache: the second run is a HIT
The same call again loads `cache/NVDA_<date>.json`. No agent runs and no tool is called, as the trace for the
run confirms.
""")
code("""
t0 = time.time()
res2 = research("NVDA", pipeline=pipeline)
calls2 = [r for r in read_trace(res2["run_id"]) if r["event"] == "tool_call"]
print(f"from cache: {res2['from_cache']} | time: {time.time()-t0:.2f}s | tool calls in this run: {len(calls2)}")
print("cached file:", cache_path("NVDA").relative_to(config.PROJECT_DIR))
assert res2["report"] == res1["report"]
""")

md("""
## 6. Observability — `logs/agent_trace.jsonl`
Every tool call is logged with its **tool name, inputs, output (truncated to 200 chars) and wall-clock duration**,
plus the calling agent, run id, status and timestamp. Handoffs and cache hits are logged as events too.
""")
code("""
trace = pd.DataFrame(read_trace())
print("events:", len(trace), "| tool calls:", int((trace.event == "tool_call").sum()),
      "| errors:", int((trace.status == "error").sum()), "| file:", config.TRACE_PATH.relative_to(config.PROJECT_DIR))
print("\\nFirst 3 lines of the file:")
with open(config.TRACE_PATH, encoding="utf-8") as f:
    for _ in range(3):
        print(f.readline().strip()[:400])
calls = trace[trace.event == "tool_call"]
display(calls.groupby("tool").agg(calls=("tool", "size"), mean_ms=("duration_ms", "mean"),
                                  errors=("status", lambda s: int((s != "ok").sum()))).round(1))
display(pd.crosstab(calls["run_id"], calls["tool"]))
""")

md("""
## 7. Bonus — Streamlit dashboard over the trace
`streamlit run dashboard.py` opens an interactive view with per-run filters, KPIs, a timeline, latency by tool,
an agent × tool matrix and error/handoff tables. Below, the app is executed headlessly with Streamlit's
`AppTest` to show it runs on this trace file. A static preview of the same views is rendered with matplotlib.
""")
code("""
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("dashboard.py", default_timeout=60).run()
print("dashboard ran without exceptions:", not at.exception)
print("metrics:", {m.label: m.value for m in at.metric})
print("sections:", [h.value for h in at.subheader])
""")
code("""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
c = calls.copy(); c["ts"] = pd.to_datetime(c["ts"])
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6bd1", "#a3a29c"]
colors = {a: palette[i % len(palette)] for i, a in enumerate(sorted(c["agent"].unique()))}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2, 1]})
t0 = c["ts"].min()
for _, r in c.sort_values("ts").iterrows():
    start = (r["ts"] - t0).total_seconds() - r["duration_ms"] / 1000
    axes[0].barh(r["tool"], max(r["duration_ms"] / 1000, 1.0), left=start, height=0.6,
                 color=colors[r["agent"]], edgecolor="black" if r["status"] != "ok" else "none", linewidth=1.5)
axes[0].legend(handles=[Patch(color=col, label=a) for a, col in colors.items()]
               + [Patch(facecolor="white", edgecolor="black", label="error")],
               frameon=False, fontsize=8, loc="lower right")
axes[0].set(title="Tool-call timeline (colour = calling agent)", xlabel="seconds since first call")
c.groupby("tool")["duration_ms"].mean().sort_values().plot.barh(ax=axes[1], color="#2a78d6")
axes[1].set(title="Mean latency by tool (ms)", ylabel="")
for ax in axes:
    for side in ("top", "right"): ax.spines[side].set_visible(False)
fig.tight_layout(); fig.savefig(config.OUTPUT_DIR / "trace_dashboard_preview.png", dpi=120); plt.close(fig)
display(Image(filename=str(config.OUTPUT_DIR / "trace_dashboard_preview.png")))
""")
code("""
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                   capture_output=True, text=True)
print(r.stdout[-400:])
""")

if os.path.exists(NB) and os.environ.get("FORCE_REBUILD") != "1":
    old = nbf.read(NB, as_version=4)
    if any(c.cell_type == "code" and c.get("outputs") for c in old.cells):
        raise SystemExit(f"{NB} has executed outputs; set FORCE_REBUILD=1 to overwrite")
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
nbf.write(nb, NB)
print("wrote", NB, len(cells), "cells")
