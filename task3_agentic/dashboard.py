"""Bonus: Streamlit dashboard over logs/agent_trace.jsonl.

Run:  streamlit run dashboard.py

Shows run-level KPIs, a per-run timeline of tool calls (which agent called
which tool, how long it took, whether it failed), tool latency/error stats,
an agent x tool matrix (evidence of tool restriction), and the raw events.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Streamlit dashboard for an
# agent tool-call trace JSONL', Date: 2026-10-07
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE = Path(__file__).resolve().parent / "logs" / "agent_trace.jsonl"

st.set_page_config(page_title="Agent Trace Dashboard", layout="wide")
st.title("Agent Trace Dashboard")
st.caption(f"Source: {TRACE.name}. Every tool call and agent handoff, as logged by @traced.")

if not TRACE.exists():
    st.warning("No trace file yet. Run the notebook first.")
    st.stop()

rows = [json.loads(l) for l in TRACE.read_text(encoding="utf-8").splitlines() if l.strip()]
df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"])
calls = df[df["event"] == "tool_call"].copy()

runs = ["(all)"] + sorted(df["run_id"].dropna().unique().tolist())
run = st.sidebar.selectbox("Run", runs)
agents = sorted(calls["agent"].dropna().unique().tolist())
picked = st.sidebar.multiselect("Agents", agents, default=agents)
if run != "(all)":
    df, calls = df[df["run_id"] == run], calls[calls["run_id"] == run]
calls = calls[calls["agent"].isin(picked)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tool calls", len(calls))
c2.metric("Total tool time (s)", round(calls["duration_ms"].sum() / 1000, 2))
c3.metric("Errors", int((calls["status"] != "ok").sum()))
c4.metric("Runs", df["run_id"].nunique())

st.subheader("Timeline")
if len(calls):
    tl = calls.assign(start=calls["ts"] - pd.to_timedelta(calls["duration_ms"], unit="ms"))
    st.dataframe(tl[["ts", "run_id", "agent", "tool", "status", "duration_ms"]]
                 .sort_values("ts"), width="stretch", hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Mean latency by tool (ms)")
    st.bar_chart(calls.groupby("tool")["duration_ms"].mean())
with right:
    st.subheader("Agent x tool (call counts)")
    st.dataframe(pd.crosstab(calls["agent"], calls["tool"]), width="stretch")

st.subheader("Errors")
st.dataframe(calls[calls["status"] != "ok"][["ts", "agent", "tool", "inputs", "output"]],
             width="stretch", hide_index=True)

st.subheader("Handoffs and other events")
st.dataframe(df[df["event"] != "tool_call"][["ts", "run_id", "event", "agent", "inputs", "output"]],
             width="stretch", hide_index=True)

st.subheader("Raw tool calls")
st.dataframe(calls[["ts", "agent", "tool", "inputs", "output", "duration_ms", "status"]],
             width="stretch", hide_index=True)
