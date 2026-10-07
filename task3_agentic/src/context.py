"""Context management for ReAct agents: bounded prompts without losing memory.

``make_pre_model_hook`` returns a LangGraph pre-model hook that builds the
messages the LLM sees on each step, separately from the stored state:
* tool observations older than the newest N are shortened (the full results
  stay in graph state / checkpointer memory, so nothing is forgotten - only
  the per-call prompt is compacted);
* once the agent has used its tool budget, a system note tells it to stop
  calling tools and answer from the evidence it already has.
This keeps every request under the free-tier input-tokens-per-minute limit
however long the agent explores, and it bounds cost per run.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph pre_model_hook that
# compacts old tool observations and enforces a tool-call budget', Date: 2026-10-07
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

import config


def make_pre_model_hook(tool_budget: int = config.AGENT_TOOL_BUDGET,
                        keep_recent: int = config.KEEP_FULL_RECENT_OBSERVATIONS,
                        old_chars: int = config.OLD_OBSERVATION_CHARS):
    def hook(state: dict) -> dict:
        msgs = state["messages"]
        tool_idx = [i for i, m in enumerate(msgs) if isinstance(m, ToolMessage)]
        recent = set(tool_idx[-keep_recent:]) if keep_recent else set()
        view = []
        for i, m in enumerate(msgs):
            if isinstance(m, ToolMessage) and i not in recent and len(str(m.content)) > old_chars:
                m = m.model_copy(update={"content": str(m.content)[:old_chars]
                                         + " ...[older observation shortened in prompt; full result kept in memory]"})
            view.append(m)
        n_calls = sum(len(m.tool_calls or []) for m in msgs if isinstance(m, AIMessage))
        if n_calls >= tool_budget:
            view.append(SystemMessage(
                f"Tool budget reached ({n_calls} calls). Do NOT call more tools; write your "
                "answer now from the evidence above, and state any data you could not obtain."))
        return {"llm_input_messages": view}
    return hook
