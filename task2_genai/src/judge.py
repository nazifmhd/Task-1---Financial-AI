"""LLM-as-judge with a defined rubric and structured (Pydantic) verdicts.

The judge sees CLAUSE + REFERENCE + CANDIDATE and returns JudgeVerdict JSON:
format / label_accuracy / faithfulness / completeness (1-5) + hallucination
flag + rationale. Base and fine-tuned candidates are judged with the identical
prompt; candidate order is irrelevant because each call grades one candidate.

Caveat (stated in the notebook): the judge (gpt-oss-120b) is also the teacher
that wrote the references, so it may favour teacher-like phrasing. The
rubric therefore grades against the CLAUSE as source of truth, and the judge
is reported alongside reference-free structured metrics.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LLM-as-judge over test
# predictions with a 4-criterion rubric and Pydantic-validated verdicts',
# Date: 2026-10-07
"""
from __future__ import annotations

import time

import config
from src.llm import GroqJSON
from src.schemas import JudgeVerdict

JUDGE_SYSTEM_PROMPT = (config.PROMPT_DIR / "judge_system_prompt.txt").read_text(encoding="utf-8")
JUDGE_USER_TMPL = "CLAUSE:\n{clause}\n\nREFERENCE:\n{reference}\n\nCANDIDATE:\n{candidate}"
JUDGE_CALL_INTERVAL_S = 4.0     # stay inside the free-tier tokens/minute budget


def judge_all(rows: list[dict], preds_by_model: dict[str, list[dict]]) -> list[dict]:
    judge = GroqJSON(config.JUDGE_MODEL, config.JUDGE_TEMPERATURE,
                     config.JUDGE_REASONING_EFFORT, max_tokens=1500)
    out = []
    for model_name, preds in preds_by_model.items():
        by_id = {p["id"]: p for p in preds}
        for r in rows:
            cand = by_id[r["id"]]["raw"] or "(empty output)"
            v = judge.call(JUDGE_SYSTEM_PROMPT, JUDGE_USER_TMPL.format(
                clause=r["messages"][1]["content"], reference=r["messages"][2]["content"],
                candidate=cand), JudgeVerdict)
            out.append({"model": model_name, "id": r["id"],
                        **(v.model_dump() if v else {"judge_error": True})})
            time.sleep(JUDGE_CALL_INTERVAL_S)
    return out
