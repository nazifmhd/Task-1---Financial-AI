"""Teacher-model data generation and quality control (Task 2A).

Diversity is engineered, not hoped for:
* a stratified *plan* fixes every example's (clause_type, industry, document
  type, jurisdiction, complexity, trigger presence) before any call is made -
  equal counts per clause type, other axes balanced by cycling shuffled lists;
* the teacher writes 3 examples per call, each with a different spec, at
  temperature 0.9;
* quality control then rejects unfaithful or malformed rows and removes
  near-duplicates (TF-IDF cosine), so the reported diversity is of the data
  actually used for training.

Generation is resumable: each validated batch is appended to raw_generated.jsonl
with its spec ids, and finished specs are skipped on re-run.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Stratified teacher data
# generation plan, resumable generation loop and faithfulness QC filters',
# Date: 2026-10-07
"""
from __future__ import annotations

import itertools
import json
import logging
import random
import re
import time
from typing import Iterable

from pydantic import BaseModel, Field

import config
from src.llm import GroqJSON
from src.schemas import Example

log = logging.getLogger(__name__)

TEACHER_SYSTEM_PROMPT = (config.PROMPT_DIR / "teacher_system_prompt.txt").read_text(encoding="utf-8")
# Share of specs that ask for a clause with no explicit trigger (null target).
NO_TRIGGER_SHARE = 0.2


class Batch(BaseModel):
    examples: list[Example] = Field(min_length=1)


def build_plan(seed: int = config.GEN_SEED) -> list[dict]:
    """Stratified list of example specs (one dict per example to generate)."""
    rng = random.Random(seed)

    def cycler(values):
        vals = list(values)
        while True:
            rng.shuffle(vals)
            yield from vals

    ind, doc, jur, cx = (cycler(v) for v in (config.INDUSTRIES, config.DOCUMENT_TYPES,
                                             config.JURISDICTIONS, config.COMPLEXITY))
    plan = []
    for ct in config.CLAUSE_TYPES:
        for _ in range(config.EXAMPLES_PER_CLAUSE_TYPE):
            plan.append({"clause_type": ct, "industry": next(ind),
                         "document_type": next(doc), "jurisdiction": next(jur),
                         "complexity": next(cx),
                         "has_trigger": rng.random() >= NO_TRIGGER_SHARE})
    rng.shuffle(plan)   # mix clause types within each call
    for i, spec in enumerate(plan):
        spec["spec_id"] = i
    return plan


def batch_user_prompt(specs: list[dict]) -> str:
    lines = [f"Generate {len(specs)} examples, in this order:"]
    for k, s in enumerate(specs, 1):
        trig = ("include an explicit trigger condition" if s["has_trigger"]
                else "make the obligation unconditional (trigger_condition must be null)")
        lines.append(
            f"{k}. clause_type={s['clause_type']}; a clause from a {s['document_type']} "
            f"in {s['industry']}, governed by the law of {s['jurisdiction']}; "
            f"written as {s['complexity']}; {trig}.")
    lines.append('Return {"examples": [...]} with exactly '
                 f"{len(specs)} items.")
    return "\n".join(lines)


def _done_spec_ids() -> set[int]:
    if not config.RAW_PATH.exists():
        return set()
    with open(config.RAW_PATH, encoding="utf-8") as fh:
        return {json.loads(l)["spec"]["spec_id"] for l in fh if l.strip()}


def chunks(seq: list, n: int) -> Iterable[list]:
    it = iter(seq)
    while batch := list(itertools.islice(it, n)):
        yield batch


def generate(plan: list[dict] | None = None, max_calls: int | None = None) -> dict:
    """Run (or resume) teacher generation. Returns simple run statistics."""
    plan = plan or build_plan()
    done = _done_spec_ids()
    todo = [s for s in plan if s["spec_id"] not in done]
    teacher = GroqJSON(config.TEACHER_MODEL, config.TEACHER_TEMPERATURE,
                       config.TEACHER_REASONING_EFFORT)
    config.DATA_DIR.mkdir(exist_ok=True)
    calls = failed = written = 0
    for specs in chunks(todo, config.EXAMPLES_PER_CALL):
        if max_calls is not None and calls >= max_calls:
            break
        if calls:
            time.sleep(config.GEN_CALL_INTERVAL_S)   # proactive rate limiting
        calls += 1
        batch = teacher.call(TEACHER_SYSTEM_PROMPT, batch_user_prompt(specs), Batch)
        if batch is None:
            failed += 1
            log.warning("batch for specs %s failed", [s["spec_id"] for s in specs])
            continue
        with open(config.RAW_PATH, "a", encoding="utf-8") as fh:
            for spec, ex in zip(specs, batch.examples):
                fh.write(json.dumps({"spec": spec, "input_text": ex.input_text,
                                     "output": ex.output.model_dump(),
                                     "teacher": config.TEACHER_MODEL},
                                    ensure_ascii=False) + "\n")
                written += 1
        if calls % 10 == 0:
            log.info("calls=%d written=%d failed=%d tokens=%d", calls, written,
                     failed, teacher.usage_tokens)
    return {"planned": len(plan), "already_done": len(done), "calls": calls,
            "failed_calls": failed, "written": written, "tokens": teacher.usage_tokens}


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def qc_reasons(row: dict) -> list[str]:
    """Reasons to reject a generated row (empty list = keep)."""
    text, out, spec = row["input_text"], row["output"], row["spec"]
    reasons = []
    if out["clause_type"] != spec["clause_type"]:
        reasons.append("label_mismatch_with_spec")
    if out["party_responsible"].lower() not in text.lower():
        reasons.append("party_not_in_text")
    n = _words(text)
    if not config.MIN_CLAUSE_WORDS <= n <= config.MAX_CLAUSE_WORDS:
        reasons.append(f"clause_length_{n}")
    if _words(out["obligation"]) > config.MAX_OBLIGATION_WORDS:
        reasons.append("obligation_too_long")
    if not spec["has_trigger"] and out["trigger_condition"] is not None:
        reasons.append("trigger_present_but_spec_unconditional")
    # Annotation leakage = a multi-word snake_case label (e.g. "force_majeure")
    # or the word "clause_type" in the text. Single-word labels are ordinary
    # English ("termination"). Plain-English words ("Termination", "Force Majeure Event") are
    # authentic legal drafting and must NOT be rejected.
    low = text.lower()
    ct = out["clause_type"]
    if ("_" in ct and ct in low) or "clause_type" in low or "clause type" in low:
        reasons.append("label_leaked_into_text")
    return reasons


def near_duplicate_pairs(texts: list[str], threshold: float = config.NEAR_DUPLICATE_COSINE):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))
    np.fill_diagonal(sim, 0)
    i, j = np.where(np.triu(sim) > threshold)
    return list(zip(i.tolist(), j.tolist(), sim[i, j].tolist()))


def run_qc() -> dict:
    """raw_generated.jsonl -> clean.jsonl (+ rejected.jsonl with reasons)."""
    with open(config.RAW_PATH, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    keep, rejected = [], []
    for r in rows:
        why = qc_reasons(r)
        (rejected if why else keep).append({**r, "qc_reasons": why} if why else r)
    # Drop the second member of every near-duplicate pair.
    dup_idx = {j for _, j, _ in near_duplicate_pairs([r["input_text"] for r in keep])}
    for j in sorted(dup_idx):
        rejected.append({**keep[j], "qc_reasons": ["near_duplicate"]})
    keep = [r for k, r in enumerate(keep) if k not in dup_idx]
    for path, data in ((config.CLEAN_PATH, keep), (config.REJECTED_PATH, rejected)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    reasons = {}
    for r in rejected:
        for why in r["qc_reasons"]:
            reasons[why] = reasons.get(why, 0) + 1
    return {"raw": len(rows), "kept": len(keep), "rejected": len(rejected),
            "reject_reasons": reasons}
