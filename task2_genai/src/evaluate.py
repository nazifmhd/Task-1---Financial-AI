"""Inference and metrics for base-vs-fine-tuned comparison (Task 2C).

Fairness controls - the only variable between the two runs is fine-tuning:
* identical system prompt (prompts/student_system_prompt.txt), test set and
  chat template;
* both models loaded with the same 4-bit NF4 quantisation;
* greedy decoding, same max_new_tokens, same stop tokens.

Metrics
* ROUGE-L (F) between the raw generation and the canonical reference JSON -
  the rubric's headline metric.
* BERTScore F1 (roberta-large) - semantic overlap, robust to paraphrase.
* Structured metrics that ROUGE cannot see: JSON validity, schema validity,
  clause_type / risk_flag accuracy, party exact match, trigger-null agreement,
  and obligation-field ROUGE-L.
* Confidence per answer = mean log-probability of the generated tokens given
  the prompt (used by the RAG fallback).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Batched greedy inference with
# per-answer mean token log-prob, ROUGE-L/BERTScore and field-level structured
# metrics for JSON extraction', Date: 2026-10-07
"""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

import config
from src.schemas import parse_prediction

STUDENT_SYSTEM_PROMPT = (config.PROMPT_DIR / "student_system_prompt.txt").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def stop_token_ids(tok) -> list[int]:
    ids = {tok.eos_token_id}
    end_id = tok.convert_tokens_to_ids("<|end|>")       # Phi-3 end-of-turn
    if isinstance(end_id, int) and end_id != tok.unk_token_id:
        ids.add(end_id)
    return [i for i in ids if i is not None]


def build_prompt_messages(clause: str, system: str = STUDENT_SYSTEM_PROMPT,
                          context: Optional[str] = None) -> list[dict]:
    user = clause if context is None else (
        f"Reference material (examples of correct extractions and label definitions):\n"
        f"{context}\n\nNow extract from this clause:\n{clause}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_batch(model, tok, message_lists: list[list[dict]],
                   max_new_tokens: int = config.MAX_NEW_TOKENS) -> list[dict]:
    """Greedy generation for a batch; returns text + mean token log-prob."""
    import torch
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
               for m in message_lists]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=stop_token_ids(tok), pad_token_id=tok.pad_token_id,
                             return_dict_in_generate=True, output_scores=True)
    gen = out.sequences[:, enc["input_ids"].shape[1]:]
    logp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    stops = set(stop_token_ids(tok)) | {tok.pad_token_id}
    results = []
    for i in range(gen.shape[0]):
        ids = gen[i].tolist()
        n = next((k for k, t in enumerate(ids) if t in stops), len(ids))
        lp = logp[i, :max(n, 1)].float().cpu().numpy()
        results.append({"raw": tok.decode(ids[:n], skip_special_tokens=True).strip(),
                        "mean_logprob": float(np.mean(lp)),
                        "n_tokens": n, "hit_max_tokens": n >= max_new_tokens})
    return results


def predict_rows(model, tok, rows: list[dict], batch_size: int = 8,
                 contexts: Optional[list[Optional[str]]] = None) -> list[dict]:
    preds = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        ctx = contexts[start:start + batch_size] if contexts else [None] * len(chunk)
        msgs = [build_prompt_messages(r["messages"][1]["content"], context=c)
                for r, c in zip(chunk, ctx)]
        for r, g in zip(chunk, generate_batch(model, tok, msgs)):
            preds.append({"id": r["id"], **g})
    return preds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def reference_of(row: dict) -> str:
    return row["messages"][-1]["content"]


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def rouge_l(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def bertscore_f1(preds: list[str], refs: list[str]) -> list[float]:
    from bert_score import score
    _, _, f1 = score(preds, refs, lang="en", verbose=False)
    return f1.tolist()


def structured_scores(raw: str, ref_json: str, clause: str) -> dict:
    ref = json.loads(ref_json)
    ext, status = parse_prediction(raw)
    d = {"json_parse": status != "invalid_json", "schema_valid": status == "valid"}
    if ext is None:
        return {**d, "clause_type_acc": 0.0, "risk_flag_acc": 0.0, "party_match": 0.0,
                "trigger_null_agree": 0.0, "party_in_text": 0.0, "obligation_rougeL": 0.0}
    p = ext.model_dump()
    return {**d,
            "clause_type_acc": float(p["clause_type"] == ref["clause_type"]),
            "risk_flag_acc": float(p["risk_flag"] == ref["risk_flag"]),
            "party_match": float(_norm(p["party_responsible"]) == _norm(ref["party_responsible"])),
            "trigger_null_agree": float((p["trigger_condition"] is None) == (ref["trigger_condition"] is None)),
            "party_in_text": float(_norm(p["party_responsible"]) in _norm(clause)),
            "obligation_rougeL": rouge_l([p["obligation"]], [ref["obligation"]])[0]}


def score_predictions(rows: list[dict], preds: list[dict], with_bertscore: bool = True) -> dict:
    """Per-example scores and aggregates for one model."""
    by_id = {p["id"]: p for p in preds}
    raws = [by_id[r["id"]]["raw"] for r in rows]
    refs = [reference_of(r) for r in rows]
    per = [structured_scores(raw, ref, r["messages"][1]["content"])
           for raw, ref, r in zip(raws, refs, rows)]
    rl = rouge_l(raws, refs)
    bs = bertscore_f1(raws, refs) if with_bertscore else [float("nan")] * len(rows)
    for d, a, b, r in zip(per, rl, bs, rows):
        d.update({"id": r["id"], "rougeL": a, "bertscore_f1": b})
    keys = ["rougeL", "bertscore_f1", "json_parse", "schema_valid", "clause_type_acc",
            "risk_flag_acc", "party_match", "trigger_null_agree", "party_in_text",
            "obligation_rougeL"]
    agg = {k: float(np.mean([float(d[k]) for d in per])) for k in keys}
    return {"aggregate": agg, "per_example": per}


def bootstrap_diff_ci(a: list[float], b: list[float], n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(b - a) - is the improvement more than noise?"""
    rng = np.random.default_rng(seed)
    diff = np.asarray(b) - np.asarray(a)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean()), float(lo), float(hi)
