"""CPU tests for schema parsing, QC filters, splitting and metrics.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for extraction schema,
# QC rules, stratified split and structured metrics', Date: 2026-10-07
"""
import json

import pytest

import config
from src.evaluate import bootstrap_diff_ci, structured_scores
from src.format_split import stratified_split, to_chat
from src.gen import build_plan, qc_reasons
from src.schemas import parse_prediction

CLAUSE = ("The Borrower shall, within five Business Days of any change of control, notify "
          "the Agent in writing and provide such information as the Agent may reasonably request.")
OUT = {"clause_type": "regulatory_reporting", "obligation": "notify the Agent in writing",
       "party_responsible": "The Borrower", "trigger_condition": "any change of control",
       "risk_flag": "medium"}


def row(**over):
    spec = {"spec_id": 1, "clause_type": "regulatory_reporting", "has_trigger": True,
            "industry": "x", "document_type": "y", "jurisdiction": "z", "complexity": "c"}
    r = {"spec": spec, "input_text": CLAUSE, "output": dict(OUT)}
    r["output"].update(over)
    return r


def test_plan_is_stratified():
    plan = build_plan()
    assert len(plan) == len(config.CLAUSE_TYPES) * config.EXAMPLES_PER_CLAUSE_TYPE
    counts = {ct: sum(s["clause_type"] == ct for s in plan) for ct in config.CLAUSE_TYPES}
    assert set(counts.values()) == {config.EXAMPLES_PER_CLAUSE_TYPE}


def test_parse_prediction_statuses():
    assert parse_prediction(json.dumps(OUT))[1] == "valid"
    assert parse_prediction("Sure! ```json\n" + json.dumps(OUT) + "\n```")[1] == "valid"
    assert parse_prediction("The clause is about reporting.")[1] == "invalid_json"
    assert parse_prediction(json.dumps({**OUT, "clause_type": "reporting"}))[1] == "schema_error"
    assert parse_prediction(json.dumps({**OUT, "extra": 1}))[1] == "schema_error"
    ext, _ = parse_prediction(json.dumps({**OUT, "trigger_condition": "null"}))
    assert ext.trigger_condition is None


def test_qc_rules():
    assert qc_reasons(row()) == []
    assert "party_not_in_text" in qc_reasons(row(party_responsible="The Lender"))
    r = row()
    r["spec"]["has_trigger"] = False
    assert "trigger_present_but_spec_unconditional" in qc_reasons(r)


def test_stratified_split_covers_every_class():
    rows = []
    for k, ct in enumerate(config.CLAUSE_TYPES * 20):
        r = row(clause_type=ct)
        r["spec"] = {**r["spec"], "spec_id": k}
        rows.append(r)
    s = stratified_split(rows)
    assert {k: len(v) for k, v in s.items()} == {"train": 160, "val": 20, "test": 20}
    for name in ("val", "test"):
        assert {r["output"]["clause_type"] for r in s[name]} == set(config.CLAUSE_TYPES)


def test_split_exact_global_fractions_with_uneven_classes():
    rows = []
    for k, ct in enumerate(config.CLAUSE_TYPES * 24):
        r = row(clause_type=ct)
        r["spec"] = {**r["spec"], "spec_id": k}
        rows.append(r)
    s = stratified_split(rows)
    assert {k: len(v) for k, v in s.items()} == {"train": 192, "val": 24, "test": 24}
    ids = [r["spec"]["spec_id"] for v in s.values() for r in v]
    assert len(ids) == len(set(ids)) == 240          # disjoint and complete


def test_chat_format_roles_and_canonical_target():
    chat = to_chat(row())
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]
    assert list(json.loads(chat["messages"][2]["content"])) == config.OUTPUT_KEYS


def test_structured_scores():
    ref = json.dumps(OUT)
    perfect = structured_scores(ref, ref, CLAUSE)
    assert perfect["schema_valid"] and perfect["clause_type_acc"] == 1 and perfect["party_in_text"] == 1
    bad = structured_scores("not json", ref, CLAUSE)
    assert not bad["json_parse"] and bad["clause_type_acc"] == 0


def test_bootstrap_ci_detects_clear_improvement():
    mean, lo, hi = bootstrap_diff_ci([0.2] * 20, [0.6] * 20)
    assert mean == pytest.approx(0.4) and lo > 0
