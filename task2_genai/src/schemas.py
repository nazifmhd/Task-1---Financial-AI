"""Extraction schema and tolerant parsing of model output.

The same ``Extraction`` model validates teacher data, student predictions and
the judge's verdicts, so every stage agrees on what a correct output is.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schema for clause
# extraction plus tolerant JSON parsing of LLM output', Date: 2026-10-07
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import config

ClauseType = Literal[
    "indemnification", "termination", "payment_terms", "confidentiality",
    "liability_cap", "force_majeure", "compliance_covenant", "data_protection",
    "aml_kyc", "regulatory_reporting"]
assert set(ClauseType.__args__) == set(config.CLAUSE_TYPES)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    clause_type: ClauseType
    obligation: str = Field(min_length=5)
    party_responsible: str = Field(min_length=2)
    trigger_condition: Optional[str] = None
    risk_flag: Literal["low", "medium", "high"]

    @field_validator("trigger_condition", mode="before")
    @classmethod
    def _null_like(cls, v):
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a"}:
            return None
        return v

    def to_json(self) -> str:
        """Canonical serialisation (fixed key order) used as the training target."""
        return json.dumps(self.model_dump(), ensure_ascii=False)


class Example(BaseModel):
    input_text: str = Field(min_length=20)
    output: Extraction


class JudgeVerdict(BaseModel):
    format: int = Field(ge=1, le=5)
    label_accuracy: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    hallucination: bool
    rationale: str


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_text(raw: str) -> Optional[str]:
    """Return the outermost {...} block in ``raw`` (handles fences/prose), or None."""
    if not raw:
        return None
    m = _FENCE.search(raw)
    text = m.group(1) if m else raw
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None


def parse_prediction(raw: str) -> tuple[Optional[Extraction], str]:
    """Parse a model's raw generation.

    Returns (extraction | None, status) where status is one of
    'valid', 'invalid_json', 'schema_error'.
    """
    block = extract_json_text(raw)
    if block is None:
        return None, "invalid_json"
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None, "invalid_json"
    try:
        return Extraction.model_validate(data), "valid"
    except ValidationError:
        return None, "schema_error"
