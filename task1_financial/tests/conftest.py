"""Keep tests from writing to the committed LLM audit log (outputs/llm_calls.jsonl).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'autouse fixture redirecting the audit log in tests', Date: 2026-10-07
"""
import pytest

from src.llm import client


@pytest.fixture(autouse=True)
def _isolated_audit_log(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "AUDIT_LOG_PATH", tmp_path / "llm_calls.jsonl")
