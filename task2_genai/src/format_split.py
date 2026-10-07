"""Chat-format conversion and stratified 80/10/10 split (Task 2A).

Each example becomes {"messages": [system, user, assistant]} - the OpenAI-style
message list that ``tokenizer.apply_chat_template`` turns into Phi-3's native
format (<|system|> ... <|end|><|user|> ... <|end|><|assistant|> ... <|end|>).
Storing messages rather than pre-rendered strings keeps the data model-agnostic;
the template is applied at tokenisation time with the student's own tokenizer.

The split is stratified by clause_type so that every class appears in val and
test - with ~20 examples per class a plain random split can leave a class out
of the 10% test set entirely.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Convert extraction rows to
# chat-format JSONL and make a stratified 80/10/10 split', Date: 2026-10-07
"""
from __future__ import annotations

import collections
import json
import random

import config
from src.schemas import Extraction

STUDENT_SYSTEM_PROMPT = (config.PROMPT_DIR / "student_system_prompt.txt").read_text(encoding="utf-8").strip()


def to_chat(row: dict) -> dict:
    target = Extraction.model_validate(row["output"]).to_json()   # canonical key order
    return {
        "id": f"ex{row['spec']['spec_id']:03d}",
        "clause_type": row["output"]["clause_type"],
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": row["input_text"]},
            {"role": "assistant", "content": target},
        ],
    }


def stratified_split(rows: list[dict], fractions=config.SPLIT_FRACTIONS,
                     seed: int = config.SPLIT_SEED) -> dict[str, list[dict]]:
    """Exact global 80/10/10 sizes, stratified by clause_type.

    Global val/test sizes are round(N * fraction). They are spread over classes
    with the largest-remainder method (ties broken randomly) so each class gets
    floor or ceil of its proportional share and every class has >= 1 item in
    val and test.
    """
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for r in rows:
        by_class[r["output"]["clause_type"]].append(r)
    classes = sorted(by_class)
    n_total = len(rows)

    def quotas(fraction: float) -> dict[str, int]:
        target = round(n_total * fraction)
        exact = {c: len(by_class[c]) * fraction for c in classes}
        q = {c: max(1, int(exact[c])) for c in classes}
        order = sorted(classes, key=lambda c: (exact[c] - int(exact[c]), rng.random()), reverse=True)
        for c in order[:max(0, target - sum(q.values()))]:
            q[c] += 1
        return q

    q_val, q_test = quotas(fractions[1]), quotas(fractions[2])
    splits = {"train": [], "val": [], "test": []}
    for c in classes:
        items = by_class[c][:]
        rng.shuffle(items)
        splits["test"] += items[:q_test[c]]
        splits["val"] += items[q_test[c]:q_test[c] + q_val[c]]
        splits["train"] += items[q_test[c] + q_val[c]:]
    for v in splits.values():
        rng.shuffle(v)
    return splits


def build(rows: list[dict] | None = None) -> dict[str, int]:
    if rows is None:
        with open(config.CLEAN_PATH, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    splits = stratified_split(rows)
    for name, data in splits.items():
        with open(config.SPLIT_PATHS[name], "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
    return {k: len(v) for k, v in splits.items()}


def load_split(name: str) -> list[dict]:
    with open(config.SPLIT_PATHS[name], encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
