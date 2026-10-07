"""Bonus: retrieval-augmented fallback for low-confidence extractions.

Confidence signal
  mean log-probability of the generated answer tokens *given the prompt*
  (= -log perplexity of the answer). The blueprint's version scores the answer
  text on its own, without the clause; that measures how "typical" the JSON is,
  not how sure the model was when extracting from this clause, so it is
  replaced here by the conditional quantity from generation scores.

Threshold
  set from data, not guessed: the RAG_CONFIDENCE_PERCENTILE-th percentile of
  confidence on the validation split. Answers below it (and any answer that
  fails schema validation) trigger retrieval.

Knowledge base (ChromaDB, in-memory, MiniLM embeddings)
  * every TRAINING clause with its gold extraction (worked examples) - never
    val/test, so there is no leakage;
  * one definition document per clause_type label and the risk_flag rubric.
  Retrieval returns the k nearest worked examples plus the definitions of the
  labels those examples carry; the model is re-queried with that context.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'ChromaDB RAG fallback triggered
# by conditional answer log-prob threshold calibrated on validation data',
# Date: 2026-10-07
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

import config
from src.evaluate import build_prompt_messages, generate_batch
from src.schemas import parse_prediction

LABEL_DEFINITIONS = {
    "indemnification": "one party must compensate or hold harmless another for losses or claims",
    "termination": "rights or duties to end the agreement, notice periods, termination events",
    "payment_terms": "amounts, fees, interest, payment timing, late-payment consequences",
    "confidentiality": "duties not to disclose or misuse confidential information",
    "liability_cap": "limits, caps or exclusions of liability",
    "force_majeure": "relief from performance due to events beyond a party's control",
    "compliance_covenant": "undertakings to comply with laws, licences, sanctions, capital or "
                           "financial ratios, or to maintain regulatory status",
    "data_protection": "personal-data processing, security, breach notification, transfers",
    "aml_kyc": "customer due diligence, identity verification, sanctions screening, "
               "suspicious-activity escalation",
    "regulatory_reporting": "filing reports or notifications with regulators or authorities",
}
RISK_RUBRIC = ("risk_flag rubric - high: uncapped/very large exposure, regulatory penalties or "
               "licence risk, immediate termination for cause, deadlines <= 72 hours, indefinite "
               "survival; medium: material but bounded exposure, deadlines of days to 30 days, "
               "notice-based termination, meaningful carve-outs; low: routine administrative "
               "duties, long deadlines, mutual well-bounded obligations.")


class ComplianceKB:
    def __init__(self, train_rows: list[dict]):
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.EphemeralClient()
        self.col = client.get_or_create_collection("compliance_kb",
                                                   metadata={"hnsw:space": "cosine"})
        docs, ids, metas = [], [], []
        for r in train_rows:
            clause = r["messages"][1]["content"]
            docs.append(clause)
            ids.append(r["id"])
            metas.append({"answer": r["messages"][2]["content"], "clause_type": r["clause_type"]})
        self.col.add(ids=ids, documents=docs, metadatas=metas,
                     embeddings=self.embedder.encode(docs, normalize_embeddings=True).tolist())

    def retrieve(self, clause: str, k: int = config.RAG_TOP_K) -> dict:
        q = self.embedder.encode([clause], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=q, n_results=k)
        hits = [{"id": i, "clause": d, "answer": m["answer"], "clause_type": m["clause_type"],
                 "distance": dist}
                for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0])]
        return {"hits": hits, "context": self.format_context(hits)}

    @staticmethod
    def format_context(hits: list[dict]) -> str:
        parts = []
        for n, h in enumerate(hits, 1):
            parts.append(f"Example {n}\nClause: {h['clause']}\nCorrect extraction: {h['answer']}")
        labels = sorted({h["clause_type"] for h in hits})
        defs = "\n".join(f"- {l}: {LABEL_DEFINITIONS[l]}" for l in labels)
        return "\n\n".join(parts) + f"\n\nLabel definitions:\n{defs}\n{RISK_RUBRIC}"


def calibrate_threshold(val_confidences: list[float],
                        percentile: float = config.RAG_CONFIDENCE_PERCENTILE) -> float:
    return float(np.percentile(val_confidences, percentile))


def needs_fallback(pred: dict, threshold: float) -> bool:
    _, status = parse_prediction(pred["raw"])
    return status != "valid" or pred["mean_logprob"] < threshold


def answer_with_fallback(model, tok, clause: str, kb: ComplianceKB, threshold: float,
                         first: Optional[dict] = None) -> dict:
    """First pass; if low-confidence or invalid, retrieve and re-query once."""
    first = first or generate_batch(model, tok, [build_prompt_messages(clause)])[0]
    if not needs_fallback(first, threshold):
        return {"final": first, "first": first, "used_rag": False, "retrieved": []}
    ret = kb.retrieve(clause)
    second = generate_batch(model, tok, [build_prompt_messages(clause, context=ret["context"])])[0]
    # Keep the retrieval answer only if it is valid; otherwise fall back to the first.
    _, status2 = parse_prediction(second["raw"])
    final = second if status2 == "valid" else first
    return {"final": final, "first": first, "second": second, "used_rag": True,
            "retrieved": [{k: h[k] for k in ("id", "clause_type", "distance")} for h in ret["hits"]]}


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, indent=1)
