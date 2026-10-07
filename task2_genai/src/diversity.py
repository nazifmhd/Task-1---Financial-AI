"""Dataset diversity evidence (Task 2A, 10 marks).

Reports, on the post-QC dataset:
1. length distribution of the inputs (words) - histogram + summary stats;
2. label distributions: clause_type, risk_flag, trigger null-share, and the
   coverage of every generation axis (industry, document type, jurisdiction,
   complexity);
3. keyword / topic frequency: top TF-IDF terms per clause type and overall
   bigram frequencies (stop-words removed);
4. redundancy: pairwise TF-IDF cosine (lexical) and MiniLM embedding cosine
   (semantic) - mean, 95th percentile, max; plus distinct-1/distinct-2.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Diversity report for a
# generated clause dataset: length histogram, label/axis coverage, TF-IDF
# keywords, lexical + semantic similarity, distinct-n', Date: 2026-10-07
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load_rows(path: Path = config.CLEAN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams / total n-grams across the corpus (higher = more varied)."""
    grams, total = set(), 0
    for t in texts:
        toks = re.findall(r"[a-z0-9]+", t.lower())
        ng = list(zip(*[toks[i:] for i in range(n)]))
        grams.update(ng)
        total += len(ng)
    return len(grams) / total if total else 0.0


def similarity_stats(sim: np.ndarray) -> dict:
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    return {"mean": float(vals.mean()), "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max())}


def lexical_similarity(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))


def semantic_similarity(texts: list[str]) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    emb = SentenceTransformer(config.EMBED_MODEL).encode(texts, normalize_embeddings=True)
    return emb @ emb.T


def top_terms_per_class(rows: list[dict], k: int = 6) -> pd.DataFrame:
    """Highest mean-TF-IDF terms for each clause type (topic sanity check)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [r["input_text"] for r in rows]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    labels = np.array([r["output"]["clause_type"] for r in rows])
    out = {}
    for ct in config.CLAUSE_TYPES:
        mask = labels == ct
        if mask.any():
            scores = np.asarray(X[mask].mean(axis=0)).ravel()
            out[ct] = ", ".join(vocab[scores.argsort()[::-1][:k]])
    return pd.DataFrame.from_dict(out, orient="index", columns=["top TF-IDF terms"])


def bigram_frequency(texts: list[str], k: int = 20) -> pd.DataFrame:
    """Most frequent bigrams with total count and document frequency."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words="english", ngram_range=(2, 2))
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    idx = counts.argsort()[::-1][:k]
    return pd.DataFrame({"bigram": vocab[idx], "count": counts[idx],
                         "share_of_examples_%": np.round(100 * doc_freq[idx] / len(texts), 1)})


def pairs_above(sim: np.ndarray, thresholds=(0.5, 0.7, 0.8, 0.9)) -> dict:
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    return {f">{t}": int((vals > t).sum()) for t in thresholds} | {"total_pairs": int(len(vals))}


def report(rows: list[dict] | None = None, plot_path: Path | None = None) -> dict:
    """Compute all diversity metrics; optionally save the figure."""
    import matplotlib.pyplot as plt
    rows = rows or load_rows()
    texts = [r["input_text"] for r in rows]
    lengths = np.array([len(t.split()) for t in texts])

    lex = lexical_similarity(texts)
    sem = semantic_similarity(texts)
    res = {
        "n_examples": len(rows),
        "length_words": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                         "mean": float(lengths.mean()), "max": int(lengths.max()),
                         "std": float(lengths.std())},
        "clause_type": dict(collections.Counter(r["output"]["clause_type"] for r in rows)),
        "risk_flag": dict(collections.Counter(r["output"]["risk_flag"] for r in rows)),
        "trigger_null_share": float(np.mean([r["output"]["trigger_condition"] is None for r in rows])),
        "axes_coverage": {ax: dict(collections.Counter(r["spec"][ax] for r in rows))
                          for ax in ("industry", "document_type", "jurisdiction", "complexity")},
        "unique_parties": len({r["output"]["party_responsible"].lower() for r in rows}),
        "distinct_1": distinct_n(texts, 1), "distinct_2": distinct_n(texts, 2),
        "lexical_tfidf_cosine": similarity_stats(lex),
        "semantic_minilm_cosine": similarity_stats(sem) if sem is not None else None,
        "lexical_pairs_above": pairs_above(lex),
        "semantic_pairs_above": pairs_above(sem) if sem is not None else None,
    }

    if plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        axes[0].hist(lengths, bins=20, color="#2a78d6", edgecolor="white")
        axes[0].set(title="Input length distribution", xlabel="words", ylabel="examples")
        ct = pd.Series(res["clause_type"]).reindex(config.CLAUSE_TYPES).fillna(0)
        axes[1].barh(ct.index[::-1], ct.values[::-1], color="#2a78d6")
        axes[1].set(title="clause_type balance", xlabel="examples")
        iu = np.triu_indices_from(lex, k=1)
        axes[2].hist(lex[iu], bins=40, alpha=.75, color="#2a78d6", label="TF-IDF (lexical)")
        if sem is not None:
            axes[2].hist(sem[iu], bins=40, alpha=.6, color="#eb6834", label="MiniLM (semantic)")
        axes[2].axvline(config.NEAR_DUPLICATE_COSINE, color="#52514e", ls="--", lw=1,
                        label=f"near-dup threshold {config.NEAR_DUPLICATE_COSINE}")
        axes[2].set(title="Pairwise cosine similarity", xlabel="cosine", ylabel="pairs")
        axes[2].legend(fontsize=8, frameon=False)
        for ax in axes:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    return res
