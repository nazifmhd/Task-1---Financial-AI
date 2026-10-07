"""Generates 04_judge_and_review.ipynb (Task 2C, CPU).

Kept separate from build_notebooks.py so regenerating it can never overwrite
the executed Colab notebooks 02/03.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Notebook for LLM-as-judge,
# manual hallucination review and qualitative analysis', Date: 2026-10-07
"""
import nbformat as nbf

# Not imported from build_notebooks.py: that module writes notebooks 01-03 on
# import and would overwrite the executed copies.
COLAB_BASE = "https://colab.research.google.com/github/nazifmhd/Task-1---Financial-AI/blob/main/task2_genai"


def badge(nb_name):
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BASE}/{nb_name})"


SETUP_LOCAL = """
# Windows: torch must be imported before scikit-learn, otherwise their OpenMP
# runtimes clash (WinError 1114 loading c10.dll). Harmless elsewhere.
try:
    import torch
except ImportError:
    pass
import os, sys, json, warnings, logging
if os.path.basename(os.getcwd()) != "task2_genai" and os.path.isdir("task2_genai"):
    os.chdir("task2_genai")
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
for n in ("httpx", "openai", "urllib3", "sentence_transformers", "chromadb", "huggingface_hub"):
    logging.getLogger(n).setLevel(logging.WARNING)
import pandas as pd
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
from IPython.display import display, Markdown, Image
import config
print("cwd:", os.getcwd())
"""

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# Task 2C (continued) — LLM-as-judge, hallucination review, qualitative analysis
{badge('04_judge_and_review.ipynb')}

This notebook consumes the artifacts produced on the GPU by notebooks 02 and 03, which are published
next to the model on the Hugging Face Hub (`training/`, `eval/`), and runs the CPU-only parts of the
evaluation: the LLM judge (Groq), the manual review of the fine-tuned outputs, and the analysis.
""")
code(SETUP_LOCAL)
code("""
# Fetch the GPU-run artifacts from the public model repo if they are not present locally.
from huggingface_hub import hf_hub_download
import shutil
config.OUTPUT_DIR.mkdir(exist_ok=True)
for f in ["eval/metrics.json", "eval/metrics_table.csv", "eval/predictions_base.jsonl",
          "eval/predictions_tuned.jsonl", "eval/rag_results.jsonl", "training/loss_per_epoch.csv",
          "training/loss_curve.png", "training/trainer_log_history.json"]:
    dst = config.OUTPUT_DIR / f.split("/")[-1]
    if not dst.exists():
        shutil.copy(hf_hub_download(config.HF_REPO_ID, f), dst)
print(sorted(p.name for p in config.OUTPUT_DIR.iterdir()))

from src.format_split import load_split
test = load_split("test")
P = lambda name: [json.loads(l) for l in open(config.OUTPUT_DIR / name, encoding="utf-8")]
preds_base, preds_tuned, rag = P("predictions_base.jsonl"), P("predictions_tuned.jsonl"), P("rag_results.jsonl")
metrics = json.load(open(config.OUTPUT_DIR / "metrics.json"))
print("test examples:", len(test), "| model:", f"https://huggingface.co/{config.HF_REPO_ID}")
""")

md("## 1. Recap of the results from notebooks 02 and 03")
code("""
loss = pd.read_csv(config.OUTPUT_DIR / "loss_per_epoch.csv")
display(loss.round(4))
display(Image(filename=str(config.OUTPUT_DIR / "loss_curve.png")))
""")
md("""
**Reading the loss curve.** Validation loss falls from 0.410 to 0.184 to 0.168 over epochs 0–2 (−59%),
then is flat at epoch 3 (0.1682, +0.0004, which is within noise for 24 validation examples) while
training loss keeps falling (0.179 → 0.146). That is the onset of over-fitting. Because training used
`load_best_model_at_end` on `eval_loss`, the **epoch-2 checkpoint** is the one that was merged and
published, so the deployed model sits at the minimum of the validation curve. With more time I would
set 2 epochs, or add early stopping with patience 1, rather than rely on checkpoint selection.
""")
code("""
table = pd.read_csv(config.OUTPUT_DIR / "metrics_table.csv")
display(Markdown(f"**Base vs fine-tuned on the held-out test set (n = {metrics['n_test']})**"))
display(table.round(3))
""")

md("""
## 2. LLM-as-judge (structured, rubric-based)
The judge (`openai/gpt-oss-120b`, temperature 0) receives the clause, the reference and the candidate.
It returns Pydantic-validated JSON with four 1–5 scores (format, label accuracy, faithfulness,
completeness), a boolean `hallucination` flag and a rationale. Base and fine-tuned outputs are graded
with the identical prompt. **Caveat:** the judge is the same model as the teacher, so it may prefer
teacher-like wording. The rubric therefore treats the *clause*, not the reference, as the source of
truth, and the judge is reported alongside the reference-free structured metrics above.
""")
code("""
from src.judge import JUDGE_SYSTEM_PROMPT, judge_all
print(JUDGE_SYSTEM_PROMPT)
""")
code("""
verdicts = judge_all(test, {"base": preds_base, "fine-tuned": preds_tuned})   # cached in outputs/judge_results.jsonl
jv = pd.DataFrame(verdicts)
print("judge errors:", int(jv.get("judge_error", pd.Series(dtype=bool)).fillna(False).sum()))
print("Example structured verdict:"); print(json.dumps(verdicts[-1], indent=1, ensure_ascii=False))
crit = ["format", "label_accuracy", "faithfulness", "completeness"]
summary = jv.groupby("model")[crit].mean()
summary["overall (mean of 4)"] = summary[crit].mean(axis=1)
summary["judge hallucination rate"] = jv.groupby("model")["hallucination"].mean()
display(summary.loc[["base", "fine-tuned"]].round(3))
""")
code("""
from src.evaluate import bootstrap_diff_ci
piv = jv.assign(overall=jv[crit].mean(axis=1)).pivot(index="id", columns="model", values="overall")
m, lo, hi = bootstrap_diff_ci(piv["base"].tolist(), piv["fine-tuned"].tolist())
print(f"Judge overall score: fine-tuned − base = {m:+.2f} points (95% paired-bootstrap CI [{lo:+.2f}, {hi:+.2f}])")
display(jv[jv.model == "fine-tuned"][["id", *crit, "hallucination", "rationale"]].head(8))
""")

md("""
## 3. Manual review of the fine-tuned outputs (hallucination rate)
**All 24** test outputs were reviewed (the brief requires ≥ 10) against the clause text, using the
definitions from notebook 01:
* **correct** — valid JSON, the right `clause_type`, and an obligation, party and trigger that are all supported by the clause;
* **partially correct** — nothing invented, but a wrong `clause_type`/`risk_flag`, an incomplete or secondary obligation, or a missed or confused trigger;
* **hallucinated** — states an obligation, party, amount, deadline or condition that is **not present in the clause**.

Process: a first pass labelled every output. The two outputs that the LLM judge flagged as
hallucinated (`ex008`, `ex098`) were then re-reviewed: `ex008` was relabelled **hallucinated** (a filing
deadline attached to a different duty), and `ex098` stayed *partially correct* (a mis-paraphrase of
"shall not be liable", but no invented fact). The reason for each label is in the table.
""")
code("""
review = json.load(open(config.OUTPUT_DIR / "manual_review.json", encoding="utf-8"))
lab = pd.DataFrame(review["labels"])
by_id = {r["id"]: r for r in test}; pt = {p["id"]: p for p in preds_tuned}
lab["clause"] = lab.id.map(lambda i: by_id[i]["messages"][1]["content"])
lab["reference"] = lab.id.map(lambda i: by_id[i]["messages"][2]["content"])
lab["fine-tuned output"] = lab.id.map(lambda i: pt[i]["raw"])
with pd.option_context("display.max_colwidth", None):
    display(lab[["id", "label", "reason", "clause", "fine-tuned output", "reference"]].style
            .set_properties(**{"text-align": "left", "white-space": "pre-wrap", "font-size": "11px"}))
""")
code("""
counts = lab.label.value_counts().reindex(["correct", "partially_correct", "hallucinated"], fill_value=0)
display(counts.to_frame("n").assign(share=lambda d: (100 * d.n / d.n.sum()).round(1).astype(str) + "%"))
rate = 100 * counts["hallucinated"] / len(lab)
print(f"Hallucination rate (manual review): {rate:.1f}% ({counts['hallucinated']}/{len(lab)} reviewed)")
partial_risk = lab[(lab.label == "partially_correct") & lab.reason.str.contains("risk_flag")]
print(f"Partially-correct outputs whose issues include risk_flag: {len(partial_risk)}/{counts['partially_correct']}")

j = jv[jv.model == "fine-tuned"].set_index("id")["hallucination"]
agree = (lab.set_index("id").label.eq("hallucinated") == j.reindex(lab.id).values).mean()
print(f"Agreement between manual hallucination labels and the LLM judge's flag: {agree:.0%}")
print("Judge hallucination rate — base:", f"{jv[jv.model=='base'].hallucination.mean():.1%}",
      "| fine-tuned:", f"{jv[jv.model=='fine-tuned'].hallucination.mean():.1%}")
""")

md("## 4. Before / after examples")
code("""
from src.schemas import parse_prediction
pb = {p["id"]: p for p in preds_base}
for i in ["ex024", "ex052", "ex217"]:
    _, st_b = parse_prediction(pb[i]["raw"]); _, st_t = parse_prediction(pt[i]["raw"])
    print(f"=== {i}  ({by_id[i]['clause_type']})\\nCLAUSE: {by_id[i]['messages'][1]['content']}")
    print(f"\\nBASE [{st_b}]:\\n{pb[i]['raw']}\\n\\nFINE-TUNED [{st_t}]:\\n{pt[i]['raw']}\\n\\nREFERENCE:\\n{by_id[i]['messages'][2]['content']}\\n")
""")
code("""
# Bonus recap: RAG fallback before/after (computed on GPU in notebook 03)
fb = [r for r in rag if r["used_rag"]]
print(f"RAG fallback triggered on {len(fb)}/{len(rag)} test clauses (threshold = val p20 of answer log-prob = {metrics['rag_threshold']:.4f})")
agg_t, agg_r = metrics["tuned"]["aggregate"], metrics["tuned_rag"]["aggregate"]
display(pd.DataFrame({"fine-tuned": agg_t, "fine-tuned + RAG": agg_r}).drop(index="bertscore_f1").round(3))
r = next(x for x in rag if x["id"] == "ex098")
print(f"\\n[ex098] confidence {r['first']['mean_logprob']:.3f} < threshold -> retrieved {[h['id'] + ':' + h['clause_type'] for h in r['retrieved']]}")
print("CLAUSE:", by_id["ex098"]["messages"][1]["content"])
print("\\nBEFORE:", r["first"]["raw"]); print("AFTER: ", r["final"]["raw"]); print("REF:   ", by_id["ex098"]["messages"][2]["content"])
""")

md("## 5. Qualitative analysis")
md("""**Where fine-tuning helped.**
* **Format.** The largest gain is reliable output. The base model wrapped every answer in a markdown fence and produced unusable JSON on 6/24 clauses. Four of those contain a stray token (`Ζ`) inside the object (`ex024`, `ex098`, `ex220`, `ex034`), and the others invent keys or labels (`"house_flag"` in `ex152`, `"AML_kyc"` in `ex217`). The fine-tuned model returned schema-valid JSON on 24/24.
* **Party extraction.** It learned to copy the party exactly as the clause names it: exact match rose from 12.5% to 95.8%. On `ex052` the base model wrote `"Insured"` where the clause says `"the Insured"`. (Part of that 83-point jump is a convention the model learned, keeping the article. The grounded-in-clause rate rose 75% → 100%, which is the faithfulness part of the gain.)
* **Obligation.** It writes a short, faithful paraphrase instead of a full sentence: obligation-field ROUGE-L 0.36 → 0.72. On `ex164` the base model wrote *"Respondent Bank must remit received funds…"*, while the fine-tuned model returned the reference wording almost verbatim.
* **Labels.** clause_type accuracy rose 62.5% → 91.7%, and the LLM judge's overall score rose 3.60 → 4.17 (format 4.00 → 4.92). Every structural improvement has a 95% paired-bootstrap interval above zero, so the gain is not test-set noise.
* **Robust to teacher errors.** On `ex152` the reference label is a teacher error (`liability_cap` for a duty to give information to a regulator), and the fine-tuned model's `regulatory_reporting` is actually correct.

**Remaining failure modes and next steps.**
* **risk_flag (62.5%) is the weak field,** and its CI includes zero. 9 of the 11 partially-correct outputs differ only or mainly in risk, mostly by one level (`ex077` medium vs low, `ex088` high vs medium). The rubric's thresholds ("material but bounded", "≤ 72 hours") need more contrastive examples than 192 synthetic clauses can supply. The fix is 20–30 human-labelled near-boundary pairs per level and, possibly, a separate classification head or a rule-based check of deadlines and caps.
* **Multi-duty clauses.** The model sometimes merges a condition from one duty into another. `ex008` attaches the filing timeframe to the duty to furnish copies; this is the single hallucination (4.2%). `ex114` puts a deadline in `trigger_condition`. The remedies are (a) training examples with two or three duties whose targets separate them, (b) constrained JSON decoding plus a verification pass that checks each extracted deadline or amount against the clause span it came from, and (c) an explicit `deadline` field so deadlines stop leaking into triggers.
* **Teacher label errors.** QC checks consistency with the generation plan, not semantic correctness, so a mislabelled reference (`ex152`) passed. A small human-validated seed set and agreement filtering between two teacher models would catch these.
* **RAG fallback.** Retrieval triggered on 7/24 clauses and left aggregate quality unchanged (ROUGE-L 0.778 → 0.775). It improved triggers (null/non-null agreement 0.71 → 0.79) and fixed `ex098`'s risk flag, but it also changed correct risk flags (`ex111`). Answer log-probability is a weak correctness signal at this scale. A calibrated confidence model trained on validation errors would route better than a percentile threshold.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
nbf.write(nb, "04_judge_and_review.ipynb")
print("wrote 04_judge_and_review.ipynb", len(cells), "cells")
