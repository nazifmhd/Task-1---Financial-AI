"""Generates the four Task 2 notebooks from the cell sources below.

01_generate_dataset.ipynb   2A - run locally/CPU (Groq teacher)
02_finetune_qlora.ipynb     2B - run on Colab T4
03_evaluate.ipynb           2C - run on Colab T4 (base vs tuned, metrics, RAG bonus)
04_judge_and_review.ipynb   2C - CPU: LLM-as-judge, manual review, analysis

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate Task 2 notebooks
# with nbformat', Date: 2026-10-07
"""
import nbformat as nbf

GITHUB_USER, REPO = "nazifmhd", "Task-1---Financial-AI"
BASE = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO}/blob/main/task2_genai"


def badge(nb_name):
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({BASE}/{nb_name})"


def _has_outputs(path) -> bool:
    import os
    if not os.path.exists(path):
        return False
    old = nbf.read(path, as_version=4)
    return any(c.cell_type == "code" and c.get("outputs") for c in old.cells)


def notebook(cells, path):
    """Write a notebook - but never overwrite one that holds executed outputs
    (e.g. the Colab runs of 02/03) unless FORCE_REBUILD=1 is set."""
    import os
    if _has_outputs(path) and os.environ.get("FORCE_REBUILD") != "1":
        print("skip (has executed outputs):", path)
        return
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
    nbf.write(nb, path)
    print("wrote", path, len(cells), "cells")


def builder():
    cells = []
    return cells, (lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))), \
        (lambda s: cells.append(nbf.v4.new_code_cell(s.strip())))


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

SETUP_COLAB = f"""
# Colab: clone the repo (data + src) and install pinned dependencies.
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Absolute paths: re-running in the same runtime must not clone inside the clone.
    if not os.path.exists("/content/{REPO}"):
        subprocess.run(["git", "clone", "-q", "https://github.com/{GITHUB_USER}/{REPO}.git", "/content/{REPO}"], check=True)
    os.chdir("/content/{REPO}/task2_genai")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
elif os.path.basename(os.getcwd()) != "task2_genai" and os.path.isdir("task2_genai"):
    os.chdir("task2_genai")
sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd(), "| Colab:", IN_COLAB)
"""

COMMON_IMPORTS_GPU = """
import json, logging, warnings, time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
for n in ("httpx", "urllib3", "sentence_transformers", "chromadb", "huggingface_hub", "datasets"):
    logging.getLogger(n).setLevel(logging.WARNING)
import torch, transformers, peft, bitsandbytes, pandas as pd
from IPython.display import display, Markdown, Image
pd.set_option("display.max_colwidth", 140); pd.set_option("display.width", 220)
import config
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | {p.total_memory/2**30:.1f} GiB | compute capability {p.major}.{p.minor}")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__)
"""

HF_LOGIN = """
# HF token from Colab Secrets (key icon) - never printed, never committed.
from src.llm import read_secret
HF_TOKEN = read_secret("HF_TOKEN")
print("HF token loaded:", bool(HF_TOKEN), "| target repo:", config.HF_REPO_ID)
"""

# =========================================================================== #
# 01 - dataset
# =========================================================================== #
c, md, code = builder()
md(f"""
# Task 2A — Use case definition and dataset engineering
{badge('01_generate_dataset.ipynb')}

## Problem statement
**Use case: financial compliance clause extraction.** Compliance and legal teams at banks,
insurers and fintechs review hundreds of contract and policy clauses (loan agreements, outsourcing
contracts, KYC policies, data-processing agreements) to log *who must do what, when, and how risky
it is*. General-purpose small LLMs return prose, invent obligations, or use labels outside the
firm's taxonomy. We fine-tune a 3.8B model to produce faithful, schema-conformant extractions.

| | Definition |
|---|---|
| **Input** | one clause (15–220 words) from a financial-sector contract or policy |
| **Output** | JSON with exactly: `clause_type` (10-label taxonomy), `obligation` (≤ 40-word faithful paraphrase of the primary duty), `party_responsible` (copied verbatim from the clause), `trigger_condition` (activating event, or `null`), `risk_flag` (`low`/`medium`/`high` per a written rubric) |
| **Correct** | valid JSON with the five keys and allowed enum values; `clause_type` matches the reference; the obligation, party and trigger are all supported by the clause text |
| **Partially correct** | valid JSON, nothing invented, but a wrong `clause_type`/`risk_flag`, a secondary obligation instead of the primary one, or a missed trigger |
| **Incorrect / hallucinated** | not parseable JSON, or states an obligation, party, amount, deadline or condition that does not appear in the clause |

**Taxonomy (10):** indemnification, termination, payment_terms, confidentiality, liability_cap,
force_majeure, compliance_covenant, data_protection, aml_kyc, regulatory_reporting.

**Teacher ≠ student:** data is generated by **`openai/gpt-oss-120b`** (Groq free tier; the blueprint's
Llama-3.3-70B has been retired from Groq), and the fine-tuned student is **`microsoft/Phi-3-mini-4k-instruct`**,
a different model family. The teacher is ~30x larger than the student.
""")
code(SETUP_LOCAL)
md("""
## 1. Generation design
Diversity is designed in before any call is made. A **stratified plan** fixes each example's
clause type (24 per type) and cycles balanced, shuffled values for industry (8), document type (10),
jurisdiction (8) and complexity (5). 20% of specs ask for an unconditional clause (null trigger).
The teacher writes 3 examples per call, each from a different spec, at temperature 0.9.
""")
code("""
from src import gen
plan = gen.build_plan()
plan_df = pd.DataFrame(plan)
print(f"{len(plan)} planned examples")
display(plan_df.head(6))
print("\\nExample user prompt sent to the teacher:\\n")
print(gen.batch_user_prompt(plan[:3]))
""")
md("""
### Full teacher system prompt (`prompts/teacher_system_prompt.txt`)
Required by the brief. It is printed verbatim below and stored in the repository.
""")
code("""print(gen.TEACHER_SYSTEM_PROMPT)""")
md("""
## 2. Run (resumable) generation
`generate()` skips specs that are already in `data/raw_generated.jsonl`. The full run was done
once with this code; the log of that run is shown, and re-running the cell confirms nothing is left to generate.
""")
code("""
print(open("outputs/generation_log.txt", encoding="utf-8").read()[-1500:])
stats = gen.generate()          # resumes; 0 calls if the dataset is complete
print("\\nThis execution:", stats)
""")
md("""
## 3. Quality control
The teacher is strong but not perfect, so every row is checked before use:
* the label matches the requested `clause_type`;
* `party_responsible` appears verbatim in the clause (a faithfulness guard);
* clause length is 15–220 words and the obligation is ≤ 45 words;
* no trigger when the spec asked for an unconditional clause;
* the label name is not leaked into the clause text;
* near-duplicates (TF-IDF cosine > 0.85) are removed.
""")
code("""
qc = gen.run_qc()
print(json.dumps(qc, indent=1))
rej = pd.read_json(config.REJECTED_PATH, lines=True) if config.REJECTED_PATH.stat().st_size else pd.DataFrame()
if len(rej):
    display(rej.assign(clause=rej.input_text.str[:110])[["qc_reasons", "clause"]].head(10))
""")
md("## 4. Sample examples")
code("""
clean = pd.read_json(config.CLEAN_PATH, lines=True)
for _, r in clean.sample(3, random_state=1).iterrows():
    print("CLAUSE:", r.input_text); print("TARGET:", json.dumps(r.output, ensure_ascii=False)); print("-" * 100)
""")
md("""
## 5. Diversity validation
Evidence that the set is not "minor variations of one scenario":
length distribution, label balance, coverage of every generation axis, keyword/topic frequency,
lexical (TF-IDF) and semantic (MiniLM embedding) pairwise similarity, and distinct-n.
""")
code("""
from src import diversity
rows = diversity.load_rows()
rep = diversity.report(rows, plot_path=config.OUTPUT_DIR / "diversity.png")
display(Image(filename=str(config.OUTPUT_DIR / "diversity.png")))
print(json.dumps({k: rep[k] for k in ["n_examples", "length_words", "unique_parties", "distinct_1",
                                      "distinct_2", "lexical_tfidf_cosine", "semantic_minilm_cosine",
                                      "lexical_pairs_above", "semantic_pairs_above",
                                      "trigger_null_share", "risk_flag"]}, indent=1))
""")
code("""
cov = rep["axes_coverage"]
display(pd.DataFrame({"axis": list(cov), "distinct values": [len(v) for v in cov.values()],
                      "min count": [min(v.values()) for v in cov.values()],
                      "max count": [max(v.values()) for v in cov.values()]}))
display(pd.DataFrame(rep["clause_type"], index=["examples"]).T.sort_index())
""")
code("""
print("Topic check - top TF-IDF terms per clause type:")
display(diversity.top_terms_per_class(rows))
print("Most frequent bigrams (stop-words removed) and the share of examples containing them:")
display(diversity.bigram_frequency([r["input_text"] for r in rows], k=15))
""")
md("""
**Reading the numbers (240 examples, 28,680 pairs).**
* **Lexical redundancy is low.** Mean TF-IDF cosine is 0.067. Only 30 pairs (0.1%) exceed 0.5, one exceeds 0.7, and none reaches the 0.85 near-duplicate threshold.
* **Semantic similarity is higher, as expected inside one domain.** The mean is 0.36. 42 pairs (0.15%) exceed 0.8 and a single pair exceeds 0.9: two clauses of the same type that share a topic but have different parties, amounts and wording. No cluster of near-copies exists.
* **Every generation axis is fully covered and balanced:** 10 clause types × 24, 8 industries × 30, 10 document types × 24, 8 jurisdictions × 30, 5 complexity levels × 48. Lengths range from 17 to 202 words.
* **Top terms per class are class-specific** (e.g. *hold harmless*, *aggregate liability*, *customer due diligence*), so the labels are learnable from content. The most common bigram, *business days*, appears in about a third of the clauses. That is ordinary contract vocabulary, not a template.
""")
md("## 6. Chat-template JSONL and stratified 80/10/10 split")
code("""
from src import format_split
sizes = format_split.build()
print("Split sizes:", sizes, "| total:", sum(sizes.values()))
print({k: f"{v / sum(sizes.values()):.0%}" for k, v in sizes.items()})
for s in ("val", "test"):
    print(s, "classes covered:", len({r["clause_type"] for r in format_split.load_split(s)}), "/ 10")
ex = format_split.load_split("train")[0]
print("\\nOne train.jsonl line (messages with system / user / assistant roles):")
print(json.dumps(ex, indent=1, ensure_ascii=False)[:1500])
""")
code("""
# The same example rendered with the student's own chat template (what the model actually sees):
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
print(tok.apply_chat_template(ex["messages"], tokenize=False))
""")
code("""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                   capture_output=True, text=True)
print(r.stdout[-800:])
""")
notebook(c, "01_generate_dataset.ipynb")

# =========================================================================== #
# 02 - fine-tune
# =========================================================================== #
c, md, code = builder()
md(f"""
# Task 2B — QLoRA fine-tuning of Phi-3-mini (Colab T4)
{badge('02_finetune_qlora.ipynb')}

**Before running:** *Runtime → Change runtime type → T4 GPU*. Add the Colab Secret **`HF_TOKEN`**
(a Hugging Face *write* token) to push the merged model. W&B is optional: if `WANDB_API_KEY` is set,
the run is also logged there. Otherwise losses are logged manually in this notebook, which the brief allows.

Student: `microsoft/Phi-3-mini-4k-instruct` (3.8B). Teacher was `openai/gpt-oss-120b` (different family).
""")
code(SETUP_COLAB)
code(COMMON_IMPORTS_GPU)
code(HF_LOGIN + """
WANDB_KEY = read_secret("WANDB_API_KEY")
REPORT_TO = "none"
if WANDB_KEY:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"])
    import wandb; wandb.login(key=WANDB_KEY); REPORT_TO = "wandb"
print("report_to:", REPORT_TO)
""")
md("## 1. Data and sequence-length check")
code("""
from src.format_split import load_split
from src import train as T
tok = T.load_tokenizer()
print({s: len(load_split(s)) for s in ("train", "val", "test")})
print("Token lengths (prompt + target, after chat template):", T.token_length_stats(tok))
""")
md("""
## 2. Hyperparameters and justification
Every value is set explicitly in `config.py` / `src/train.py`. None is left at a library default without a reason.

| Parameter | Value | Justification |
|---|---|---|
| `load_in_4bit`, `bnb_4bit_quant_type` | True, **nf4** | QLoRA. NF4 is the information-theoretically optimal 4-bit code for normally distributed weights, so the 3.8B base fits in about 2.3 GB of VRAM. |
| `bnb_4bit_use_double_quant` | True | Quantises the quantisation constants too, saving about 0.37 bits per parameter (QLoRA paper). This is free headroom on a 15 GB T4. |
| `bnb_4bit_compute_dtype` / mixed precision | **float16** / `fp16=True` | The T4 (compute capability 7.5) has **no native bfloat16**. The blueprint's bf16 would error or fall back to slow emulation. fp16 with loss scaling is the correct choice on Turing GPUs. |
| `target_modules` | `qkv_proj, o_proj, gate_up_proj, down_proj` | All attention **and** MLP projections. Phi-3 **fuses** Q/K/V into `qkv_proj` and gate/up into `gate_up_proj`, so the blueprint's `q_proj/k_proj/...` names would match nothing in this model. Adapting the MLPs as well as attention clearly helps on new output formats (QLoRA paper: all-linear LoRA is needed to match full fine-tuning). |
| LoRA `r` | 16 | The task is narrow (fixed schema, 10 labels). r=8 risks under-fitting the label/risk decisions, while r=64 adds about 4x the parameters, which 192 examples cannot constrain (over-fitting risk). |
| `lora_alpha` | 32 | α/r = 2 sets the update scale. This widely used ratio keeps the effective learning rate of the adapter stable if r is changed. |
| `lora_dropout` | 0.05 | Light regularisation for a small dataset. Higher dropout slows convergence within 3 epochs. |
| `learning_rate` | 2e-4 | The standard QLoRA learning rate for 3–7B models. Adapters start at zero (B=0) and need a larger step than full fine-tuning (about 1e-5). |
| `lr_scheduler_type` | cosine | Smooth decay towards 0 over a short run gives a lower final loss than a constant rate and avoids a hard step change. |
| `warmup_ratio` | 0.1 | 4 warm-up steps out of 36 (192 examples / effective batch 16 = 12 steps per epoch × 3). Because there are so few steps, 0.03 would round to a single step. Warm-up protects the freshly initialised adapters and the fp16 loss scaler. |
| `num_train_epochs` | 3 | With 192 examples, 1 epoch is 12 optimiser steps, which is too few to learn the schema. More than 3 epochs over-fits 192 synthetic examples. Validation loss is checked every epoch and the **best epoch is kept** (`load_best_model_at_end`). |
| `per_device_train_batch_size` | 4 | Sequences are at most 600 tokens (mean about 400), so 4 fits easily on a T4 with gradient checkpointing (peak memory is printed below). |
| `gradient_accumulation_steps` | 4 | Effective batch 16 gives a smoother gradient and stable fp16 training, with 12 updates per epoch. |
| `max_seq_length` | 1024 | The longest example is 600 tokens (measured above), so **nothing is truncated**, and the limit leaves headroom for longer real clauses. Padding is dynamic, so the cap costs no memory. |
| `optim` | `paged_adamw_8bit` | The QLoRA optimiser. 8-bit Adam states plus paging to CPU absorb memory spikes during long sequences. |
| `max_grad_norm` | 0.3 | The QLoRA paper's clipping value. It stabilises fp16 updates of 4-bit LoRA. |
| `weight_decay` | 0.0 | The adapters are small and dropout already regularises them. Decaying LoRA weights towards 0 only shrinks the update. |
| `gradient_checkpointing` | True (non-reentrant) | Trades about 30% compute for a large activation-memory saving, which is the main protection against OOM on 15 GB. |
| loss masking | assistant tokens only | Loss is computed on the JSON answer plus `<|end|>`, not on the system/user prompt. The model learns to extract, not to repeat the prompt, and the loss values become interpretable. |
| `group_by_length` | True | Batches similar-length examples together to reduce padding waste. |
| eval / save strategy | per epoch, best by `eval_loss` | Produces the per-epoch validation curve the brief requires and guards against over-fitting. |
| `seed` / `data_seed` | 42 | Makes runs reproducible. |
""")
code("""
display(pd.DataFrame([{k: getattr(config, k) for k in [
    "BASE_MODEL", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LEARNING_RATE", "LR_SCHEDULER", "WARMUP_RATIO",
    "NUM_EPOCHS", "PER_DEVICE_BATCH", "GRAD_ACCUM", "MAX_SEQ_LENGTH", "OPTIMIZER", "MAX_GRAD_NORM",
    "WEIGHT_DECAY"]}]).T.rename(columns={0: "value"}))
print("LoRA targets:", config.LORA_TARGET_MODULES)
""")
md("## 3. Load the 4-bit NF4 base model and attach LoRA adapters")
code("""
model = T.load_4bit_model()
print(model.config.quantization_config)
print(f"4-bit model memory footprint: {model.get_memory_footprint()/2**30:.2f} GiB")
model = T.prepare_peft_model(model)
model.print_trainable_parameters()
""")
md("## 4. Train, logging train and validation loss per epoch")
code("""
ds = T.build_datasets(tok)
print(ds)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer, cb, epoch0, result = T.train(model, tok, ds, output_dir="checkpoints", report_to=REPORT_TO)
print(f"Training time: {(time.time()-t0)/60:.1f} min | peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print(result)
""")
code("""
table = T.loss_table(trainer, cb, epoch0)
display(table.style.format({"train_loss": "{:.4f}", "val_loss": "{:.4f}"}).hide(axis="index"))
v = table["val_loss"].tolist()
print("Validation loss by epoch:", [round(x, 4) for x in v])
print("Strictly decreasing across epochs:", all(b < a for a, b in zip(v, v[1:])))
print("Best checkpoint kept:", trainer.state.best_model_checkpoint)
config.OUTPUT_DIR.mkdir(exist_ok=True)
table.to_csv(config.OUTPUT_DIR / "loss_per_epoch.csv", index=False)
json.dump(trainer.state.log_history, open(config.OUTPUT_DIR / "trainer_log_history.json", "w"), indent=1)
fig = T.plot_losses(trainer, table, config.OUTPUT_DIR / "loss_curve.png")
""")
code("""
# Raw per-step log (evidence of manual logging)
display(pd.DataFrame(trainer.state.log_history).round(4))
""")
md("""
### Memory / OOM notes
The memory budget was planned to avoid OOM on the 15 GB T4: a 4-bit base (about 2.3 GB),
LoRA adapters plus 8-bit paged optimiser states (under 0.5 GB), and activations kept small by
gradient checkpointing and dynamic padding of ≤ 600-token sequences. Peak usage is printed above.
If an OOM does occur on a smaller GPU, apply these fixes in order:
(1) `PER_DEVICE_BATCH=2, GRAD_ACCUM=8`, which keeps the effective batch at 16;
(2) `MAX_SEQ_LENGTH=768`;
(3) keep the paged optimiser and checkpointing enabled.
The merge step loads an fp16 base (about 7.6 GB), so the training model is deleted and the CUDA cache emptied first.
""")
md("## 5. Save the adapter, merge into an fp16 base, and push to the Hugging Face Hub")
code("""
ADAPTER_DIR, MERGED_DIR = "adapter", "merged"
trainer.model.save_pretrained(ADAPTER_DIR)        # best epoch (load_best_model_at_end)
tok.save_pretrained(ADAPTER_DIR)
del trainer, model; T.free_gpu()
print(f"GPU memory after freeing: {torch.cuda.memory_allocated()/2**30:.2f} GiB")
merged = T.merge_adapter(ADAPTER_DIR, MERGED_DIR)   # PeftModel(...).merge_and_unload() on fp16 base
print(type(merged).__name__, "| merged params:", f"{sum(p.numel() for p in merged.parameters())/1e9:.2f}B")
print(sorted(os.listdir(MERGED_DIR)))
""")
code("""
# Sanity check: the merged model answers a validation clause in the target format.
from src.evaluate import generate_batch, build_prompt_messages
tok_l = T.load_tokenizer(MERGED_DIR)
row = load_split("val")[0]
out = generate_batch(merged, tok_l, [build_prompt_messages(row["messages"][1]["content"])])[0]
print("CLAUSE:", row["messages"][1]["content"]); print("MERGED MODEL:", out["raw"]); print("REFERENCE:", row["messages"][2]["content"])
""")
code("""
T.write_model_card(MERGED_DIR, table)
if HF_TOKEN:
    url = T.push(MERGED_DIR, token=HF_TOKEN)
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    for f in ("loss_per_epoch.csv", "loss_curve.png", "trainer_log_history.json"):
        api.upload_file(path_or_fileobj=str(config.OUTPUT_DIR / f), path_in_repo=f"training/{f}",
                        repo_id=config.HF_REPO_ID)
    print("Merged model:", url)
else:
    print("HF_TOKEN missing - model saved locally only in", MERGED_DIR)
""")
notebook(c, "02_finetune_qlora.ipynb")

# =========================================================================== #
# 03 - evaluate (GPU)
# =========================================================================== #
c, md, code = builder()
md(f"""
# Task 2C — Base vs fine-tuned evaluation, plus the RAG fallback bonus (Colab T4)
{badge('03_evaluate.ipynb')}

**Fairness:** both models get the **same system prompt**, the **same held-out test set** (never seen in
training), the same 4-bit NF4 loading, and greedy decoding with the same stop tokens. The only variable is fine-tuning.
The fine-tuned model is loaded **from the public Hugging Face repo**, which also proves the published link works.
""")
code(SETUP_COLAB)
code(COMMON_IMPORTS_GPU)
code(HF_LOGIN)
code("""
from transformers import AutoModelForCausalLM
from src import train as T, evaluate as E
from src.format_split import load_split
test, val, train_rows = load_split("test"), load_split("val"), load_split("train")
print({"train": len(train_rows), "val": len(val), "test": len(test)})

def load_eval_model(name):
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=T.bnb_config(), device_map={"": 0},
                                             torch_dtype=torch.float16, attn_implementation="sdpa")
    m.eval(); return m, T.load_tokenizer(name)
""")
md("## 1. Baseline: untuned Phi-3-mini with the same system prompt")
code("""
print("System prompt used for BOTH models:\\n" + E.STUDENT_SYSTEM_PROMPT + "\\n")
base, tok_b = load_eval_model(config.BASE_MODEL)
t0 = time.time(); preds_base = E.predict_rows(base, tok_b, test); print(f"base: {time.time()-t0:.0f}s")
del base; T.free_gpu()
""")
md("## 2. Fine-tuned (merged) model from the Hub")
code("""
tuned, tok_t = load_eval_model(config.HF_REPO_ID)
t0 = time.time(); preds_tuned = E.predict_rows(tuned, tok_t, test); print(f"tuned: {time.time()-t0:.0f}s")
preds_val_tuned = E.predict_rows(tuned, tok_t, val)          # used only to calibrate the RAG threshold
""")
md("## 3. ROUGE-L, BERTScore and structured metrics")
code("""
sb, st = E.score_predictions(test, preds_base), E.score_predictions(test, preds_tuned)
labels = {"rougeL": "ROUGE-L (F)", "bertscore_f1": "BERTScore F1", "json_parse": "JSON parse rate",
          "schema_valid": "Schema-valid rate", "clause_type_acc": "clause_type accuracy",
          "risk_flag_acc": "risk_flag accuracy", "party_match": "party exact match",
          "trigger_null_agree": "trigger null/non-null agreement", "party_in_text": "party grounded in clause",
          "obligation_rougeL": "obligation-field ROUGE-L"}
rows_ = []
for k, name in labels.items():
    a = [float(d[k]) for d in sb["per_example"]]; b = [float(d[k]) for d in st["per_example"]]
    mean, lo, hi = E.bootstrap_diff_ci(a, b)
    rows_.append({"metric": name, "base": sb["aggregate"][k], "fine-tuned": st["aggregate"][k],
                  "Δ": mean, "95% CI of Δ (paired bootstrap)": f"[{lo:+.3f}, {hi:+.3f}]"})
results = pd.DataFrame(rows_)
display(Markdown(f"**Held-out test set: n = {len(test)}**"))
display(results.style.format({"base": "{:.3f}", "fine-tuned": "{:.3f}", "Δ": "{:+.3f}"}).hide(axis="index"))
""")
code("""
print("| Model | ROUGE-L | BERTScore F1 |\\n|---|---|---|")
print(f"| Phi-3-mini base (system prompt, no fine-tuning) | {sb['aggregate']['rougeL']:.3f} | {sb['aggregate']['bertscore_f1']:.3f} |")
print(f"| Phi-3-mini QLoRA fine-tuned | {st['aggregate']['rougeL']:.3f} | {st['aggregate']['bertscore_f1']:.3f} |")
""")
md("## 4. Side-by-side outputs")
code("""
side = pd.DataFrame({"id": [r["id"] for r in test],
                     "clause": [r["messages"][1]["content"][:150] for r in test],
                     "base": [p["raw"][:220] for p in preds_base],
                     "fine-tuned": [p["raw"][:220] for p in preds_tuned],
                     "reference": [r["messages"][2]["content"][:220] for r in test]})
display(side.head(8))
""")
md("""
## 5. Bonus — RAG fallback for low-confidence answers
* **Confidence** is the mean log-probability of the generated tokens *given the prompt* (from generation scores).
* **Threshold** is the 20th percentile of that confidence on the **validation** set, so roughly the least-confident fifth triggers retrieval. Any schema-invalid answer also triggers it.
* **Knowledge base:** ChromaDB with MiniLM embeddings over the **training** clauses and their gold extractions, plus label definitions and the risk rubric. There is no val/test leakage.
""")
code("""
from src.rag_fallback import ComplianceKB, calibrate_threshold, answer_with_fallback
kb = ComplianceKB(train_rows)
thr = calibrate_threshold([p["mean_logprob"] for p in preds_val_tuned])
print(f"KB documents: {kb.col.count()} | confidence threshold (val p{config.RAG_CONFIDENCE_PERCENTILE}): {thr:.4f}")
rag = [answer_with_fallback(tuned, tok_t, r["messages"][1]["content"], kb, thr, first=p)
       for r, p in zip(test, preds_tuned)]
preds_rag = [{"id": r["id"], **x["final"]} for r, x in zip(test, rag)]
s_rag = E.score_predictions(test, preds_rag, with_bertscore=False)
n_fb = sum(x["used_rag"] for x in rag)
print(f"Fallback triggered on {n_fb}/{len(test)} test clauses")
cmp = pd.DataFrame([{"pipeline": "fine-tuned", **{labels[k]: st["aggregate"][k] for k in ("rougeL", "schema_valid", "clause_type_acc", "risk_flag_acc", "party_match")}},
                    {"pipeline": "fine-tuned + RAG fallback", **{labels[k]: s_rag["aggregate"][k] for k in ("rougeL", "schema_valid", "clause_type_acc", "risk_flag_acc", "party_match")}}])
display(cmp.round(3))
""")
code("""
# Concrete before/after: the fallback case with the largest gain (ROUGE-L + label correctness).
per_t = {d["id"]: d for d in st["per_example"]}; per_r = {d["id"]: d for d in s_rag["per_example"]}
def gain(i): return (per_r[i]["rougeL"] + per_r[i]["clause_type_acc"] + per_r[i]["risk_flag_acc"]) - \\
                    (per_t[i]["rougeL"] + per_t[i]["clause_type_acc"] + per_t[i]["risk_flag_acc"])
fb = [(r, x) for r, x in zip(test, rag) if x["used_rag"]]
if fb:
    r, x = max(fb, key=lambda rx: gain(rx[0]["id"]))
    print(f"[{r['id']}] confidence {x['first']['mean_logprob']:.4f} < threshold {thr:.4f} -> retrieved {x['retrieved']}")
    print("\\nCLAUSE:\\n" + r["messages"][1]["content"])
    print("\\nBEFORE (fine-tuned only):\\n" + x["first"]["raw"])
    print("\\nAFTER (with retrieved context):\\n" + x["final"]["raw"])
    print("\\nREFERENCE:\\n" + r["messages"][2]["content"])
    print(f"\\nROUGE-L {per_t[r['id']]['rougeL']:.3f} -> {per_r[r['id']]['rougeL']:.3f} | "
          f"clause_type {per_t[r['id']]['clause_type_acc']:.0f} -> {per_r[r['id']]['clause_type_acc']:.0f} | "
          f"risk {per_t[r['id']]['risk_flag_acc']:.0f} -> {per_r[r['id']]['risk_flag_acc']:.0f}")
""")
md("## 6. Save artifacts (used by notebook 04) and publish them next to the model")
code("""
config.OUTPUT_DIR.mkdir(exist_ok=True)
def dump(name, rows):
    with open(config.OUTPUT_DIR / name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
dump("predictions_base.jsonl", preds_base); dump("predictions_tuned.jsonl", preds_tuned)
dump("rag_results.jsonl", [{"id": r["id"], **x} for r, x in zip(test, rag)])
json.dump({"n_test": len(test), "base": sb, "tuned": st, "tuned_rag": s_rag, "rag_threshold": thr,
           "rag_triggered": n_fb}, open(config.OUTPUT_DIR / "metrics.json", "w"), indent=1)
results.to_csv(config.OUTPUT_DIR / "metrics_table.csv", index=False)
if HF_TOKEN:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    for f in ("predictions_base.jsonl", "predictions_tuned.jsonl", "rag_results.jsonl", "metrics.json", "metrics_table.csv"):
        api.upload_file(path_or_fileobj=str(config.OUTPUT_DIR / f), path_in_repo=f"eval/{f}", repo_id=config.HF_REPO_ID)
    print("Uploaded eval artifacts to", f"https://huggingface.co/{config.HF_REPO_ID}/tree/main/eval")
""")
notebook(c, "03_evaluate.ipynb")
