# Task 2 — Domain-Specific Fine-Tuning: Financial Compliance Clause Extraction

| Notebook | What it does | Where it runs |
|---|---|---|
| [`01_generate_dataset.ipynb`](01_generate_dataset.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nazifmhd/Task-1---Financial-AI/blob/main/task2_genai/01_generate_dataset.ipynb) | 2A: problem statement, teacher generation, QC, diversity metrics, chat JSONL, 80/10/10 split | CPU (executed) |
| [`02_finetune_qlora.ipynb`](02_finetune_qlora.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nazifmhd/Task-1---Financial-AI/blob/main/task2_genai/02_finetune_qlora.ipynb) | 2B: QLoRA (4-bit NF4) training, hyperparameter justification, per-epoch loss, merge, push to the Hub | Colab T4 |
| [`03_evaluate.ipynb`](03_evaluate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nazifmhd/Task-1---Financial-AI/blob/main/task2_genai/03_evaluate.ipynb) | 2C: base vs fine-tuned on the held-out test set (ROUGE-L, BERTScore, structured metrics, bootstrap CIs) and the RAG fallback bonus | Colab T4 |
| `04_judge_and_review.ipynb` | 2C: LLM-as-judge, manual hallucination review, qualitative analysis | CPU |

**Fine-tuned model:** https://huggingface.co/nazifmhd/phi3-mini-compliance-extractor (public, merged fp16 weights)

## Use case
A model receives one clause from a financial-sector contract or policy (loan agreement, outsourcing
contract, KYC policy, DPA, ISDA schedule, and so on) and returns JSON:

```json
{"clause_type": "data_protection", "obligation": "delete or return all personal data relating to the individual and notify the Client",
 "party_responsible": "The Service Provider", "trigger_condition": "receipt of a data subject's request", "risk_flag": "medium"}
```

* `clause_type` comes from a fixed 10-label taxonomy.
* `party_responsible` must be copied verbatim from the clause.
* `trigger_condition` may be `null`.
* `risk_flag` follows a written rubric.

**Correct** means valid JSON, the right label, and nothing invented. **Hallucinated** means any
obligation, party, amount, deadline or condition that is not in the clause. The full definitions are
in notebook 01.

## Pipeline
| Stage | Choice | Why |
|---|---|---|
| Teacher | `openai/gpt-oss-120b` on Groq (free) | The strongest free model. Llama-3.3-70B is retired on Groq. |
| Student | `microsoft/Phi-3-mini-4k-instruct` (3.8B) | Different family from the teacher, and fits a T4 in 4-bit. |
| Data | 240 examples from a stratified plan: 10 types × 24, balanced across 8 industries, 10 document types, 8 jurisdictions and 5 complexity levels | Diversity is designed in, then measured: mean TF-IDF cosine 0.067, no lexical pair above 0.85. |
| QC | label = requested type, party appears verbatim in the clause, length bounds, null-trigger consistency, no label leakage, near-duplicate removal | Training targets have to be faithful, or the student learns to hallucinate. |
| Split | stratified, exactly 192 / 24 / 24 | Every class appears in val and test. |
| Training | QLoRA NF4 + double quantisation, LoRA r16/α32 on `qkv_proj, o_proj, gate_up_proj, down_proj`, fp16, loss on assistant tokens only | Every hyperparameter is justified in notebook 02. |
| Merge | adapter merged into an **fp16** base (not the 4-bit one) | Merging into 4-bit weights would bake quantisation error into the published model. |
| Eval | same prompt, test set, quantisation and greedy decoding for both models; paired-bootstrap 95% CIs | Fine-tuning is the only variable, and the CIs show whether the gain is real or noise. |
| Bonus | ChromaDB RAG over **train** clauses + label definitions, triggered when answer log-prob < the val 20th percentile | The threshold is calibrated on data, and there is no test leakage. |

## Deviations from the blueprint (and why)
* **Teacher model:** `openai/gpt-oss-120b` instead of Llama-3.3-70B, because the latter is no longer on Groq.
* **LoRA target names:** Phi-3 fuses its projections, so the blueprint's `q_proj/k_proj/v_proj/gate_proj/up_proj` match nothing. The fused names are used instead.
* **fp16 instead of bf16:** the T4 has no native bf16.
* **Plain `Trainer` with explicit assistant-only loss masking instead of `SFTTrainer`:** the TRL API changes between versions, and the mask should be explicit and tested.
* **RAG confidence:** the answer's log-prob is computed *given the prompt*, not as the perplexity of the answer on its own.

## Run
```bash
pip install -r requirements.txt
python -m pytest            # CPU unit tests
python smoke_test_cpu.py    # CPU end-to-end smoke test of the GPU code with a tiny random Phi-3
```
* **Notebook 01:** needs `GROQ_API_KEY` in a git-ignored `.env` file or in Colab Secrets.
* **Notebooks 02 and 03:** open in Colab with a T4 runtime and the Secret `HF_TOKEN` (write access), then *Run all*.
* **Notebook 04:** needs `GROQ_API_KEY`.

## Files
`prompts/teacher_system_prompt.txt` (the full generation prompt, required by the brief) ·
`prompts/student_system_prompt.txt` · `prompts/judge_system_prompt.txt` ·
`data/raw_generated.jsonl`, `clean.jsonl`, `train/val/test.jsonl` · `outputs/` (plots, metrics, predictions).
