"""Task 2 configuration: taxonomy, diversity axes, models, hyperparameters, paths.

Every number used by the pipeline lives here, next to the reason it was chosen.
Hyperparameters are justified in more depth in 02_finetune_qlora.ipynb.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Task 2 config for compliance
# clause extraction QLoRA pipeline', Date: 2026-10-07
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PROMPT_DIR = PROJECT_DIR / "prompts"

RAW_PATH = DATA_DIR / "raw_generated.jsonl"
CLEAN_PATH = DATA_DIR / "clean.jsonl"
REJECTED_PATH = DATA_DIR / "rejected.jsonl"
SPLIT_PATHS = {s: DATA_DIR / f"{s}.jsonl" for s in ("train", "val", "test")}

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
CLAUSE_TYPES = [
    "indemnification", "termination", "payment_terms", "confidentiality",
    "liability_cap", "force_majeure", "compliance_covenant", "data_protection",
    "aml_kyc", "regulatory_reporting",
]
RISK_FLAGS = ["low", "medium", "high"]
OUTPUT_KEYS = ["clause_type", "obligation", "party_responsible",
               "trigger_condition", "risk_flag"]

# --------------------------------------------------------------------------- #
# Diversity axes for teacher generation (sampled per example)
# --------------------------------------------------------------------------- #
INDUSTRIES = [
    "retail banking", "investment banking", "insurance", "fintech payments",
    "asset management", "trade finance", "digital asset custody",
    "microfinance and leasing",
]
DOCUMENT_TYPES = [
    "syndicated loan agreement", "ISDA master agreement schedule",
    "cloud outsourcing contract", "KYC/AML onboarding policy",
    "insurance policy wording", "fund prospectus", "payment services agreement",
    "data processing agreement", "correspondent banking agreement",
    "investment management agreement",
]
JURISDICTIONS = [
    "England and Wales", "New York", "Singapore", "Sri Lanka", "DIFC (Dubai)",
    "Ireland (EU)", "India", "Hong Kong",
]
COMPLEXITY = [
    "a single plain sentence",
    "two to three sentences with a defined deadline or amount",
    "a multi-part clause with an exception or carve-out ('save that', 'provided that')",
    "a dense clause with cross-references and nested conditions",
    "an ambiguous clause where the trigger is implicit or there is no trigger",
]

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
EXAMPLES_PER_CLAUSE_TYPE = 24         # 10 types x 24 = 240 raw -> >=200 after QC
EXAMPLES_PER_CALL = 3                 # amortises the system prompt under 8k TPM
GEN_SEED = 7
TEACHER_PROVIDER_BASE_URL = "https://api.groq.com/openai/v1"
# Teacher (data generator) != student (fine-tuned). Llama-3.3-70B, the blueprint's
# teacher, was retired from Groq in 2026; gpt-oss-120b (OpenAI, 120B MoE) is the
# strongest free model there. Student is Microsoft Phi-3-mini: different family.
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "openai/gpt-oss-120b")
TEACHER_TEMPERATURE = 0.9             # high temperature for lexical diversity
TEACHER_REASONING_EFFORT = "low"      # generation needs fluency more than deep reasoning
TEACHER_MAX_TOKENS = 4000
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
JUDGE_TEMPERATURE = 0.0
JUDGE_REASONING_EFFORT = "medium"
LLM_MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_S = 15.0
# ~1.3k tokens per 3-example call; 10 s spacing keeps us under the 8k tokens/min free tier.
GEN_CALL_INTERVAL_S = 10.0

# Quality-control thresholds
MIN_CLAUSE_WORDS = 15
MAX_CLAUSE_WORDS = 220
MAX_OBLIGATION_WORDS = 45
NEAR_DUPLICATE_COSINE = 0.85          # TF-IDF cosine above this => near-duplicate
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42

# --------------------------------------------------------------------------- #
# Student model + QLoRA hyperparameters (justified in notebook 02)
# --------------------------------------------------------------------------- #
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "nazifmhd/phi3-mini-compliance-extractor")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Phi-3 fuses Q/K/V into qkv_proj and gate/up into gate_up_proj.
LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4                        # effective batch 16
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.3                   # QLoRA paper value
OPTIMIZER = "paged_adamw_8bit"
TRAIN_SEED = 42

# Inference
MAX_NEW_TOKENS = 320
OUTPUT_TRUNCATE_FOR_LOG = 200

# RAG fallback
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 3
# Confidence = mean token log-prob of the generated answer given the prompt;
# threshold is set from the validation-set distribution (see notebook 03).
RAG_CONFIDENCE_PERCENTILE = 20
