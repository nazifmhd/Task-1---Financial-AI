# CDAZZDEV-MLE-MohamedNazif

Senior Machine Learning Engineer technical assessment for Ceylon Dazzling Dev Holding (Pvt.) Ltd.

| Task | Folder | Status | Notebook |
|---|---|---|---|
| 1 — Financial AI: LLM-powered equity research assistant | [`task1_financial/`](task1_financial/) | Complete (1A, 1B, bonus report) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nazifmhd/Task-1---Financial-AI/blob/main/task1_financial/task1_equity_research.ipynb) |
| 2 — Generative AI: QLoRA fine-tuning of Phi-3-mini for compliance clause extraction | [`task2_genai/`](task2_genai/) | Complete (2A, 2B, 2C, RAG bonus). Model: [nazifmhd/phi3-mini-compliance-extractor](https://huggingface.co/nazifmhd/phi3-mini-compliance-extractor) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nazifmhd/Task-1---Financial-AI/blob/main/task2_genai/02_finetune_qlora.ipynb) |
| 3 — Agentic workflows: multi-agent research | `task3_agentic/` | — | — |

* [`CITATIONS.md`](CITATIONS.md) lists all AI assistance and the external sources used.
* [`REFLECTION.md`](REFLECTION.md) covers architecture decisions, limitations and next steps.

## Credentials
No keys are committed. The notebooks read `GROQ_API_KEY` / `OPENROUTER_API_KEY` from Colab
Secrets or from a git-ignored `.env` file (see [`.env.example`](.env.example)).
