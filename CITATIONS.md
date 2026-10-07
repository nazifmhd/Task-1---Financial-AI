# Citations

This file follows Section 2.2 of the assessment brief. Inline `# AI-ASSISTED:` / `# SOURCE:`
comments also appear at the top of each module.

## AI assistance

All code was written with Claude acting as a pair-programmer (Claude Code in VS Code). I set the
architecture by following my own Task 1 blueprint. I reviewed, ran and tested every module, and I
can explain and defend each design choice.

```
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement Task 1 config module per CDAZZDEV blueprint - all magic numbers centralised', Date: 2026-10-07
#   -> task1_financial/config.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Robust yfinance OHLCV + fundamentals fetch with no hardcoded dates and MultiIndex flattening', Date: 2026-10-07
#   -> task1_financial/src/data_fetch.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement SMA, Wilder RSI, MACD and Bollinger Bands from first principles in pandas without TA-Lib, with correct warm-up handling', Date: 2026-10-07
#   -> task1_financial/src/indicators.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'News retrieval with yfinance primary and RSS fallbacks, dedupe and normalisation', Date: 2026-10-07
#   -> task1_financial/src/news.py (incl. relevance ranking and 13F-boilerplate filter)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build summary dict (price, 52w range, P/E, YTD, momentum) with null-safe fallbacks and a run_task1a orchestrator', Date: 2026-10-07
#   -> task1_financial/src/summary.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic v2 schemas for headline sentiment and Buy/Hold/Sell signal with sentence-count validation', Date: 2026-10-07
#   -> task1_financial/src/llm/schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write system/user prompt templates for financial headline sentiment and an indicator-combination Buy/Hold/Sell signal that forbids restating values', Date: 2026-10-07
#   -> task1_financial/src/llm/prompts.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'OpenAI-compatible Groq / OpenRouter client with JSON mode, Pydantic validation, repair-retry loop, 429 back-off and audit logging', Date: 2026-10-07
#   -> task1_financial/src/llm/client.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Per-headline sentiment via LLM with Pydantic validation and confidence-weighted aggregation', Date: 2026-10-07
#   -> task1_financial/src/llm/sentiment.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Derive relational indicator features (cross recency, histogram slope, bandwidth percentile) and call LLM for a validated Buy/Hold/Sell with rule-based fallback', Date: 2026-10-07
#   -> task1_financial/src/llm/signal.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Orchestrator wiring Task 1A outputs into LLM sentiment + signal with graceful degradation', Date: 2026-10-07
#   -> task1_financial/src/pipeline.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Render a one-page equity research brief from pipeline outputs: matplotlib charts, Markdown and a self-contained Jinja2 HTML page with risk disclaimer', Date: 2026-10-07
#   -> task1_financial/report.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest suite validating SMA/RSI/MACD/Bollinger against naive reference implementations', Date: 2026-10-07
#   -> task1_financial/tests/test_indicators.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for Pydantic validation, repair-retry loop and sentiment aggregation using a fake OpenAI-compatible transport', Date: 2026-10-07
#   -> task1_financial/tests/test_llm.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate the Task 1 notebook cells with nbformat', Date: 2026-10-07
#   -> task1_financial/build_notebook.py, task1_equity_research.ipynb
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Annotate how the executed LLM signal reasons over indicator combinations', Date: 2026-10-07
#   -> task1_financial/annotate_notebook.py (notebook §8.1)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Draft README, CITATIONS and REFLECTION for Task 1', Date: 2026-10-07
#   -> README.md, task1_financial/README.md, CITATIONS.md, REFLECTION.md
```

## Runtime LLM (part of the product, not used to write code)
* Groq API, `openai/gpt-oss-120b` (free tier, reasoning_effort=medium): classifies headline sentiment and produces
  the Buy/Hold/Sell signal. The full prompts are in `task1_financial/src/llm/prompts.py`, and the
  notebook (§6) prints them.

## External references (formulas and conventions; no code copied)
```
# SOURCE: J. Welles Wilder Jr., "New Concepts in Technical Trading Systems" (1978) - RSI and Wilder smoothing definition
# SOURCE: https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi - RSI seed/smoothing procedure
# SOURCE: https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd - MACD(12,26,9)
# SOURCE: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands - Bollinger Bands (20, 2, population std)
# SOURCE: yfinance documentation https://github.com/ranaroussi/yfinance - download/Ticker.info/news APIs
# SOURCE: Groq OpenAI-compatible API docs https://console.groq.com/docs/openai - base URL and JSON mode
```

## Free data sources used at runtime
yfinance (Yahoo Finance) for prices and fundamentals; Yahoo Finance RSS and Google News RSS for
headlines.

---

# Task 2 — Generative AI (task2_genai/)

## AI assistance (Claude Code, claude-opus-5-5, 2026-10-07)
I chose the use case, the taxonomy and the evaluation design. Claude helped write the code. I ran
notebooks 02 and 03 on Colab and reviewed all outputs.
```
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Task 2 config for compliance clause extraction QLoRA pipeline', Date: 2026-10-07  -> config.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schema for clause extraction plus tolerant JSON parsing of LLM output', Date: 2026-10-07  -> src/schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Groq JSON-mode client with rate-limit back-off and Pydantic validation for data generation', Date: 2026-10-07  -> src/llm.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Stratified teacher data generation plan, resumable generation loop and faithfulness QC filters', Date: 2026-10-07  -> src/gen.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Diversity report for a generated clause dataset: length histogram, label/axis coverage, TF-IDF keywords, lexical + semantic similarity, distinct-n', Date: 2026-10-07  -> src/diversity.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Convert extraction rows to chat-format JSONL and make a stratified 80/10/10 split', Date: 2026-10-07  -> src/format_split.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'QLoRA training module for Phi-3-mini on T4: NF4 config, LoRA on fused Phi-3 projections, assistant-only loss masking, per-epoch loss callback, fp16 merge and push', Date: 2026-10-07  -> src/train.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Batched greedy inference with per-answer mean token log-prob, ROUGE-L/BERTScore and field-level structured metrics for JSON extraction', Date: 2026-10-07  -> src/evaluate.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'ChromaDB RAG fallback triggered by conditional answer log-prob threshold calibrated on validation data', Date: 2026-10-07  -> src/rag_fallback.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LLM-as-judge over test predictions with a 4-criterion rubric and Pydantic-validated verdicts', Date: 2026-10-07  -> src/judge.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for extraction schema, QC rules, stratified split and structured metrics', Date: 2026-10-07  -> tests/test_pipeline.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'CPU smoke test for the QLoRA pipeline with a tiny random Phi-3 model', Date: 2026-10-07  -> smoke_test_cpu.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate Task 2 notebooks with nbformat', Date: 2026-10-07  -> build_notebooks.py, build_notebook_04.py, *.ipynb
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Draft manual hallucination-review labels with reasons for the 24 fine-tuned test outputs', Date: 2026-10-07  -> outputs/manual_review.json (drafted by Claude; each label checked by me against the clause text)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Teacher, student and judge system prompts for compliance clause extraction', Date: 2026-10-07  -> prompts/*.txt
```

## Models used at runtime
* **Teacher (data generation):** `openai/gpt-oss-120b` via the Groq free tier, temperature 0.9. The full system prompt is in `task2_genai/prompts/teacher_system_prompt.txt` and is printed in notebook 01.
* **Student (fine-tuned):** `microsoft/Phi-3-mini-4k-instruct` (MIT licence). Published as https://huggingface.co/nazifmhd/phi3-mini-compliance-extractor
* **Judge:** `openai/gpt-oss-120b` via Groq, temperature 0. The prompt is in `prompts/judge_system_prompt.txt`.
* **Embeddings for diversity and RAG:** `sentence-transformers/all-MiniLM-L6-v2`. BERTScore uses `roberta-large` (bert_score defaults).

## External references (methods; no code copied)
```
# SOURCE: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) https://arxiv.org/abs/2305.14314 - NF4, double quantisation, paged optimiser, lr 2e-4, max_grad_norm 0.3, all-linear LoRA
# SOURCE: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) https://arxiv.org/abs/2106.09685 - rank/alpha scaling
# SOURCE: Hugging Face PEFT / Transformers / bitsandbytes documentation - LoraConfig, prepare_model_for_kbit_training, BitsAndBytesConfig, merge_and_unload
# SOURCE: Lin, "ROUGE" (2004); Zhang et al., "BERTScore" (2020) https://arxiv.org/abs/1904.09675 - evaluation metrics
# SOURCE: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023) https://arxiv.org/abs/2306.05685 - LLM-as-judge design and self-preference bias caveat
```

---

# Task 3 — Agentic Workflows (task3_agentic/)

## AI assistance (Claude Code, claude-opus-5-5, 2026-10-07)
I set the architecture, following my Task 3 blueprint, and directed the design changes listed in the
Task 3 README. Claude helped write the code. The notebook was executed end-to-end on live data.
```
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Config for a LangGraph multi-agent financial research system on Groq free tier', Date: 2026-10-07  -> config.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Tracing decorator writing tool name, inputs, truncated output, duration, agent and run id to JSONL', Date: 2026-10-07  -> src/observability/trace.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schemas for DataBrief handoff, clarification request/response and three-section research report', Date: 2026-10-07  -> src/schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'ChatOpenAI factory for Groq with Colab-secret / env key loading', Date: 2026-10-07  -> src/llm.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Five LangChain tools (price data with indicators, news with RSS fallback, volatility, LLM sentiment, DuckDuckGo search) with structured errors, tracing and fault injection', Date: 2026-10-07  -> src/tools.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pretty-print LangGraph ReAct message streams with action/observation/replan markers', Date: 2026-10-07  -> src/console.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph pre_model_hook that compacts old tool observations and enforces a tool-call budget', Date: 2026-10-07  -> src/context.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph ReAct research agent with five tools, checkpointer memory and structured three-section report', Date: 2026-10-07  -> src/single_agent.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph two-agent pipeline with tool-restricted ReAct agents, Pydantic handoffs, one-round critique loop and handoff trace', Date: 2026-10-07  -> src/multi_agent.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Per-ticker dated JSON cache for research briefs with cache-hit tracing', Date: 2026-10-07  -> src/memory.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Streamlit dashboard for an agent tool-call trace JSONL', Date: 2026-10-07  -> dashboard.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Offline pytest for agent tools, tracing, tool restriction, context hook and cache', Date: 2026-10-07  -> tests/test_offline.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate the Task 3 notebook with nbformat', Date: 2026-10-07  -> build_notebook.py, task3_agentic.ipynb
# SOURCE: Adapted from task1_financial/src/indicators.py (my Task 1 code in this repository) -> task3_agentic/src/indicators.py
```

## Models and services used at runtime
* **Agents:** `openai/gpt-oss-20b` via Groq (free tier).
* **Sentiment tool:** `openai/gpt-oss-120b` via Groq.
* **Web search:** DuckDuckGo via the `ddgs` package.
* **Market data:** yfinance; news from the Yahoo Finance RSS and Google News RSS feeds.

## External references (APIs and patterns; no code copied)
```
# SOURCE: LangGraph documentation https://langchain-ai.github.io/langgraph/ - create_react_agent, pre_model_hook, StateGraph conditional edges, MemorySaver checkpointer
# SOURCE: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022) https://arxiv.org/abs/2210.03629 - reason/act/observe loop
# SOURCE: Streamlit docs https://docs.streamlit.io - st.metric, st.bar_chart, streamlit.testing.v1.AppTest
```
