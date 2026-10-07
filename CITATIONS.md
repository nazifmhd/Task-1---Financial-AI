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
