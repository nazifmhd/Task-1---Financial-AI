# Task 1 — LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedNazif/CDAZZDEV-MLE-MohamedNazif/blob/main/task1_financial/task1_equity_research.ipynb)

**Executed notebook (outputs visible):** [`task1_equity_research.ipynb`](task1_equity_research.ipynb)
**Sample brief:** [`outputs/brief.html`](outputs/brief.html) · [`outputs/brief.md`](outputs/brief.md)

The pipeline pulls real market data for a ticker (default **NVDA**), computes five technical
indicators from first principles, gathers news, and then uses an LLM (Groq
`openai/gpt-oss-120b` by default, or OpenRouter) to score headline sentiment and produce
a reasoned Buy/Hold/Sell call. Every LLM output is validated before use. The result is a
one-page research brief.

## Architecture

```
config.py                  every tunable number (no magic numbers in logic)
src/
  data_fetch.py            yfinance OHLCV (runtime-computed dates) + fundamentals, null-safe
  indicators.py            SMA50/200, Wilder RSI14, MACD(12,26,9), Bollinger(20,2), momentum vote
  news.py                  yfinance → Yahoo RSS → Google News RSS; dedupe + relevance ranking
  summary.py               summary dict + run_task1a() orchestrator
  pipeline.py              run_task1b(), save_outputs()
  llm/
    prompts.py             all prompt templates (system/user split, versioned)
    schemas.py             Pydantic contracts: HeadlineSentiment, TradeSignal, SentimentAggregate
    client.py              OpenAI-compatible client: JSON mode, validate, repair-retry, audit log
    sentiment.py           per-headline scoring + confidence-weighted aggregate
    signal.py              relational indicator context → validated TradeSignal (+ flagged fallback)
report.py                  bonus: matplotlib charts → Markdown + self-contained HTML brief
tests/                     indicator reference tests + LLM validation tests (no network)
build_notebook.py          generates the notebook from source
annotate_notebook.py       inserts the post-run analysis of the LLM signal
```

Data logic, LLM logic and presentation never mix. `config.py` is the only place numbers
live, and `prompts.py` is the only place prompt text lives.

## Run it

**Colab:** click the badge, add `GROQ_API_KEY` (or `OPENROUTER_API_KEY`) under **Secrets**
(the key icon in the sidebar), then *Runtime → Run all*. The first cell clones the repo and
installs the requirements.

**Locally:**
```bash
cd task1_financial
pip install -r requirements.txt
cp ../.env.example .env        # add GROQ_API_KEY=...  (.env is git-ignored)
python -m pytest               # 15 tests, no network or key needed
python build_notebook.py       # regenerate the notebook from source (optional)
jupyter nbconvert --to notebook --execute --inplace task1_equity_research.ipynb
python annotate_notebook.py    # add the reviewer note on the signal (markdown only)
```
To analyse another ticker, set `TICKER=MSFT` before running. Without a key, the pipeline still
completes. Sentiment is reported as *unavailable* and the signal uses a clearly flagged rule-based
fallback.

## Indicator correctness (no TA-Lib)

| Indicator | Convention |
|---|---|
| SMA 50/200 | arithmetic rolling mean. NaN until the window is full, so values are never made up during warm-up |
| RSI 14 | **Wilder smoothing**: the first average is a simple mean of 14 gains/losses, then `avg = (prev·13 + x)/14`. All gains → 100, flat → 50 |
| MACD 12/26/9 | recursive EMA (`adjust=False`, α = 2/(n+1)), masked for the first 26 bars; signal = EMA9(MACD); hist = MACD − signal |
| Bollinger 20/2 | SMA20 ± 2 × **population** std (`ddof=0`), as Bollinger defines it. Also outputs %B and bandwidth |

`tests/test_indicators.py` checks each indicator against separate loop-based implementations of
the textbook formulas. The notebook (§3.1) repeats this check on the live NVDA series. Errors
are around 1e-13. It also shows that a naive rolling-mean RSI gives materially different
values.

## LLM design

* **Structured output.** Requests use JSON mode. The JSON is pulled out even if the model wraps it
  in fences or prose, then validated with Pydantic (`Literal` enums, `confidence ∈ [0,1]`, a
  3–5-sentence justification counted so that "67.3" is not a sentence break). On failure, the
  error is logged and sent back to the model as a repair turn, up to 2 retries. After that the
  result is `None` and is **excluded and counted**, never treated as neutral. Every attempt is
  written to `outputs/llm_calls.jsonl`. Notebook §9 shows the failure path.
* **Sentiment aggregation.** `Σ(polarity·confidence) / Σ(confidence)` gives a score in [−1, 1],
  with a ±0.15 neutral band.
* **Signal reasoning.** The model gets relationships rather than raw levels: % distance from each
  SMA, how long ago the golden/death cross and MACD cross happened, histogram slope, 5-day RSI
  change, %B, bandwidth percentile, volume ratio, and news sentiment. The system prompt forbids
  restating values. It asks the model to say whether indicators confirm or contradict each other
  and to weigh at least one conflicting signal.
* **Provider-agnostic.** The client uses the `openai` SDK against Groq or OpenRouter, so switching
  provider only means changing an environment variable.

## Outputs (`outputs/`)
`summary.json`, `headlines.json`, `sentiment.json`, `signal.json`,
`prices_with_indicators.csv`, `llm_calls.jsonl`, `chart.png`, `sentiment.png`, `brief.md`,
`brief.html`.
