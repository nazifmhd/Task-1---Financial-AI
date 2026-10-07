"""Generates task1_equity_research.ipynb from cell sources below.

Keeping the notebook as code makes it reviewable in diffs; execute it with
  jupyter nbconvert --to notebook --execute --inplace task1_equity_research.ipynb

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate the Task 1
# notebook cells with nbformat', Date: 2026-10-07
"""
import nbformat as nbf

GITHUB_USER = "MohamedNazif"           # update if your GitHub handle differs
REPO = "CDAZZDEV-MLE-MohamedNazif"
NB_PATH = "task1_financial/task1_equity_research.ipynb"
COLAB = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO}/blob/main/{NB_PATH}"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# Task 1 — LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

**Pipeline:** yfinance OHLCV → five indicators from first principles → multi-source news →
summary dict (**Task 1A**) → per-headline LLM sentiment + reasoned Buy/Hold/Sell with Pydantic
validation (**Task 1B**) → one-page HTML/Markdown brief with embedded matplotlib charts (**Bonus**).

All logic lives in `src/` (data, indicators, news, summary, `llm/`), numbers in `config.py`,
prompts in `src/llm/prompts.py`. This notebook only orchestrates and displays results.

| Section | Rubric item |
|---|---|
| 1 | Setup and key handling (no credentials in code) |
| 2 | 1A: OHLCV fetch (≥2 years, dates computed at runtime) |
| 3 | 1A: Indicators + numerical validation against reference implementations |
| 4 | 1A: News retrieval (≥10 headlines) |
| 5 | 1A: Summary dictionary + robustness demo |
| 6 | 1B: Prompts (system/user separation) |
| 7 | 1B: Per-headline sentiment JSON + aggregation |
| 8 | 1B: Reasoned signal over indicator combinations |
| 9 | 1B: Validation-failure handling demo |
| 10 | Bonus: research brief rendering |
""")

md("## 1. Setup")
code(f"""
# In Colab: clone the repo and install deps. Locally: run from task1_financial/.
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("{REPO}"):
        subprocess.run(["git", "clone", "-q", "https://github.com/{GITHUB_USER}/{REPO}.git"], check=True)
    os.chdir("{REPO}/task1_financial")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
if os.path.basename(os.getcwd()) != "task1_financial" and os.path.isdir("task1_financial"):
    os.chdir("task1_financial")
sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd(), "| Colab:", IN_COLAB, "| Python", sys.version.split()[0])
""")
code("""
import json, logging, warnings
import pandas as pd
from IPython.display import display, HTML, Markdown, Image
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
for noisy in ("httpx", "openai", "yfinance", "urllib3", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 100)

import config
from src import data_fetch, indicators, news, summary as summ
from src.llm import client as llm_client, prompts
print("Ticker:", config.TICKER, "| run date:", config.TODAY)
""")
code("""
# API key: read from Colab Secrets (key icon in the left sidebar) or a git-ignored .env file.
# The key itself is never printed or stored in the notebook.
try:
    provider, model, _key = llm_client.resolve_provider()
    print(f"LLM provider: {provider} | model: {model} | key loaded: {bool(_key)} (len hidden)")
    del _key
except llm_client.LLMUnavailable as e:
    print("WARNING:", e)
""")

md("""
## 2. Task 1A — OHLCV data (≥ 2 years, no hardcoded dates)
`compute_date_range()` derives `start` from `date.today()` minus `LOOKBACK_YEARS` (2) plus a
300-calendar-day warm-up, so SMA-200 already has valid values at the start of the 2-year window.
""")
code("""
start, end = data_fetch.compute_date_range()
print(f"Requested window: {start} -> {end} (computed at runtime)")
raw = data_fetch.fetch_ohlcv(config.TICKER)
span_years = (raw.index[-1] - raw.index[0]).days / 365.25
print(f"{len(raw)} daily bars | {raw.index[0].date()} -> {raw.index[-1].date()} | {span_years:.2f} years")
print("Nulls per column:", raw.isna().sum().to_dict())
assert span_years >= config.LOOKBACK_YEARS
display(raw.tail())
""")

md("""
## 3. Task 1A — Technical indicators from first principles (no TA-Lib)

| Indicator | Implementation (`src/indicators.py`) |
|---|---|
| SMA 50 / 200 | `rolling(n, min_periods=n).mean()` — NaN until a full window exists |
| RSI 14 | **Wilder smoothing**: seed = simple mean of first 14 gains/losses, then `avg = (prev·13 + x)/14`. Edge cases: no losses → 100, flat → 50 |
| MACD 12/26/9 | recursive EMAs (`adjust=False`, α = 2/(n+1)); MACD masked during the 26-bar warm-up; signal = EMA9 of MACD; hist = MACD − signal |
| Bollinger 20/2 | SMA20 ± 2 × **population** std (ddof=0), plus %B and bandwidth |
""")
code("""
df = indicators.add_all_indicators(raw)
ind_cols = ["Close", "sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_lower", "bb_mid", "bb_upper", "bb_pct_b"]
print("Warm-up NaNs (expected = window-1):", df[["sma_50", "sma_200", "rsi_14", "macd"]].isna().sum().to_dict())
display(df[ind_cols].tail(8).round(3))
""")
md("""
### 3.1 Indicator validation
Each indicator is checked on the **real** price series against an independent, loop-based
implementation of the textbook formula (written separately in `tests/test_indicators.py`).
We also show that a naïve rolling-mean RSI differs materially — the common error the rubric warns about.
""")
code("""
import numpy as np, math
from tests.test_indicators import ref_sma, ref_rsi, ref_ema

x = df["Close"].tolist()
checks = {}
checks["SMA-50"]  = np.nanmax(np.abs(df["sma_50"].to_numpy() - np.array(ref_sma(x, 50))))
checks["SMA-200"] = np.nanmax(np.abs(df["sma_200"].to_numpy() - np.array(ref_sma(x, 200))))
checks["RSI-14 (Wilder)"] = np.nanmax(np.abs(df["rsi_14"].to_numpy() - np.array(ref_rsi(x, 14))))
line = np.array(ref_ema(x, 12)) - np.array(ref_ema(x, 26))
checks["MACD line"] = np.nanmax(np.abs(df["macd"].to_numpy()[25:] - line[25:]))
i = len(x) - 1; w = x[i-19:i+1]; mu = sum(w)/20; sd = math.sqrt(sum((v-mu)**2 for v in w)/20)
checks["Bollinger upper (last bar)"] = abs(df["bb_upper"].iloc[-1] - (mu + 2*sd))
display(pd.DataFrame({"max abs error vs reference": {k: f"{v:.2e}" for k, v in checks.items()}}))

d = df["Close"].diff()
naive = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
print(f"Last-bar RSI  Wilder: {df['rsi_14'].iloc[-1]:.2f}  |  naive rolling-mean RSI: {naive.iloc[-1]:.2f}"
      f"  |  max divergence over history: {(df['rsi_14'] - naive).abs().max():.1f} pts")
""")
code("""
# Full unit-test suite: indicators vs references, edge cases, LLM validation/repair logic.
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                   capture_output=True, text=True)
print(r.stdout[-1500:] or r.stderr[-1500:])
""")

md("""
## 4. Task 1A — News headlines (≥ 10, free sources)
Sources are layered and each is isolated in its own `try/except`: yfinance `.news` (its schema
changed in 2024 and often returns nothing) → Yahoo Finance RSS → Google News RSS. Results are
de-duplicated and **ranked**: company-specific news first, auto-generated 13F holding-change
headlines ("X LLC buys 2,218 shares of …") next, generic market pieces last.
""")
code("""
fundamentals = data_fetch.fetch_fundamentals(config.TICKER)
headlines = news.get_news(config.TICKER, company=fundamentals["company_name"])
print(f"{len(headlines)} headlines retrieved (minimum {config.MIN_HEADLINES})")
assert len(headlines) >= config.MIN_HEADLINES
display(pd.DataFrame(headlines)[["published", "source", "publisher", "relevant", "boilerplate", "title"]])
""")

md("## 5. Task 1A — Summary dictionary")
code("""
summary = summ.build_summary(df, fundamentals, config.TICKER)
print(json.dumps(summary, indent=2, default=str))
required = ["current_price", "week52_high", "week52_low", "pe_ratio", "ytd_return_pct", "momentum_signal"]
print("All required fields present:", all(k in summary for k in required))
""")
md("""
### 5.1 Robustness: invalid ticker and missing data
A non-existent ticker and a frame with injected gaps must degrade gracefully (logged warnings,
`None` fields) — never raise.
""")
code("""
bad = summ.run_task1a("ZZZZ-NOT-A-TICKER")
print("Bars:", len(bad.prices), "| headlines:", len(bad.headlines))
print(json.dumps({k: bad.summary[k] for k in ["current_price", "week52_high", "pe_ratio",
                                              "ytd_return_pct", "momentum_signal"]}, indent=1))

gappy = raw.copy()
gappy.iloc[-30:-25, gappy.columns.get_loc("Close")] = np.nan     # inject missing closes
gappy.iloc[-10:, gappy.columns.get_loc("Volume")] = np.nan
g = indicators.add_all_indicators(gappy.ffill())
print("With gaps -> momentum:", indicators.momentum_signal(g)["label"],
      "| last RSI:", round(g["rsi_14"].iloc[-1], 2))
print("Empty frame -> summary:", summ.build_summary(pd.DataFrame(), {}, "EMPTY")["momentum_signal"])
""")
code("""
# Single orchestrator used by the rest of the notebook (same results as the steps above).
task1a = summ.run_task1a(config.TICKER)
print(json.dumps(task1a.summary, default=str)[:400], "...")
""")

md("""
## 6. Task 1B — Prompt engineering
Prompts are versioned constants in `src/llm/prompts.py` — no prompt text is built inline in business
logic. **System** messages hold persona, rules and the output contract; **user** messages carry only
per-call data. The signal system prompt explicitly forbids restating values and requires reasoning
about how indicators *interact* and weighing at least one conflicting signal.
""")
code("""
print("PROMPT_VERSION:", prompts.PROMPT_VERSION)
print("=" * 30, "SENTIMENT_SYSTEM", "=" * 30); print(prompts.SENTIMENT_SYSTEM)
print("=" * 30, "SIGNAL_SYSTEM", "=" * 30); print(prompts.SIGNAL_SYSTEM)
""")

md("""
## 7. Task 1B — Per-headline sentiment (structured JSON) + aggregation
One LLM call per headline; each reply is validated against `HeadlineSentiment`
(`headline`, `sentiment ∈ {positive, negative, neutral}`, `confidence ∈ [0,1]`, `brief_reason`).

**Aggregate** = Σ(polarity × confidence) / Σ(confidence) ∈ [−1, 1], with a ±0.15 neutral dead-band.
Confidence-weighting means a hesitant call cannot cancel a confident one; neutral items stay in the
denominator so routine news dilutes the score; headlines that never validated are **excluded and
counted**, not silently defaulted to neutral.
""")
code("""
from src.llm.sentiment import score_headlines
sentiment = score_headlines(task1a.headlines, task1a.ticker, task1a.summary["company_name"])
print("Per-headline JSON (first 3):")
for h in sentiment.per_headline[:3]:
    print(h.model_dump_json(indent=2))
""")
code("""
display(pd.DataFrame([h.model_dump() for h in sentiment.per_headline]))
print(json.dumps({k: v for k, v in sentiment.model_dump().items() if k != "per_headline"}, indent=2))
""")

md("""
## 8. Task 1B — Reasoned Buy / Hold / Sell
`build_signal_context()` turns raw indicator levels into **relationships** (distance from each SMA,
crossover recency, histogram slope, RSI change, %B, bandwidth percentile, volume ratio) so the
model can reason about interactions. Output is validated as `TradeSignal` (enum signal,
3–5-sentence justification counted with a decimal-safe sentence splitter, 2–4 key drivers).
""")
code("""
from src.llm.signal import generate_signal
sig = generate_signal(task1a.prices, task1a.summary, sentiment)
print("USER PROMPT SENT TO THE MODEL:\\n" + sig["prompt"])
""")
code("""
s = sig["signal"]
print(f"SOURCE: {sig['source']}  |  SIGNAL: {s.signal}  |  CONVICTION: {s.conviction}\\n")
print("JUSTIFICATION:\\n" + s.justification + "\\n")
print("KEY DRIVERS:", *[f"  - {d}" for d in s.key_drivers], sep="\\n")
print("CONFLICTING SIGNALS:", *[f"  - {d}" for d in s.conflicting_signals], sep="\\n")
""")
# Filled in after execution by annotate_notebook.py, because the annotation
# must quote the actual model output from the run.
md("{{SIGNAL_ANNOTATION}}")

md("""
## 9. Task 1B — Structured-output validation failures are caught, logged, and handled
A fake OpenAI-compatible transport injects realistic failures (prose instead of JSON, an
out-of-enum label, an out-of-range confidence). The client logs each failure, feeds the
validation error back to the model, retries (max 2), and finally returns `None`, which the
aggregator excludes and counts.
""")
code("""
from types import SimpleNamespace
from src.llm.client import LLMClient, VALIDATION_EVENTS
from src.llm.schemas import HeadlineSentiment

class ScriptedTransport:
    def __init__(self, replies): self.replies = list(replies)
    def create(self, **kw):
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

ok = '{"headline": "%s", "sentiment": "negative", "confidence": 0.8, "brief_reason": "Export curbs hit revenue."}'
VALIDATION_EVENTS.clear()
scripted = LLMClient(transport=ScriptedTransport([
    # headline A: prose, then wrong enum, then valid  -> recovered on attempt 3
    "Sure! The sentiment is clearly bearish.",
    '{"headline": "A", "sentiment": "bearish", "confidence": 0.8, "brief_reason": "x y z"}',
    ok % "A",
    # headline B: three invalid replies -> excluded
    '{"headline": "B", "sentiment": "positive", "confidence": 1.7, "brief_reason": "too sure"}',
    '{"headline": "B", "sentiment": "mixed", "confidence": 0.5, "brief_reason": "unsure"}',
    "```json\\n{ broken json\\n```",
    # headline C: valid first time
    '{"headline": "C", "sentiment": "positive", "confidence": 0.9, "brief_reason": "Record data-center sales."}',
]))
demo = score_headlines([{"title": t} for t in "ABC"], "DEMO", "Demo Corp", client=scripted)
print(f"\\nscored={demo.n_scored} failed={demo.n_failed} aggregate={demo.aggregate_score:+.3f} ({demo.aggregate_label})")
display(pd.DataFrame(VALIDATION_EVENTS))
""")
code("""
# The same machinery on the live model: every real call is audited to outputs/llm_calls.jsonl
audit = pd.read_json(llm_client.AUDIT_LOG_PATH, lines=True)
live = audit[audit["model"] != "fake-model"]
print(f"Live LLM attempts: {len(live)} | first-try validation success: {live['ok'].mean():.0%}")
display(live.groupby("task")[["ok", "latency_s"]].agg(["count", "mean"]).round(3))
""")

md("## 10. Bonus — One-page equity research brief")
code("""
from src.pipeline import Task1BResult, save_outputs
import report
model_name = f"{provider}/{model}" if "provider" in dir() else "unavailable"
task1b = Task1BResult(sentiment=sentiment, signal=sig, llm_model=model_name)
print(json.dumps(save_outputs(task1a, task1b), indent=1))
paths = report.render(task1a, task1b)
print(json.dumps(paths, indent=1))
""")
code("""
display(Image(filename=paths["chart"]))
""")
code("""
display(Markdown(open(paths["markdown"], encoding="utf-8").read().replace("](chart.png)", "](outputs/chart.png)")))
""")
code("""
# The styled, self-contained HTML brief (charts embedded as base64), rendered inline.
# Open outputs/brief.html in a browser for the full-page version.
display(HTML(open(paths["html"], encoding="utf-8").read()))
""")

md("""
## Appendix — design notes
* **Why Wilder RSI?** It is the original definition used by trading platforms; a rolling-mean RSI is a
  different indicator (Section 3.1 shows the divergence on this very series).
* **Why confidence-weighted sentiment?** A low-confidence negative should not cancel a high-confidence
  positive; failed validations are excluded, not imputed.
* **How is LLM output guaranteed valid?** JSON mode → fence/prose-tolerant extraction → Pydantic
  validation → bounded repair loop with the error fed back → `None` + exclusion. Every attempt is
  audited (`outputs/llm_calls.jsonl`).
* **Graceful degradation:** with no API key, sentiment is reported as *unavailable* and the signal falls
  back to a clearly-flagged rule-based vote, so the report still renders.
* **Known limits:** no fundamentals/macro in the signal; yfinance is a scraped, rate-limited source;
  free-tier LLM latency bounds headline count (`MAX_HEADLINES=15`).
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []}})
nbf.write(nb, "task1_equity_research.ipynb")
print("wrote task1_equity_research.ipynb with", len(cells), "cells")
