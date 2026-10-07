# Reflection

## Task 1 — Financial AI

### Architectural decisions
**Strict layering.** Data (`data_fetch`, `indicators`, `news`, `summary`), LLM reasoning
(`llm/`) and presentation (`report.py`) are kept in separate modules. All numbers live in
`config.py` and all prompt text lives in `prompts.py`. Every behaviour can be
unit-tested without Jupyter, the network or an API key.

**Correctness over convenience for indicators.** RSI uses true Wilder smoothing with an SMA seed
rather than `ewm` from the first bar. MACD is masked during EMA warm-up, and Bollinger Bands use
population std. Every indicator is tested against separate loop implementations of the textbook
formulas, and the notebook repeats that check on live data. Two years of data plus a 300-day
warm-up buffer means SMA-200 is valid across the whole analysis window.

**LLM output is untrusted input.** Each call goes through JSON mode, then tolerant extraction,
then Pydantic validation, then a bounded repair loop that sends the validation error back to the
model. If that still fails, the result is `None`, and the headline is excluded and counted rather
than defaulted to neutral, which would bias the aggregate. Every attempt is audited to JSONL.
The client uses the OpenAI-compatible protocol, so Groq and OpenRouter can be swapped with an
environment variable.

**The model reasons over relationships, not raw values.** The rubric rewards reasoning over
indicator *combinations*. So instead of handing the model raw levels, I compute relational
features: distance from each SMA, how recent the crosses are, histogram slope, RSI change,
bandwidth percentile and volume ratio. The system prompt forbids echoing values and asks the
model to weigh a conflicting signal. Schema fields (`key_drivers`, `conflicting_signals`) make
that reasoning inspectable.

**News quality is a data problem.** Live testing showed that yfinance `.news` returned zero items,
Yahoo RSS was full of generic market pieces, and Google News was dominated by auto-generated 13F
headlines ("X LLC buys 2,218 shares"). Classifying the raw feed would have measured noise. I
added layered sources, deduplication, a relevance check against company keywords, and a
boilerplate regex that ranks those items last.

**Graceful degradation.** A bad ticker, empty data, missing P/E or a missing API key never raises.
Fields become `None`, sentiment becomes "unavailable" (deliberately not "neutral"), and the
signal falls back to a flagged rule-based vote so the brief still renders.

### Limitations
* The signal is purely technical plus headline sentiment. It ignores valuation, earnings
  revisions, macro conditions and options-implied volatility.
* Headline-only sentiment misses article context, and duplicate coverage of one event can
  over-weight that event.
* yfinance is a scraped, rate-limited source. `.info` fields are inconsistent, and prices are
  split/dividend-adjusted, so the 52-week range can differ slightly from quoted unadjusted values.
* LLM outputs are non-deterministic even at temperature 0.1, and there is no ground truth to
  measure signal quality.
* The boilerplate filter is a heuristic regex and can miss new phrasings.

### What I would do with more time
1. **Evaluate the signal.** Run a walk-forward backtest of the Buy/Hold/Sell calls against forward
   1–3-month returns, compared with the rule-based vote, to check whether the LLM adds value.
2. **Better sentiment.** Use article bodies, cluster near-duplicate stories into events, weight by
   recency and source credibility, and calibrate the LLM's confidence against labelled data.
3. **Fundamental context.** Add earnings surprises, estimate revisions and peer-relative valuation
   to the prompt.
4. **Production hardening.** Add a cached data layer (Parquet), async batched scoring, and response
   caching keyed by prompt version. Add CI that runs the test suite and a secret scan.
5. **Prompt evaluation.** Compare prompt versions and models on a labelled headline set.
