# Reflection

## Task 1 — Financial AI
**Decisions.** Data, LLM and presentation layers are kept separate. All numbers live in `config.py` and all prompts in `prompts.py`, so everything is unit-testable without a network or key. The indicators are tested against independent reference implementations: Wilder RSI with an SMA seed, MACD masked during warm-up, and population-std Bollinger Bands. LLM output is treated as untrusted. Each call goes through JSON mode, tolerant extraction, Pydantic validation and a bounded repair loop. Invalid results are excluded and counted, never treated as neutral. The signal prompt receives *relational* features (SMA distance, cross recency, histogram slope, bandwidth percentile), so the model reasons over interactions instead of echoing values. News turned out to be a data-quality problem: yfinance returned nothing, and the RSS feeds were dominated by 13F boilerplate. I solved it with layered sources plus relevance and boilerplate ranking.

**Limitations / next.** The signal ignores valuation and macro conditions, and it has not been backtested. My first step would be a walk-forward test of its calls against forward returns. Sentiment uses headlines only. I would cluster stories into events and calibrate the confidence scores.

## Task 2 — Fine-tuning
**Decisions.** I chose compliance-clause extraction because it is objectively gradeable: fixed schema, 10-label taxonomy, verbatim party. Diversity was designed in through a stratified generation plan rather than hoped for. QC filters (label = spec, party appears in the clause) protect faithfulness. The split is stratified to exactly 192/24/24. Teacher `gpt-oss-120b` ≠ student Phi-3-mini. I corrected three blueprint issues:
- Phi-3's fused `qkv_proj`/`gate_up_proj` target names.
- fp16 instead of bf16, which the T4 lacks.
- Merging into an fp16 base rather than the 4-bit model.

Loss is computed on assistant tokens only, with a plain `Trainer` so the mask is explicit and tested. A CPU smoke test with a tiny random Phi-3 caught API errors before any GPU time was spent.

**Results.** On held-out test data, ROUGE-L rose from 0.633 to 0.778 and BERTScore from 0.905 to 0.967. Schema validity went from 75% to 100% and clause-type accuracy from 63% to 92%. Paired-bootstrap confidence intervals exclude zero for all of these. LLM-judge score rose from 3.60 to 4.17 out of 5. Manual review of all 24 outputs found 1 hallucination (4.2%): a deadline moved onto the wrong duty. Risk flag remains weak at 63%.

**Limitations.**
- Validation loss plateaued at epoch 3, so the best checkpoint (epoch 2) was the one published.
- The evaluation is synthetic, n = 24, and the judge is also the teacher model.
- The teacher made label errors that QC cannot catch (one test reference is mislabelled).
- The RAG fallback was neutral overall: it fixed some triggers but changed some correct risk flags.

**With more time.** Human-validate a seed set, add risk-rubric examples, try constrained JSON decoding, and calibrate confidence for RAG routing.
