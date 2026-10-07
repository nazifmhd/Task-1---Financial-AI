"""Inserts the analyst annotation of the LLM signal into the EXECUTED notebook.

The annotation quotes the model's actual output, so it can only be written
after a run. This edits a markdown cell only - no code is re-executed, so the
recorded outputs stay exactly as produced.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Annotate how the executed
# LLM signal reasons over indicator combinations', Date: 2026-10-07
"""
import nbformat

NB = "task1_equity_research.ipynb"
PLACEHOLDER = "{{SIGNAL_ANNOTATION}}"

ANNOTATION = """
### 8.1 Reviewer note: does the model reason over combinations? (run of 2026-10-07)

The output is not a list of indicator values. It **combines** them in several places:

| Interaction in the justification | Why it is combination reasoning |
|---|---|
| *"price firmly above both the 50-day and 200-day SMAs … while the MACD line sits above its signal with a rising histogram"* | Treats the trend regime and MACD momentum as **mutually confirming**, not as two separate facts. |
| *"RSI has climbed to 67, indicating strong but not yet exhausted buying pressure"* | Reads RSI **in the context of the uptrend** (67 counts as confirmation inside a bullish regime), which is the regime-conditional reading the system prompt asks for. It also uses the 5-session change (56.7 → 67.3). |
| *"trading near the upper Bollinger band and just 1.7% below its 52-week high … volume slightly below its 20-day average, hinting at waning participation"* | Weighs **conflicting evidence**: price stretch (%B 0.97) and soft volume (0.93×) against the trend. It concludes these are near-term cautions, which is why conviction is *medium* rather than *high*. |
| *"positive news sentiment adds … support"* | Uses the LLM news aggregate (+0.62) as **confirmation** of the technical picture rather than a separate verdict. |

The structured `key_drivers` and `conflicting_signals` fields make this reasoning auditable. Note
that the model did **not** just copy the rule-based vote (5/5 bullish → Buy). It lowered conviction
because of the Bollinger stretch, the proximity to the 52-week high and the volume, none of which
the rule-based vote looks at.

*Limitation:* the call is purely technical plus headline sentiment. Valuation (trailing P/E ≈ 30)
and macro risk are outside its inputs (see REFLECTION.md).
"""


def main():
    nb = nbformat.read(NB, as_version=4)
    hits = [c for c in nb.cells if c.cell_type == "markdown" and c.source.strip() == PLACEHOLDER]
    if not hits:
        print("placeholder not found (already annotated?)")
        return
    hits[0].source = ANNOTATION.strip()
    nbformat.write(nb, NB)
    print("annotation inserted")


if __name__ == "__main__":
    main()
