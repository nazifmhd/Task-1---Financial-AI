"""Bonus: one-page equity research brief (Markdown + styled HTML [+ PDF]).

Outputs (in config.OUTPUT_DIR):
* chart.png       - 3-panel matplotlib figure: price/SMAs/Bollinger, RSI, MACD
* sentiment.png   - per-headline confidence-weighted polarity bars
* brief.md        - the brief in Markdown (as the bonus spec asks)
* brief.html      - self-contained styled page (images embedded as base64,
                    so the single file can be emailed or opened anywhere)
* brief.pdf       - only if WeasyPrint and its native libs are installed

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Render a one-page equity
# research brief from pipeline outputs: matplotlib charts, Markdown and a
# self-contained Jinja2 HTML page with risk disclaimer', Date: 2026-10-07
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import BaseLoader, Environment  # noqa: E402

import config  # noqa: E402
from src.indicators import COL_RSI, COL_SMA_LONG, COL_SMA_SHORT  # noqa: E402

log = logging.getLogger(__name__)

# Palette (validated categorical slots 1-3 from the dataviz reference palette)
C_PRICE, C_SMA_S, C_SMA_L = "#2a78d6", "#eb6834", "#1baf7a"
C_POS, C_NEG, C_NEU = "#2a78d6", "#e34948", "#a3a29c"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e6e5e0"
TOP_HEADLINES = 3
CHART_DPI = 130

DISCLAIMER = (
    "This brief was generated automatically by an LLM-assisted research "
    "pipeline for a technical assessment. It is not investment advice, an offer "
    "or a solicitation to buy or sell any security. Technical indicators are "
    "backward-looking, news sentiment is model-estimated and may be wrong, and "
    "the recommendation ignores valuation, fundamentals, macro conditions and "
    "your personal circumstances. Markets involve risk, including loss of "
    "principal. Verify all data independently and consult a licensed adviser "
    "before making any investment decision.")


def _style_axis(ax):
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def make_chart(df, ticker: str, path: Path) -> Path:
    """Price + SMA-50/200 + Bollinger band, RSI(14) and MACD(12,26,9) panels."""
    tail = df.tail(config.CHART_LOOKBACK_DAYS)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1.2], "hspace": 0.12})

    ax1.fill_between(tail.index, tail["bb_lower"], tail["bb_upper"],
                     color=C_PRICE, alpha=0.08, linewidth=0,
                     label=f"Bollinger ({config.BB_WINDOW}, {config.BB_STD:g}σ)")
    ax1.plot(tail.index, tail["Close"], color=C_PRICE, lw=1.6, label="Close")
    ax1.plot(tail.index, tail[COL_SMA_SHORT], color=C_SMA_S, lw=1.3,
             label=f"SMA {config.SMA_SHORT}")
    ax1.plot(tail.index, tail[COL_SMA_LONG], color=C_SMA_L, lw=1.3,
             label=f"SMA {config.SMA_LONG}")
    ax1.set_title(f"{ticker} — price, trend and volatility (last 12 months)",
                  loc="left", fontsize=11, color=C_INK, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)
    ax1.set_ylabel("Price", fontsize=8, color=C_MUTED)

    ax2.plot(tail.index, tail[COL_RSI], color=C_PRICE, lw=1.2)
    for level in (config.RSI_OVERBOUGHT, config.RSI_OVERSOLD):
        ax2.axhline(level, color=C_MUTED, lw=0.8, ls="--")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"RSI {config.RSI_PERIOD}", fontsize=8, color=C_MUTED)

    hist = tail["macd_hist"]
    ax3.bar(tail.index, hist, color=[C_POS if v >= 0 else C_NEG for v in hist.fillna(0)],
            width=1.0, alpha=0.55, label="Histogram")
    ax3.plot(tail.index, tail["macd"], color=C_PRICE, lw=1.2, label="MACD")
    ax3.plot(tail.index, tail["macd_signal"], color=C_SMA_S, lw=1.2, label="Signal")
    ax3.axhline(0, color=C_MUTED, lw=0.6)
    ax3.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    ax3.set_ylabel("MACD", fontsize=8, color=C_MUTED)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    for ax in (ax1, ax2, ax3):
        _style_axis(ax)
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_sentiment_chart(sentiment, path: Path) -> Path:
    """Horizontal bars: polarity x confidence per headline."""
    items = sentiment.per_headline
    fig, ax = plt.subplots(figsize=(10, 0.32 * max(len(items), 3) + 0.8))
    if items:
        sign = {"positive": 1, "negative": -1, "neutral": 0}
        vals = [sign[h.sentiment] * h.confidence for h in items]
        # Neutral items get a small stub so they remain visible.
        widths = [v if v else 0.03 for v in vals]
        colors = [C_POS if v > 0 else C_NEG if v < 0 else C_NEU for v in vals]
        labels = [(h.headline[:70] + "…") if len(h.headline) > 70 else h.headline
                  for h in items]
        y = range(len(items))[::-1]
        ax.barh(list(y), widths, color=colors, height=0.7)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7.5, color=C_INK)
        ax.axvline(0, color=C_MUTED, lw=0.8)
        ax.axvline(sentiment.aggregate_score, color=C_INK, lw=1.2, ls="--")
        ax.text(sentiment.aggregate_score, len(items) - 0.4,
                f" aggregate {sentiment.aggregate_score:+.2f}",
                fontsize=8, color=C_INK, va="bottom")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("sentiment × confidence  (−1 negative … +1 positive)",
                  fontsize=8, color=C_MUTED)
    ax.set_title("Per-headline LLM sentiment", loc="left", fontsize=11,
                 color=C_INK, fontweight="bold")
    _style_axis(ax)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _top_headlines(sentiment):
    """Top N by |polarity| x confidence; neutral items only fill remaining slots."""
    sign = {"positive": 1, "negative": -1, "neutral": 0}
    ranked = sorted(sentiment.per_headline,
                    key=lambda h: (abs(sign[h.sentiment]), h.confidence),
                    reverse=True)
    return ranked[:TOP_HEADLINES]


def _fmt_money(v):
    return "n/a" if v is None else f"${v:,.2f}"


def _fmt_pct(v):
    return "n/a" if v is None else f"{v:+.2f}%"


def _fmt_num(v, nd=2):
    return "n/a" if v is None else f"{v:,.{nd}f}"


def _context(a, b) -> dict:
    s, snap, sig = a.summary, a.snapshot, b.signal["signal"]
    return dict(
        s=s, snap=snap, sig=sig, sent=b.sentiment, src=b.signal["source"],
        model=b.llm_model, prompt_version=b.signal["prompt_version"],
        top=_top_headlines(b.sentiment), disclaimer=DISCLAIMER,
        asof=s.get("as_of"), generated=config.TODAY.isoformat(),
        money=_fmt_money, pct=_fmt_pct, num=_fmt_num,
        checks=(s.get("momentum_detail") or {}).get("checks", {}),
        rsi_col=COL_RSI, sma_s=COL_SMA_SHORT, sma_l=COL_SMA_LONG,
    )


MD_TEMPLATE = """\
# {{ s.company_name }} ({{ s.ticker }}) — Equity Research Brief
*Data as of {{ asof }} · generated {{ generated }} · model `{{ model }}` · prompts v{{ prompt_version }}*

## Recommendation: **{{ sig.signal | upper }}** ({{ sig.conviction }} conviction{% if src != 'llm' %}, rule-based fallback{% endif %})
{{ sig.justification }}

**Key drivers:** {{ sig.key_drivers | join('; ') }}
{% if sig.conflicting_signals %}**Conflicting signals:** {{ sig.conflicting_signals | join('; ') }}{% endif %}

## Company snapshot
| Metric | Value |
|---|---|
| Sector | {{ s.sector or 'n/a' }} |
| Current price | {{ money(s.current_price) }} |
| 52-week range | {{ money(s.week52_low) }} – {{ money(s.week52_high) }} ({{ pct(s.pct_from_52w_high) }} from high) |
| P/E (trailing / forward) | {{ num(s.pe_ratio) }} / {{ num(s.forward_pe) }} |
| YTD return | {{ pct(s.ytd_return_pct) }} |
| Momentum signal | {{ s.momentum_signal }} ({{ s.momentum_detail.score }}) |

## Technical outlook
| Indicator | Value |
|---|---|
| SMA {{ sma_s[4:] }} / SMA {{ sma_l[4:] }} | {{ money(snap[sma_s]) }} / {{ money(snap[sma_l]) }} |
| RSI(14) | {{ num(snap[rsi_col], 1) }} |
| MACD / signal / hist | {{ num(snap.macd, 3) }} / {{ num(snap.macd_signal, 3) }} / {{ num(snap.macd_hist, 3) }} |
| Bollinger lower / mid / upper | {{ money(snap.bb_lower) }} / {{ money(snap.bb_mid) }} / {{ money(snap.bb_upper) }} |
| Bollinger %B | {{ num(snap.bb_pct_b) }} |

![Price and indicators](chart.png)

## News sentiment: {{ sent.aggregate_label }} ({{ '%+.3f' % sent.aggregate_score }})
Scored {{ sent.n_scored }}/{{ sent.n_total }} headlines — {{ sent.counts.positive }} positive, {{ sent.counts.neutral }} neutral, {{ sent.counts.negative }} negative.

**Top {{ top | length }} headlines**
{% for h in top %}
{{ loop.index }}. *{{ h.headline }}* — **{{ h.sentiment }}** ({{ '%.2f' % h.confidence }}): {{ h.brief_reason }}
{%- endfor %}

---
**Risk disclaimer.** {{ disclaimer }}
"""

HTML_TEMPLATE = """\
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ s.ticker }} Research Brief</title>
<style>
:root{--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e0;--surface:#ffffff;--panel:#f7f6f2;
 --accent:#1f3a5f;--buy:#1d7a4c;--hold:#9a6b00;--sell:#b3261e}
*{box-sizing:border-box}
body{margin:0;background:#eceae4;color:var(--ink);
 font:14px/1.5 "Inter","Segoe UI",system-ui,-apple-system,sans-serif}
.page{max-width:900px;margin:24px auto;background:var(--surface);padding:32px 40px;
 box-shadow:0 1px 3px rgba(0,0,0,.08)}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;
 border-bottom:3px solid var(--accent);padding-bottom:12px}
h1{margin:0;font-size:24px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12px}
.rec{text-align:right}
.badge{display:inline-block;padding:6px 16px;border-radius:4px;color:#fff;font-weight:700;
 font-size:18px;letter-spacing:.06em}
.badge.Buy{background:var(--buy)}.badge.Hold{background:var(--hold)}.badge.Sell{background:var(--sell)}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--accent);
 margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi{background:var(--panel);border-radius:4px;padding:10px 12px}
.kpi .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .n{font-size:11px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}
img{width:100%;height:auto;display:block}
.just{font-size:14.5px}
ul.tags{list-style:none;padding:0;margin:6px 0;display:flex;flex-wrap:wrap;gap:6px}
ul.tags li{background:var(--panel);border-radius:12px;padding:2px 10px;font-size:12px}
ul.tags.warn li{background:#fbefe0}
ol.news{padding-left:18px;margin:6px 0}
ol.news li{margin-bottom:6px}
.pill{font-size:11px;padding:1px 8px;border-radius:10px;color:#fff}
.pill.positive{background:#2a78d6}.pill.negative{background:#e34948}.pill.neutral{background:#7d7c77}
.check{font-size:12px}.ok{color:var(--buy)}.no{color:var(--sell)}
.disclaimer{margin-top:24px;padding:10px 12px;border-left:3px solid var(--sell);
 background:#fbf3f2;font-size:11.5px;color:var(--muted)}
.fallback{color:var(--sell);font-weight:600}
@media (max-width:700px){.page{padding:20px 16px;margin:0}.grid{grid-template-columns:repeat(2,1fr)}
 .two{grid-template-columns:1fr}header{flex-direction:column;align-items:flex-start}.rec{text-align:left}}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;padding:16px}}
</style></head><body><div class="page">
<header>
 <div><h1>{{ s.company_name }} <span class="sub">({{ s.ticker }})</span></h1>
  <div class="sub">{{ s.sector or '' }} · Equity research brief · data as of {{ asof }} ·
   generated {{ generated }}</div></div>
 <div class="rec"><span class="badge {{ sig.signal }}">{{ sig.signal | upper }}</span>
  <div class="sub">{{ sig.conviction }} conviction · {{ model }}
  {% if src != 'llm' %}<br><span class="fallback">rule-based fallback</span>{% endif %}</div></div>
</header>

<h2>Company snapshot</h2>
<div class="grid">
 <div class="kpi"><div class="l">Price</div><div class="v">{{ money(s.current_price) }}</div>
  <div class="n">{{ pct(s.pct_from_52w_high) }} vs 52w high</div></div>
 <div class="kpi"><div class="l">52-week range</div>
  <div class="v" style="font-size:15px">{{ money(s.week52_low) }} – {{ money(s.week52_high) }}</div>
  <div class="n">source: {{ s.week52_source }}</div></div>
 <div class="kpi"><div class="l">P/E (ttm)</div><div class="v">{{ num(s.pe_ratio) }}</div>
  <div class="n">forward {{ num(s.forward_pe) }}</div></div>
 <div class="kpi"><div class="l">YTD return</div><div class="v">{{ pct(s.ytd_return_pct) }}</div>
  <div class="n">momentum: {{ s.momentum_signal }} ({{ s.momentum_detail.score }})</div></div>
</div>

<h2>Recommendation &amp; reasoning</h2>
<p class="just">{{ sig.justification }}</p>
<div class="two">
 <div><b style="font-size:12px">Key drivers</b>
  <ul class="tags">{% for d in sig.key_drivers %}<li>{{ d }}</li>{% endfor %}</ul></div>
 <div><b style="font-size:12px">Conflicting signals</b>
  <ul class="tags warn">{% for d in sig.conflicting_signals %}<li>{{ d }}</li>{% else %}<li>none flagged</li>{% endfor %}</ul></div>
</div>

<h2>Technical outlook</h2>
<img src="data:image/png;base64,{{ chart_b64 }}" alt="{{ s.ticker }} price with SMA 50, SMA 200 and Bollinger Bands; RSI and MACD panels">
<div class="two" style="margin-top:10px">
 <table>
  <tr><th>Indicator</th><th>Latest</th></tr>
  <tr><td>SMA 50 / SMA 200</td><td>{{ money(snap[sma_s]) }} / {{ money(snap[sma_l]) }}</td></tr>
  <tr><td>RSI (14, Wilder)</td><td>{{ num(snap[rsi_col], 1) }}</td></tr>
  <tr><td>MACD / signal / hist</td><td>{{ num(snap.macd, 2) }} / {{ num(snap.macd_signal, 2) }} / {{ num(snap.macd_hist, 2) }}</td></tr>
  <tr><td>Bollinger L / M / U</td><td>{{ num(snap.bb_lower) }} / {{ num(snap.bb_mid) }} / {{ num(snap.bb_upper) }}</td></tr>
  <tr><td>Bollinger %B</td><td>{{ num(snap.bb_pct_b) }}</td></tr>
 </table>
 <table>
  <tr><th>Momentum check</th><th>Result</th></tr>
  {% for k, v in checks.items() %}<tr><td class="check">{{ k.replace('_', ' ') }}</td>
   <td class="check {{ 'ok' if v else 'no' }}">{{ 'yes' if v else ('n/a' if v is none else 'no') }}</td></tr>{% endfor %}
 </table>
</div>

<h2>News sentiment — {{ sent.aggregate_label }} ({{ '%+.3f' % sent.aggregate_score }})</h2>
<div class="sub">{{ sent.n_scored }}/{{ sent.n_total }} headlines scored · {{ sent.counts.positive }} positive ·
 {{ sent.counts.neutral }} neutral · {{ sent.counts.negative }} negative · confidence-weighted</div>
<ol class="news">{% for h in top %}
 <li><b>{{ h.headline }}</b> <span class="pill {{ h.sentiment }}">{{ h.sentiment }} {{ '%.2f' % h.confidence }}</span>
  <div class="sub">{{ h.brief_reason }}</div></li>{% endfor %}</ol>
{% if sent_b64 %}<img src="data:image/png;base64,{{ sent_b64 }}" alt="Per-headline sentiment bars">{% endif %}

<div class="disclaimer"><b>Risk disclaimer.</b> {{ disclaimer }}</div>
</div></body></html>
"""

# Markdown output must not be HTML-escaped; the HTML env below escapes.
_env = Environment(loader=BaseLoader(), autoescape=False)


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def render(a, b, out_dir: Path = config.OUTPUT_DIR) -> dict:
    """Render chart(s), Markdown, HTML and (optionally) PDF. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = make_chart(a.prices, a.ticker, out_dir / "chart.png")
    sent_chart = make_sentiment_chart(b.sentiment, out_dir / "sentiment.png")
    ctx = _context(a, b)

    md = _env.from_string(MD_TEMPLATE).render(**ctx)
    (out_dir / "brief.md").write_text(md, encoding="utf-8")

    # Autoescape on for HTML: headlines and LLM text are untrusted input.
    html_env = Environment(loader=BaseLoader(), autoescape=True)
    html = html_env.from_string(HTML_TEMPLATE).render(
        **ctx, chart_b64=_b64(chart), sent_b64=_b64(sent_chart))
    (out_dir / "brief.html").write_text(html, encoding="utf-8")

    paths = {"chart": str(chart), "sentiment_chart": str(sent_chart),
             "markdown": str(out_dir / "brief.md"), "html": str(out_dir / "brief.html")}
    try:
        from weasyprint import HTML  # optional; needs native Pango/Cairo
        HTML(string=html).write_pdf(out_dir / "brief.pdf")
        paths["pdf"] = str(out_dir / "brief.pdf")
    except Exception as exc:
        log.info("PDF not rendered (HTML is the primary output): %s", exc)
    return paths
