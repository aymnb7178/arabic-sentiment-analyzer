"""Self-contained HTML dashboard (inline CSS, inline base64 PNGs, no scripts)."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import charts
from .aspects import aspect_summary
from .config import LABELS
from .models import top_ngrams

__all__ = ["build_dashboard"]

_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#222b36}
main{max-width:1000px;margin:0 auto;padding:24px 16px}
h1{margin:0 0 4px}h2{margin-top:32px;border-bottom:1px solid #d8dde3;padding-bottom:6px}
.note{background:#fff7e0;border:1px solid #e9c46a;padding:10px 14px;border-radius:6px}
.cards{display:flex;gap:12px;flex-wrap:wrap}.card{background:#fff;border:1px solid #e1e5ea;border-radius:8px;padding:12px 18px;min-width:130px}
.card b{display:block;font-size:1.6em}
img{max-width:100%;height:auto;background:#fff;border:1px solid #e1e5ea;border-radius:6px}
table{border-collapse:collapse;background:#fff}td,th{border:1px solid #e1e5ea;padding:4px 10px;text-align:left}
.ar{direction:rtl;text-align:right;font-size:1.1em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
"""


def _img(fig: Any, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{charts.png_b64(charts.fig_to_png(fig))}">'


def _ngram_table(label: str, items: list[tuple[str, float]]) -> str:
    rows = "".join(f'<tr><td class="ar">{html.escape(w)}</td><td>{s:.2f}</td></tr>' for w, s in items)
    return f"<div><h3>{label}</h3><table><tr><th>n-gram</th><th>weight</th></tr>{rows}</table></div>"


def build_dashboard(df: pd.DataFrame, model: Any, *, title: str = "Arabic Sentiment Dashboard",
                    ngram_model: Any | None = None, synthetic: bool = True) -> str:
    """Render the dashboard for ``df`` (columns ``text`` and optionally ``date``).

    Predictions are made with ``model``. ``ngram_model`` must be a fitted linear
    pipeline (logreg / linear_svc); if omitted, ``model`` is used when possible.
    """
    df = df.copy()
    df["pred"] = model.predict(df["text"])
    share = df["pred"].value_counts(normalize=True)
    asp = aspect_summary(df["text"], model)
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(df):,} texts</p>"]
    if synthetic:
        parts.append('<p class="note"><b>Synthetic data.</b> Texts were generated from templates for demonstration. '
                     "Percentages describe the generator, not real customers.</p>")
    parts.append('<div class="cards">' + "".join(
        f'<div class="card"><b>{share.get(l, 0) * 100:.1f}%</b>{l}</div>' for l in LABELS) +
        f'<div class="card"><b>{len(df):,}</b>texts</div></div>')
    parts += ["<h2>Sentiment share</h2>", _img(charts.sentiment_share_chart(df["pred"]), "Sentiment share")]
    if "date" in df and df["date"].notna().any():
        parts += ["<h2>Monthly trend</h2>", _img(charts.monthly_trend_chart(df["date"], df["pred"]), "Monthly sentiment trend")]
    if not asp.empty:
        rows = "".join(f"<tr><td>{a}</td>" + "".join(f"<td>{int(r[l])}</td>" for l in LABELS) +
                       f"<td>{int(r['total'])}</td></tr>" for a, r in asp.iterrows())
        parts += ["<h2>Top aspects</h2>",
                  "<p>Keyword-matched; sentiment is predicted on the clause that mentions the aspect.</p>",
                  _img(charts.aspect_chart(asp), "Aspects by sentiment"),
                  f"<table><tr><th>aspect</th>{''.join(f'<th>{l}</th>' for l in LABELS)}<th>total</th></tr>{rows}</table>"]
    try:
        ng = top_ngrams(ngram_model or model, k=8)
        parts += ["<h2>Top n-grams per class</h2><p>Highest logistic-regression weights among word 1-2-grams "
                  "(<code>not_</code> marks a negated word).</p>",
                  '<div class="grid">' + "".join(_ngram_table(c, ng[c]) for c in LABELS if c in ng) + "</div>"]
    except (ValueError, KeyError, AttributeError):
        pass
    body = "\n".join(parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>")
