"""Keyword-based aspect extraction (delivery, price, service, quality).

Aspects are found by matching normalised tokens against small hand-written
keyword lists. Sentiment per aspect is predicted on the *clause* that mentions
the aspect (clauses are split on punctuation and contrast words), so
"الأكل لذيذ، لكن التوصيل بطيء" gives quality=positive and delivery=negative.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd

from .config import LABELS
from .preprocess import CONTRAST, norm_word

__all__ = ["ASPECTS", "extract_aspects", "split_clauses", "aspect_sentiment", "aspect_summary"]

_KW_SRC = {
    "delivery": "توصيل مندوب سائق شحن وصول وصل وصلني تاخر تاخير تحضير توصيله",
    "price": "سعر اسعار غالي غاليه رخيص رخيصه فاتوره الفاتوره باقه رسوم مبالغ",
    "service": "خدمه موظف موظفين الموظف دعم تعامل استرجاع عملاء استقبال",
    "quality": "جوده طعم طعام اكل منتج تغليف نظيف شبكه انترنت تغطيه مطعم وجبه جلسه",
}
ASPECTS: dict[str, frozenset[str]] = {
    a: frozenset(norm_word(w) for w in src.split()) for a, src in _KW_SRC.items()
}
_PREFIXES = ("وال", "بال", "لل", "ال", "و", "ب", "ل", "ف")
_SPLIT_RE = re.compile(r"[،,.;؛!؟?\n]+")
_CONTRAST_RE = re.compile(r"\s(?:%s)\s" % "|".join(sorted(CONTRAST, key=len, reverse=True)))


def _token_aspects(tok: str) -> set[str]:
    cands = {tok}
    for pre in _PREFIXES:
        if tok.startswith(pre) and len(tok) - len(pre) >= 3:
            cands.add(tok[len(pre):])
    return {a for a, kws in ASPECTS.items() if cands & kws}


def extract_aspects(text: str) -> list[str]:
    """Aspects mentioned in ``text`` (alphabetical, deduplicated)."""
    found: set[str] = set()
    for tok in re.sub(r"[^\w\s]", " ", norm_word(str(text))).split():
        found |= _token_aspects(tok)
    return sorted(found)


def split_clauses(text: str) -> list[str]:
    """Split on punctuation and contrast words (لكن / بس / ولكن)."""
    out: list[str] = []
    for part in _SPLIT_RE.split(str(text)):
        out.extend(p.strip() for p in _CONTRAST_RE.split(f" {part} ") if p.strip())
    return out or [str(text)]


def aspect_sentiment(texts: Iterable[str], model: Any) -> pd.DataFrame:
    """Long table ``row, aspect, label`` with one row per (text, aspect) mention."""
    texts = list(texts)
    items: list[tuple[int, str, str]] = []  # (row, aspect, clause)
    for i, text in enumerate(texts):
        for clause in split_clauses(text):
            for asp in extract_aspects(clause):
                items.append((i, asp, clause))
    if not items:
        return pd.DataFrame(columns=["row", "aspect", "label"])
    labels = model.predict([c for _, _, c in items])
    return pd.DataFrame({"row": [r for r, _, _ in items], "aspect": [a for _, a, _ in items],
                         "label": labels})


def aspect_summary(texts: Iterable[str], model: Any) -> pd.DataFrame:
    """Per-aspect counts of positive / neutral / negative mentions plus total."""
    long = aspect_sentiment(texts, model)
    if long.empty:
        return pd.DataFrame(columns=["aspect", *LABELS, "total"]).set_index("aspect")
    tab = long.groupby(["aspect", "label"]).size().unstack(fill_value=0)
    for lab in LABELS:
        if lab not in tab:
            tab[lab] = 0
    tab = tab[list(LABELS)]
    tab["total"] = tab.sum(axis=1)
    return tab.sort_values("total", ascending=False)
