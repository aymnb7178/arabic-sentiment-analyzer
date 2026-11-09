"""Hand-written MSA + Gulf sentiment lexicon baseline.

The lexicon is deliberately small (a few dozen entries per polarity) and was
written by hand for this project. It works on preprocessed tokens, so negation
(``NOT_`` prefix) simply flips the sign of the word score.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import LABELS
from .preprocess import CONTRAST, norm_word, tokenize

__all__ = ["LEXICON", "LexiconClassifier", "lexicon_score"]

_POS = """
ممتاز رائع جيد مميز جميل حلو طيب لذيذ سريع نظيف محترم متعاون مناسب معقول رخيص فخم راقي مرتب منظم
تمام خرافي يجنن كشخه محترف احسن افضل شكرا اعجبني عجبني ينصح انصح انصحكم يستاهل تستاهل استمتعت
سعيد مبسوط راضي رضا وفقكم الله يعطيكم العافيه ماشاءالله تباركالله ممتازه رائعه جيده مميزه طازج ساخن
emopos ثابته احترافي ابداع مبدع
"""
_NEG = """
سيء سيئه رديء ضعيف ضعيفه مزعج وحش وحشه بايخ زبال زباله خايس كارثه بطيء بطيئه متاخر متاخره غالي غاليه مرتفع
مبالغ فوضوي بارد باردة قديم متسخ تعبان مخيب مخيبه ندمت ندم ضيعت تضايقت تاخير تاخر مكسور تالف نصب
سرقه احتيال مشكله مشاكل اسوا اسوء خاب emoneg يقطع تقطع انقطاع هدر مضيعه خطا اخطاء يعاب
"""
_STRONG = {"كارثه", "اسوا", "اسوء", "احتيال", "نصب", "زباله", "زبال", "خرافي", "يجنن", "ممتاز"}

LEXICON: dict[str, float] = {}
for _w in _POS.split():
    LEXICON[norm_word(_w)] = 1.0
for _w in _NEG.split():
    LEXICON[norm_word(_w)] = -1.0
for _w in _STRONG:
    LEXICON[norm_word(_w)] = 2.0 * (1 if LEXICON.get(norm_word(_w), 0) > 0 else -1)

_PREFIXES = ("وال", "بال", "لل", "ال", "و", "ب", "ل", "ف")


def _lookup(word: str) -> float:
    """Score a single token, trying light prefix stripping."""
    if word in LEXICON:
        return LEXICON[word]
    for pre in _PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 3 and word[len(pre):] in LEXICON:
            return LEXICON[word[len(pre):]]
    return 0.0


def lexicon_score(text: str) -> float:
    """Signed sentiment score; words after the last contrast word count double."""
    tokens = tokenize(text)
    last = max((i for i, t in enumerate(tokens) if t in CONTRAST), default=-1)
    total = 0.0
    for i, tok in enumerate(tokens):
        sign = 1.0
        if tok.startswith("NOT_"):
            tok, sign = tok[4:], -0.8
        total += sign * _lookup(tok) * (2.0 if i > last >= 0 else 1.0)
    return total


class LexiconClassifier:
    """Rule-based baseline with the scikit-learn ``predict`` interface."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.classes_ = np.array(LABELS)

    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "LexiconClassifier":
        """No-op (kept for API symmetry)."""
        return self

    def predict(self, X: Iterable[str]) -> np.ndarray:
        """Predict labels for raw texts."""
        out = []
        for text in X:
            s = lexicon_score(text)
            out.append("positive" if s >= self.threshold else "negative" if s <= -self.threshold else "neutral")
        return np.array(out)
