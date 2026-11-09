"""SYNTHETIC Arabic sentiment corpus generator and a hand-written hard-case set.

Everything produced here is artificial: sentences are assembled from templates
and vocabulary lists written for this project (MSA plus Saudi/Gulf flavour).
Because the generator reuses a small vocabulary, scores measured on this corpus
are an *optimistic upper bound* and say nothing about real social-media text.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

__all__ = ["generate_corpus", "hard_cases", "DOMAINS"]

# (text, gender, aspect); gender: "m" or "f" for adjective agreement.
Subject = tuple[str, str, str]

DOMAINS: dict[str, list[Subject]] = {
    "restaurant": [
        ("الطعام", "m", "quality"), ("الأكل", "m", "quality"), ("الطعم", "m", "quality"),
        ("الخدمة", "f", "service"), ("تعامل الموظفين", "m", "service"),
        ("الأسعار", "f", "price"), ("المطعم", "m", "quality"), ("الجلسة", "f", "quality"),
        ("وجبة الغداء", "f", "quality"), ("وقت التحضير", "m", "delivery"),
    ],
    "delivery": [
        ("التوصيل", "m", "delivery"), ("المندوب", "m", "delivery"), ("السائق", "m", "delivery"),
        ("وقت الوصول", "m", "delivery"), ("تغليف الطلب", "m", "quality"),
        ("التطبيق", "m", "quality"), ("رسوم التوصيل", "f", "price"), ("خدمة التوصيل", "f", "service"),
    ],
    "telecom": [
        ("الشبكة", "f", "quality"), ("سرعة الانترنت", "f", "quality"), ("خدمة العملاء", "f", "service"),
        ("الباقة", "f", "price"), ("الفاتورة", "f", "price"), ("الدعم الفني", "m", "service"),
        ("التغطية", "f", "quality"), ("تعامل موظف الاستقبال", "m", "service"),
    ],
    "ecommerce": [
        ("المنتج", "m", "quality"), ("جودة المنتج", "f", "quality"), ("التغليف", "m", "quality"),
        ("الشحن", "m", "delivery"), ("سعر المنتج", "m", "price"),
        ("خدمة ما بعد البيع", "f", "service"), ("الاسترجاع", "m", "service"), ("الموقع", "m", "quality"),
    ],
}

# Adjectives: (masculine, feminine). Generic lists are shared across aspects.
_POS_GENERIC_MSA = [("ممتاز", "ممتازة"), ("رائع", "رائعة"), ("جيد", "جيدة"), ("مميز", "مميزة"),
                    ("ممتاز جدا", "ممتازة جدا"), ("محترف", "محترفة"), ("راقي", "راقية")]
_POS_GENERIC_GULF = [("حلو", "حلوة"), ("طيب", "طيبة"), ("خرافي", "خرافية"), ("تمام", "تمام"),
                     ("يجنن", "يجنن"), ("فخم", "فخمة"), ("كشخة", "كشخة"), ("ولا غلطة", "ولا غلطة")]
_NEG_GENERIC_MSA = [("سيء", "سيئة"), ("رديء", "رديئة"), ("ضعيف", "ضعيفة"), ("مخيب للآمال", "مخيبة للآمال"),
                    ("مزعج", "مزعجة"), ("غير مقبول", "غير مقبولة")]
_NEG_GENERIC_GULF = [("وحش", "وحشة"), ("بايخ", "بايخة"), ("زبالة", "زبالة"), ("مو طبيعي بالسوء", "مو طبيعية بالسوء"),
                     ("خايس", "خايسة"), ("كارثة", "كارثة")]

_ASPECT_POS = {
    "delivery": [("سريع", "سريعة"), ("في الوقت المحدد", "في الوقت المحدد"), ("منظم", "منظمة")],
    "price": [("مناسب", "مناسبة"), ("معقول", "معقولة"), ("رخيص", "رخيصة"), ("ممتاز مقابل الجودة", "ممتازة مقابل الجودة")],
    "service": [("محترم", "محترمة"), ("متعاون", "متعاونة"), ("سريع الرد", "سريعة الرد")],
    "quality": [("لذيذ", "لذيذة"), ("نظيف", "نظيفة"), ("ممتاز", "ممتازة"), ("مرتب", "مرتبة"), ("ثابتة", "ثابتة")],
}
_ASPECT_NEG = {
    "delivery": [("بطيء", "بطيئة"), ("متأخر", "متأخرة"), ("فوضوي", "فوضوية")],
    "price": [("غالي", "غالية"), ("مرتفع", "مرتفعة"), ("مبالغ فيه", "مبالغ فيها"), ("نار", "نار")],
    "service": [("بارد", "باردة"), ("غير متعاون", "غير متعاونة"), ("متجاهل", "متجاهلة")],
    "quality": [("بارد", "باردة"), ("قديم", "قديمة"), ("متسخ", "متسخة"), ("تعبان", "تعبانة")],
}

_INT_MSA = ["", "", "جدا", "للغاية"]
_INT_GULF = ["", "", "مرة", "وايد", "كثير", "حيل"]
_OPEN_MSA = ["", "", "في الحقيقة ", "للأمانة ", "بشكل عام ", "تجربتي: "]
_OPEN_GULF = ["", "", "والله ", "صراحة ", "بصراحة ", "يا جماعة ", "الصدق "]
_CLOSE_POS = ["أنصح به بشدة", "سأكرر التجربة", "وراح أطلب مرة ثانية", "يستاهل التجربة", "شكرا لكم", "الله يعطيكم العافية", "وفقكم الله"]
_CLOSE_NEG = ["لا أنصح به", "لن أكرر التجربة", "ما راح أطلب مرة ثانية", "ضيعت فلوسي", "ندمت", "تجربة مخيبة", "ما انصح فيه"]
_EMO_POS = ["😍", "👍", "❤", "👏", "🔥", "😊"]
_EMO_NEG = ["😡", "👎", "😤", "😒", "💔", "🤬"]

_STORES = ["مطعم الواحة", "متجر النخيل", "كافيه الريف", "سوبرماركت الحي", "مطعم السفرة", "متجر الندى"]
_DAYS = ["الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]
_CITIES = ["الرياض", "جدة", "مكة", "الدمام", "الطائف", "المدينة"]
_ITEMS = ["الوجبة", "الباقة الشهرية", "الجهاز", "الطلب", "الاشتراك", "القسيمة"]
_DEPTS = ["قسم الدعم", "قسم المبيعات", "الفرع الرئيسي", "إدارة الحسابات"]
_NEUTRAL_ADJ = [("عادي", "عادية"), ("مقبول", "مقبولة"), ("متوسط", "متوسطة"), ("زي غيره", "زي غيرها")]

_CONTRAST = ["لكن", "بس", "ولكن"]


def _adj(pair: tuple[str, str], gender: str) -> str:
    return pair[0] if gender == "m" else pair[1]


def _pick_adj(rng: random.Random, aspect: str, polarity: str, gulf: bool) -> tuple[str, str]:
    generic = (_POS_GENERIC_GULF if gulf else _POS_GENERIC_MSA) if polarity == "pos" else (
        _NEG_GENERIC_GULF if gulf else _NEG_GENERIC_MSA)
    specific = (_ASPECT_POS if polarity == "pos" else _ASPECT_NEG)[aspect]
    pool = specific * 2 + generic
    return rng.choice(pool)


def _intens(rng: random.Random, gulf: bool) -> str:
    return rng.choice(_INT_GULF if gulf else _INT_MSA)


def _clause(rng: random.Random, subj: Subject, polarity: str, gulf: bool) -> str:
    """Single opinion clause about ``subj``; negated opposite adjectives appear too."""
    text, gender, aspect = subj
    r = rng.random()
    intens = _intens(rng, gulf)
    if r < 0.62:  # direct adjective
        a = _adj(_pick_adj(rng, aspect, polarity, gulf), gender)
        return f"{text} {a} {intens}".strip()
    if r < 0.84:  # negated opposite adjective
        opp = "neg" if polarity == "pos" else "pos"
        a = _adj(_pick_adj(rng, aspect, opp, gulf), gender)
        # keep the negated adjective a simple one-word form
        a = a.split()[0] if a.split()[0] not in {"غير", "في", "ولا", "مو"} else a
        neg = rng.choice(["مو", "مب", "ما هو"]) if gulf else rng.choice(["ليس", "لم يكن", "لا يعتبر"])
        if neg == "لم يكن":
            return f"لم يكن {text} {a}"
        if neg == "لا يعتبر":
            return f"لا يعتبر {text} {a}"
        return f"{text} {neg} {a}"
    # verb-style
    if polarity == "pos":
        return rng.choice([f"عجبني {text}", f"أعجبني {text} {intens}".strip(), f"استمتعت ب{text}",
                           f"ما في اي مشكلة في {text}" if gulf else f"لا يوجد اي مشكلة في {text}"])
    return rng.choice([f"ما عجبني {text}", f"لم يعجبني {text}", f"ما ارتحت ل{text}" if gulf else f"لم أرتح ل{text}",
                       f"تضايقت من {text}"])


def _neutral(rng: random.Random, domain: str, gulf: bool) -> str:
    subj = rng.choice(DOMAINS[domain])
    n = rng.randint(2, 29)
    num = rng.randint(1000, 9999)
    opts = [
        f"طلبت من {rng.choice(_STORES)} يوم {rng.choice(_DAYS)}",
        f"استلمت {rng.choice(_ITEMS)} الساعة {n % 12 + 1}",
        f"هل يوجد {subj[0]} في {rng.choice(_CITIES)}؟",
        f"الطلب رقم {num} وصل يوم {rng.choice(_DAYS)}",
        f"سعر {rng.choice(_ITEMS)} {n * 10} ريال",
        f"تم تحويل طلبي الى {rng.choice(_DEPTS)}",
        f"متى يفتح {rng.choice(_STORES)}؟" if not gulf else f"وش وقت يفتح {rng.choice(_STORES)}؟",
        f"فتحت حساب جديد في التطبيق يوم {rng.choice(_DAYS)}",
        f"{subj[0]} {_adj(rng.choice(_NEUTRAL_ADJ), subj[1])}",
        f"{subj[0]} {_adj(rng.choice(_NEUTRAL_ADJ), subj[1])} لا اكثر ولا اقل",
        f"{subj[0]} ليس سيئا وليس ممتازا" if not gulf else f"{subj[0]} مو وحش ولا ممتاز",
        f"طلبت مرتين هذا الاسبوع من {rng.choice(_STORES)}",
        f"اريد معرفة {subj[0]} قبل الاشتراك",
    ]
    return rng.choice(opts)


def _decorate(rng: random.Random, text: str, polarity: str) -> str:
    """Add surface noise that preprocessing must handle (emoji, tatweel, elongation, URLs)."""
    if rng.random() < 0.10:
        pool = _EMO_POS if polarity == "pos" else _EMO_NEG if polarity == "neg" else []
        if pool:
            text += " " + rng.choice(pool)
    if rng.random() < 0.05:
        text += " " + rng.choice(["@support", "https://example.com/x", "#تجربتي"])
    if rng.random() < 0.06:
        words = text.split()
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 3 and w[1] not in "ـ":
            words[i] = w[:2] + w[1] * 3 + w[2:]
        text = " ".join(words)
    if rng.random() < 0.04:
        words = text.split()
        i = rng.randrange(len(words))
        if len(words[i]) > 2:
            words[i] = words[i][0] + "ـ" + words[i][1:]
        text = " ".join(words)
    return text


def generate_corpus(n: int = 1800, seed: int = 42, start: str = "2025-01-01",
                    months: int = 12) -> pd.DataFrame:
    """Create a SYNTHETIC labelled corpus.

    Columns: ``id, text, label, domain, dialect, date``. About 22% of rows are neutral; the rest are positive/negative.
    Mixed-sentiment rows (roughly 14% of all rows) carry the polarity of the
    clause after the contrast word.
    Dates are assigned with a mild upward positive drift so the monthly-trend
    chart has something to show; the drift is an artefact of the generator.
    """
    rng = random.Random(seed)
    base = date.fromisoformat(start)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < n and attempts < n * 30:
        attempts += 1
        domain = rng.choice(list(DOMAINS))
        gulf = rng.random() < 0.5
        month = rng.randrange(months)
        day_offset = month * 30 + rng.randrange(28)
        d = base + timedelta(days=day_offset)
        p_pos = 0.35 + 0.30 * month / max(months - 1, 1)  # P(positive | non-neutral): 0.35 -> 0.65
        r = rng.random()
        if r < 0.22:
            label = "neutral"
        else:
            label = "positive" if rng.random() < p_pos else "negative"
        opener = rng.choice(_OPEN_GULF if gulf else _OPEN_MSA)
        if label == "neutral":
            text = _neutral(rng, domain, gulf)
        else:
            pol = "pos" if label == "positive" else "neg"
            s1, s2 = rng.sample(DOMAINS[domain], 2)
            shape = rng.random()
            if shape < 0.55:
                text = opener + _clause(rng, s1, pol, gulf)
            elif shape < 0.70:
                text = f"{opener}{_clause(rng, s1, pol, gulf)}، {rng.choice(_CLOSE_POS if pol == 'pos' else _CLOSE_NEG)}"
            elif shape < 0.82:
                text = f"{opener}{_clause(rng, s1, pol, gulf)} و{_clause(rng, s2, pol, gulf)}"
            else:  # mixed: label follows clause after contrast word
                other = "neg" if pol == "pos" else "pos"
                text = (f"{opener}{_clause(rng, s1, other, gulf)}، {rng.choice(_CONTRAST)} "
                        f"{_clause(rng, s2, pol, gulf)}")
            text = _decorate(rng, text, pol)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": label, "domain": domain,
                     "dialect": "gulf" if gulf else "msa", "date": d.isoformat()})
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    return df.sort_values("date").reset_index(drop=True).assign(id=lambda x: range(1, len(x) + 1))


def hard_cases() -> pd.DataFrame:
    """Hand-written difficult sentences (negation, mixed sentiment, idioms, light sarcasm).

    Written separately from the generator vocabulary on purpose. Labels follow
    the dominant / final opinion; a few are debatable, which is part of the point.
    """
    rows: list[tuple[str, str, str]] = [
        # negation of positive
        ("ما كان الأكل لذيذ ابدا", "negative", "negation"),
        ("الخدمة مو زينة والله", "negative", "negation"),
        ("لا أعتقد أن المنتج يستحق هذا السعر", "negative", "negation"),
        ("ما انصح احد يجرب هالمطعم", "negative", "negation"),
        ("التوصيل ما وصل في الوقت", "negative", "negation"),
        ("لم تكن الباقة مناسبة لاحتياجي", "negative", "negation"),
        ("مب معقولة هالاسعار", "negative", "negation"),
        ("الشبكة ضعيفة ولا تتحمل اكثر من جهاز", "negative", "negation"),
        # negation of negative
        ("ما في تأخير ولا اخطاء في الطلب", "positive", "negation"),
        ("ما لقيت اي مشكلة في الجهاز", "positive", "negation"),
        ("الأكل مو سيء بالعكس لذيذ", "positive", "negation"),
        ("لا يوجد ما يعاب على الخدمة", "positive", "negation"),
        # mixed sentiment
        ("الاكل لذيذ بس الخدمة بطيئة جدا", "negative", "mixed"),
        ("الطعم ممتاز لكن السعر مبالغ فيه", "neutral", "mixed"),
        ("التوصيل تأخر ساعة لكن المندوب كان محترم والأكل وصل ساخن", "positive", "mixed"),
        ("المنتج حلو بس التغليف سيء", "neutral", "mixed"),
        ("الباقة رخيصة لكن الشبكة تقطع كثير", "negative", "mixed"),
        ("في البداية كنت متحمس لكن بعد التجربة ندمت", "negative", "mixed"),
        ("الأسعار غالية لكنها تستاهل بصراحة", "positive", "mixed"),
        # dialect / idiom
        ("تجربة تجنن والله ما شاء الله", "positive", "idiom"),
        ("يا شيخ خرااافي المطعم هذا", "positive", "idiom"),
        ("ما قصروا الله يعطيهم العافية", "positive", "idiom"),
        ("الموظف تعامله زفت", "negative", "idiom"),
        ("بصراحة فلوسي رايحة هدر", "negative", "idiom"),
        ("وصلني الطلب وانا جوعان ومات الجوع قبله", "negative", "idiom"),
        # sarcasm (very hard for bag-of-words)
        ("برافو عليكم تأخرتوا ثلاث ساعات", "negative", "sarcasm"),
        ("ممتاز جدا استلمت المنتج مكسور", "negative", "sarcasm"),
        ("خدمة عملاء رائعة تنتظر ساعة وما يردون", "negative", "sarcasm"),
        # neutral / factual with sentiment-looking words
        ("هل الخدمة ممتازة في فرع جدة؟", "neutral", "neutral"),
        ("اشتريت الجهاز يوم الخميس وسعره ٣٠٠ ريال", "neutral", "neutral"),
        ("الطلب وصل الساعة الخامسة", "neutral", "neutral"),
        ("الأكل عادي مو احسن شي ولا اسوأ شي", "neutral", "neutral"),
        ("اتصلت بخدمة العملاء وحولوني على القسم المختص", "neutral", "neutral"),
        # plain but outside generator vocabulary
        ("ما توقعت اصلا ان الجودة تكون بهالمستوى العالي", "positive", "plain"),
        ("اسوأ تجربة مرت علي في حياتي", "negative", "plain"),
        ("تجربة جميلة وسأعود مجددا", "positive", "plain"),
        ("الحمد لله وصل سالم وبسرعة", "positive", "plain"),
        ("خاب ظني في هذا المتجر", "negative", "plain"),
        ("المكان هادئ والعاملين لطيفين", "positive", "plain"),
        ("المنتج لا يشبه الصور اطلاقا", "negative", "plain"),
        ("طلبت مرتين وفي كل مرة نفس المشكلة", "negative", "plain"),
    ]
    return pd.DataFrame(rows, columns=["text", "label", "kind"])


def iter_texts(df: pd.DataFrame) -> Iterable[str]:
    """Yield the text column (small convenience for callers)."""
    return iter(df["text"].tolist())
