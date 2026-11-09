"""Arabic text preprocessing for sentiment analysis.

Pipeline (all steps are rule-based and deterministic):

1. Unicode NFKC normalisation (also folds Arabic presentation forms).
2. URL / e-mail / @mention removal; hashtags keep their words.
3. Emoji handling: sentiment emojis become ``emopos`` / ``emoneg`` tokens,
   everything else is dropped.
4. Tashkeel (diacritics) and tatweel (kashida) removal.
5. Letter normalisation (alef variants, ya/alef-maqsura, ta-marbuta, ...).
6. Repeated-letter squeeze (``جمييييل`` -> ``جميل``).
7. Protection of fixed phrases (``ما شاء الله``) that contain a negator.
8. Tokenisation with clause boundaries, stop-word removal and simple negation
   marking (``ما عجبني`` -> ``NOT_عجبني``).
"""

from __future__ import annotations

import re
import unicodedata

from .config import PreprocessConfig

__all__ = [
    "normalize_text",
    "basic_tokens",
    "tokenize",
    "preprocess",
    "STOPWORDS",
    "NEGATORS",
    "CONTRAST",
    "norm_word",
]

DEFAULT_CONFIG = PreprocessConfig()

_TASHKEEL_RE = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]"
)
_TATWEEL = "ـ"
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_PUNCT_RE = re.compile(r"[^\w\s]|_")

_LETTER_MAP = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ی": "ي", "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ک": "ك", "گ": "ك", "چ": "ك",
        "پ": "ب", "ڤ": "ف",
    }
)
_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"
)

POS_EMOJI = set("😍😊😀😁😃😄🥰😘👍👌❤♥💖💕🔥👏🙏🤩😎✨💯🌹")
NEG_EMOJI = set("😡😠😤👎💔😢😭🤬😞😒🤮😫😩😖🙄😔😕")
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿⭐⭕]")
_INVISIBLE = {"️", "‍", "‎", "‏", "‪", "‫", "‬"}


def norm_word(word: str) -> str:
    """Apply the letter-level normalisation to a single word (no tokenising)."""
    word = _TASHKEEL_RE.sub("", word).replace(_TATWEEL, "")
    return word.translate(_LETTER_MAP)


# Fixed phrases that contain a negator or contrast word but are not negation.
# Patterns operate on already-normalised text.
_PROTECTED = [
    (re.compile(r"\bما شاء الله\b"), "ماشاءالله"),
    (re.compile(r"\bتبارك الله\b"), "تباركالله"),
    (re.compile(r"\bان شاء الله\b"), "انشاءالله"),
    (re.compile(r"\bلا باس\b"), "لاباس"),
    (re.compile(r"\bما قصر\w*"), "ماقصروا"),
    (re.compile(r"\b(?:مو|مب|مش|ماهو) بس\b"), "موبس"),
]

_STOPWORD_SRC = """
في من الى على عن ان انه انها انهم هذا هذه هذي هاذي هاذا ذلك تلك هو هي هم هن نحن انا انت انتي انتم
كان كانت كانوا يكون تكون قد ثم او و ايضا كما حتى عند بعد قبل مع هنا هناك هنالك الذي التي الذين اللذين
لقد اي اذا اذ بين حيث لان لانه لانها منذ خلال بدون دون ضد لدى لدي لها له لهم لنا لي لك
فيه فيها فيهم منه منها منهم عليه عليها عنه عنها به بها بهم
احنا اللي الي شي شيء ذا ذي كذا هالـ هالشي ايش وش ليش علشان عشان لين الحين الان ذلحين
هل ماذا لماذا كيف متى اين كم ماذا هيك
او ام اما ان انما كلا
يا ايها
هم هما
هذان هاتان
عندي عندنا عندك
معي معنا معك
و ف
"""

CONTRAST = frozenset({"لكن", "لاكن", "ولكن", "بس"})
NEGATORS = frozenset(
    norm_word(w)
    for w in (
        "لا ما لم لن مو مب مش ليس ليست ليسوا ماهو ماهي ماني مانا مانك غير"
    ).split()
)
STOPWORDS = frozenset(
    w for w in (norm_word(x) for x in _STOPWORD_SRC.split()) if w not in NEGATORS
) - CONTRAST


def normalize_text(text: object, cfg: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Normalise raw Arabic text without tokenising or removing stop-words."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", text)

    out: list[str] = []
    for ch in text:
        if ch in _INVISIBLE:
            continue
        if ch in POS_EMOJI:
            out.append(" emopos " if cfg.emoji_tokens else " ")
        elif ch in NEG_EMOJI:
            out.append(" emoneg " if cfg.emoji_tokens else " ")
        elif _EMOJI_RE.match(ch):
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out).translate(_DIGIT_MAP).lower()

    text = _TASHKEEL_RE.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_LETTER_MAP)
    text = _REPEAT_RE.sub(r"\1", text)
    for pattern, repl in _PROTECTED:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


def basic_tokens(text: object, cfg: PreprocessConfig = DEFAULT_CONFIG) -> list[str]:
    """Normalised tokens with ``|`` marking punctuation (clause) boundaries."""
    norm = normalize_text(text, cfg)
    norm = _PUNCT_RE.sub(" | ", norm)
    tokens: list[str] = []
    for tok in norm.split():
        if tok.isdigit():
            tok = "NUM"
        if tok == "|" and tokens and tokens[-1] == "|":
            continue
        tokens.append(tok)
    while tokens and tokens[-1] == "|":
        tokens.pop()
    return tokens


def tokenize(text: object, cfg: PreprocessConfig = DEFAULT_CONFIG) -> list[str]:
    """Tokenise, drop stop-words and mark negation scope.

    A negator opens a scope covering the next ``cfg.negation_window`` content
    words (stop-words are skipped, not counted). Punctuation and contrast words
    (لكن / بس) close the scope.
    """
    out: list[str] = []
    pending = 0
    for tok in basic_tokens(text, cfg):
        if tok == "|":
            pending = 0
            continue
        if tok in CONTRAST:
            pending = 0
            out.append(tok)
            continue
        if cfg.mark_negation and tok in NEGATORS:
            pending = cfg.negation_window
            if cfg.keep_negators:
                out.append(tok)
            continue
        if cfg.remove_stopwords and tok in STOPWORDS:
            continue
        if pending > 0:
            out.append("NOT_" + tok)
            pending -= 1
        else:
            out.append(tok)
    return out


def preprocess(text: object, cfg: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Return the fully preprocessed text as a space-joined token string."""
    return " ".join(tokenize(text, cfg))
