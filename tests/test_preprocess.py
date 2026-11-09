from arabic_sentiment.config import PreprocessConfig
from arabic_sentiment.preprocess import STOPWORDS, NEGATORS, normalize_text, preprocess, tokenize


def test_tashkeel_and_tatweel_removed():
    assert normalize_text("جَمِيـــــل") == "جميل"


def test_letter_normalisation():
    assert normalize_text("أحمد إبراهيم آمن") == "احمد ابراهيم امن"
    assert normalize_text("مدرسة") == "مدرسه"
    assert normalize_text("على") == "علي"


def test_repeated_letters_squeezed():
    assert normalize_text("جمييييل") == "جميل"
    assert normalize_text("خرااااافي") == "خرافي"


def test_urls_mentions_removed_hashtag_words_kept():
    out = normalize_text("شكرا @support https://example.com/x www.a.com #تجربة_رائعة")
    assert "http" not in out and "@" not in out and "www" not in out
    assert out == "شكرا تجربه رايعه"


def test_emoji_become_tokens_or_dropped():
    out = normalize_text("حلو 😍 لكن 😡 🚗")
    assert "emopos" in out and "emoneg" in out
    assert "🚗" not in out
    assert "emopos" not in normalize_text("😍", PreprocessConfig(emoji_tokens=False))


def test_eastern_digits_become_ascii():
    assert normalize_text("٣٠٠ ريال") == "300 ريال"


def test_none_and_nan_safe():
    assert preprocess(None) == "" and preprocess(float("nan")) == ""


def test_stopwords_removed_negators_kept_out_of_stopwords():
    toks = tokenize("هذا الطعام في المطعم لذيذ")
    assert "هذا" not in toks and "في" not in toks
    assert not (STOPWORDS & NEGATORS)
    assert {"لا", "ما", "لم", "مو", "مب"} <= NEGATORS


def test_negation_marks_following_words_only_within_window():
    assert tokenize("ما عجبني الاكل") == ["NOT_عجبني", "NOT_الاكل"]
    assert tokenize("مو حلو الخدمة") == ["NOT_حلو", "NOT_الخدمه"]
    toks = tokenize("لم يعجبني الطعام ابدا", PreprocessConfig(negation_window=1))
    assert toks[0].startswith("NOT_") and not toks[1].startswith("NOT_")


def test_negation_scope_closed_by_punctuation_and_contrast():
    assert tokenize("ما عجبني، الاكل لذيذ") == ["NOT_عجبني", "الاكل", "لذيذ"]
    toks = tokenize("مب حلو لكن الاكل لذيذ")
    assert toks == ["NOT_حلو", "لكن", "الاكل", "لذيذ"]


def test_fixed_phrase_is_not_negation():
    toks = tokenize("ما شاء الله الاكل لذيذ")
    assert not any(t.startswith("NOT_") for t in toks)


def test_negation_can_be_disabled():
    out = tokenize("ما عجبني", PreprocessConfig(mark_negation=False, keep_negators=True))
    assert out == ["ما", "عجبني"]
