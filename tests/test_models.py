import joblib
import numpy as np
import pandas as pd
import pytest

from arabic_sentiment.config import LABELS
from arabic_sentiment.data import generate_corpus, hard_cases
from arabic_sentiment.lexicon import LexiconClassifier, lexicon_score
from arabic_sentiment.models import (build_pipeline, cross_validate_models, evaluate, load_model,
                                     predict_texts, top_ngrams, train_and_save)


def test_corpus_deterministic_and_valid():
    a, b = generate_corpus(300, seed=1), generate_corpus(300, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["label"]) == set(LABELS)
    assert a["text"].is_unique and set(a["dialect"]) == {"msa", "gulf"}
    assert set(a["domain"]) == {"restaurant", "delivery", "telecom", "ecommerce"}
    assert not generate_corpus(300, seed=2)["text"].equals(a["text"])


def test_hard_cases_shape():
    h = hard_cases()
    assert len(h) >= 30 and set(h["label"]) == set(LABELS)
    assert {"negation", "mixed"} <= set(h["kind"])


def test_lexicon_polarity_and_negation_flip():
    assert lexicon_score("الخدمة ممتازة") > 0
    assert lexicon_score("الخدمة سيئة") < 0
    assert lexicon_score("ما عجبني الاكل") < 0
    assert lexicon_score("الطلب وصل يوم الخميس") == 0
    pred = LexiconClassifier().predict(["الخدمة ممتازة", "كارثة", "الطلب وصل الخميس"])
    assert list(pred) == ["positive", "negative", "neutral"]


@pytest.mark.parametrize("name", ["logreg", "linear_svc", "complement_nb", "ensemble"])
def test_each_model_trains_and_beats_majority(corpus, name):
    m = build_pipeline(name).fit(corpus["text"], corpus["label"])
    res = evaluate(m, corpus)  # in-sample: only checks the pipeline learns
    majority = corpus["label"].value_counts(normalize=True).max()
    assert res["accuracy"] > majority + 0.2


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_pipeline("bert")


def test_cross_validation_table(corpus):
    cv = cross_validate_models(corpus, ("logreg", "complement_nb"))
    assert list(cv["model"]) == ["logreg", "complement_nb", "lexicon"]
    assert (cv["accuracy_mean"] > 0.5).all() and (cv["macro_f1_mean"] <= 1).all()


def test_evaluate_confusion_matrix_consistent(model, corpus):
    res = evaluate(model, corpus)
    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (3, 3) and cm.sum() == len(corpus)
    assert res["accuracy"] == pytest.approx(np.trace(cm) / cm.sum())


def test_hard_cases_are_harder_than_synthetic_holdout(corpus):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(corpus, test_size=0.25, stratify=corpus["label"], random_state=0)
    m = build_pipeline("logreg").fit(tr["text"], tr["label"])
    assert evaluate(m, hard_cases())["accuracy"] < evaluate(m, te)["accuracy"]


def test_train_save_load_roundtrip(tmp_path, corpus):
    path = tmp_path / "sub" / "m.joblib"
    res = train_and_save(corpus, path, name="logreg")
    assert path.exists() and res["n"] == int(round(len(corpus) * 0.2))
    loaded = load_model(path)
    texts = ["الخدمة ممتازة جدا", "التوصيل بطيء وسيء"]
    assert list(loaded.predict(texts)) == ["positive", "negative"]
    assert joblib.load(path)["labels"] == list(LABELS)


def test_predict_texts_confidence(model):
    out = predict_texts(model, ["الاكل لذيذ", "الاكل بارد"])
    assert list(out.columns) == ["text", "label", "confidence"]
    assert out["confidence"].between(0, 1).all()


def test_top_ngrams_reflect_polarity(model):
    ng = top_ngrams(model, k=15)
    pos_words = " ".join(w for w, _ in ng["positive"])
    neg_words = " ".join(w for w, _ in ng["negative"])
    assert "ممتاز" in pos_words or "رايع" in pos_words
    assert "not_" in neg_words or "كارثه" in neg_words or "بارد" in neg_words
