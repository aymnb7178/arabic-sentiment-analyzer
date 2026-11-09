import json

import pandas as pd

from arabic_sentiment import charts
from arabic_sentiment.aspects import ASPECTS, aspect_summary, extract_aspects, split_clauses
from arabic_sentiment.cli import main
from arabic_sentiment.report import build_dashboard


def test_extract_aspects():
    assert extract_aspects("التوصيل بطيء والسعر غالي") == ["delivery", "price"]
    assert extract_aspects("خدمة العملاء ممتازة") == ["service"]
    assert extract_aspects("الطلب وصل") == ["delivery"]
    assert extract_aspects("مرحبا بكم") == []
    assert set(ASPECTS) == {"delivery", "price", "service", "quality"}


def test_split_clauses_on_contrast_and_punctuation():
    assert split_clauses("الأكل لذيذ، لكن التوصيل بطيء") == ["الأكل لذيذ", "التوصيل بطيء"]
    assert len(split_clauses("الطعام جيد بس السعر غالي")) == 2


def test_aspect_summary_counts(model):
    s = aspect_summary(["الطعام لذيذ ممتاز، لكن التوصيل بطيء ومتأخر", "الأسعار غالية"], model)
    assert s.loc["quality", "positive"] == 1
    assert s.loc["delivery", "negative"] == 1
    assert s.loc["price", "negative"] == 1
    assert (s["total"] == s[["negative", "neutral", "positive"]].sum(axis=1)).all()


def test_charts_return_png_bytes(model):
    pngs = [charts.fig_to_png(charts.confusion_chart([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            charts.fig_to_png(charts.sentiment_share_chart(pd.Series(["positive", "negative", "positive"])))]
    assert all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)


def test_dashboard_is_self_contained(corpus, model):
    html = build_dashboard(corpus, model)
    assert html.startswith("<!doctype html>")
    assert html.count("data:image/png;base64,") >= 3
    assert "<script" not in html and "http://" not in html and "https://" not in html
    assert "Synthetic data" in html and 'class="ar"' in html


def test_cli_train_predict_batch_evaluate(tmp_path, capsys):
    mp = str(tmp_path / "m.joblib")
    assert main(["train", "--n", "500", "--model", "logreg", "--model-path", mp]) == 0
    assert "accuracy=" in capsys.readouterr().out

    assert main(["predict", "التوصيل سريع والخدمة ممتازة", "--model-path", mp]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["label"] == "positive" and "delivery" in out["aspects"]

    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["الاكل لذيذ", "الاكل سيء", "هل يوجد فرع في جدة؟"], "id": [1, 2, 3]}).to_csv(src, index=False)
    assert main(["batch", str(src), "--model-path", mp]) == 0
    res = pd.read_csv(tmp_path / "in_predictions.csv")
    assert list(res["label"][:2]) == ["positive", "negative"] and "confidence" in res

    assert main(["evaluate", "--n", "400", "--models", "logreg"]) == 0
    assert "Hard cases" in capsys.readouterr().out


def test_cli_missing_model_returns_error(tmp_path, capsys):
    assert main(["predict", "نص", "--model-path", str(tmp_path / "none.joblib")]) == 1
