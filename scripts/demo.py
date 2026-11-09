"""End-to-end demo: SYNTHETIC corpus -> CV -> train -> hard cases -> charts -> HTML dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from arabic_sentiment import charts  # noqa: E402
from arabic_sentiment.aspects import aspect_summary  # noqa: E402
from arabic_sentiment.config import Config  # noqa: E402
from arabic_sentiment.data import generate_corpus, hard_cases  # noqa: E402
from arabic_sentiment.models import (build_pipeline, cross_validate_models, dialect_transfer,  # noqa: E402
                                     evaluate, train_and_save, load_model)
from arabic_sentiment.report import build_dashboard  # noqa: E402

DOCS = ROOT / "docs"


def cm_text(res: dict) -> str:
    lines = ["            " + "".join(f"{l:>10}" for l in res["labels"])]
    for lab, row in zip(res["labels"], res["confusion_matrix"]):
        lines.append(f"{lab:<12}" + "".join(f"{v:>10}" for v in row))
    return "\n".join(lines)


def main() -> None:
    import logging
    logging.disable(logging.CRITICAL)
    cfg = Config()
    df = generate_corpus(n=1800, seed=42)
    DOCS.mkdir(exist_ok=True)
    df.to_csv(DOCS / "sample_corpus.csv", index=False, encoding="utf-8-sig")
    print(f"SYNTHETIC corpus: {len(df)} rows, labels {df.label.value_counts().to_dict()}, "
          f"dialects {df.dialect.value_counts().to_dict()}")

    cv = cross_validate_models(df, cfg=cfg)
    print("\n5-fold CV (synthetic, optimistic):")
    print(cv.round(3).to_string(index=False))
    charts.fig_to_png(charts.model_comparison_chart(cv), DOCS / "model_comparison.png")

    metrics = train_and_save(df, ROOT / "models" / "sentiment.joblib", name="ensemble", cfg=cfg)
    print(f"\nEnsemble holdout ({metrics['n']} rows): accuracy={metrics['accuracy']:.3f} macro-F1={metrics['macro_f1']:.3f}")
    print(cm_text(metrics))
    charts.fig_to_png(charts.confusion_chart(metrics["confusion_matrix"], "Holdout confusion matrix (synthetic)"),
                      DOCS / "confusion_matrix.png")

    tr, te = train_test_split(df, test_size=cfg.test_size, stratify=df["label"], random_state=cfg.model.seed)
    contrast = te["text"].str.contains(r"\b(?:لكن|بس|ولكن)\b")
    m = build_pipeline("ensemble", cfg).fit(tr["text"], tr["label"])
    for name, sub in (("with contrast word", te[contrast]), ("without contrast word", te[~contrast])):
        print(f"  holdout {name}: n={len(sub)} accuracy={evaluate(m, sub)['accuracy']:.3f}")

    hc = hard_cases()
    lr = build_pipeline("logreg", cfg).fit(df["text"], df["label"])
    hres = evaluate(lr, hc)
    print(f"\nHard cases ({hres['n']} hand-written sentences, logreg trained on synthetic corpus): "
          f"accuracy={hres['accuracy']:.3f} macro-F1={hres['macro_f1']:.3f}")
    print(cm_text(hres))
    hc["pred"] = hres["pred"]
    print("accuracy by kind:", hc.assign(ok=hc.label == hc.pred).groupby("kind")["ok"].mean().round(2).to_dict())
    charts.fig_to_png(charts.confusion_chart(hres["confusion_matrix"], "Hard cases confusion matrix"),
                      DOCS / "hard_cases_confusion.png")
    print("dialect transfer (logreg):", {k: round(v, 3) for k, v in dialect_transfer(df, cfg=cfg).items()})

    model = load_model(ROOT / "models" / "sentiment.joblib")
    pred = model.predict(df["text"])
    charts.fig_to_png(charts.monthly_trend_chart(df["date"], pd.Series(pred)), DOCS / "monthly_trend.png")
    asp = aspect_summary(df["text"], model)
    print("\nAspect mentions (predicted sentiment):")
    print(asp.to_string())
    charts.fig_to_png(charts.aspect_chart(asp), DOCS / "aspects.png")
    charts.fig_to_png(charts.sentiment_share_chart(pd.Series(pred)), DOCS / "sentiment_share.png")

    (DOCS / "sample_dashboard.html").write_text(
        build_dashboard(df, model, ngram_model=lr), encoding="utf-8")
    print("\nWrote:", ", ".join(sorted(p.name for p in DOCS.iterdir())))


if __name__ == "__main__":
    main()
