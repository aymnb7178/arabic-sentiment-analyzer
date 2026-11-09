"""Command line interface: ``python -m arabic_sentiment <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .aspects import aspect_summary
from .config import Config
from .data import generate_corpus, hard_cases
from .logging_utils import setup_logging
from .models import (MODEL_NAMES, cross_validate_models, dialect_transfer, evaluate, load_model,
                     predict_texts, train_and_save, build_pipeline)

DEFAULT_MODEL = Path("models/sentiment.joblib")


def _load_data(path: str | None, n: int, seed: int) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        if not {"text", "label"} <= set(df.columns):
            raise SystemExit("data file needs 'text' and 'label' columns")
        return df
    return generate_corpus(n=n, seed=seed)


def _fmt_cm(res: dict) -> str:
    labs = res["labels"]
    lines = ["confusion matrix (rows=true, cols=pred)", " " * 10 + "".join(f"{l:>10}" for l in labs)]
    for lab, row in zip(labs, res["confusion_matrix"]):
        lines.append(f"{lab:<10}" + "".join(f"{v:>10}" for v in row))
    return "\n".join(lines)


def cmd_generate(a: argparse.Namespace) -> int:
    df = generate_corpus(n=a.n, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} SYNTHETIC rows to {a.out}")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    df = _load_data(a.data, a.n, a.seed)
    res = train_and_save(df, a.model_path, name=a.model)
    print(f"Trained {a.model} on {len(df)} rows; saved to {a.model_path}")
    print(f"Holdout ({res['n']} rows): accuracy={res['accuracy']:.3f} macro-F1={res['macro_f1']:.3f}")
    print(_fmt_cm(res))
    if not a.data:
        print("NOTE: data is synthetic; these scores are an optimistic upper bound.")
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    model = load_model(a.model_path)
    out = predict_texts(model, [a.text])
    row = out.iloc[0].to_dict()
    row["aspects"] = {k: v for k, v in aspect_summary([a.text], model)[list(("negative", "neutral", "positive"))]
                      .to_dict("index").items()}
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_batch(a: argparse.Namespace) -> int:
    model = load_model(a.model_path)
    df = pd.read_csv(a.file)
    if a.column not in df:
        raise SystemExit(f"column {a.column!r} not found; columns: {list(df.columns)}")
    pred = predict_texts(model, df[a.column].astype(str).tolist())
    df["label"] = pred["label"].to_numpy()
    if "confidence" in pred:
        df["confidence"] = pred["confidence"].to_numpy()
    out = a.out or str(Path(a.file).with_suffix("")) + "_predictions.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} predictions to {out}")
    print(df["label"].value_counts().to_string())
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    cfg = Config()
    df = _load_data(a.data, a.n, a.seed)
    models = a.models or list(MODEL_NAMES)
    cv = cross_validate_models(df, models, cfg)
    print(f"{cfg.cv_folds}-fold cross-validation on {len(df)} rows")
    print(cv.round(3).to_string(index=False))
    if not a.data:
        hc = hard_cases()
        m = build_pipeline("logreg", cfg).fit(df["text"], df["label"])
        res = evaluate(m, hc)
        print(f"\nHard cases ({res['n']} hand-written sentences, logreg trained on the synthetic corpus)")
        print(f"accuracy={res['accuracy']:.3f} macro-F1={res['macro_f1']:.3f}")
        print(_fmt_cm(res))
        print("\nDialect transfer (logreg):", {k: round(v, 3) for k, v in dialect_transfer(df).items()})
        print("NOTE: synthetic corpus scores are an optimistic upper bound.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arabic_sentiment", description="Arabic sentiment analysis (classical ML).")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write the SYNTHETIC corpus to CSV")
    g.add_argument("--n", type=int, default=1800)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/synthetic_corpus.csv")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train, hold out a split, report, save with joblib")
    t.add_argument("--data", help="labelled CSV with text,label (default: synthetic corpus)")
    t.add_argument("--n", type=int, default=1800)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--model", choices=MODEL_NAMES, default="ensemble")
    t.add_argument("--model-path", default=str(DEFAULT_MODEL))
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="predict one text")
    pr.add_argument("text")
    pr.add_argument("--model-path", default=str(DEFAULT_MODEL))
    pr.set_defaults(func=cmd_predict)

    b = sub.add_parser("batch", help="predict every row of a CSV")
    b.add_argument("file")
    b.add_argument("--column", default="text")
    b.add_argument("--out")
    b.add_argument("--model-path", default=str(DEFAULT_MODEL))
    b.set_defaults(func=cmd_batch)

    e = sub.add_parser("evaluate", help="cross-validate models (+ hard cases on synthetic data)")
    e.add_argument("--data")
    e.add_argument("--n", type=int, default=1800)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
