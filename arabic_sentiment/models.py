"""Model construction, cross-validation, evaluation and persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from .config import LABELS, Config, ModelConfig, PreprocessConfig
from .lexicon import LexiconClassifier
from .preprocess import preprocess

log = logging.getLogger(__name__)

__all__ = ["build_pipeline", "MODEL_NAMES", "cross_validate_models", "evaluate",
           "train_and_save", "load_model", "predict_texts", "top_ngrams", "dialect_transfer"]

MODEL_NAMES = ("logreg", "linear_svc", "complement_nb", "ensemble")


def _prep_many(texts: Iterable[str]) -> list[str]:
    """Preprocess a batch of raw texts (module-level so it can be pickled)."""
    return [preprocess(t) for t in texts]


def _make_features(mc: ModelConfig) -> FeatureUnion:
    return FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=mc.word_ngram, min_df=mc.min_df,
                              sublinear_tf=True, token_pattern=r"\S+")),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=mc.char_ngram, min_df=mc.min_df,
                              sublinear_tf=True)),
    ])


def _make_classifier(name: str, mc: ModelConfig) -> Any:
    if name == "logreg":
        return LogisticRegression(C=mc.logreg_c, max_iter=2000, random_state=mc.seed)
    if name == "linear_svc":
        return LinearSVC(C=mc.svc_c, random_state=mc.seed)
    if name == "complement_nb":
        return ComplementNB(alpha=mc.nb_alpha)
    if name == "ensemble":
        return VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=mc.logreg_c, max_iter=2000, random_state=mc.seed)),
                ("svc", CalibratedClassifierCV(LinearSVC(C=mc.svc_c, random_state=mc.seed), cv=3)),
                ("nb", ComplementNB(alpha=mc.nb_alpha)),
            ],
            voting="soft",
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def build_pipeline(name: str = "ensemble", cfg: Config | None = None) -> Pipeline:
    """Build an untrained raw-text -> label pipeline."""
    cfg = cfg or Config()
    return Pipeline([
        ("prep", FunctionTransformer(_prep_many, validate=False)),
        ("feats", _make_features(cfg.model)),
        ("clf", _make_classifier(name, cfg.model)),
    ])


def cross_validate_models(df: pd.DataFrame, names: Iterable[str] = MODEL_NAMES,
                          cfg: Config | None = None) -> pd.DataFrame:
    """Stratified k-fold CV for each model; returns mean/std accuracy and macro-F1.

    Note: on the templated synthetic corpus near-duplicate sentences can land in
    different folds, so these numbers are optimistic.
    """
    cfg = cfg or Config()
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.model.seed)
    rows = []
    for name in names:
        res = cross_validate(build_pipeline(name, cfg), df["text"], df["label"], cv=cv,
                             scoring=["accuracy", "f1_macro"])
        rows.append({"model": name,
                     "accuracy_mean": res["test_accuracy"].mean(), "accuracy_std": res["test_accuracy"].std(),
                     "macro_f1_mean": res["test_f1_macro"].mean(), "macro_f1_std": res["test_f1_macro"].std()})
        log.info("CV %-14s acc=%.3f macroF1=%.3f", name, rows[-1]["accuracy_mean"], rows[-1]["macro_f1_mean"])
    # lexicon baseline needs no training; score it on the same folds' union (= all rows)
    lex = LexiconClassifier()
    pred = lex.predict(df["text"])
    rows.append({"model": "lexicon", "accuracy_mean": accuracy_score(df["label"], pred), "accuracy_std": float("nan"),
                 "macro_f1_mean": f1_score(df["label"], pred, average="macro", labels=list(LABELS)),
                 "macro_f1_std": float("nan")})
    return pd.DataFrame(rows)


def evaluate(model: Any, df: pd.DataFrame) -> dict[str, Any]:
    """Accuracy, macro-F1, confusion matrix (rows=true, cols=pred, LABELS order) and report."""
    pred = model.predict(df["text"])
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(df["label"], pred)),
        "macro_f1": float(f1_score(df["label"], pred, average="macro", labels=list(LABELS), zero_division=0)),
        "confusion_matrix": confusion_matrix(df["label"], pred, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
        "report": classification_report(df["label"], pred, labels=list(LABELS), zero_division=0),
        "pred": list(pred),
    }


def dialect_transfer(df: pd.DataFrame, name: str = "logreg", cfg: Config | None = None) -> dict[str, float]:
    """Train on one dialect and test on the other (a harsher, still synthetic, check)."""
    out: dict[str, float] = {}
    for src, dst in (("msa", "gulf"), ("gulf", "msa")):
        m = build_pipeline(name, cfg).fit(df[df.dialect == src]["text"], df[df.dialect == src]["label"])
        te = df[df.dialect == dst]
        out[f"{src}->{dst}"] = float(accuracy_score(te["label"], m.predict(te["text"])))
    return out


def train_and_save(df: pd.DataFrame, path: str | Path, name: str = "ensemble",
                   cfg: Config | None = None, test_size: float | None = None) -> dict[str, Any]:
    """Hold out a stratified test split, fit, evaluate, persist with joblib."""
    from sklearn.model_selection import train_test_split

    cfg = cfg or Config()
    tr, te = train_test_split(df, test_size=test_size or cfg.test_size, stratify=df["label"],
                              random_state=cfg.model.seed)
    model = build_pipeline(name, cfg).fit(tr["text"], tr["label"])
    metrics = evaluate(model, te)
    final = build_pipeline(name, cfg).fit(df["text"], df["label"])  # refit on everything for deployment
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "name": name, "labels": list(LABELS)}, path)
    log.info("saved %s model to %s (holdout acc=%.3f)", name, path, metrics["accuracy"])
    metrics["model_path"] = str(path)
    return metrics


def load_model(path: str | Path) -> Pipeline:
    """Load a pipeline saved by :func:`train_and_save`."""
    obj = joblib.load(path)
    return obj["model"] if isinstance(obj, dict) else obj


def predict_texts(model: Any, texts: list[str]) -> pd.DataFrame:
    """Predict labels (and confidence where the model has ``predict_proba``)."""
    pred = model.predict(texts)
    out = pd.DataFrame({"text": texts, "label": pred})
    if hasattr(model, "predict_proba"):
        out["confidence"] = np.round(model.predict_proba(texts).max(axis=1), 3)
    return out


def top_ngrams(model: Pipeline, k: int = 10) -> dict[str, list[tuple[str, float]]]:
    """Most indicative word n-grams per class from a fitted logreg pipeline."""
    feats: FeatureUnion = model.named_steps["feats"]
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise ValueError("top_ngrams needs a linear model (logreg or linear_svc)")
    names = np.array(feats.get_feature_names_out())
    mask = np.array([n.startswith("w__") for n in names])
    words = np.array([n[3:] for n in names[mask]])
    out: dict[str, list[tuple[str, float]]] = {}
    for i, cls in enumerate(clf.classes_):
        coef = clf.coef_[i][mask]
        idx = np.argsort(coef)[::-1][:k]
        out[str(cls)] = [(str(words[j]), float(coef[j])) for j in idx]
    return out
