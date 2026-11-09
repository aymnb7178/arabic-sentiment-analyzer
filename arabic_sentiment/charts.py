"""Matplotlib charts (PNG). Labels are English on purpose.

``arabic_reshaper`` and ``python-bidi`` are not available in this project's
dependency set, and matplotlib alone renders Arabic as disconnected, reversed
glyphs. Arabic tokens therefore appear only in the HTML dashboard, where the
browser shapes them correctly.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import LABELS  # noqa: E402

__all__ = ["COLORS", "fig_to_png", "png_b64", "confusion_chart", "model_comparison_chart",
           "sentiment_share_chart", "monthly_trend_chart", "aspect_chart"]

COLORS = {"negative": "#d1495b", "neutral": "#8d99ae", "positive": "#2a9d8f"}
_INK = "#222b36"


def _style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=_INK)


def fig_to_png(fig: plt.Figure, path: str | Path | None = None) -> bytes:
    """Render a figure to PNG bytes (optionally also writing ``path``) and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
    return data


def png_b64(data: bytes) -> str:
    """Base64 string for embedding in ``<img src="data:image/png;base64,...">``."""
    return base64.b64encode(data).decode("ascii")


def confusion_chart(cm: list[list[int]], title: str = "Confusion matrix") -> plt.Figure:
    """Heat-map of a confusion matrix (rows = true, columns = predicted)."""
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, color=_INK)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else _INK)
    return fig


def model_comparison_chart(cv: pd.DataFrame) -> plt.Figure:
    """Grouped bars of CV accuracy and macro-F1 per model."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(cv))
    w = 0.38
    ax.bar(x - w / 2, cv["accuracy_mean"], w, yerr=cv["accuracy_std"].fillna(0), label="Accuracy", color="#264653")
    ax.bar(x + w / 2, cv["macro_f1_mean"], w, yerr=cv["macro_f1_std"].fillna(0), label="Macro-F1", color="#e9c46a")
    ax.set_xticks(x, cv["model"])
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (synthetic corpus, 5-fold CV)", fontsize=11, color=_INK)
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    return fig


def sentiment_share_chart(labels: pd.Series) -> plt.Figure:
    """Horizontal bars with the share of each sentiment label."""
    share = labels.value_counts(normalize=True).reindex(LABELS).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(list(share.index), share.values, color=[COLORS[l] for l in share.index])
    for i, v in enumerate(share.values):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", color=_INK)
    ax.set_xlim(0, max(share.max() * 1.18, 10))
    ax.set_xlabel("Share of texts (%)")
    ax.set_title("Sentiment share", fontsize=11, color=_INK)
    _style(ax)
    return fig


def monthly_trend_chart(dates: pd.Series, labels: pd.Series) -> plt.Figure:
    """Monthly share of each sentiment label."""
    df = pd.DataFrame({"m": pd.to_datetime(dates).dt.strftime("%Y-%m"), "l": labels.values})
    tab = pd.crosstab(df["m"], df["l"], normalize="index").reindex(columns=list(LABELS), fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for lab in LABELS:
        ax.plot(tab.index, tab[lab], marker="o", ms=4, color=COLORS[lab], label=lab)
    ax.set_ylabel("Share of month's texts (%)")
    ax.set_title("Monthly sentiment trend", fontsize=11, color=_INK)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, ncol=3)
    _style(ax)
    return fig


def aspect_chart(summary: pd.DataFrame) -> plt.Figure:
    """Stacked bars: mentions per aspect split by predicted sentiment."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    left = np.zeros(len(summary))
    for lab in LABELS:
        vals = summary[lab].to_numpy()
        ax.barh(summary.index, vals, left=left, color=COLORS[lab], label=lab)
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("Aspect mentions")
    ax.set_title("Top aspects by sentiment", fontsize=11, color=_INK)
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    return fig
