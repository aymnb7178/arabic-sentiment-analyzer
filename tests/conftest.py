"""Shared fixtures: a small SYNTHETIC corpus and a trained logistic-regression model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arabic_sentiment.data import generate_corpus  # noqa: E402
from arabic_sentiment.models import build_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    return generate_corpus(n=600, seed=3)


@pytest.fixture(scope="session")
def model(corpus):
    return build_pipeline("logreg").fit(corpus["text"], corpus["label"])
