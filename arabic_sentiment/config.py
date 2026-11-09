"""Configuration dataclasses for the Arabic sentiment analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class PreprocessConfig:
    """Options controlling Arabic text cleaning.

    Attributes:
        remove_stopwords: Drop MSA / Gulf stopwords (negators are never dropped
            before negation marking is applied).
        mark_negation: Attach a ``NOT_`` prefix to the content words that follow
            a negator (لا، ما، لم، لن، مو، مب، ...).
        negation_window: Number of content words affected by a negator.
        keep_negators: Keep the negator token itself in the output.
        emoji_tokens: Map sentiment emojis to ``emopos`` / ``emoneg`` tokens.
    """

    remove_stopwords: bool = True
    mark_negation: bool = True
    negation_window: int = 2
    keep_negators: bool = False
    emoji_tokens: bool = True


@dataclass
class ModelConfig:
    """Feature extraction and classifier hyper-parameters."""

    word_ngram: tuple[int, int] = (1, 2)
    char_ngram: tuple[int, int] = (2, 5)
    min_df: int = 2
    logreg_c: float = 4.0
    svc_c: float = 0.5
    nb_alpha: float = 0.3
    lexicon_features: bool = False
    seed: int = 42


@dataclass
class Config:
    """Top-level configuration."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    labels: tuple[str, ...] = LABELS
    model_path: Path = Path("models/sentiment.joblib")
    cv_folds: int = 5
    test_size: float = 0.2
    contrast_words: tuple[str, ...] = ("لكن", "لاكن", "ولكن", "بس")
