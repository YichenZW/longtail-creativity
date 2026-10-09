"""General-purpose scorers and an extension point for task-specific scoring."""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoringExample:
    """Normalized inputs available to task-performance scorers."""

    task_id: str
    prompt: str
    prediction: str
    answer: Any
    metadata: dict[str, Any]


class Scorer(ABC):
    """Per-example scorer with an overridable batched execution path."""

    requires_answer = True

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def score(self, prediction: str, answer: Any, metadata: dict[str, Any]) -> float:
        """Return a task-performance score for one prediction."""

    def score_batch(self, examples: list[ScoringExample]) -> list[float]:
        """Score a batch, falling back to the per-example interface."""

        return [
            self.score(example.prediction, example.answer, example.metadata)
            for example in examples
        ]

    def summary_metadata(self) -> dict[str, Any]:
        """Return serializable scorer metadata for the evaluation summary."""

        return {}

    def close(self) -> None:
        """Release model or client resources held by the scorer."""


class ExactMatchScorer(Scorer):
    """Normalized string exact match."""

    @staticmethod
    def normalize(value: Any) -> str:
        text = str(value).strip().casefold()
        return re.sub(r"\s+", " ", text)

    def score(self, prediction: str, answer: Any, metadata: dict[str, Any]) -> float:
        del metadata
        return float(self.normalize(prediction) == self.normalize(answer))


class NumericMatchScorer(Scorer):
    """Compare the last numeric value in the prediction with a tolerance."""

    _NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

    def score(self, prediction: str, answer: Any, metadata: dict[str, Any]) -> float:
        del metadata
        values = self._NUMBER.findall(prediction.replace(",", ""))
        if not values:
            return 0.0
        try:
            predicted = float(values[-1])
            expected = float(str(answer).replace(",", ""))
        except ValueError:
            return 0.0
        tolerance = float(self.options.get("tolerance", 1e-8))
        return float(abs(predicted - expected) <= tolerance)


def load_scorer(name: str, options: dict[str, Any] | None = None) -> Scorer:
    builtins = {
        "exact_match": ExactMatchScorer,
        "numeric_match": NumericMatchScorer,
    }
    if name in builtins:
        return builtins[name](options)
    if name == "reward_model":
        from longtail_creativity.evaluation.reward_model import RewardModelScorer

        return RewardModelScorer(options)
    if ":" not in name:
        raise ValueError("Scorer plugins must use 'package.module:ClassName'")
    module_name, object_name = name.split(":", 1)
    scorer_type = getattr(importlib.import_module(module_name), object_name)
    scorer = scorer_type(options or {})
    if not isinstance(scorer, Scorer):
        raise TypeError(f"{name} must construct a Scorer")
    return scorer
