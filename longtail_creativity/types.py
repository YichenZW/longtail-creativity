"""Shared, serializable types used by generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Transformation(str, Enum):
    """How creativity enters a task in the paper taxonomy."""

    INHERENT = "inherent"
    RARE_OUTPUT = "rare_output"
    CONSTRAINT_ELICITED = "constraint_elicited"


class CreativityDepth(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"


class EvaluationFocus(str, Enum):
    OUTCOME = "outcome"
    PROCESS = "process"


class ReferenceType(str, Enum):
    SELF = "self"
    COMMUNITY = "community"
    HISTORICAL = "historical"


@dataclass(frozen=True)
class TaxonomyConfig:
    """Labels and an optional prompt template for one experimental condition."""

    transformation: Transformation | None = None
    depth: CreativityDepth | None = None
    focus: EvaluationFocus = EvaluationFocus.OUTCOME
    reference: ReferenceType | None = None
    prompt_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.value if isinstance(value, Enum) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class TaskExample:
    id: str
    prompt: str
    answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    text: str
    raw_output: str | None = None
    final_answer: str | None = None
    reasoning_content: str | None = None
    normalization: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    trace: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
