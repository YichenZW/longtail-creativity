"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from longtail_creativity.types import (
    CreativityDepth,
    EvaluationFocus,
    ReferenceType,
    TaxonomyConfig,
    Transformation,
)


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    model_id: str
    revision: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskConfig:
    source: str
    path: str
    split: str | None = None
    subset: str | None = None
    prompt_field: str = "prompt"
    id_field: str = "id"
    answer_field: str | None = "answer"
    metadata_fields: tuple[str, ...] = ()
    limit: int | None = None


@dataclass(frozen=True)
class MethodConfig:
    plugin: str = "direct"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizationConfig:
    """Output normalizer used before generations are serialized."""

    plugin: str = "reasoning_tags"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    task: TaskConfig
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    generation: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "outputs/run"
    seed: int = 42
    batch_size: int = 8
    resume: bool = False


def _enum(enum_type, value):
    return enum_type(value) if value is not None else None


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML run config and reject missing required fields early."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise TypeError("The top-level configuration must be a mapping")
    for section in ("model", "task"):
        if not isinstance(raw.get(section), dict):
            raise TypeError(f"Missing required '{section}' mapping")

    model_raw = raw["model"]
    task_raw = raw["task"]
    taxonomy_raw = raw.get("taxonomy") or {}
    method_raw = raw.get("method") or {}
    normalization_raw = raw.get("normalization") or {}
    run_raw = raw.get("run") or {}

    model = ModelConfig(
        backend=str(model_raw["backend"]),
        model_id=str(model_raw["model_id"]),
        revision=model_raw.get("revision"),
        options=dict(model_raw.get("options") or {}),
    )
    task = TaskConfig(
        source=str(task_raw.get("source", "local")),
        path=str(task_raw["path"]),
        split=task_raw.get("split"),
        subset=task_raw.get("subset"),
        prompt_field=str(task_raw.get("prompt_field", "prompt")),
        id_field=str(task_raw.get("id_field", "id")),
        answer_field=task_raw.get("answer_field", "answer"),
        metadata_fields=tuple(task_raw.get("metadata_fields") or ()),
        limit=task_raw.get("limit"),
    )
    taxonomy = TaxonomyConfig(
        transformation=_enum(Transformation, taxonomy_raw.get("transformation")),
        depth=_enum(CreativityDepth, taxonomy_raw.get("depth")),
        focus=_enum(EvaluationFocus, taxonomy_raw.get("focus", "outcome")),
        reference=_enum(ReferenceType, taxonomy_raw.get("reference")),
        prompt_template=taxonomy_raw.get("prompt_template"),
    )
    if taxonomy.prompt_template and "{prompt}" not in taxonomy.prompt_template:
        raise ValueError("taxonomy.prompt_template must contain '{prompt}'")

    method = MethodConfig(
        plugin=str(method_raw.get("plugin", "direct")),
        options=dict(method_raw.get("options") or {}),
    )
    normalization = NormalizationConfig(
        plugin=str(normalization_raw.get("plugin", "reasoning_tags")),
        options=dict(normalization_raw.get("options") or {}),
    )
    generation = dict(raw.get("generation") or {})
    if int(generation.get("n", 1)) < 1:
        raise ValueError("generation.n must be at least 1")

    return RunConfig(
        model=model,
        task=task,
        taxonomy=taxonomy,
        method=method,
        normalization=normalization,
        generation=generation,
        output_dir=str(run_raw.get("output_dir", "outputs/run")),
        seed=int(run_raw.get("seed", 42)),
        batch_size=max(1, int(run_raw.get("batch_size", 8))),
        resume=bool(run_raw.get("resume", False)),
    )
