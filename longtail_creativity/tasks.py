"""Dataset-agnostic task loading.

Datasets are deliberately not distributed with this repository. Local JSON or
JSONL files and Hugging Face datasets are normalized into the same TaskExample
schema at runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from longtail_creativity.config import TaskConfig
from longtail_creativity.types import TaskExample


def _stable_id(record: Mapping[str, Any], index: int) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"item-{index}-{digest}"


def _records_from_local(path: Path, split: str | None) -> Iterable[Mapping[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    yield value
        return

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        if split:
            if split not in payload:
                raise KeyError(f"Split '{split}' is not present in {path}")
            payload = payload[split]
        else:
            for candidate in ("data", "test", "validation", "train", "items", "examples"):
                if isinstance(payload.get(candidate), list):
                    payload = payload[candidate]
                    break
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list or a mapping containing a split in {path}")
    for item in payload:
        if not isinstance(item, Mapping):
            raise TypeError(f"All records in {path} must be JSON objects")
        yield item


def _records_from_huggingface(config: TaskConfig) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face loading requires: pip install 'longtail-creativity[datasets]'"
        ) from exc
    if not config.split:
        raise ValueError("A split is required for a Hugging Face dataset")
    dataset = load_dataset(config.path, config.subset, split=config.split)
    yield from dataset


def load_tasks(config: TaskConfig) -> list[TaskExample]:
    """Load and normalize tasks without making assumptions about a benchmark."""

    if config.source == "local":
        records = _records_from_local(Path(config.path), config.split)
    elif config.source == "huggingface":
        records = _records_from_huggingface(config)
    else:
        raise ValueError("task.source must be 'local' or 'huggingface'")

    tasks: list[TaskExample] = []
    for index, record in enumerate(records):
        if config.limit is not None and len(tasks) >= config.limit:
            break
        if config.prompt_field not in record:
            raise KeyError(f"Record {index} does not contain prompt field '{config.prompt_field}'")
        prompt = record[config.prompt_field]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Record {index} has an empty or non-string prompt")

        item_id = record.get(config.id_field)
        answer = record.get(config.answer_field) if config.answer_field else None
        metadata = {field: record[field] for field in config.metadata_fields if field in record}
        tasks.append(
            TaskExample(
                id=str(item_id) if item_id is not None else _stable_id(record, index),
                prompt=prompt,
                answer=answer,
                metadata=metadata,
            )
        )
    return tasks
