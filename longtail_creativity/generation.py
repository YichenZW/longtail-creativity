"""Reproducible, resumable generation orchestration."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from longtail_creativity.backends import create_backend
from longtail_creativity.config import RunConfig
from longtail_creativity.methods import load_method
from longtail_creativity.normalization import load_normalizer, normalize_model_output
from longtail_creativity.tasks import load_tasks
from longtail_creativity.taxonomy import prepare_prompt


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_serialize)
        handle.write("\n")
    temporary.replace(path)


def _completed_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()
    completed: set[str] = set()
    with output_file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["task_id"]))
    return completed


def run_generation(config: RunConfig) -> dict[str, Any]:
    """Generate model outputs according to a validated RunConfig."""

    random.seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "outputs.jsonl"
    manifest_file = output_dir / "run_manifest.json"

    if output_file.exists() and not config.resume:
        raise FileExistsError(
            f"{output_file} already exists; choose another output directory or set run.resume"
        )

    tasks = load_tasks(config.task)
    done = _completed_ids(output_file) if config.resume else set()
    pending = [task for task in tasks if task.id not in done]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": _serialize(asdict(config)),
        "n_tasks": len(tasks),
    }
    _write_json(manifest_file, manifest)

    method = load_method(config.method.plugin, config.method.options)
    normalizer = load_normalizer(config.normalization.plugin, config.normalization.options)
    generated = 0
    if not pending:
        return {
            "output_dir": str(output_dir),
            "n_tasks": len(tasks),
            "n_previously_completed": len(done),
            "n_generated": 0,
        }

    with (
        create_backend(config.model) as backend,
        output_file.open("a", encoding="utf-8") as output_handle,
    ):
        for start in range(0, len(pending), config.batch_size):
            batch = pending[start : start + config.batch_size]
            prompts = [prepare_prompt(task, config.taxonomy) for task in batch]
            outputs = method.generate(backend, prompts, config.generation, batch)
            if len(outputs) != len(batch):
                raise RuntimeError(f"Method returned {len(outputs)} batches for {len(batch)} tasks")
            for task, prompt, samples in zip(batch, prompts, outputs):
                samples = [normalize_model_output(sample, normalizer) for sample in samples]
                record = {
                    "task_id": task.id,
                    "prompt": prompt,
                    "taxonomy": config.taxonomy.to_dict(),
                    "method": config.method.plugin,
                    "outputs": [sample.to_dict() for sample in samples],
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                generated += 1

    return {
        "output_dir": str(output_dir),
        "n_tasks": len(tasks),
        "n_previously_completed": len(done),
        "n_generated": generated,
    }
