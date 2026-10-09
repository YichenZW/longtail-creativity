"""Evaluate generated outputs against task answers."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from longtail_creativity.config import TaskConfig
from longtail_creativity.evaluation.scorers import ScoringExample, load_scorer
from longtail_creativity.normalization import extract_record_text
from longtail_creativity.tasks import load_tasks


def _prediction(record: dict[str, Any], sample_index: int) -> str | None:
    value, _ = extract_record_text(record, "auto", sample_index)
    outputs = record.get("outputs")
    if value is None and (not isinstance(outputs, list) or sample_index >= len(outputs)):
        raise IndexError(f"Task {record.get('task_id')} has no output at index {sample_index}")
    return value


def evaluate_outputs(
    outputs_path: str | Path,
    task_config: TaskConfig,
    scorer_name: str = "exact_match",
    scorer_options: dict[str, Any] | None = None,
    sample_index: int = 0,
    destination: str | Path | None = None,
    include_length_truncated: bool = False,
) -> dict[str, Any]:
    tasks = {task.id: task for task in load_tasks(task_config)}
    scorer = load_scorer(scorer_name, scorer_options)
    outputs_path = Path(outputs_path)
    destination_path = Path(destination) if destination else outputs_path.parent / "metrics.jsonl"
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    scoring_examples: list[ScoringExample] = []
    scoring_positions: list[int] = []
    with outputs_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id not in tasks:
                raise KeyError(f"Output task '{task_id}' is absent from the configured dataset")
            task = tasks[task_id]
            if task.answer is None and scorer.requires_answer:
                metrics.append({"task_id": task_id, "score": None, "skip_reason": "no_answer"})
                continue
            outputs = record.get("outputs")
            selected_output = (
                outputs[sample_index]
                if isinstance(outputs, list) and sample_index < len(outputs)
                else {}
            )
            if (
                isinstance(selected_output, dict)
                and selected_output.get("finish_reason") == "length"
                and not include_length_truncated
            ):
                metrics.append(
                    {"task_id": task_id, "score": None, "skip_reason": "length_limit"}
                )
                continue
            prediction = _prediction(record, sample_index)
            if prediction is None:
                metrics.append(
                    {"task_id": task_id, "score": None, "skip_reason": "missing_output"}
                )
                continue
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = task.prompt
            scoring_positions.append(len(metrics))
            scoring_examples.append(
                ScoringExample(
                    task_id=task_id,
                    prompt=prompt,
                    prediction=prediction,
                    answer=task.answer,
                    metadata=task.metadata,
                )
            )
            metrics.append({"task_id": task_id, "score": None})

    try:
        scores = scorer.score_batch(scoring_examples)
        scorer_metadata = scorer.summary_metadata()
    finally:
        scorer.close()
    if len(scores) != len(scoring_examples):
        raise RuntimeError(
            f"Scorer returned {len(scores)} scores for {len(scoring_examples)} examples"
        )
    for position, value in zip(scoring_positions, scores):
        metrics[position]["score"] = float(value)

    with destination_path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric, ensure_ascii=False) + "\n")

    values = [metric["score"] for metric in metrics if metric["score"] is not None]
    summary = {
        "scorer": scorer_name,
        "n_examples": len(metrics),
        "n_scored": len(values),
        "mean": statistics.fmean(values) if values else None,
        **scorer_metadata,
    }
    summary_path = destination_path.with_name("summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary
