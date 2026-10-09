from __future__ import annotations

import json

from longtail_creativity.config import TaskConfig
from longtail_creativity.evaluation import runner
from longtail_creativity.evaluation.runner import evaluate_outputs
from longtail_creativity.evaluation.scorers import Scorer


def test_evaluation_skips_missing_and_length_limited_final_answers(tmp_path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps({"id": item_id, "prompt": "prompt", "answer": "answer"}) + "\n"
            for item_id in ("missing", "limited")
        ),
        encoding="utf-8",
    )
    outputs_path.write_text(
        json.dumps(
            {
                "task_id": "missing",
                "outputs": [{"text": "fallback", "final_answer": None}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "task_id": "limited",
                "outputs": [{"text": "answer", "finish_reason": "length"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = evaluate_outputs(
        outputs_path,
        TaskConfig(source="local", path=str(tasks_path)),
        destination=metrics_path,
    )
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [metric["skip_reason"] for metric in metrics] == ["missing_output", "length_limit"]
    assert summary["n_scored"] == 0


class FakeReferenceFreeBatchScorer(Scorer):
    requires_answer = False

    def __init__(self) -> None:
        super().__init__()
        self.examples = []

    def score(self, prediction, answer, metadata):
        raise AssertionError("The runner should use score_batch")

    def score_batch(self, examples):
        self.examples = examples
        return [0.75 for _ in examples]

    def summary_metadata(self):
        return {"reward_model": "test/model"}


def test_reference_free_batch_scorer_receives_prompt_and_normalized_output(
    tmp_path,
    monkeypatch,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    tasks_path.write_text(
        json.dumps({"id": "writing-1", "prompt": "raw task prompt"}) + "\n",
        encoding="utf-8",
    )
    outputs_path.write_text(
        json.dumps(
            {
                "task_id": "writing-1",
                "prompt": "generation prompt",
                "outputs": [
                    {
                        "text": "fallback",
                        "final_answer": "normalized response",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scorer = FakeReferenceFreeBatchScorer()
    monkeypatch.setattr(runner, "load_scorer", lambda *_: scorer)

    summary = evaluate_outputs(
        outputs_path,
        TaskConfig(source="local", path=str(tasks_path), answer_field=None),
        scorer_name="reward_model",
        destination=metrics_path,
    )
    metric = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metric["score"] == 0.75
    assert scorer.examples[0].prompt == "generation prompt"
    assert scorer.examples[0].prediction == "normalized response"
    assert summary["reward_model"] == "test/model"
