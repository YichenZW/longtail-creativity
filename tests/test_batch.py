from __future__ import annotations

import json
import threading
import time

from longtail_creativity.novelty.batch import evaluate_jsonl
from longtail_creativity.novelty.tokenization import WhitespaceTokenizer


def test_batch_evaluation_preserves_order_and_reports_skips(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    records = [
        {"task_id": "first", "outputs": [{"text": "a b c"}]},
        {"task_id": "second", "outputs": []},
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = evaluate_jsonl(
        input_path,
        output_path,
        query_factories=[lambda: lambda _: 0],
        tokenizer_factory=WhitespaceTokenizer,
        window=None,
        minimum_ngram=2,
        n_values=[2],
    )

    results = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [result["task_id"] for result in results] == ["first", "second"]
    assert results[0]["creativity_index_normalized"] == 1.0
    assert results[1]["skip_reason"] == "missing_text"
    assert summary == {
        "n_examples": 2,
        "n_scored": 1,
        "n_skipped": 1,
        "n_errors": 0,
        "n_previously_completed": 0,
        "n_quarantined_endpoints": 0,
        "creativity_index_normalized_mean": 1.0,
        "mean_n_novelty_5_13_mean": None,
    }


def test_prompt_novelty_uses_the_same_record_identity(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "prompt_metrics.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "prompt": "input prompt words",
                "outputs": [{"text": "generated answer words"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluate_jsonl(
        input_path,
        output_path,
        query_factories=[lambda: lambda _: 0],
        tokenizer_factory=WhitespaceTokenizer,
        window=None,
        minimum_ngram=2,
        n_values=[2],
        text_source="prompt",
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["task_id"] == "task-1"
    assert result["record_id"] == "task-1"
    assert result["text_source"] == "prompt"
    assert result["creativity_index_normalized"] == 1.0


def test_length_limited_generation_is_skipped_but_prompt_is_scoreable(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    record = {
        "task_id": "task-1",
        "prompt": "prompt has enough words",
        "outputs": [{"text": "unfinished output words", "finish_reason": "length"}],
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    options = {
        "input_path": input_path,
        "output_path": output_path,
        "query_factories": [lambda: lambda _: 0],
        "tokenizer_factory": WhitespaceTokenizer,
        "window": None,
        "minimum_ngram": 2,
        "n_values": [2],
    }
    evaluate_jsonl(**options)
    assert json.loads(output_path.read_text())["skip_reason"] == "length_limit"

    evaluate_jsonl(**options, text_source="prompt")
    assert json.loads(output_path.read_text())["creativity_index_normalized"] == 1.0


def test_resume_skips_completed_evaluation_keys(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps({"task_id": "task-1", "outputs": [{"text": "a b c"}]}) + "\n",
        encoding="utf-8",
    )
    evaluate_jsonl(
        input_path,
        output_path,
        query_factories=[lambda: lambda _: 0],
        tokenizer_factory=WhitespaceTokenizer,
        window=None,
        minimum_ngram=2,
        n_values=[2],
    )

    def should_not_initialize():
        raise AssertionError("resume initialized an endpoint for completed work")

    summary = evaluate_jsonl(
        input_path,
        output_path,
        query_factories=[should_not_initialize],
        tokenizer_factory=WhitespaceTokenizer,
        window=None,
        minimum_ngram=2,
        n_values=[2],
        resume=True,
    )
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
    assert summary["n_previously_completed"] == 1


def test_resume_fingerprint_does_not_reuse_different_metric_settings(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps({"task_id": "task-1", "outputs": [{"text": "a b c"}]}) + "\n",
        encoding="utf-8",
    )
    common = {
        "input_path": input_path,
        "output_path": output_path,
        "query_factories": [lambda: lambda _: 0],
        "tokenizer_factory": WhitespaceTokenizer,
        "window": None,
        "minimum_ngram": 2,
    }
    evaluate_jsonl(**common, n_values=[2])
    summary = evaluate_jsonl(**common, n_values=[3], resume=True)
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert summary["n_previously_completed"] == 0


def test_failed_endpoint_requeues_item_on_another_endpoint(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps({"task_id": "task-1", "outputs": [{"text": "a b c"}]}) + "\n",
        encoding="utf-8",
    )
    first_failed = threading.Event()

    def failing_factory():
        def fail(_):
            first_failed.set()
            raise RuntimeError("endpoint unavailable")

        return fail

    def healthy_factory():
        assert first_failed.wait(timeout=1)
        return lambda _: 0

    summary = evaluate_jsonl(
        input_path,
        output_path,
        query_factories=[failing_factory, healthy_factory],
        tokenizer_factory=WhitespaceTokenizer,
        window=None,
        minimum_ngram=2,
        n_values=[2],
        watchdog_interval=0.01,
        per_item_timeout=1,
    )
    assert summary["n_scored"] == 1
    assert summary["n_quarantined_endpoints"] == 1


def test_watchdog_requeues_a_stalled_item(tmp_path) -> None:
    input_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps({"task_id": "task-1", "outputs": [{"text": "a b c"}]}) + "\n",
        encoding="utf-8",
    )
    slow_query_started = threading.Event()

    def slow_factory():
        def slow_query(_):
            slow_query_started.set()
            time.sleep(0.1)
            return 0

        return slow_query

    def healthy_factory():
        assert slow_query_started.wait(timeout=1)
        return lambda _: 0

    summary = evaluate_jsonl(
        input_path,
        output_path,
        query_factories=[slow_factory, healthy_factory],
        tokenizer_factory=WhitespaceTokenizer,
        window=None,
        minimum_ngram=2,
        n_values=[2],
        watchdog_interval=0.005,
        per_item_timeout=0.02,
    )
    assert summary["n_scored"] == 1
    assert summary["n_quarantined_endpoints"] == 1
