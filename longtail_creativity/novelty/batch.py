"""Fault-tolerant JSONL novelty evaluation across count endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import random
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longtail_creativity.normalization import extract_record_text
from longtail_creativity.novelty.algorithm import compute_creativity_metrics
from longtail_creativity.novelty.clients import SQLiteCountCache, with_memory_cache
from longtail_creativity.novelty.tokenization import Tokenizer

LOGGER = logging.getLogger(__name__)


def extract_text(
    record: dict[str, Any],
    text_source: str = "auto",
    sample_index: int = 0,
) -> str | None:
    """Backward-compatible wrapper around the shared record text selector."""

    return extract_record_text(record, text_source, sample_index)[0]


def _record_id(record: dict[str, Any], index: int) -> str:
    for field_name in ("variant_id", "task_id", "id"):
        value = record.get(field_name)
        if value is not None and str(value):
            return str(value)
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"record-{index}-{digest}"


def _evaluation_key(
    record_id: str,
    text_source: str,
    sample_index: int,
    settings_digest: str,
) -> str:
    return (
        f"{record_id}::source={text_source}::sample={sample_index}"
        f"::settings={settings_digest}"
    )


def _score_record(
    record: dict[str, Any],
    record_id: str,
    evaluation_key: str,
    query,
    tokenizer: Tokenizer,
    window: int | None,
    minimum_ngram: int,
    n_values: Sequence[int],
    text_source: str,
    sample_index: int,
    seed: int,
    include_details: bool,
    include_length_truncated: bool,
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or record.get("id") or record_id)
    text, selected_source = extract_record_text(record, text_source, sample_index)
    identity = {
        "record_id": record_id,
        "task_id": task_id,
        "evaluation_key": evaluation_key,
        "text_source": selected_source,
        "sample_index": sample_index,
    }
    if record.get("variant_id") is not None:
        identity["variant_id"] = str(record["variant_id"])
    if record.get("base_task_id") is not None:
        identity["base_task_id"] = str(record["base_task_id"])
    outputs = record.get("outputs")
    selected_output = (
        outputs[sample_index]
        if isinstance(outputs, list) and sample_index < len(outputs)
        else {}
    )
    if (
        text_source != "prompt"
        and isinstance(selected_output, dict)
        and selected_output.get("finish_reason") == "length"
        and not include_length_truncated
    ):
        return {**identity, "skip_reason": "length_limit"}
    if not text or not text.strip():
        return {**identity, "skip_reason": "missing_text"}
    tokens = tokenizer.tokenize(text)
    if len(tokens) < minimum_ngram:
        return {
            **identity,
            "skip_reason": "too_short",
            "n_tokens_total": len(tokens),
        }

    start = 0
    if window and len(tokens) > window:
        window_identity = f"{record_id}::source={text_source}::sample={sample_index}"
        digest = hashlib.sha256(window_identity.encode("utf-8")).digest()
        item_seed = int.from_bytes(digest[:8], "big") ^ seed
        start = random.Random(item_seed).randint(0, len(tokens) - window)
        score_tokens = tokens[start : start + window]
    else:
        score_tokens = tokens

    metrics = compute_creativity_metrics(
        score_tokens,
        query,
        minimum_ngram=minimum_ngram,
        n_values=n_values,
        include_details=include_details,
    )
    return {
        **identity,
        "n_tokens_total": len(tokens),
        "window_start": start,
        **metrics,
    }


@dataclass(frozen=True)
class _WorkItem:
    index: int
    record: dict[str, Any]
    record_id: str
    evaluation_key: str


class _EndpointWorker(threading.Thread):
    """One worker bound to one count endpoint for its entire lifetime."""

    def __init__(
        self,
        *,
        number: int,
        query_factory: Callable[[], Callable[[tuple[str, ...]], int]],
        tokenizer_factory: Callable[[], Tokenizer],
        work_queue: queue.Queue,
        event_queue: queue.Queue,
        stop_event: threading.Event,
        cache: SQLiteCountCache | None,
        score_options: dict[str, Any],
    ) -> None:
        super().__init__(name=f"novelty-worker-{number}", daemon=True)
        self.query_factory = query_factory
        self.tokenizer_factory = tokenizer_factory
        self.work_queue = work_queue
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.cache = cache
        self.score_options = score_options
        self.state_lock = threading.Lock()
        self.current_item: _WorkItem | None = None
        self.item_started_at = 0.0
        self.quarantined = False

    def quarantine(self) -> _WorkItem | None:
        """Quarantine this endpoint and return its in-flight item once."""

        with self.state_lock:
            if self.quarantined:
                return None
            self.quarantined = True
            item = self.current_item
            self.current_item = None
            return item

    def run(self) -> None:
        try:
            tokenizer = self.tokenizer_factory()
            query = self.query_factory()
            if self.cache is not None:
                query = self.cache.wrap(query)
            query = with_memory_cache(query)
        except Exception as exc:  # noqa: BLE001
            self.quarantine()
            self.event_queue.put(("worker_failed", self, None, exc))
            return

        while not self.stop_event.is_set():
            with self.state_lock:
                if self.quarantined:
                    return
            try:
                item = self.work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.state_lock:
                if self.quarantined:
                    self.work_queue.put(item)
                    return
                self.current_item = item
                self.item_started_at = time.monotonic()
            try:
                result = _score_record(
                    item.record,
                    item.record_id,
                    item.evaluation_key,
                    query,
                    tokenizer,
                    **self.score_options,
                )
            except Exception as exc:  # noqa: BLE001
                failed_item = self.quarantine()
                self.event_queue.put(("worker_failed", self, failed_item, exc))
                return

            with self.state_lock:
                self.current_item = None
            self.event_queue.put(("result", self, item, result))


class _Watchdog(threading.Thread):
    """Requeue work and quarantine an endpoint when one item stalls."""

    def __init__(
        self,
        workers: Sequence[_EndpointWorker],
        event_queue: queue.Queue,
        stop_event: threading.Event,
        timeout: float,
        interval: float,
    ) -> None:
        super().__init__(name="novelty-watchdog", daemon=True)
        self.workers = workers
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.timeout = timeout
        self.interval = interval

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            now = time.monotonic()
            for worker in self.workers:
                with worker.state_lock:
                    item = worker.current_item
                    age = now - worker.item_started_at if item is not None else 0.0
                if item is not None and age > self.timeout:
                    stalled_item = worker.quarantine()
                    if stalled_item is not None:
                        self.event_queue.put(
                            (
                                "worker_stalled",
                                worker,
                                stalled_item,
                                TimeoutError(f"item exceeded {self.timeout:g} seconds"),
                            )
                        )


def _load_completed(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    completed: dict[str, dict[str, Any]] = {}
    malformed = 0
    if not path.exists():
        return completed, malformed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            key = result.get("evaluation_key")
            if isinstance(key, str):
                completed.setdefault(key, result)
    return completed, malformed


def _summarize(
    results: Sequence[dict[str, Any]],
    *,
    n_examples: int,
    n_previously_completed: int,
    n_quarantined_endpoints: int,
) -> dict[str, Any]:
    ci_scores = [
        float(result["creativity_index_normalized"])
        for result in results
        if result.get("creativity_index_normalized") is not None
    ]
    aggregate_scores = [
        float(result["mean_n_novelty_5_13"])
        for result in results
        if result.get("mean_n_novelty_5_13") is not None
    ]
    return {
        "n_examples": n_examples,
        "n_scored": len(ci_scores),
        "n_skipped": sum("skip_reason" in result for result in results),
        "n_errors": sum("error" in result for result in results),
        "n_previously_completed": n_previously_completed,
        "n_quarantined_endpoints": n_quarantined_endpoints,
        "creativity_index_normalized_mean": (
            statistics.fmean(ci_scores) if ci_scores else None
        ),
        "mean_n_novelty_5_13_mean": (
            statistics.fmean(aggregate_scores) if aggregate_scores else None
        ),
    }


def evaluate_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    query_factories: Sequence[Callable[[], Callable[[tuple[str, ...]], int]]],
    tokenizer_factory: Callable[[], Tokenizer],
    *,
    window: int | None = 512,
    minimum_ngram: int = 5,
    n_values: Sequence[int] = tuple(range(4, 14)),
    text_source: str = "auto",
    sample_index: int = 0,
    seed: int = 42,
    include_details: bool = False,
    include_length_truncated: bool = False,
    cache_path: str | Path | None = None,
    cache_namespace: str = "default",
    evaluation_namespace: str = "default",
    resume: bool = False,
    per_item_timeout: float = 600.0,
    watchdog_interval: float = 10.0,
) -> dict[str, Any]:
    """Evaluate JSONL with resume, one writer, retries, and endpoint quarantine."""

    if not query_factories:
        raise ValueError("At least one query endpoint is required")
    if minimum_ngram < 1:
        raise ValueError("minimum_ngram must be positive")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if window is not None and window < minimum_ngram:
        raise ValueError("window must be at least minimum_ngram, or 0/None for full text")
    if per_item_timeout <= 0 or watchdog_interval <= 0:
        raise ValueError("watchdog timeouts must be positive")

    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    settings_payload = {
        "window": window,
        "minimum_ngram": minimum_ngram,
        "n_values": list(n_values),
        "seed": seed,
        "include_details": include_details,
        "include_length_truncated": include_length_truncated,
        "cache_namespace": cache_namespace,
        "evaluation_namespace": evaluation_namespace,
    }
    settings_json = json.dumps(settings_payload, sort_keys=True, separators=(",", ":"))
    settings_digest = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()[:12]
    items: list[_WorkItem] = []
    keys: set[str] = set()
    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        key = _evaluation_key(record_id, text_source, sample_index, settings_digest)
        if key in keys:
            raise ValueError(f"Duplicate evaluation identity: {key}")
        keys.add(key)
        items.append(_WorkItem(index, record, record_id, key))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed, malformed_lines = _load_completed(output_path) if resume else ({}, 0)
    if malformed_lines:
        LOGGER.warning("Ignored %d malformed resume lines in %s", malformed_lines, output_path)
    completed = {key: value for key, value in completed.items() if key in keys}
    pending = [item for item in items if item.evaluation_key not in completed]
    if not resume:
        output_path.write_text("", encoding="utf-8")

    if not pending:
        ordered = [completed[item.evaluation_key] for item in items]
        summary = _summarize(
            ordered,
            n_examples=len(items),
            n_previously_completed=len(completed),
            n_quarantined_endpoints=0,
        )
        summary_path = output_path.with_name(output_path.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    work_queue: queue.Queue = queue.Queue()
    event_queue: queue.Queue = queue.Queue()
    for item in pending:
        work_queue.put(item)
    stop_event = threading.Event()
    shared_cache = SQLiteCountCache(cache_path, cache_namespace) if cache_path else None
    score_options = {
        "window": window,
        "minimum_ngram": minimum_ngram,
        "n_values": n_values,
        "text_source": text_source,
        "sample_index": sample_index,
        "seed": seed,
        "include_details": include_details,
        "include_length_truncated": include_length_truncated,
    }
    workers = [
        _EndpointWorker(
            number=index,
            query_factory=query_factory,
            tokenizer_factory=tokenizer_factory,
            work_queue=work_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            cache=shared_cache,
            score_options=score_options,
        )
        for index, query_factory in enumerate(query_factories)
    ]
    for worker in workers:
        worker.start()
    watchdog = _Watchdog(
        workers,
        event_queue,
        stop_event,
        timeout=per_item_timeout,
        interval=watchdog_interval,
    )
    watchdog.start()

    newly_completed: dict[str, dict[str, Any]] = {}
    quarantined_workers: set[str] = set()
    try:
        with output_path.open("a", encoding="utf-8") as output_handle:
            while len(newly_completed) < len(pending):
                try:
                    event, worker, item, payload = event_queue.get(timeout=0.5)
                except queue.Empty:
                    available = [
                        worker
                        for worker in workers
                        if worker.is_alive() and not worker.quarantined
                    ]
                    if not available:
                        remaining = len(pending) - len(newly_completed)
                        raise RuntimeError(
                            f"All count endpoints were quarantined with {remaining} records pending"
                        )
                    continue

                if event == "result":
                    result = payload
                    if item.evaluation_key in newly_completed or item.evaluation_key in completed:
                        continue
                    newly_completed[item.evaluation_key] = result
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    continue

                quarantined_workers.add(worker.name)
                LOGGER.warning("Quarantined %s: %s", worker.name, payload)
                if item is not None and item.evaluation_key not in newly_completed:
                    work_queue.put(item)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=0.2)
        watchdog.join(timeout=0.2)

    all_results = {**completed, **newly_completed}
    ordered = [all_results[item.evaluation_key] for item in items]
    summary = _summarize(
        ordered,
        n_examples=len(items),
        n_previously_completed=len(completed),
        n_quarantined_endpoints=len(quarantined_workers),
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
