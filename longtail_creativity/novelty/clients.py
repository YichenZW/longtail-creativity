"""Count-query clients and persistent caching."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import requests


class HTTPCountClient:
    """Client for a local ``longtail-creativity serve-index`` process."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def __call__(self, tokens: tuple[str, ...]) -> int:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/count",
                    json={"tokens": list(tokens)},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return int(response.json()["count"])
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"Count query failed after retries: {self.base_url}") from last_error


class InfiniGramAPIClient:
    """Client for the public Infini-gram or Infini-gram Mini API."""

    ENDPOINTS: ClassVar[dict[str, str]] = {
        "infini-gram": "https://api.infini-gram.io/",
        "infini-gram-mini": "https://api.infini-gram-mini.io/",
    }

    def __init__(
        self,
        detokenize: Callable[[Sequence[str]], str],
        service: str = "infini-gram",
        index: str | None = None,
        timeout: float = 30.0,
        throttle_seconds: float = 0.15,
        max_retries: int = 4,
    ) -> None:
        if service not in self.ENDPOINTS:
            raise ValueError(f"Unknown Infini-gram service: {service}")
        self.detokenize = detokenize
        self.service = service
        self.index = index or (
            "v2_dclm_all" if service == "infini-gram-mini" else "v4_rpj_llama_s4"
        )
        self.timeout = timeout
        self.throttle_seconds = throttle_seconds
        self.max_retries = max_retries
        self.session = requests.Session()

    def _payload(self, text: str) -> dict[str, str]:
        if self.service == "infini-gram-mini":
            return {"index": self.index, "query_type": "count", "query": text}
        return {
            "corpus": self.index,
            "engine": "c++",
            "query_type": "count",
            "query": text,
        }

    def __call__(self, tokens: tuple[str, ...]) -> int:
        text = self.detokenize(tokens)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.throttle_seconds:
                time.sleep(self.throttle_seconds)
            try:
                response = self.session.post(
                    self.ENDPOINTS[self.service],
                    json=self._payload(text),
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return int(response.json()["count"])
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Infini-gram query failed after retries: {text!r}") from last_error


class SQLiteCountCache:
    """Thread-safe persistent cache keyed by token sequence and corpus namespace."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS counts (
        namespace TEXT NOT NULL,
        token_json TEXT NOT NULL,
        count INTEGER NOT NULL,
        queried_at TEXT NOT NULL,
        PRIMARY KEY (namespace, token_json)
    )
    """

    def __init__(self, path: str | Path, namespace: str) -> None:
        self.path = str(path)
        self.namespace = namespace
        self._local = threading.local()
        self._write_lock = threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = self._connection()
        connection.execute(self._SCHEMA)
        connection.commit()

    def _connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "connection"):
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = connection
        return self._local.connection

    @staticmethod
    def _key(tokens: Sequence[str]) -> str:
        return json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":"))

    def get(self, tokens: Sequence[str]) -> int | None:
        row = (
            self._connection()
            .execute(
                "SELECT count FROM counts WHERE namespace = ? AND token_json = ?",
                (self.namespace, self._key(tokens)),
            )
            .fetchone()
        )
        return int(row[0]) if row is not None else None

    def put(self, tokens: Sequence[str], count: int) -> None:
        with self._write_lock:
            self._connection().execute(
                "INSERT OR REPLACE INTO counts VALUES (?, ?, ?, ?)",
                (
                    self.namespace,
                    self._key(tokens),
                    int(count),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._connection().commit()

    def wrap(self, query: Callable[[tuple[str, ...]], int]):
        def cached(tokens: tuple[str, ...]) -> int:
            value = self.get(tokens)
            if value is not None:
                return value
            value = int(query(tokens))
            self.put(tokens, value)
            return value

        return cached


def with_memory_cache(query: Callable[[tuple[str, ...]], int], maxsize: int = 65_536):
    """Add an in-process LRU in front of any count-query callable."""

    return lru_cache(maxsize=maxsize)(query)
