"""HTTP count server for a separately downloaded Infini-gram Mini engine."""

from __future__ import annotations

import importlib
import logging
import socket
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_engine(engine_root: str | Path, index_dir: str | Path, load_to_ram: bool = True):
    """Load all numeric index shards through an Infini-gram Mini checkout."""

    engine_root = Path(engine_root).expanduser().resolve()
    engine_package = engine_root / "engine"
    if not engine_package.is_dir():
        raise FileNotFoundError(
            f"{engine_package} does not exist; --engine-root must point to infini-gram-mini"
        )
    sys.path.insert(0, str(engine_package))
    engine_module = importlib.import_module("src.engine")

    index_dir = Path(index_dir).expanduser().resolve()
    if not index_dir.is_dir():
        raise FileNotFoundError(index_dir)
    shards = sorted(
        (path for path in index_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    index_paths = shards or [index_dir]
    logger.info("Loading %d index shard(s) from %s", len(index_paths), index_dir)
    started = time.monotonic()
    engine = engine_module.InfiniGramMiniEngine(
        index_dirs=[str(path) for path in index_paths],
        load_to_ram=load_to_ram,
        get_metadata=False,
    )
    logger.info("Index ready after %.1f seconds", time.monotonic() - started)
    return engine, index_paths


def create_app(engine: Any, index_name: str, n_shards: int):
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:
        raise RuntimeError(
            "The count server requires: pip install 'longtail-creativity[ci]'"
        ) from exc

    try:
        from sacremoses import MosesDetokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The count server requires: pip install 'longtail-creativity[ci]'"
        ) from exc
    detokenizer = MosesDetokenizer(lang="en")
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "index": index_name, "n_shards": n_shards})

    @app.post("/count")
    def count():
        payload = request.get_json(force=True)
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            return jsonify({"error": "tokens must be a list of strings"}), 400
        if not tokens:
            return jsonify({"count": 0})
        result = engine.count(detokenizer.detokenize(tokens))
        return jsonify({"count": int(result.get("count", 0))})

    return app


def run_server(
    engine_root: str | Path,
    index_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 5555,
    advertise_host: str | None = None,
    address_file: str | Path | None = None,
    load_to_ram: bool = True,
) -> None:
    engine, index_paths = load_engine(engine_root, index_dir, load_to_ram)
    app = create_app(engine, Path(index_dir).name, len(index_paths))
    public_host = advertise_host or socket.gethostname()
    address = f"http://{public_host}:{port}"
    if address_file:
        path = Path(address_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(address + "\n", encoding="utf-8")
        logger.info("Wrote server address to %s", path)
    logger.info("Serving count queries at %s", address)
    app.run(host=host, port=port, threaded=True)
