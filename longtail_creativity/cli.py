"""Command-line interface for generation, task scoring, and novelty evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

from longtail_creativity.config import load_config


def _add_generate(subparsers) -> None:
    parser = subparsers.add_parser("generate", help="Generate model outputs from a YAML config")
    parser.add_argument("config", help="Path to a run YAML file")
    parser.set_defaults(handler=_generate)


def _generate(args) -> None:
    from longtail_creativity.generation import run_generation

    print(json.dumps(run_generation(load_config(args.config)), indent=2))


def _add_evaluate(subparsers) -> None:
    parser = subparsers.add_parser("evaluate", help="Score generated task outputs")
    parser.add_argument("config", help="Run YAML whose task mapping should be used")
    parser.add_argument("--outputs", required=True, help="Generated outputs.jsonl")
    parser.add_argument("--scorer", default="exact_match")
    parser.add_argument("--scorer-options", default="{}", help="JSON object passed to the scorer")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--destination", default=None)
    parser.add_argument("--include-length-truncated", action="store_true")
    parser.set_defaults(handler=_evaluate)


def _evaluate(args) -> None:
    from longtail_creativity.evaluation.runner import evaluate_outputs

    config = load_config(args.config)
    summary = evaluate_outputs(
        args.outputs,
        config.task,
        scorer_name=args.scorer,
        scorer_options=json.loads(args.scorer_options),
        sample_index=args.sample_index,
        destination=args.destination,
        include_length_truncated=args.include_length_truncated,
    )
    print(json.dumps(summary, indent=2))


def _add_novelty(subparsers) -> None:
    parser = subparsers.add_parser(
        "novelty", help="Compute Creativity Index and n-gram novelty for JSONL records"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    endpoints = parser.add_mutually_exclusive_group()
    endpoints.add_argument("--servers", nargs="+", help="Explicit local count-server URLs")
    endpoints.add_argument(
        "--service",
        choices=["infini-gram", "infini-gram-mini"],
        help="Fall back to a public Infini-gram API",
    )
    endpoints.add_argument(
        "--server-address-file",
        default=None,
        help="Local server URLs, one per line (default: runtime/ci-server.address)",
    )
    parser.add_argument("--index", default=None, help="Corpus index for the public API")
    parser.add_argument("--tokenizer", choices=["moses", "whitespace"], default="moses")
    parser.add_argument("--window", type=int, default=512, help="0 scores the full text")
    parser.add_argument("--minimum-ngram", type=int, default=5)
    parser.add_argument("--n-values", type=int, nargs="+", default=list(range(4, 14)))
    parser.add_argument(
        "--text-source",
        "--text-field",
        dest="text_source",
        choices=["auto", "prompt", "text", "final_answer", "raw_output"],
        default="auto",
        help="Text to score; use 'prompt' for input novelty",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default=None, help="Optional SQLite cache path")
    parser.add_argument(
        "--cache-namespace",
        default="default",
        help="Stable corpus name used by the cache and resume fingerprint",
    )
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--include-length-truncated", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append and skip completed IDs")
    parser.add_argument("--per-item-timeout", type=float, default=600.0)
    parser.add_argument("--watchdog-interval", type=float, default=10.0)
    parser.set_defaults(handler=_novelty)


def _http_factory(url: str):
    from longtail_creativity.novelty.clients import HTTPCountClient

    return HTTPCountClient(url)


def _api_factory(service: str, index: str | None, tokenizer_name: str):
    from longtail_creativity.novelty.clients import InfiniGramAPIClient
    from longtail_creativity.novelty.tokenization import create_tokenizer

    tokenizer = create_tokenizer(tokenizer_name)
    return InfiniGramAPIClient(tokenizer.detokenize, service=service, index=index)


def _read_server_addresses(path: str | Path) -> list[str]:
    address_path = Path(path)
    if not address_path.is_file():
        raise FileNotFoundError(
            f"Local CI server address file not found: {address_path}. "
            "Start 'longtail-creativity serve-index', pass --servers, or use --service."
        )
    addresses = [
        line.strip()
        for line in address_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not addresses:
        raise ValueError(f"No server URLs found in {address_path}")
    return addresses


def _novelty(args) -> None:
    from longtail_creativity.novelty.batch import evaluate_jsonl
    from longtail_creativity.novelty.tokenization import create_tokenizer

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else input_path.with_name("novelty_metrics.jsonl")
    )
    tokenizer_factory = partial(create_tokenizer, args.tokenizer)
    if not args.service:
        server_urls = args.servers or _read_server_addresses(
            args.server_address_file or "runtime/ci-server.address"
        )
        factories = [partial(_http_factory, url) for url in server_urls]
        cache_namespace = args.cache_namespace
        evaluation_namespace = f"{args.tokenizer}:local:{cache_namespace}"
    else:
        factories = [partial(_api_factory, args.service, args.index, args.tokenizer)]
        default_index = (
            "v2_dclm_all" if args.service == "infini-gram-mini" else "v4_rpj_llama_s4"
        )
        resolved_index = args.index or default_index
        cache_namespace = (
            args.cache_namespace
            if args.cache_namespace != "default"
            else f"{args.service}:{resolved_index}:{args.tokenizer}"
        )
        evaluation_namespace = f"{args.tokenizer}:{args.service}:{resolved_index}"
    summary = evaluate_jsonl(
        input_path,
        output_path,
        factories,
        tokenizer_factory,
        window=args.window or None,
        minimum_ngram=args.minimum_ngram,
        n_values=args.n_values,
        text_source=args.text_source,
        sample_index=args.sample_index,
        seed=args.seed,
        include_details=args.include_details,
        include_length_truncated=args.include_length_truncated,
        cache_path=args.cache,
        cache_namespace=cache_namespace,
        evaluation_namespace=evaluation_namespace,
        resume=args.resume,
        per_item_timeout=args.per_item_timeout,
        watchdog_interval=args.watchdog_interval,
    )
    print(json.dumps(summary, indent=2))


def _json_object(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def _add_likelihood(subparsers) -> None:
    parser = subparsers.add_parser(
        "likelihood",
        help="Compute input and conditional-output perplexity with vLLM",
    )
    parser.add_argument("--input", required=True, help="Generation outputs.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", required=True, help="Scoring model name or path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer", default=None, help="Defaults to the scoring model")
    parser.add_argument("--tokenizer-options", type=_json_object, default={})
    parser.add_argument("--model-options", type=_json_object, default={})
    parser.add_argument(
        "--text-source",
        choices=["auto", "text", "final_answer", "raw_output"],
        default="auto",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    formatting = parser.add_mutually_exclusive_group()
    formatting.add_argument("--apply-chat-template", action="store_true")
    formatting.add_argument(
        "--plain-template",
        default="{prompt}\n{answer}",
        help="Template containing one {prompt} and one {answer}",
    )
    parser.add_argument("--max-model-length", type=int, default=None)
    parser.add_argument("--truncate-output-tokens", type=int, default=0)
    parser.add_argument("--include-length-truncated", action="store_true")
    parser.set_defaults(handler=_likelihood)


def _likelihood(args) -> None:
    from longtail_creativity.likelihood import evaluate_likelihood_jsonl

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else input_path.with_name("likelihood_metrics.jsonl")
    )
    summary = evaluate_likelihood_jsonl(
        input_path,
        output_path,
        model_id=args.model,
        revision=args.revision,
        tokenizer_id=args.tokenizer,
        tokenizer_options=args.tokenizer_options,
        model_options=args.model_options,
        text_source=args.text_source,
        sample_index=args.sample_index,
        apply_chat_template=args.apply_chat_template,
        plain_template=args.plain_template,
        max_model_length=args.max_model_length,
        truncate_output_tokens=args.truncate_output_tokens,
        include_length_truncated=args.include_length_truncated,
    )
    print(json.dumps(summary, indent=2))


def _add_serve_index(subparsers) -> None:
    parser = subparsers.add_parser("serve-index", help="Serve a local Infini-gram Mini index")
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--advertise-host", default=None)
    parser.add_argument("--address-file", default="runtime/ci-server.address")
    parser.add_argument("--no-load-to-ram", action="store_true")
    parser.set_defaults(handler=_serve_index)


def _serve_index(args) -> None:
    from longtail_creativity.novelty.server import run_server

    run_server(
        engine_root=args.engine_root,
        index_dir=args.index_dir,
        host=args.host,
        port=args.port,
        advertise_host=args.advertise_host,
        address_file=args.address_file,
        load_to_ram=not args.no_load_to_ram,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="longtail-creativity")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate(subparsers)
    _add_evaluate(subparsers)
    _add_novelty(subparsers)
    _add_likelihood(subparsers)
    _add_serve_index(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.handler(args)


if __name__ == "__main__":
    main()
