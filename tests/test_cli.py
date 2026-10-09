from __future__ import annotations

import pytest

from longtail_creativity.cli import _read_server_addresses, build_parser


def test_novelty_defaults_to_the_local_address_file() -> None:
    args = build_parser().parse_args(["novelty", "--input", "outputs.jsonl"])
    assert args.servers is None
    assert args.server_address_file is None
    assert args.service is None


def test_server_address_file_supports_multiple_servers_and_comments(tmp_path) -> None:
    address_file = tmp_path / "servers.address"
    address_file.write_text(
        "# warm count servers\nhttp://node-a:5555\n\nhttp://node-b:5555\n",
        encoding="utf-8",
    )
    assert _read_server_addresses(address_file) == [
        "http://node-a:5555",
        "http://node-b:5555",
    ]


def test_missing_default_server_explains_the_api_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--service"):
        _read_server_addresses(tmp_path / "missing.address")
