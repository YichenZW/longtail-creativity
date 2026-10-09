"""Tokenizer pairs for corpus-consistent novelty queries."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...

    def detokenize(self, tokens: Sequence[str]) -> str: ...


class MosesTokenizer:
    """NLTK casual tokenization with Moses detokenization, matching CI tooling."""

    def __init__(self) -> None:
        try:
            import nltk
            from sacremoses import MosesDetokenizer
            from unidecode import unidecode
        except ImportError as exc:
            raise RuntimeError(
                "Moses tokenization requires: pip install 'longtail-creativity[ci]'"
            ) from exc
        self._nltk = nltk
        self._detokenizer = MosesDetokenizer(lang="en")
        self._unidecode = unidecode

    def tokenize(self, text: str) -> list[str]:
        return self._nltk.tokenize.casual.casual_tokenize(self._unidecode(text))

    def detokenize(self, tokens: Sequence[str]) -> str:
        return self._detokenizer.detokenize(list(tokens))


class WhitespaceTokenizer:
    """Dependency-free tokenizer for indexes built with whitespace tokenization."""

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\S+", text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


def create_tokenizer(name: str) -> Tokenizer:
    if name == "moses":
        return MosesTokenizer()
    if name == "whitespace":
        return WhitespaceTokenizer()
    raise ValueError("tokenizer must be 'moses' or 'whitespace'")
