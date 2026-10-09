"""Configurable normalization for generated model responses.

The framework keeps the untouched backend response in ``raw_output`` and uses
the normalized final answer for task and novelty evaluation.  Built-in
normalizers intentionally cover only model-independent conventions; projects
with a custom response schema can provide a plugin.
"""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from longtail_creativity.types import ModelOutput


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized response text plus extraction provenance."""

    final_answer: str | None
    reasoning_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OutputNormalizer(ABC):
    """Interface for built-in and external output normalizers."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def normalize(self, raw_output: str) -> NormalizationResult:
        """Extract the scoreable final answer from a raw backend response."""


class IdentityNormalizer(OutputNormalizer):
    """Trim surrounding whitespace without interpreting response structure."""

    def normalize(self, raw_output: str) -> NormalizationResult:
        text = raw_output.strip()
        return NormalizationResult(
            final_answer=text or None,
            metadata={"extractor": "identity"},
        )


class ReasoningTagNormalizer(OutputNormalizer):
    """Remove complete reasoning-tag blocks and optionally extract with a regex.

    ``reasoning_tags`` is a sequence of ``[opening, closing]`` pairs. Complete
    pairs are removed. An unclosed block is treated as unfinished reasoning by
    default and is recorded in the provenance; set ``drop_unclosed_reasoning``
    to false to retain it. ``final_answer_pattern`` may be a regular expression;
    its first capture group (or the full match when no group exists) becomes
    the final answer.
    """

    DEFAULT_TAGS = (("<think>", "</think>"), ("<analysis>", "</analysis>"))

    def normalize(self, raw_output: str) -> NormalizationResult:
        configured_tags = self.options.get("reasoning_tags", self.DEFAULT_TAGS)
        tags = tuple(tuple(pair) for pair in configured_tags)
        text = raw_output
        reasoning: list[str] = []
        unclosed_reasoning_tags: list[str] = []

        for pair in tags:
            if len(pair) != 2 or not all(isinstance(value, str) and value for value in pair):
                raise ValueError("Each reasoning_tags entry must contain two non-empty strings")
            opening, closing = pair
            pattern = re.compile(
                re.escape(opening) + r"(.*?)" + re.escape(closing),
                flags=re.DOTALL,
            )

            def remove_block(match: re.Match[str]) -> str:
                reasoning.append(match.group(1).strip())
                return " "

            text = pattern.sub(remove_block, text)
            unclosed_start = text.rfind(opening)
            if unclosed_start >= 0 and text.find(closing, unclosed_start + len(opening)) < 0:
                unclosed_reasoning_tags.append(opening)
                if self.options.get("drop_unclosed_reasoning", True):
                    reasoning.append(text[unclosed_start + len(opening) :].strip())
                    text = text[:unclosed_start]

        stripped_tokens: list[str] = []
        for token in self.options.get("strip_tokens", ()):
            if not isinstance(token, str):
                raise TypeError("strip_tokens must contain strings")
            if token and token in text:
                text = text.replace(token, " ")
                stripped_tokens.append(token)

        extraction_pattern = self.options.get("final_answer_pattern")
        pattern_matched = False
        if extraction_pattern:
            matches = list(re.finditer(str(extraction_pattern), text, flags=re.DOTALL))
            if matches:
                match = matches[-1]
                text = match.group(1) if match.lastindex else match.group(0)
                pattern_matched = True

        text = text.strip()
        return NormalizationResult(
            final_answer=text or None,
            reasoning_content="\n\n".join(part for part in reasoning if part) or None,
            metadata={
                "extractor": "reasoning_tags",
                "removed_reasoning_blocks": len(reasoning),
                "unclosed_reasoning_tags": unclosed_reasoning_tags,
                "stripped_tokens": stripped_tokens,
                "final_answer_pattern_matched": pattern_matched,
            },
        )


def load_normalizer(
    plugin: str = "reasoning_tags",
    options: dict[str, Any] | None = None,
) -> OutputNormalizer:
    """Load a built-in normalizer or ``package.module:ClassName`` plugin."""

    builtins = {
        "identity": IdentityNormalizer,
        "reasoning_tags": ReasoningTagNormalizer,
    }
    if plugin in builtins:
        return builtins[plugin](options)
    if ":" not in plugin:
        raise ValueError("Normalizer plugins must use 'package.module:ClassName'")
    module_name, object_name = plugin.split(":", 1)
    normalizer_type = getattr(importlib.import_module(module_name), object_name)
    normalizer = normalizer_type(options or {})
    if not isinstance(normalizer, OutputNormalizer):
        raise TypeError(f"{plugin} must construct an OutputNormalizer")
    return normalizer


def normalize_model_output(output: ModelOutput, normalizer: OutputNormalizer) -> ModelOutput:
    """Return a model output with normalized text and preserved raw content."""

    raw_output = output.raw_output if output.raw_output is not None else output.text
    result = normalizer.normalize(raw_output)
    return ModelOutput(
        text=result.final_answer or "",
        raw_output=raw_output,
        final_answer=result.final_answer,
        reasoning_content=result.reasoning_content,
        normalization=result.metadata,
        finish_reason=output.finish_reason,
        usage=output.usage,
        trace=output.trace,
    )


def extract_record_text(
    record: dict[str, Any],
    text_source: str = "auto",
    sample_index: int = 0,
) -> tuple[str | None, str]:
    """Select prompt, plain text, or one generated sample from a JSON record.

    The returned source records the field that was actually selected.  In
    ``auto`` mode, normalized ``final_answer`` is preferred over ``text`` and
    ``raw_output``.  Legacy records without ``final_answer`` remain supported.
    """

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if text_source == "prompt":
        value = record.get("prompt")
        return (value if isinstance(value, str) else None), "prompt"

    outputs = record.get("outputs")
    if not isinstance(outputs, list):
        value = record.get(text_source if text_source != "auto" else "text")
        source = text_source if text_source != "auto" else "text"
        return (value if isinstance(value, str) else None), source
    if sample_index >= len(outputs) or not isinstance(outputs[sample_index], dict):
        return None, text_source

    output = outputs[sample_index]
    if text_source == "auto":
        for field_name in ("final_answer", "text", "raw_output"):
            if field_name in output:
                value = output[field_name]
                return (value if isinstance(value, str) else None), field_name
        return None, "auto"

    value = output.get(text_source)
    if text_source == "final_answer" and "final_answer" not in output:
        value = output.get("text")
        return (value if isinstance(value, str) else None), "text"
    return (value if isinstance(value, str) else None), text_source
