from __future__ import annotations

from longtail_creativity.normalization import (
    ReasoningTagNormalizer,
    extract_record_text,
    normalize_model_output,
)
from longtail_creativity.types import ModelOutput


def test_reasoning_tags_are_separated_from_final_answer() -> None:
    output = normalize_model_output(
        ModelOutput(
            text="<think>private reasoning</think> The answer is 7.",
            raw_output="<think>private reasoning</think> The answer is 7.",
        ),
        ReasoningTagNormalizer(),
    )
    assert output.raw_output == "<think>private reasoning</think> The answer is 7."
    assert output.text == "The answer is 7."
    assert output.final_answer == "The answer is 7."
    assert output.reasoning_content == "private reasoning"
    assert output.normalization["removed_reasoning_blocks"] == 1


def test_configured_regex_and_special_tokens_are_recorded() -> None:
    result = ReasoningTagNormalizer(
        {
            "strip_tokens": ["<end>"],
            "final_answer_pattern": r"FINAL:\s*(.*)",
        }
    ).normalize("draft FINAL: result <end>")
    assert result.final_answer == "result"
    assert result.metadata["stripped_tokens"] == ["<end>"]
    assert result.metadata["final_answer_pattern_matched"] is True


def test_auto_text_selection_prefers_explicit_final_answer() -> None:
    text, source = extract_record_text(
        {
            "outputs": [
                {
                    "text": "fallback",
                    "raw_output": "reasoning plus answer",
                    "final_answer": "answer",
                }
            ]
        }
    )
    assert (text, source) == ("answer", "final_answer")


def test_unclosed_reasoning_is_not_treated_as_a_final_answer() -> None:
    result = ReasoningTagNormalizer().normalize("<think>unfinished reasoning")
    assert result.final_answer is None
    assert result.reasoning_content == "unfinished reasoning"
    assert result.metadata["unclosed_reasoning_tags"] == ["<think>"]
