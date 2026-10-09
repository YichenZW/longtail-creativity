from __future__ import annotations

import math

from longtail_creativity.likelihood import (
    SequenceBoundaries,
    common_prefix_length,
    prepare_likelihood_sequence,
    score_logprob_segments,
)


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        tokens = [ord(character) for character in text]
        return ([1] + tokens) if add_special_tokens else tokens

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        user = messages[0]["content"]
        assistant = messages[1]["content"]
        return f"<user>{user}</user><assistant>{assistant}</assistant>"


def test_common_prefix_length_handles_boundary_divergence() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4, 5]) == 2


def test_chat_template_boundaries_separate_content_and_answer() -> None:
    sequence = prepare_likelihood_sequence(
        CharacterTokenizer(),
        "question",
        "answer",
        apply_chat_template=True,
    )
    boundaries = sequence.boundaries
    assert 0 < boundaries.content_start < boundaries.content_end < boundaries.input_end
    assert boundaries.input_end < len(sequence.token_ids)


def test_logprob_scoring_excludes_the_unscored_first_token() -> None:
    metrics = score_logprob_segments(
        [None, -1.0, -1.0, -2.0, -2.0],
        SequenceBoundaries(content_start=1, content_end=3, input_end=3),
        truncate_output_tokens=1,
    )
    assert metrics["num_scored_tokens_x"] == 2
    assert metrics["ppl_x"] == math.e
    assert metrics["ppl_y_given_x"] == math.exp(2)
    assert metrics["ppl_y_given_x_trunc1"] == math.exp(2)
