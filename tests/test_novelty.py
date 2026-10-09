from __future__ import annotations

import random

import pytest

from longtail_creativity.novelty.algorithm import (
    compute_creativity_metrics,
    compute_familiarity_horizons,
    compute_longest_matches,
)


def corpus_query(corpus: list[str]):
    def count(tokens: tuple[str, ...]) -> int:
        width = len(tokens)
        return sum(
            tuple(corpus[start : start + width]) == tokens
            for start in range(len(corpus) - width + 1)
        )

    return count


def test_complete_match_has_zero_novelty() -> None:
    metrics = compute_creativity_metrics(
        ["a", "b", "c"],
        corpus_query(["a", "b", "c"]),
        minimum_ngram=2,
        n_values=[2, 3],
    )
    assert metrics["creativity_index_normalized"] == 0.0
    assert metrics["n_novelty"] == {2: 0.0, 3: 0.0}


def test_unseen_text_has_maximum_normalized_ci() -> None:
    metrics = compute_creativity_metrics(
        ["a", "b", "c"],
        lambda _: 0,
        minimum_ngram=2,
        n_values=[2],
    )
    assert metrics["creativity_index_normalized"] == 1.0
    assert metrics["n_novelty"] == {2: 1.0}


def test_paper_novelty_aggregate_has_an_unambiguous_name() -> None:
    metrics = compute_creativity_metrics(
        [str(index) for index in range(14)],
        lambda _: 0,
        minimum_ngram=5,
        n_values=range(4, 14),
    )
    assert metrics["mean_n_novelty_5_13"] == 1.0


def test_longest_match_reuses_suffix_lower_bound() -> None:
    calls = 0

    def counted(tokens: tuple[str, ...]) -> int:
        nonlocal calls
        calls += 1
        return corpus_query(["a", "b", "c", "d"])(tokens)

    assert compute_longest_matches(["a", "b", "c", "d"], counted) == [4, 3, 2, 1]
    assert calls <= 8


def test_heap_horizons_match_naive_implementation() -> None:
    rng = random.Random(7)
    for length in range(1, 30):
        longest = [rng.randint(0, length - index) for index in range(length)]
        expected = [0] * length
        for start, width in enumerate(longest):
            for position in range(start, start + width):
                expected[position] = max(expected[position], width)
        assert compute_familiarity_horizons(longest) == expected


def test_invalid_ngram_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_creativity_metrics(["a"], lambda _: 0, minimum_ngram=0)
