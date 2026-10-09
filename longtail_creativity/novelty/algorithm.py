"""Efficient exact-match Creativity Index and n-gram novelty computation.

For a token sequence, ``longest_match[i]`` is the longest prefix beginning at
position ``i`` that occurs in a reference corpus. The suffix property of exact
matches gives ``longest_match[i + 1] >= longest_match[i] - 1``. Reusing that
lower bound reduces corpus count requests to at most one failed request per
position plus a linear number of successful extensions.
"""

from __future__ import annotations

import heapq
import statistics
from collections.abc import Callable, Sequence
from typing import Any

CountQuery = Callable[[tuple[str, ...]], int]


def compute_longest_matches(tokens: Sequence[str], count_query: CountQuery) -> list[int]:
    """Compute the longest corpus match beginning at every token position."""

    longest = [0] * len(tokens)
    lower_bound = 0
    for start in range(len(tokens)):
        while start + lower_bound < len(tokens):
            candidate = tuple(tokens[start : start + lower_bound + 1])
            if count_query(candidate) <= 0:
                break
            lower_bound += 1
        longest[start] = lower_bound
        lower_bound = max(0, lower_bound - 1)
    return longest


def compute_familiarity_horizons(longest_matches: Sequence[int]) -> list[int]:
    """Compute the largest matching n-gram covering each token in O(N log N)."""

    horizons = [0] * len(longest_matches)
    active: list[tuple[int, int]] = []
    for position, length in enumerate(longest_matches):
        if length < 0 or position + length > len(longest_matches):
            raise ValueError(f"Invalid longest-match length {length} at position {position}")
        if length:
            heapq.heappush(active, (-length, position + length))
        while active and active[0][1] <= position:
            heapq.heappop(active)
        if active:
            horizons[position] = -active[0][0]
    return horizons


def compute_n_novelty(
    longest_matches: Sequence[int],
    n_values: Sequence[int],
) -> dict[int, float]:
    """Return the fraction of novel windows for every requested n-gram size."""

    length = len(longest_matches)
    scores: dict[int, float] = {}
    for n in n_values:
        if n < 1:
            raise ValueError("n-gram lengths must be positive")
        windows = length - n + 1
        if windows <= 0:
            continue
        known = sum(longest_matches[index] >= n for index in range(windows))
        scores[int(n)] = 1.0 - (known / windows)
    return scores


def compute_creativity_index(horizons: Sequence[int], minimum_ngram: int) -> tuple[float, float]:
    """Return raw and length-normalized reconstruction-based Creativity Index."""

    if minimum_ngram < 1:
        raise ValueError("minimum_ngram must be positive")
    length = len(horizons)
    if length == 0:
        return 0.0, 0.0
    total = sum(max(0, length - max(minimum_ngram - 1, horizon)) for horizon in horizons)
    raw = total / length
    maximum = length - minimum_ngram + 1
    normalized = raw / maximum if maximum > 0 else 0.0
    return raw, normalized


def compute_creativity_metrics(
    tokens: Sequence[str],
    count_query: CountQuery,
    minimum_ngram: int = 5,
    n_values: Sequence[int] = tuple(range(4, 14)),
    include_details: bool = False,
) -> dict[str, Any]:
    """Compute CI and multiple n-novelty scores from one set of corpus queries."""

    longest = compute_longest_matches(tokens, count_query)
    horizons = compute_familiarity_horizons(longest)
    raw, normalized = compute_creativity_index(horizons, minimum_ngram)
    n_novelty = compute_n_novelty(longest, n_values)
    paper_range_scores = [score for n, score in n_novelty.items() if 5 <= n <= 13]
    result: dict[str, Any] = {
        "creativity_index": raw,
        "creativity_index_normalized": normalized,
        "n_novelty": n_novelty,
        "mean_n_novelty_5_13": (
            statistics.fmean(paper_range_scores) if paper_range_scores else None
        ),
        "n_tokens": len(tokens),
        "minimum_ngram": minimum_ngram,
    }
    if include_details:
        result["longest_matches"] = longest
        result["familiarity_horizons"] = horizons
    return result
