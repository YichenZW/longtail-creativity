"""Input and conditional-output likelihood evaluation with vLLM.

Token boundaries are located by comparing prefixes against the tokenization of
the complete sequence.  This avoids assuming that separately tokenized prompt
and answer strings concatenate cleanly across a BPE boundary.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from longtail_creativity.normalization import extract_record_text


@dataclass(frozen=True)
class SequenceBoundaries:
    """Token offsets for user content, the complete input, and output."""

    content_start: int
    content_end: int
    input_end: int


@dataclass(frozen=True)
class PreparedSequence:
    """A complete token sequence and its likelihood segments."""

    token_ids: list[int]
    boundaries: SequenceBoundaries


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of identical leading values in two sequences."""

    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _encode(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _boundaries_from_character_offsets(
    tokenizer,
    full_text: str,
    *,
    content_start_character: int,
    content_end_character: int,
    answer_start_character: int,
    add_special_tokens: bool,
    max_sequence_length: int | None,
) -> PreparedSequence:
    full_tokens = _encode(tokenizer, full_text, add_special_tokens=add_special_tokens)
    input_tokens = _encode(
        tokenizer,
        full_text[:answer_start_character],
        add_special_tokens=add_special_tokens,
    )
    content_prefix_tokens = _encode(
        tokenizer,
        full_text[:content_start_character],
        add_special_tokens=add_special_tokens,
    )
    content_tokens = _encode(
        tokenizer,
        full_text[:content_end_character],
        add_special_tokens=add_special_tokens,
    )

    content_start = common_prefix_length(content_prefix_tokens, full_tokens)
    content_end = max(content_start, common_prefix_length(content_tokens, full_tokens))
    input_end = max(content_end, common_prefix_length(input_tokens, full_tokens))
    if input_end == 0:
        raise ValueError("Tokenizer produced no common prefix for the input boundary")

    if max_sequence_length is not None:
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        full_tokens = full_tokens[:max_sequence_length]
        input_end = min(input_end, len(full_tokens))
        content_end = min(content_end, input_end)
        content_start = min(content_start, content_end)

    return PreparedSequence(
        token_ids=full_tokens,
        boundaries=SequenceBoundaries(content_start, content_end, input_end),
    )


def prepare_likelihood_sequence(
    tokenizer,
    prompt: str,
    answer: str,
    *,
    apply_chat_template: bool = False,
    plain_template: str = "{prompt}\n{answer}",
    max_sequence_length: int | None = None,
) -> PreparedSequence:
    """Render and tokenize a prompt/answer pair with robust token boundaries."""

    if not prompt:
        raise ValueError("prompt must not be empty")
    if not answer:
        raise ValueError("answer must not be empty")

    if apply_chat_template:
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        answer_start = full_text.rfind(answer)
        prompt_start = full_text.find(prompt)
        if answer_start < 0:
            raise ValueError("Assistant content was changed by the tokenizer chat template")
        if prompt_start < 0 or prompt_start > answer_start:
            raise ValueError("User content was changed or reordered by the tokenizer chat template")
        return _boundaries_from_character_offsets(
            tokenizer,
            full_text,
            content_start_character=prompt_start,
            content_end_character=prompt_start + len(prompt),
            answer_start_character=answer_start,
            add_special_tokens=False,
            max_sequence_length=max_sequence_length,
        )

    if plain_template.count("{prompt}") != 1 or plain_template.count("{answer}") != 1:
        raise ValueError("plain_template must contain {prompt} and {answer} exactly once")
    before_answer, after_answer = plain_template.split("{answer}", 1)
    input_text = before_answer.replace("{prompt}", prompt)
    suffix = after_answer.replace("{prompt}", prompt)
    full_text = input_text + answer + suffix
    prompt_start = input_text.find(prompt)
    if prompt_start < 0:
        raise ValueError("Could not locate prompt in the rendered plain template")
    return _boundaries_from_character_offsets(
        tokenizer,
        full_text,
        content_start_character=prompt_start,
        content_end_character=prompt_start + len(prompt),
        answer_start_character=len(input_text),
        add_special_tokens=True,
        max_sequence_length=max_sequence_length,
    )


def _segment_stats(
    logprobs: Sequence[float | None],
    start: int,
    end: int,
) -> tuple[float, float | None, int]:
    values = [value for value in logprobs[start:end] if value is not None]
    total = sum(values)
    perplexity = math.exp(-total / len(values)) if values else None
    return total, perplexity, len(values)


def score_logprob_segments(
    logprobs: Sequence[float | None],
    boundaries: SequenceBoundaries,
    *,
    truncate_output_tokens: int = 0,
) -> dict[str, Any]:
    """Aggregate token log probabilities into input and conditional-output PPL."""

    if not (0 <= boundaries.content_start <= boundaries.content_end <= boundaries.input_end):
        raise ValueError("Invalid likelihood boundaries")
    if boundaries.input_end > len(logprobs):
        raise ValueError("Input boundary exceeds the returned log probabilities")

    p_x_log, ppl_x, scored_x = _segment_stats(logprobs, 0, boundaries.input_end)
    p_x_content_log, ppl_x_content, scored_content = _segment_stats(
        logprobs,
        boundaries.content_start,
        boundaries.content_end,
    )
    p_y_log, ppl_y, scored_y = _segment_stats(
        logprobs,
        boundaries.input_end,
        len(logprobs),
    )
    result: dict[str, Any] = {
        "p_x_log": p_x_log,
        "p_x_content_log": p_x_content_log,
        "p_y_given_x_log": p_y_log,
        "ppl_x": ppl_x,
        "ppl_x_content": ppl_x_content,
        "ppl_y_given_x": ppl_y,
        "num_tokens_x": boundaries.input_end,
        "num_tokens_x_content": boundaries.content_end - boundaries.content_start,
        "num_tokens_y": len(logprobs) - boundaries.input_end,
        "num_scored_tokens_x": scored_x,
        "num_scored_tokens_x_content": scored_content,
        "num_scored_tokens_y": scored_y,
        "content_start": boundaries.content_start,
        "content_end": boundaries.content_end,
        "total_logprob": sum(value for value in logprobs if value is not None),
    }
    if truncate_output_tokens > 0:
        output_values = [
            value for value in logprobs[boundaries.input_end :] if value is not None
        ][:truncate_output_tokens]
        total = sum(output_values)
        count = len(output_values)
        result[f"p_y_given_x_log_trunc{truncate_output_tokens}"] = total
        result[f"ppl_y_given_x_trunc{truncate_output_tokens}"] = (
            math.exp(-total / count) if count else None
        )
        result[f"num_tokens_y_trunc{truncate_output_tokens}"] = count
    return result


def _actual_token_logprobs(output) -> list[float | None]:
    values: list[float | None] = []
    for token_id, candidates in zip(output.prompt_token_ids, output.prompt_logprobs):
        if candidates is None:
            values.append(None)
            continue
        candidate = candidates.get(token_id)
        if candidate is None:
            raise RuntimeError(f"vLLM did not return a log probability for token {token_id}")
        value = candidate.logprob if hasattr(candidate, "logprob") else candidate
        values.append(float(value))
    return values


def _item_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("variant_id") or record.get("task_id") or record.get("id") or index)


def evaluate_likelihood_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_id: str,
    revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_options: dict[str, Any] | None = None,
    model_options: dict[str, Any] | None = None,
    text_source: str = "auto",
    sample_index: int = 0,
    apply_chat_template: bool = False,
    plain_template: str = "{prompt}\n{answer}",
    max_model_length: int | None = None,
    truncate_output_tokens: int = 0,
    include_length_truncated: bool = False,
) -> dict[str, Any]:
    """Compute input and conditional-output perplexity for generated JSONL."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if truncate_output_tokens < 0:
        raise ValueError("truncate_output_tokens must be non-negative")
    if max_model_length is not None and max_model_length < 3:
        raise ValueError("max_model_length must be at least 3")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "Likelihood evaluation requires: pip install 'longtail-creativity[vllm]'"
        ) from exc

    tokenizer_kwargs = dict(tokenizer_options or {})
    trust_remote_code = bool(tokenizer_kwargs.pop("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id or model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    prepared: list[tuple[int, dict[str, Any], PreparedSequence]] = []
    indexed_results: dict[int, dict[str, Any]] = {}
    sequence_limit = max_model_length - 1 if max_model_length is not None else None
    for index, record in enumerate(records):
        item_id = _item_id(record, index)
        identity = {"record_id": item_id, "task_id": str(record.get("task_id") or item_id)}
        prompt = record.get("prompt")
        answer, selected_source = extract_record_text(record, text_source, sample_index)
        identity.update({"text_source": selected_source, "sample_index": sample_index})
        outputs = record.get("outputs")
        output = (
            outputs[sample_index]
            if isinstance(outputs, list) and sample_index < len(outputs)
            else {}
        )
        if record.get("variant_id") is not None:
            identity["variant_id"] = str(record["variant_id"])
        if record.get("base_task_id") is not None:
            identity["base_task_id"] = str(record["base_task_id"])
        if not isinstance(prompt, str) or not prompt.strip():
            indexed_results[index] = {**identity, "skip_reason": "missing_prompt"}
            continue
        if not answer or not answer.strip():
            indexed_results[index] = {**identity, "skip_reason": "missing_output"}
            continue
        if output.get("finish_reason") == "length" and not include_length_truncated:
            indexed_results[index] = {**identity, "skip_reason": "length_limit"}
            continue
        sequence = prepare_likelihood_sequence(
            tokenizer,
            prompt,
            answer,
            apply_chat_template=apply_chat_template,
            plain_template=plain_template,
            max_sequence_length=sequence_limit,
        )
        if sequence.boundaries.input_end >= len(sequence.token_ids):
            indexed_results[index] = {**identity, "skip_reason": "output_truncated"}
            continue
        prepared.append((index, identity, sequence))

    prepared.sort(key=lambda item: item[2].token_ids[: item[2].boundaries.input_end])
    if prepared:
        llm_kwargs = dict(model_options or {})
        llm_kwargs.setdefault("enable_prefix_caching", True)
        if max_model_length is not None:
            llm_kwargs["max_model_len"] = max_model_length
        model = LLM(model=model_id, revision=revision, **llm_kwargs)
        requests = [{"prompt_token_ids": item[2].token_ids} for item in prepared]
        sampling = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
        model_outputs = model.generate(requests, sampling_params=sampling, use_tqdm=False)
        if len(model_outputs) != len(prepared):
            raise RuntimeError(
                f"vLLM returned {len(model_outputs)} results for {len(prepared)} inputs"
            )
        for (index, identity, sequence), model_output in zip(prepared, model_outputs):
            logprobs = _actual_token_logprobs(model_output)
            indexed_results[index] = {
                **identity,
                **score_logprob_segments(
                    logprobs,
                    sequence.boundaries,
                    truncate_output_tokens=truncate_output_tokens,
                ),
            }

    results = [indexed_results[index] for index in range(len(records))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    conditional_ppl = [
        float(result["ppl_y_given_x"])
        for result in results
        if result.get("ppl_y_given_x") is not None
    ]
    summary = {
        "model_id": model_id,
        "n_examples": len(records),
        "n_scored": len(conditional_ppl),
        "n_skipped": sum("skip_reason" in result for result in results),
        "ppl_y_given_x_mean": statistics.fmean(conditional_ppl) if conditional_ppl else None,
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
