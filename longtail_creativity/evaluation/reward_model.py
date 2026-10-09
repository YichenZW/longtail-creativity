"""Batched Hugging Face reward-model evaluation for open-ended tasks."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from longtail_creativity.evaluation.scorers import Scorer, ScoringExample

LOGGER = logging.getLogger(__name__)

DEFAULT_REWARD_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


def format_reward_model_input(
    tokenizer,
    prompt: str,
    response: str,
    *,
    remove_leading_bos: bool = True,
) -> str:
    """Render a two-turn user/assistant conversation for a reward model."""

    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    bos_token = getattr(tokenizer, "bos_token", None)
    if remove_leading_bos and bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]
    return text


class RewardModelScorer(Scorer):
    """Assign scalar rewards to ``(prompt, response)`` pairs in batches.

    The default model is Skywork-Reward-V2-Llama-3.1-8B. Any compatible
    Hugging Face sequence-classification checkpoint with one output logit can
    be selected through ``model_id``.
    """

    requires_answer = False

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        tokenizer=None,
        model=None,
        torch_module=None,
    ) -> None:
        super().__init__(options)
        self.model_id = str(self.options.get("model_id", DEFAULT_REWARD_MODEL))
        self.tokenizer_id = str(self.options.get("tokenizer_id", self.model_id))
        self.revision = self.options.get("revision")
        self.batch_size = int(self.options.get("batch_size", 4))
        self.max_length = int(self.options.get("max_length", 4096))
        self.remove_leading_bos = bool(self.options.get("remove_leading_bos", True))
        if self.batch_size < 1:
            raise ValueError("reward_model batch_size must be positive")
        if self.max_length < 1:
            raise ValueError("reward_model max_length must be positive")

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._device: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            if self._device is None:
                self._device = str(self.options.get("device") or "cpu")
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Reward-model evaluation requires: pip install 'longtail-creativity[reward]'"
            ) from exc

        self._torch = torch
        requested_device = self.options.get("device")
        if requested_device in (None, "auto"):
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = str(requested_device)

        trust_remote_code = bool(self.options.get("trust_remote_code", False))
        tokenizer_options = dict(self.options.get("tokenizer_options") or {})
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
            **tokenizer_options,
        )
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError("Reward-model tokenizer has no pad or EOS token")
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_options = dict(self.options.get("model_options") or {})
        model_options.setdefault("num_labels", 1)
        dtype_name = self.options.get("dtype")
        if dtype_name is None:
            dtype_name = "bfloat16" if self._device.startswith("cuda") else "float32"
        if dtype_name != "auto":
            if not hasattr(torch, str(dtype_name)):
                raise ValueError(f"Unknown torch dtype: {dtype_name}")
            model_options.setdefault("torch_dtype", getattr(torch, str(dtype_name)))

        attention = str(self.options.get("attn_implementation", "auto"))
        if attention == "auto":
            has_flash_attention = importlib.util.find_spec("flash_attn") is not None
            attention = (
                "flash_attention_2"
                if self._device.startswith("cuda") and has_flash_attention
                else "eager"
            )
        model_options.setdefault("attn_implementation", attention)

        device_map = self.options.get("device_map")
        if device_map is not None:
            model_options.setdefault("device_map", device_map)
        LOGGER.info("Loading reward model %s on %s", self.model_id, self._device)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
            **model_options,
        )
        if device_map is None:
            self._model.to(self._device)
        self._model.eval()

    def _input_device(self):
        try:
            return next(self._model.parameters()).device
        except (AttributeError, StopIteration):
            return self._device

    @staticmethod
    def _logits_to_scores(logits, expected: int) -> list[float]:
        shape = tuple(logits.shape)
        if len(shape) == 2 and shape[-1] == 1:
            logits = logits.squeeze(-1)
        elif len(shape) != 1:
            raise ValueError(
                "Reward model must return one scalar logit per example; "
                f"received shape {shape}"
            )
        values = logits.detach().float().cpu().reshape(-1).tolist()
        if not isinstance(values, list):
            values = [values]
        scores = [float(value) for value in values]
        if len(scores) != expected:
            raise RuntimeError(
                f"Reward model returned {len(scores)} scores for {expected} examples"
            )
        return scores

    def score_batch(self, examples: list[ScoringExample]) -> list[float]:
        if not examples:
            return []
        self._ensure_loaded()
        scores: list[float] = []
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            formatted = [
                format_reward_model_input(
                    self._tokenizer,
                    example.prompt,
                    example.prediction,
                    remove_leading_bos=self.remove_leading_bos,
                )
                for example in batch
            ]
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = inputs.to(self._input_device())
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits
            batch_scores = self._logits_to_scores(logits, len(batch))
            scores.extend(batch_scores)
            LOGGER.info(
                "Reward-scored examples %d-%d (range %.3f to %.3f)",
                start,
                start + len(batch) - 1,
                min(batch_scores),
                max(batch_scores),
            )
        return scores

    def score(self, prediction: str, answer: Any, metadata: dict[str, Any]) -> float:
        """Support direct use when ``metadata['prompt']`` is supplied."""

        del answer
        prompt = metadata.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Direct reward scoring requires metadata['prompt']")
        example = ScoringExample("direct", prompt, prediction, None, metadata)
        return self.score_batch([example])[0]

    def summary_metadata(self) -> dict[str, Any]:
        return {
            "reward_model": self.model_id,
            "reward_tokenizer": self.tokenizer_id,
            "reward_batch_size": self.batch_size,
            "reward_max_length": self.max_length,
        }

    def close(self) -> None:
        self._model = None
        if self._torch is not None and getattr(self._torch, "cuda", None) is not None:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
