"""Hugging Face Transformers backend."""

from __future__ import annotations

from typing import Any

from longtail_creativity.backends.base import ModelBackend
from longtail_creativity.types import ModelOutput


class HuggingFaceBackend(ModelBackend):
    def __init__(self, config) -> None:
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires: pip install 'longtail-creativity[hf]'"
            ) from exc

        options = dict(config.options)
        self.use_chat_template = bool(options.pop("use_chat_template", True))
        trust_remote_code = bool(options.pop("trust_remote_code", False))
        tokenizer_options = dict(options.pop("tokenizer_options", {}))
        dtype_name = options.pop("dtype", None)
        if dtype_name:
            options["torch_dtype"] = getattr(torch, dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=trust_remote_code,
            **tokenizer_options,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=trust_remote_code,
            **options,
        )

    def _format(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate_batch(
        self,
        prompts: list[str],
        generation_config: dict[str, Any],
        n: int = 1,
    ) -> list[list[ModelOutput]]:
        import torch

        formatted = [self._format(prompt) for prompt in prompts]
        encoded = self.tokenizer(formatted, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        config = dict(generation_config)
        if "max_tokens" in config:
            config["max_new_tokens"] = config.pop("max_tokens")
        max_new_tokens = int(config.get("max_new_tokens", 256))
        config.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        config.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        config["num_return_sequences"] = n

        with torch.inference_mode():
            sequences = self.model.generate(**encoded, **config)
        prompt_width = encoded["input_ids"].shape[1]
        generated = sequences[:, prompt_width:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        grouped: list[list[ModelOutput]] = []
        for prompt_index in range(len(prompts)):
            samples: list[ModelOutput] = []
            for sample_index in range(n):
                row = prompt_index * n + sample_index
                token_count = int(generated[row].shape[0])
                samples.append(
                    ModelOutput(
                        text=decoded[row].strip(),
                        raw_output=decoded[row],
                        finish_reason="length" if token_count >= max_new_tokens else "stop",
                        usage={"completion_tokens": token_count},
                    )
                )
            grouped.append(samples)
        return grouped

    def close(self) -> None:
        self.model = None
