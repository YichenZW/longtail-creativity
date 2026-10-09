"""vLLM backend for high-throughput local inference."""

from __future__ import annotations

from typing import Any

from longtail_creativity.backends.base import ModelBackend
from longtail_creativity.types import ModelOutput


class VLLMBackend(ModelBackend):
    def __init__(self, config) -> None:
        super().__init__(config)
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "The vLLM backend requires: pip install 'longtail-creativity[vllm]'"
            ) from exc
        options = dict(config.options)
        self.use_chat_template = bool(options.pop("use_chat_template", True))
        self.model = LLM(model=config.model_id, revision=config.revision, **options)

    def _format(self, prompt: str) -> str:
        tokenizer = self.model.get_tokenizer()
        if self.use_chat_template and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
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
        from vllm import SamplingParams

        config = dict(generation_config)
        if "max_new_tokens" in config:
            config["max_tokens"] = config.pop("max_new_tokens")
        config["n"] = n
        results = self.model.generate(
            [self._format(prompt) for prompt in prompts],
            SamplingParams(**config),
            use_tqdm=False,
        )
        return [
            [
                ModelOutput(
                    text=sample.text.strip(),
                    raw_output=sample.text,
                    finish_reason=str(sample.finish_reason) if sample.finish_reason else None,
                    usage={"completion_tokens": len(sample.token_ids)},
                )
                for sample in request_output.outputs
            ]
            for request_output in results
        ]

    def close(self) -> None:
        self.model = None
