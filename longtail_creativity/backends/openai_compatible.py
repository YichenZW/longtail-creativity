"""Backend for OpenAI-compatible chat-completions APIs."""

from __future__ import annotations

import os
from typing import Any

from longtail_creativity.backends.base import ModelBackend
from longtail_creativity.types import ModelOutput


class OpenAICompatibleBackend(ModelBackend):
    def __init__(self, config) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The API backend requires: pip install 'longtail-creativity[api]'"
            ) from exc

        options = dict(config.options)
        api_key_env = str(options.pop("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        client_options = dict(options.pop("client_options", {}))
        if "base_url" in options:
            client_options["base_url"] = options.pop("base_url")
        self.request_options = options
        self.client = OpenAI(api_key=api_key, **client_options)

    def generate_batch(
        self,
        prompts: list[str],
        generation_config: dict[str, Any],
        n: int = 1,
    ) -> list[list[ModelOutput]]:
        config = dict(self.request_options)
        config.update(generation_config)
        if "max_new_tokens" in config:
            config["max_tokens"] = config.pop("max_new_tokens")

        batches: list[list[ModelOutput]] = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                **config,
            )
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            batches.append(
                [
                    ModelOutput(
                        text=(choice.message.content or "").strip(),
                        raw_output=choice.message.content or "",
                        finish_reason=choice.finish_reason,
                        usage=usage,
                    )
                    for choice in response.choices
                ]
            )
        return batches

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()
