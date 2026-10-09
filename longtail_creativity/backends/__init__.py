"""Model backend factory."""

from longtail_creativity.backends.base import ModelBackend
from longtail_creativity.config import ModelConfig


def create_backend(config: ModelConfig) -> ModelBackend:
    if config.backend == "hf":
        from longtail_creativity.backends.huggingface import HuggingFaceBackend

        return HuggingFaceBackend(config)
    if config.backend == "vllm":
        from longtail_creativity.backends.vllm import VLLMBackend

        return VLLMBackend(config)
    if config.backend in {"api", "openai"}:
        from longtail_creativity.backends.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(config)
    raise ValueError(f"Unknown model backend: {config.backend}")


__all__ = ["ModelBackend", "create_backend"]
