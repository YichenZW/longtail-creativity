"""Backend interface shared by local and hosted models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from longtail_creativity.config import ModelConfig
from longtail_creativity.types import ModelOutput


class ModelBackend(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        generation_config: dict[str, Any],
        n: int = 1,
    ) -> list[list[ModelOutput]]:
        """Return one list of samples per prompt."""

    def close(self) -> None:
        """Release resources held by the backend."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
