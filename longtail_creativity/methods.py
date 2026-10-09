"""Plugin API for generation methods.

Only direct generation ships with the repository. Concrete improvement methods
belong in external experiment packages and can be loaded with
``package.module:ClassName``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from longtail_creativity.backends.base import ModelBackend
from longtail_creativity.types import ModelOutput, TaskExample


class GenerationMethod(ABC):
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def generate(
        self,
        backend: ModelBackend,
        prompts: list[str],
        generation_config: dict[str, Any],
        tasks: list[TaskExample],
    ) -> list[list[ModelOutput]]:
        """Generate outputs for a batch of prompts."""


class DirectGeneration(GenerationMethod):
    """Unmodified, single-pass generation used as the baseline."""

    def generate(
        self,
        backend: ModelBackend,
        prompts: list[str],
        generation_config: dict[str, Any],
        tasks: list[TaskExample],
    ) -> list[list[ModelOutput]]:
        del tasks
        config = dict(generation_config)
        n = int(config.pop("n", 1))
        return backend.generate_batch(prompts, config, n=n)


def load_method(plugin: str, options: dict[str, Any] | None = None) -> GenerationMethod:
    """Load the built-in baseline or an external GenerationMethod class."""

    if plugin == "direct":
        return DirectGeneration(options)
    if ":" not in plugin:
        raise ValueError("Method plugins must use 'package.module:ClassName'")
    module_name, object_name = plugin.split(":", 1)
    module = importlib.import_module(module_name)
    method_type = getattr(module, object_name)
    method = method_type(options or {})
    if not isinstance(method, GenerationMethod):
        raise TypeError(f"{plugin} must construct a GenerationMethod")
    return method
