"""Generic application of paper-taxonomy conditions."""

from __future__ import annotations

from longtail_creativity.types import TaskExample, TaxonomyConfig


def prepare_prompt(task: TaskExample, taxonomy: TaxonomyConfig) -> str:
    """Apply a user-supplied condition template while preserving taxonomy labels.

    The repository intentionally provides no experiment-specific transformation
    text. A paper reproduction can supply its exact template in YAML, while new
    studies can use the same labels with their own operationalization.
    """

    if taxonomy.prompt_template is None:
        return task.prompt
    return taxonomy.prompt_template.format(prompt=task.prompt)
