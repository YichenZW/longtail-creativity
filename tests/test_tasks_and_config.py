from __future__ import annotations

import json

import pytest

from longtail_creativity.config import TaskConfig, load_config
from longtail_creativity.tasks import load_tasks
from longtail_creativity.taxonomy import prepare_prompt


def test_jsonl_loader_maps_fields(tmp_path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        json.dumps({"key": "x", "question": "Prompt", "target": "Answer", "group": 2}) + "\n",
        encoding="utf-8",
    )
    tasks = load_tasks(
        TaskConfig(
            source="local",
            path=str(dataset),
            prompt_field="question",
            id_field="key",
            answer_field="target",
            metadata_fields=("group",),
        )
    )
    assert tasks[0].id == "x"
    assert tasks[0].answer == "Answer"
    assert tasks[0].metadata == {"group": 2}


def test_config_parses_taxonomy_and_template(tmp_path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
model:
  backend: hf
  model_id: organization/model
task:
  source: local
  path: tasks.jsonl
taxonomy:
  transformation: rare_output
  depth: semantic
  focus: outcome
  reference: historical
  prompt_template: "Instruction: {prompt}"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    from longtail_creativity.types import TaskExample

    assert prepare_prompt(TaskExample("1", "hello"), config.taxonomy) == "Instruction: hello"


def test_template_must_preserve_prompt(tmp_path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
model: {backend: hf, model_id: organization/model}
task: {source: local, path: tasks.jsonl}
taxonomy: {prompt_template: "missing placeholder"}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must contain"):
        load_config(config_path)
