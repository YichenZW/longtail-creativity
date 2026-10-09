from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from longtail_creativity.evaluation.reward_model import (
    DEFAULT_REWARD_MODEL,
    RewardModelScorer,
    format_reward_model_input,
)
from longtail_creativity.evaluation.scorers import ScoringExample


class FakeEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    bos_token = "<s>"
    bos_token_id = 1
    eos_token = "</s>"
    eos_token_id = 2
    pad_token = "</s>"
    pad_token_id = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def apply_chat_template(self, conversation, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        assert add_generation_prompt is False
        return (
            f"<s>user:{conversation[0]['content']}|"
            f"assistant:{conversation[1]['content']}"
        )

    def __call__(self, texts, **options):
        assert options == {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 32,
        }
        self.batches.append(texts)
        return FakeEncoding(texts=texts)


class FakeVector:
    def __init__(self, values) -> None:
        self.values = values
        self.shape = (len(values),)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_):
        return self

    def tolist(self):
        return self.values


class FakeLogits:
    def __init__(self, values) -> None:
        self.values = values
        self.shape = (len(values), 1)

    def squeeze(self, dimension):
        assert dimension == -1
        return FakeVector(self.values)


class FakeModel:
    def __init__(self) -> None:
        self.offset = 0
        self.batch_sizes: list[int] = []

    def parameters(self):
        yield SimpleNamespace(device="fake-device")

    def __call__(self, **inputs):
        assert inputs["device"] == "fake-device"
        size = len(inputs["texts"])
        self.batch_sizes.append(size)
        values = [float(self.offset + index) for index in range(size)]
        self.offset += size
        return SimpleNamespace(logits=FakeLogits(values))


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        return nullcontext()


def test_skywork_is_the_default_reward_model_example() -> None:
    scorer = RewardModelScorer()
    assert scorer.model_id == DEFAULT_REWARD_MODEL


def test_reward_input_is_two_turn_and_removes_leading_bos() -> None:
    text = format_reward_model_input(FakeTokenizer(), "prompt", "response")
    assert text == "user:prompt|assistant:response"


def test_reward_scorer_batches_pairs_and_loads_model_once() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    scorer = RewardModelScorer(
        {"batch_size": 2, "max_length": 32},
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
    )
    examples = [
        ScoringExample(str(index), f"prompt {index}", f"response {index}", None, {})
        for index in range(3)
    ]
    assert scorer.score_batch(examples) == [0.0, 1.0, 2.0]
    assert model.batch_sizes == [2, 1]
    assert len(tokenizer.batches) == 2
    assert tokenizer.batches[0][0] == "user:prompt 0|assistant:response 0"
