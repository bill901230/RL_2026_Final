# pyright: reportMissingImports=false

import math
from typing import Any, cast

import pytest
import torch

from train.grpo.critic import Critic
from train.grpo.rewards import RewardOrchestrator


class _FakeInputs(dict[str, torch.Tensor]):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        self.device = device
        return self


class _FakeProcessor:
    def __init__(self, decoded):
        self.decoded = decoded

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "chat"

    def __call__(self, text, images, videos, padding, return_tensors):
        return _FakeInputs(input_ids=torch.zeros((1, 3), dtype=torch.long))

    def decode(self, tokens, skip_special_tokens):
        return self.decoded


class _FakeModel:
    def generate(self, **kwargs):
        return torch.zeros((1, 5), dtype=torch.long)


def _vlm_critic(decoded):
    critic = Critic.__new__(Critic)
    critic.backend = "vlm"
    critic.device = "cpu"
    critic.processor = _FakeProcessor(decoded)
    critic_any = cast(Any, critic)
    critic_any.model = _FakeModel()
    critic_any._process_vision_info = lambda messages: ([], [])
    return critic


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        pytest.param("7/10", 0.7, id="slash-rating"),
        pytest.param("7", 0.7, id="integer-rating"),
        pytest.param("15", 1.0, id="high-clamp"),
        pytest.param("-3", 0.0, id="low-clamp"),
        pytest.param("banana", 0.0, id="fallback-banana"),
    ],
)
def test_vlm_rating_parser_clamps_and_falls_back(decoded, expected):
    assert _vlm_critic(decoded).score(object(), "sharp moss texture") == pytest.approx(expected)


class _FakeClip:
    def __init__(self, cosine_value):
        self.cosine_value = cosine_value

    def embed_image(self, crop_pil):
        return torch.tensor([1.0, 0.0])

    def embed_text(self, prompt):
        return torch.tensor([1.0, 0.0])

    def cosine(self, image_embedding, text_embedding):
        return torch.tensor(self.cosine_value)


@pytest.mark.parametrize(
    ("cosine_value", "expected"),
    [
        pytest.param(0.42, 0.42, id="inside-range"),
        pytest.param(1.25, 1.0, id="high-clamp"),
        pytest.param(-0.25, 0.0, id="low-clamp"),
        pytest.param(float("nan"), 0.0, id="nan-fallback"),
    ],
)
def test_clipscore_backend_returns_finite_unit_interval(cosine_value, expected):
    critic = Critic(
        {"rewards": {"r_crit": {"backend": "clipscore"}}},
        clip_embedder=_FakeClip(cosine_value),
        device="cpu",
    )

    score = critic.score(object(), "sharp moss texture")

    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(expected)


def _phr_orchestrator(fillers):
    rewards = RewardOrchestrator.__new__(RewardOrchestrator)
    rewards.cfg = {"r_phr": {"fillers": fillers}}
    return rewards


def test_r_phr_penalizes_blacklisted_filler_and_keeps_clean_prompt_neutral():
    rewards = _phr_orchestrator(["the image shows", "i can see"])

    assert rewards._r_phr("The image shows bright moss on bark", None) < 0.0
    assert rewards._r_phr("bright moss on bark, macro texture", None) == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Default fillers miss inserted-adjective variants like 'the first image shows'; "
        "rewards.py is owned by the R_phr validator."
    ),
)
def test_r_phr_default_fillers_should_penalize_first_image_shows_variant():
    default_fillers = [
        "the image shows",
        "i can see",
        "this is a",
        "this image",
        "in the image",
        "there is",
        "there are",
        "appears to be",
        "it looks like",
        "a set of words",
    ]
    rewards = _phr_orchestrator(default_fillers)

    assert rewards._r_phr("the first image shows a red bicycle", None) < 0.0
