import importlib
import sys
import types

import torch
from PIL import Image


if "qwen_vl_utils" not in sys.modules:
    qwen_stub = types.ModuleType("qwen_vl_utils")
    setattr(qwen_stub, "process_vision_info", lambda messages: ([], []))
    sys.modules["qwen_vl_utils"] = qwen_stub


rollout = importlib.import_module("train.grpo.rollout")


class _FakeInputs(dict[str, torch.Tensor]):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        self.device = device
        return self


class _FakePolicy:
    def __init__(self):
        self.tokens = [10, 11, 12]
        self.cursor = 0

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        count = kwargs.get("num_return_sequences", 1)
        rows = []
        for _ in range(count):
            token = self.tokens[self.cursor % len(self.tokens)]
            self.cursor += 1
            generated = torch.tensor([token], dtype=input_ids.dtype)
            rows.append(torch.cat([input_ids[0], generated]))
        return torch.stack(rows)


class _FakeProcessor:
    text_by_token = {10: "p0", 11: "p1", 12: "p2"}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(self.text_by_token[int(token)] for token in tokens)


class _FakeRewards:
    def __init__(self):
        self.values = {"p0": 1.0, "p1": 5.0, "p2": 2.0}

    def compute(self, text, ctx):
        return self.values[text], {"score": self.values[text]}


class _FakeSR:
    def __init__(self):
        self.prompts = []

    def render(self, crop, prompt):
        self.prompts.append(prompt)
        return crop, {}


def _cfg(advance_policy=None):
    cfg = {
        "generation": {
            "max_new_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "group_size": 3,
            "sample_micro_batch": None,
        },
        "rollout": {
            "upscale": 2,
            "rec_num": 1,
            "scales": [1],
        },
    }
    if advance_policy is not None:
        cfg["rollout"]["advance_policy"] = advance_policy
    return cfg


def _fake_process_state(processor, messages, device):
    return _FakeInputs(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 1), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )


def _make_rollout(monkeypatch, advance_policy=None):
    monkeypatch.setattr(rollout, "process_state", _fake_process_state)
    monkeypatch.setattr(
        rollout.Rollout,
        "_anchor_caption",
        lambda self, x0, base_path: "anchor",
    )
    monkeypatch.setattr(
        rollout.Rollout,
        "_completion_logprobs",
        lambda self, seq_ids, prompt_len, inputs: torch.zeros(
            seq_ids.shape[1] - prompt_len,
            dtype=torch.float32,
        ),
    )
    sr = _FakeSR()
    runner = rollout.Rollout(
        _cfg(advance_policy),
        _FakePolicy(),
        _FakeProcessor(),
        sr,
        _FakeRewards(),
        "cpu",
    )
    return runner, sr


def test_group_has_exactly_group_size_completions(monkeypatch):
    runner, _ = _make_rollout(monkeypatch, advance_policy="sampled")

    groups = runner.run_episode(Image.new("RGB", (8, 8)), "img.png")

    assert len(groups) == 1
    assert len(groups[0].completions) == 3
    assert all(completion.old_logp is not None for completion in groups[0].completions)


def test_advance_policy_best_picks_argmax_reward(monkeypatch):
    runner, sr = _make_rollout(monkeypatch, advance_policy="best")

    runner.run_episode(Image.new("RGB", (8, 8)), "img.png")

    assert sr.prompts == ["p1"]


def test_advance_policy_sampled_picks_tracked_sampled_index(monkeypatch):
    runner, sr = _make_rollout(monkeypatch, advance_policy="sampled")

    groups = runner.run_episode(Image.new("RGB", (8, 8)), "img.png")

    group = groups[0]

    assert group.sampled_idx == 0
    assert group.completions[group.sampled_idx].text == "p0"
    assert sr.prompts == ["p0"]


def test_advance_sampled_default(monkeypatch):
    runner, sr = _make_rollout(monkeypatch)

    runner.run_episode(Image.new("RGB", (8, 8)), "img.png")

    assert runner.advance_policy == "sampled"
    assert sr.prompts == ["p0"]
