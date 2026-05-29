import importlib
import sys
import types

import pytest
import torch


try:
    state = importlib.import_module("train.grpo.state")
except ModuleNotFoundError as exc:
    if exc.name != "qwen_vl_utils":
        raise
    qwen_stub = types.ModuleType("qwen_vl_utils")
    setattr(qwen_stub, "process_vision_info", lambda messages: ([], []))
    sys.modules["qwen_vl_utils"] = qwen_stub
    state = importlib.import_module("train.grpo.state")


def _image_payloads(messages):
    user_content = messages[1]["content"]
    return [item["image"] for item in user_content if item["type"] == "image"]


@pytest.mark.parametrize(
    ("x_prev2", "expected_images"),
    [
        pytest.param(None, ["x0", "x_prev1"], id="without-ar2-state"),
        pytest.param("x_prev2", ["x0", "x_prev2", "x_prev1"], id="with-ar2-state"),
    ],
)
def test_build_messages_orders_non_none_state_images(x_prev2, expected_images):
    messages = state.build_messages("x0", x_prev2, "x_prev1", scale_factor=4)

    assert _image_payloads(messages) == expected_images
    assert len(_image_payloads(messages)) == sum(
        image is not None for image in ("x0", x_prev2, "x_prev1")
    )


def test_scale_token_required():
    messages = state.build_messages("x0", "x_prev2", "x_prev1", scale_factor=8)

    assert "zoom factor=8x" in messages[0]["content"]
    assert "{scale}" not in messages[0]["content"]


class _FakeBatch(dict[str, torch.Tensor]):
    device: str | None = None

    def to(self, device: str) -> "_FakeBatch":
        self.device = device
        return self


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert tokenize is False
        assert add_generation_prompt is True
        return "<chat>"

    def __call__(self, text, images, videos, padding, return_tensors):
        assert text == ["<chat>"]
        assert images == ["x0", "x_prev1"]
        assert videos == []
        assert padding is True
        assert return_tensors == "pt"
        return _FakeBatch(
            {
                "input_ids": torch.zeros((1, 5), dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "pixel_values": torch.zeros((2, 3, 14, 14), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 14, 14], [1, 14, 14]]),
            }
        )


def test_process_state_returns_qwen_processor_batch_contract(monkeypatch):
    messages = state.build_messages("x0", None, "x_prev1", scale_factor=2)

    def fake_process_vision_info(seen_messages):
        assert seen_messages is messages
        return ["x0", "x_prev1"], []

    monkeypatch.setattr(state, "process_vision_info", fake_process_vision_info)

    inputs = state.process_state(_FakeProcessor(), messages, device="cuda:0")

    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }
    assert inputs["input_ids"].shape == (1, 5)
    assert inputs["attention_mask"].shape == (1, 5)
    assert inputs["pixel_values"].shape == (2, 3, 14, 14)
    assert inputs["image_grid_thw"].shape == (2, 3)
    assert inputs.device == "cuda:0"
