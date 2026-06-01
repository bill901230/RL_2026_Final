from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Protocol, cast

import pytest
import torch
from PIL import Image


class _SRBackbone(Protocol):
    def render(self, crop_pil: Image.Image, prompt: str) -> tuple[Image.Image, torch.Tensor]: ...


class _SRBackboneClass(Protocol):
    def __call__(self, cfg: object, device: str) -> _SRBackbone: ...


class _SREnvModule(Protocol):
    FrozenSRBackbone: _SRBackboneClass


def _install_osediff_stub(monkeypatch: pytest.MonkeyPatch) -> _SREnvModule:
    class _FrozenPart:
        device: object | None = None
        requires_grad: bool | None = None

        def to(self, device: object) -> "_FrozenPart":
            self.device = device
            return self

        def requires_grad_(self, requires_grad: bool) -> "_FrozenPart":
            self.requires_grad = requires_grad
            return self

    class _FakeSD3Euler:
        device: object
        text_enc_1: _FrozenPart
        text_enc_2: _FrozenPart
        text_enc_3: _FrozenPart
        transformer: _FrozenPart
        vae: _FrozenPart

        def __init__(self, device: object = "cuda") -> None:
            self.device = device
            self.text_enc_1 = _FrozenPart()
            self.text_enc_2 = _FrozenPart()
            self.text_enc_3 = _FrozenPart()
            self.transformer = _FrozenPart()
            self.vae = _FrozenPart()

    class _FakeOSEDiffSD3Test:
        args: object
        model: object

        def __init__(self, args: object, model: object) -> None:
            self.args = args
            self.model = model

        def __call__(self, lq: torch.Tensor, prompt: str) -> torch.Tensor:
            assert prompt == "fine texture"
            assert lq.ndim == 4
            assert lq.shape[1] == 3
            assert lq.device.type == "cpu"
            assert float(lq.amin()) >= -1.0
            assert float(lq.amax()) <= 1.0
            return torch.full_like(lq, 0.25)

    stub = types.ModuleType("osediff_sd3")
    setattr(stub, "SD3Euler", _FakeSD3Euler)
    setattr(stub, "OSEDiff_SD3_TEST", _FakeOSEDiffSD3Test)
    monkeypatch.setitem(sys.modules, "osediff_sd3", stub)
    _ = sys.modules.pop("train.grpo.sr_env", None)
    module = cast(object, importlib.import_module("train.grpo.sr_env"))
    return cast(_SREnvModule, module)


def test_sr_env_render_returns_unit_interval_float_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sr_env = _install_osediff_stub(monkeypatch)
    cfg = {
        "model": {
            "sr_lora_path": "ckpt/SR_LoRA/model_20001.pkl",
            "sr_vae_path": "ckpt/SR_VAE/vae_encoder_20001.pt",
            "sr_lora_rank": 4,
        }
    }
    crop = Image.new("RGB", (16, 16), color=(64, 128, 192))

    sr_pil, sr01 = sr_env.FrozenSRBackbone(cfg, device="cpu").render(crop, "fine texture")

    assert isinstance(sr_pil, Image.Image)
    assert sr_pil.size == crop.size
    assert isinstance(sr01, torch.Tensor)
    assert torch.is_floating_point(sr01)
    assert sr01.ndim in (3, 4)
    channel_dim = 0 if sr01.ndim == 3 else 1
    assert sr01.shape[channel_dim] == 3
    assert float(sr01.amin()) >= 0.0
    assert float(sr01.amax()) <= 1.0


def test_zoom_dataset_yields_512_rgb_pil_from_valid_manifest() -> None:
    from train.dataset.zoom_dataset import ZoomBaseImageDataset

    dataset = ZoomBaseImageDataset("data/manifests/valid.txt", process_size=512, limit=1)

    item = dataset[0]
    image = item["image"]
    path = item["path"]

    assert set(item) == {"image", "path"}
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (512, 512)
    assert isinstance(path, str)
    assert Path(path).is_file()
