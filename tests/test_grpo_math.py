import importlib
import sys
import types

import pytest
import torch


def _stub_optional_dependency(name):
    module = types.ModuleType(name)
    if name == "transformers":
        setattr(module, "Qwen2_5_VLForConditionalGeneration", object)
        setattr(module, "AutoProcessor", object)
        setattr(module, "get_scheduler", lambda *args, **kwargs: None)
    elif name == "peft":
        setattr(module, "PeftModel", object)
        setattr(module, "LoraConfig", object)
    sys.modules[name] = module


def _import_trainer():
    while True:
        try:
            return importlib.import_module("train.grpo.trainer")
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformers", "peft"}:
                raise
            _stub_optional_dependency(exc.name)


trainer = _import_trainer()


def test_group_advantage_zero_mean_unit_scale():
    rewards = torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.float32)

    adv = trainer._group_advantages(rewards, adv_eps=0.0)

    assert adv.mean().item() == pytest.approx(0.0, abs=1.0e-7)
    assert adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1.0e-7)


def test_k3_kl_non_negative_and_zero_when_logprobs_match():
    new_logp = torch.tensor([-2.0, -0.5, -3.0], dtype=torch.float32)

    zero_kl = trainer._k3_kl(new_logp, new_logp)
    shifted_kl = trainer._k3_kl(
        new_logp,
        torch.tensor([-2.3, -0.1, -4.5], dtype=torch.float32),
    )

    assert torch.allclose(zero_kl, torch.zeros_like(zero_kl))
    assert torch.all(shifted_kl >= 0.0)


def test_clamp_bind_fraction_zero_without_cache():
    new_logp = torch.tensor([-0.2, -1.0, -1.7], dtype=torch.float32)
    old_logp = new_logp.clone()

    ratio = trainer._importance_ratio(new_logp, old_logp)
    bind_fraction = trainer._clamp_bind_fraction(ratio, clip_eps=0.2)

    assert torch.allclose(ratio, torch.ones_like(ratio))
    assert bind_fraction.item() == pytest.approx(0.0)


def test_clamp_can_bind_with_cached_sample_logp():
    old_logp = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
    new_logp = torch.tensor([-0.6, -1.0, -1.4], dtype=torch.float32)

    ratio = trainer._importance_ratio(new_logp, old_logp)
    bind_fraction = trainer._clamp_bind_fraction(ratio, clip_eps=0.2)

    assert ratio.std(unbiased=False).item() > 0.0
    assert bind_fraction.item() == pytest.approx(2.0 / 3.0)
