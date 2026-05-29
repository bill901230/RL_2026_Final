# pyright: reportMissingImports=false
import math
from typing import Any

import pytest
from PIL import Image

from train.grpo.rewards import RewardContext, RewardOrchestrator, _RunningNorm


REWARD_KEYS = ("r_anc", "r_rep", "r_fb", "r_crit", "r_phr")
EPS = 1.0e-6


def _cfg(*, normalize=True, normalization_mode=None, weights=None) -> dict[str, Any]:
    weights = weights or {}
    rewards: dict[str, Any] = {"normalize": normalize}
    if normalization_mode is not None:
        rewards["normalization_mode"] = normalization_mode

    for key in REWARD_KEYS:
        rewards[key] = {"enabled": True, "weight": weights.get(key, 1.0)}

    rewards["r_rep"]["ngram"] = 2
    rewards["r_fb"].update({"quality_metric": "musiq", "consistency_weight": 0.0})
    rewards["r_phr"]["fillers"] = []
    return {"rewards": rewards, "generation": {"group_size": 4}}


def _ctx():
    return RewardContext(Image.new("RGB", (1, 1)), x0_caption="anchor")


def _orchestrator(values, *, normalize=True, normalization_mode=None, weights=None):
    orchestrator = RewardOrchestrator(
        _cfg(normalize=normalize, normalization_mode=normalization_mode, weights=weights)
    )
    for key, fn_name in RewardOrchestrator._FNS.items():
        setattr(orchestrator, fn_name, lambda prompt, ctx, key=key: values[prompt][key])
    return orchestrator


def _standardized_uses(values, prompts):
    raws = [values[prompt] for prompt in prompts]
    uses = [{key: 0.0 for key in REWARD_KEYS} for _ in prompts]
    for key in REWARD_KEYS:
        column = [raw[key] for raw in raws]
        mean = sum(column) / len(column)
        var = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(var)
        if std <= EPS:
            continue
        for use, raw in zip(uses, raws):
            use[key] = (raw[key] - mean) / (std + EPS)
    return uses


def _expected_totals(values, prompts, weights):
    return [
        sum(weights.get(key, 1.0) * use[key] for key in REWARD_KEYS)
        for use in _standardized_uses(values, prompts)
    ]


def test_weight_application_matches_weighted_normalized_components():
    prompts = ["p0", "p1", "p2", "p3"]
    weights = {"r_anc": 2.0, "r_rep": 0.5, "r_fb": 1.5, "r_crit": 0.25, "r_phr": 0.75}
    values = {
        "p0": {"r_anc": 1.0, "r_rep": 20.0, "r_fb": -3.0, "r_crit": 0.1, "r_phr": 0.0},
        "p1": {"r_anc": 2.0, "r_rep": 10.0, "r_fb": 4.0, "r_crit": 0.4, "r_phr": -1.0},
        "p2": {"r_anc": 4.0, "r_rep": 5.0, "r_fb": 9.0, "r_crit": 0.8, "r_phr": -2.0},
        "p3": {"r_anc": 8.0, "r_rep": 1.0, "r_fb": 16.0, "r_crit": 1.6, "r_phr": -3.0},
    }
    orchestrator = _orchestrator(values, weights=weights)

    results = orchestrator.compute_group(prompts, _ctx())
    expected = _expected_totals(values, prompts, weights)

    assert [raw for _, raw in results] == [values[prompt] for prompt in prompts]
    assert [total for total, _ in results] == pytest.approx(expected, abs=1e-12)


def test_per_group_normalization_balances_mixed_scale_components():
    prompts = ["low", "mid_low", "mid_high", "high"]
    values = {
        "low": {"r_anc": 1_000.0, "r_rep": 0.001, "r_fb": -40.0, "r_crit": -2.0, "r_phr": 0.0},
        "mid_low": {"r_anc": 2_000.0, "r_rep": 0.002, "r_fb": -10.0, "r_crit": -1.0, "r_phr": 1.0},
        "mid_high": {"r_anc": 3_000.0, "r_rep": 0.003, "r_fb": 10.0, "r_crit": 1.0, "r_phr": 0.0},
        "high": {"r_anc": 4_000.0, "r_rep": 0.004, "r_fb": 40.0, "r_crit": 2.0, "r_phr": 1.0},
    }
    orchestrator = _orchestrator(values)

    results = orchestrator.compute_group(prompts, _ctx())
    assert [total for total, _ in results] == pytest.approx(
        _expected_totals(values, prompts, {}), abs=1e-12
    )

    for use in _standardized_uses(values, prompts):
        contributions = [abs(use[key]) for key in REWARD_KEYS]
        total_abs = sum(contributions)
        assert max(contributions) / total_abs < 0.5


def test_global_norm_is_order_dependent():
    prompts = ["low", "mid", "high"]
    reversed_prompts = list(reversed(prompts))
    values = {
        "low": {key: 0.0 for key in REWARD_KEYS},
        "mid": {key: 10.0 for key in REWARD_KEYS},
        "high": {key: 20.0 for key in REWARD_KEYS},
    }

    running_a = _orchestrator(values, normalization_mode="running")
    running_b = _orchestrator(values, normalization_mode="running")
    running_by_prompt_a = {
        prompt: running_a.compute(prompt, _ctx())[0] for prompt in prompts
    }
    running_by_prompt_b = {
        prompt: running_b.compute(prompt, _ctx())[0] for prompt in reversed_prompts
    }
    assert running_by_prompt_a["mid"] != pytest.approx(running_by_prompt_b["mid"])

    per_group_a = _orchestrator(values)
    per_group_b = _orchestrator(values)
    grouped_a = per_group_a.compute_group(prompts, _ctx())
    grouped_b = per_group_b.compute_group(reversed_prompts, _ctx())
    by_prompt_a = dict(zip(prompts, [total for total, _ in grouped_a]))
    by_prompt_b = dict(zip(reversed_prompts, [total for total, _ in grouped_b]))

    assert by_prompt_a == pytest.approx(by_prompt_b, abs=1e-12)


def test_running_norm_state_can_reset_and_restore():
    norm = _RunningNorm()
    for value in (1.0, 2.0, 4.0):
        norm.update(value)

    state = norm.state_dict()
    restored = _RunningNorm()
    restored.load_state_dict(state)

    assert restored.normalize(3.0) == pytest.approx(norm.normalize(3.0))

    norm.reset()
    assert norm.state_dict() == {"n": 0, "mean": 0.0, "m2": 0.0}
