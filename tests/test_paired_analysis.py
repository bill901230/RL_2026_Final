"""TDD spec for the upgraded paired ablation analysis (scripts/abl_paired_analysis.py)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_rel

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "abl_paired_analysis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("abl_paired_analysis", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


apa = _load_module()


def _frames(base_vals, arm_vals, axis="vlm_grounding_deep"):
    n = len(base_vals)
    ids = [f"{i:04d}" for i in range(1, n + 1)]
    base_df = pd.DataFrame({"image": ids, axis: list(base_vals)}).set_index("image")
    arm_df = pd.DataFrame({"image": ids, axis: list(arm_vals)}).set_index("image")
    return base_df, arm_df


def test_paired_stats_pvalue_and_tstat_match_scipy():
    axis = "vlm_grounding_deep"
    base_vals = [7.0, 8.0, 6.5, 9.0, 7.5, 8.5, 6.0, 7.0, 8.0, 7.5]
    arm_vals = [7.5, 8.5, 7.0, 9.0, 8.5, 9.0, 6.5, 8.0, 8.0, 8.5]
    base_df, arm_df = _frames(base_vals, arm_vals, axis)

    s = apa.paired_stats(base_df, arm_df, axis)

    ref = ttest_rel(np.asarray(arm_vals, float), np.asarray(base_vals, float))
    assert s["t_stat"] == pytest.approx(float(ref.statistic), abs=1e-6)
    assert s["p_value"] == pytest.approx(float(ref.pvalue), abs=1e-6)


def test_mean_delta_and_sign_match():
    axis = "unique_token_ratio"
    base_vals = [0.40, 0.50, 0.45, 0.60, 0.50]
    arm_vals = [0.50, 0.55, 0.50, 0.65, 0.60]
    base_df, arm_df = _frames(base_vals, arm_vals, axis)

    s = apa.paired_stats(base_df, arm_df, axis)

    expected = float(np.mean(np.asarray(arm_vals) - np.asarray(base_vals)))
    assert s["mean_delta"] == pytest.approx(expected, abs=1e-9)
    assert s["mean_delta"] > 0
    assert s["t_stat"] > 0
    assert math.copysign(1.0, s["t_stat"]) == math.copysign(1.0, s["mean_delta"])


def test_negative_delta_keeps_sign_and_matches_scipy():
    axis = "vlm_grounding_deep"
    base_vals = [8.0, 9.0, 7.5, 8.5, 9.0, 8.0]
    arm_vals = [7.0, 8.0, 7.0, 8.0, 8.0, 7.5]
    base_df, arm_df = _frames(base_vals, arm_vals, axis)

    s = apa.paired_stats(base_df, arm_df, axis)

    assert s["mean_delta"] < 0
    assert s["t_stat"] < 0
    ref = ttest_rel(np.asarray(arm_vals, float), np.asarray(base_vals, float))
    assert s["p_value"] == pytest.approx(float(ref.pvalue), abs=1e-6)
    assert math.copysign(1.0, s["t_stat"]) == math.copysign(1.0, s["mean_delta"])


def test_bootstrap_ci_brackets_mean_delta_and_reproducible():
    axis = "vlm_grounding_all"
    base_vals = [8.0, 8.5, 7.5, 9.0, 8.0, 7.0, 8.5, 9.0, 7.5, 8.0, 8.5, 7.0]
    arm_vals = [8.5, 8.8, 8.3, 9.2, 8.6, 7.4, 9.2, 9.1, 8.4, 8.3, 9.0, 7.4]
    base_df, arm_df = _frames(base_vals, arm_vals, axis)

    s = apa.paired_stats(base_df, arm_df, axis)

    assert s["ci_low"] <= s["mean_delta"] <= s["ci_high"]
    assert s["ci_low"] < s["ci_high"]
    s2 = apa.paired_stats(base_df, arm_df, axis)
    assert s["ci_low"] == pytest.approx(s2["ci_low"])
    assert s["ci_high"] == pytest.approx(s2["ci_high"])


def test_bootstrap_ci_function_reproducible_and_sane():
    delta = np.array([0.10, 0.20, -0.10, 0.30, 0.0, 0.15, 0.25, -0.05])
    lo1, hi1 = apa.bootstrap_ci(delta, B=3000)
    lo2, hi2 = apa.bootstrap_ci(delta, B=3000)
    assert lo1 == pytest.approx(lo2)
    assert hi1 == pytest.approx(hi2)
    assert lo1 < hi1
    assert lo1 <= float(np.mean(delta)) <= hi1


def test_paired_stats_keeps_existing_fields():
    axis = "vlm_grounding_deep"
    base_df, arm_df = _frames([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 4.0, 3.0], axis)
    s = apa.paired_stats(base_df, arm_df, axis)
    for key in ("axis", "n", "mean_delta", "std_delta", "wins", "losses",
                "ties", "win_rate", "loss_rate", "tie_rate", "t_read"):
        assert key in s
    for key in ("t_stat", "p_value", "ci_low", "ci_high"):
        assert key in s
    assert s["n"] == 4


def test_parse_args_arm_mapping_custom():
    args = apa.parse_args([
        "--base", "control",
        "--arm", "control=results/control.csv",
        "--arm", "A_drgrpo=results/A_drgrpo.csv",
    ])
    cfg = apa.resolve_config(args)
    assert cfg.is_default is False
    assert cfg.base_label == "control"
    assert cfg.arm_files["control"].endswith("results/control.csv")
    assert cfg.arm_files["A_drgrpo"].endswith("results/A_drgrpo.csv")
    assert "A_drgrpo" in cfg.paired_arms
    assert "control" not in cfg.paired_arms


def test_parse_args_no_args_reproduces_default_ladder():
    args = apa.parse_args([])
    cfg = apa.resolve_config(args)
    assert cfg.is_default is True
    assert cfg.base_label == "BASE"
    assert cfg.paired_arms == apa.PAIRED_ARMS
    assert cfg.ladder_order == apa.LADDER_ORDER
    assert set(apa.ARM_FILES).issubset(set(cfg.arm_files))


def test_resolve_config_rejects_base_not_in_arms():
    args = apa.parse_args(["--base", "missing", "--arm", "A=results/A.csv"])
    with pytest.raises((ValueError, KeyError, SystemExit)):
        apa.resolve_config(args)
