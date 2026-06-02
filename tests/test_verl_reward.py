import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeClip:
    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            low = text.lower()
            rows.append(
                [
                    float("red" in low or "bird" in low),
                    float("blue" in low or "sky" in low or "mountain" in low),
                    1.0,
                ]
            )
        feats = torch.tensor(rows, dtype=torch.float32)
        return torch.nn.functional.normalize(feats, dim=-1)

    @staticmethod
    def cosine(a, b):
        return ((a @ b.t()).clamp(-1.0, 1.0) + 1.0) / 2.0


def _fresh_reward(monkeypatch):
    sys.modules.pop("train.grpo.critic", None)
    sys.modules.pop("train.grpo.metrics", None)
    module = importlib.import_module("verl_custom_reward")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "_CLIP_EMBEDDER", FakeClip())
    monkeypatch.setattr(module, "_ORCHESTRATOR", None)
    return module


def test_compute_score_returns_finite_float(monkeypatch):
    reward = _fresh_reward(monkeypatch)
    score = reward.compute_score(
        data_source="coz",
        solution_str="blue sky mountain ridges crisp horizon bright clouds distant valley detail",
        ground_truth="",
        extra_info={"x0_caption": "blue sky mountain", "prev_prompt": "red bird", "scale": 2},
    )
    assert isinstance(score, float)
    assert torch.isfinite(torch.tensor(score))


def test_r_rep_penalizes_repetition(monkeypatch):
    reward = _fresh_reward(monkeypatch)
    repeated = reward.compute_score(
        "coz",
        "red bird red bird red bird feather beak wing detail texture",
        "",
        {"x0_caption": "red bird", "prev_prompt": "red bird", "scale": 2},
    )
    novel = reward.compute_score(
        "coz",
        "blue sky mountain ridges bright clouds distant valley crisp detail",
        "",
        {"x0_caption": "red bird", "prev_prompt": "red bird", "scale": 2},
    )
    assert novel > repeated


def test_no_sr_or_critic_import(monkeypatch):
    reward = _fresh_reward(monkeypatch)
    reward.compute_score(
        "coz",
        "blue sky mountain ridges bright clouds distant valley crisp detail",
        "",
        {"x0_caption": "blue sky", "prev_prompt": "red bird", "scale": 1},
    )
    assert "train.grpo.critic" not in sys.modules
    assert "train.grpo.metrics" not in sys.modules


def test_validity_guard_makes_short_and_cjk_completions_worst(monkeypatch):
    reward = _fresh_reward(monkeypatch)
    extra = {"x0_caption": "red bird", "prev_prompt": "red bird", "scale": 2, "image_id": "toy"}
    healthy = "crisp feather barbs golden rim highlights layered wing texture natural shadow detail edges"
    short = "red bird dog"
    cjk = "新规发育"

    guarded = reward.compute_score(
        data_sources=["coz", "coz", "coz"],
        solution_strs=[healthy, short, cjk],
        ground_truths=["", "", ""],
        extra_infos=[extra, extra, extra],
    )

    assert isinstance(guarded, list)
    healthy_rec, short_rec, cjk_rec = guarded
    worst_r_rep = min(record["raw_r_rep"] for record in guarded)
    assert short_rec["validity_invalid"] == 1
    assert cjk_rec["validity_invalid"] == 1
    assert short_rec["raw_r_rep"] == worst_r_rep
    assert cjk_rec["raw_r_rep"] == worst_r_rep
    assert short_rec["raw_validity_total"] == -2.0
    assert cjk_rec["raw_validity_total"] == -2.0
    assert short_rec["score"] == -2.0
    assert cjk_rec["score"] == -2.0
    assert healthy_rec["validity_invalid"] == 0

    monkeypatch.setenv("COZ_VALIDITY_GUARD", "0")
    unguarded = reward.compute_score(
        data_sources=["coz"],
        solution_strs=[healthy],
        ground_truths=[""],
        extra_infos=[extra],
    )
    assert isinstance(unguarded, list)
    assert healthy_rec["raw_r_rep"] == unguarded[0]["raw_r_rep"]
    assert healthy_rec["raw_r_anc"] == unguarded[0]["raw_r_anc"]
    assert healthy_rec["raw_r_phr"] == unguarded[0]["raw_r_phr"]
