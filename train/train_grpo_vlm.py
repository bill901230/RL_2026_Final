"""Entry point: GRPO fine-tuning of the Chain-of-Zoom VLM prompt extractor.

    python -m train.train_grpo_vlm --config train/configs/grpo_default.yaml [overrides...]

Heavy reward sub-models are built only if the corresponding reward is enabled,
so a text-only smoke run (R_anc + R_rep) loads neither the SR backbone nor the
critic VLM.
"""
import argparse
import os
import sys

import yaml
import torch

# make repo root importable (osediff_sd3, inference_coz, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset.zoom_dataset import ZoomBaseImageDataset
from train.grpo.rewards import RewardOrchestrator
from train.grpo.rollout import Rollout
from train.grpo.trainer import GRPOTrainer


def parse_overrides(pairs):
    """Allow `--set a.b=c` dotted overrides on top of the YAML."""
    out = {}
    for p in pairs:
        key, _, val = p.partition("=")
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        out[key] = val
    return out


def apply_override(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides a.b=c")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in parse_overrides(args.set).items():
        apply_override(cfg, k, v)

    torch.manual_seed(cfg["seed"])

    rcfg = cfg["rewards"]
    need_clip = (rcfg["r_anc"]["enabled"] or rcfg["r_rep"]["enabled"]
                 or rcfg["r_fb"]["enabled"]
                 or (rcfg["r_crit"]["enabled"] and rcfg["r_crit"]["backend"] == "clipscore"))
    need_metrics = rcfg["r_fb"]["enabled"]
    need_sr = rcfg["r_fb"]["enabled"]
    need_critic = rcfg["r_crit"]["enabled"]

    clip = metrics = sr_backbone = critic = None
    if need_clip:
        from train.grpo.text_sim import ClipEmbedder
        clip = ClipEmbedder(device=cfg["device_critic"])
    if need_metrics:
        from train.grpo.metrics import IQAMetrics
        metrics = IQAMetrics([rcfg["r_fb"]["quality_metric"]], device=cfg["device_sr"])
    if need_sr:
        from train.grpo.sr_env import FrozenSRBackbone
        sr_backbone = FrozenSRBackbone(cfg, device=cfg["device_sr"])
    if need_critic:
        from train.grpo.critic import Critic
        critic = Critic(cfg, clip_embedder=clip, device=cfg["device_critic"])

    rewards = RewardOrchestrator(
        cfg, clip=clip, metrics=metrics, sr_backbone=sr_backbone, critic=critic
    )
    dataset = ZoomBaseImageDataset(
        cfg["dataset"]["train_txt"], process_size=cfg["rollout"]["process_size"]
    )
    trainer = GRPOTrainer(cfg, rewards, Rollout, dataset, sr_backbone)
    trainer.train()


if __name__ == "__main__":
    main()
