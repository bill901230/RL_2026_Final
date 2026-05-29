#!/usr/bin/env bash
# Cheap end-to-end smoke test: text-only rewards (no SR backbone, no critic),
# tiny group, 5 steps, 1 GPU, no wandb. Confirms rollout -> reward -> GRPO step
# -> checkpoint all work before launching a real experiment.
#   bash scripts/train/experiments/smoke.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
GPUS="${GPUS:-0}"
NAME="${NAME:-smoke}"

run_grpo \
  rewards.r_fb.enabled=false \
  rewards.r_crit.enabled=false \
  rewards.r_phr.enabled=false \
  rewards.r_anc.enabled=true \
  rewards.r_rep.enabled=true \
  generation.group_size=2 \
  rollout.images_per_step=1 \
  optim.max_steps=5 \
  logging.ckpt_every=5 \
  eval.enabled=false \
  logging.wandb=false \
  device_policy=cuda:0 \
  device_critic=cuda:0
