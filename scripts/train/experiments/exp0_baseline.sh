#!/usr/bin/env bash
# exp0 — BASELINE: original Chain-of-Zoom paper reward (R_crit + R_phr only).
# Our three contributions (R_anc / R_rep / R_fb) are OFF. No SR backbone loaded.
# Hardware/device placement comes from the HW profile (see _common.sh):
#   bash scripts/train/experiments/exp0_baseline.sh            # HW=4090 (default)
#   HW=a6000   bash scripts/train/experiments/exp0_baseline.sh
#   HW=pro6000 bash scripts/train/experiments/exp0_baseline.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME="${NAME:-exp0_baseline}"

run_grpo \
  rewards.r_anc.enabled=false \
  rewards.r_rep.enabled=false \
  rewards.r_fb.enabled=false \
  rewards.r_crit.enabled=true \
  rewards.r_phr.enabled=true
