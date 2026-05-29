#!/usr/bin/env bash
# exp4 — FULL proposed method: all five rewards (R_anc + R_rep + R_fb + R_crit + R_phr).
# Loads the frozen SR backbone + critic. On HW=4090 this needs 3 GPUs; on
# a6000/pro6000 everything shares one card. Device placement comes from the HW profile:
#   bash scripts/train/experiments/exp4_full.sh                # HW=4090 (default)
#   HW=a6000   bash scripts/train/experiments/exp4_full.sh
#   HW=pro6000 bash scripts/train/experiments/exp4_full.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME="${NAME:-exp4_full}"
NEED_SR=1   # loads the frozen SR backbone -> 3 cards on HW=4090

run_grpo \
  rewards.r_anc.enabled=true \
  rewards.r_rep.enabled=true \
  rewards.r_fb.enabled=true \
  rewards.r_crit.enabled=true \
  rewards.r_phr.enabled=true
