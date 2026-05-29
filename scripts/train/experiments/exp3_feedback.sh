#!/usr/bin/env bash
# exp3 — baseline + R_fb (SR-output feedback: pyiqa quality + CLIP consistency).
# Loads the frozen SR backbone. On HW=4090 this needs 3 GPUs; on a6000/pro6000
# everything shares one card. Device placement comes from the HW profile:
#   bash scripts/train/experiments/exp3_feedback.sh            # HW=4090 (default)
#   HW=a6000   bash scripts/train/experiments/exp3_feedback.sh
#   HW=pro6000 bash scripts/train/experiments/exp3_feedback.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME="${NAME:-exp3_feedback}"
NEED_SR=1   # loads the frozen SR backbone -> 3 cards on HW=4090

run_grpo \
  rewards.r_anc.enabled=false \
  rewards.r_rep.enabled=false \
  rewards.r_fb.enabled=true \
  rewards.r_crit.enabled=true \
  rewards.r_phr.enabled=true
